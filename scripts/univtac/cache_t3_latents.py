#!/usr/bin/env python3
"""Precompute frozen T3-large latents for the SimpleTac world model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

# Prefer this worktree over any older editable RLinf installation.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlinf.data.datasets.openpi_pytorch.univtac.univtac_sft_dataset import (
    TASK_PROMPTS,
    _decode_rgb,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_cache_manifest import (
    T3_WORLD_CACHE_INCOMPLETE,
    T3_WORLD_CACHE_MANIFEST,
    build_t3_world_cache_manifest,
    exclusive_t3_world_cache_lock,
    load_t3_world_cache_manifest,
    validate_t3_world_cache_files,
    validate_t3_world_cache_manifest,
    validate_t3_world_cache_sources,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_preprocess_profile import (
    T3_PREPROCESS_PROFILES,
    resolve_t3_preprocess_profile,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_tactile_encoder import (
    T3_LARGE_WIDTH,
    T3LargeTactileEncoder,
)

_BASE_EPISODE_DATASETS = ("embodiment/joint",)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_PROMPTS), default="insert_hole")
    parser.add_argument(
        "--data-root",
        required=True,
    )
    parser.add_argument(
        "--output-root",
        required=True,
    )
    parser.add_argument(
        "--tactile-encoder",
        required=True,
    )
    parser.add_argument(
        "--tactile-preprocess-profile",
        choices=T3_PREPROCESS_PROFILES,
        required=True,
        help=(
            "Exact T3 normalization profile. Use legacy_imagenet_v1 only to "
            "validate/regenerate a verified pre-v11 cache."
        ),
    )
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=49)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--pool-grid",
        type=int,
        default=14,
        help="14 keeps all T3 patch tokens; smaller values enable pooling",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--validate-samples",
        type=int,
        default=4,
        help="Re-encode this many evenly spaced frames per episode",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Verify manifest, shapes, and sampled online/cache equality without writing"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _decode_batch(
    file: h5py.File,
    start: int,
    end: int,
    tactile_datasets: tuple[str, str],
) -> torch.Tensor:
    sensors = []
    for key in tactile_datasets:
        frames = [_decode_rgb(file[key][index]) for index in range(start, end)]
        sensors.append(np.stack(frames))
    images = torch.from_numpy(np.stack(sensors, axis=1))
    images = images.to(torch.float32) / 255.0 * 2.0 - 1.0
    images = images.permute(0, 1, 4, 2, 3)
    # The encoder owns the resolved resize/crop/normalization profile so online
    # and cached paths cannot silently diverge.
    return images


def _inspect_episode(episode_path: Path, tactile_datasets: tuple[str, str]) -> int:
    """Validate tactile/state alignment; policy camera requirements are separate."""
    with h5py.File(episode_path, "r") as file:
        episode_datasets = (*_BASE_EPISODE_DATASETS, *tactile_datasets)
        missing = [key for key in episode_datasets if key not in file]
        if missing:
            raise KeyError(
                f"UniVTAC episode {episode_path} is missing required streams: {missing}"
            )
        lengths = {key: int(file[key].shape[0]) for key in episode_datasets}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"UniVTAC episode {episode_path} has inconsistent stream "
                f"lengths: {lengths}"
            )
        length = next(iter(lengths.values()))
        if length <= 0:
            raise ValueError(f"UniVTAC episode {episode_path} is empty")
        joint = file["embodiment/joint"]
        if joint.ndim != 2 or joint.shape[1] < 8:
            raise ValueError(
                f"UniVTAC episode {episode_path} requires joint shape "
                f"[T, >=8], got {joint.shape}"
            )
    return length


@torch.inference_mode()
def _cache_episode(
    encoder: T3LargeTactileEncoder,
    episode_path: Path,
    output_path: Path,
    *,
    batch_size: int,
    pool_grid: int,
    device: torch.device,
    overwrite: bool,
    tactile_datasets: tuple[str, str],
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path}", flush=True)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.npy")
    with h5py.File(episode_path, "r") as file:
        length = int(file[tactile_datasets[0]].shape[0])
        tokens_per_sensor = 1 + pool_grid**2
        output = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=np.float16,
            shape=(length, 2, tokens_per_sensor, T3_LARGE_WIDTH),
        )
        for start in range(0, length, batch_size):
            end = min(start + batch_size, length)
            images = _decode_batch(file, start, end, tactile_datasets).to(
                device, non_blocking=True
            )
            latents = encoder.encode_world_tokens(images, pool_grid=pool_grid)
            output[start:end] = latents.float().cpu().numpy().astype(np.float16)
        output.flush()
        del output
    os.replace(temporary_path, output_path)
    print(f"[done] {output_path} frames={length}", flush=True)


@torch.inference_mode()
def _validate_episode(
    encoder: T3LargeTactileEncoder | None,
    episode_path: Path,
    output_path: Path,
    *,
    pool_grid: int,
    device: torch.device | None,
    sample_count: int,
    tactile_datasets: tuple[str, str],
) -> None:
    cached = np.load(output_path, mmap_mode="r")
    with h5py.File(episode_path, "r") as file:
        length = int(file[tactile_datasets[0]].shape[0])
        expected_shape = (
            length,
            2,
            1 + pool_grid**2,
            T3_LARGE_WIDTH,
        )
        if cached.shape != expected_shape or cached.dtype != np.float16:
            raise ValueError(
                f"Invalid T3 cache {output_path}: shape/dtype "
                f"{cached.shape}/{cached.dtype}, expected {expected_shape}/float16"
            )
        for start in range(0, length, 32):
            if not np.isfinite(cached[start : start + 32]).all():
                raise ValueError(f"T3 cache contains non-finite values: {output_path}")
        if sample_count <= 0 or length == 0:
            print(f"[verified] {output_path} finite=True", flush=True)
            return
        if encoder is None or device is None:
            raise RuntimeError(
                "Online cache comparison requires a loaded T3 encoder and device"
            )
        indices = np.linspace(
            0,
            length - 1,
            min(sample_count, length),
            dtype=np.int64,
        )
        frames = [
            _decode_batch(file, int(index), int(index) + 1, tactile_datasets)[0]
            for index in indices
        ]
        images = torch.stack(frames).to(device, non_blocking=True)
    online = encoder.encode_world_tokens(images, pool_grid=pool_grid)
    online = online.float().cpu().numpy()
    cached_samples = np.asarray(cached[indices], dtype=np.float32)
    if not np.isfinite(online).all() or not np.isfinite(cached_samples).all():
        raise ValueError(f"Non-finite online/cache T3 latents found for {output_path}")
    max_error = float(np.max(np.abs(online - cached_samples)))
    if max_error > 0.02:
        raise ValueError(
            f"Cached/online T3 latent mismatch for {output_path}: "
            f"max_abs_error={max_error:.6f} > 0.02"
        )
    print(
        f"[verified] {output_path} samples={len(indices)} "
        f"max_abs_error={max_error:.6f}",
        flush=True,
    )


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_incomplete_marker(path: Path) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "Cache generation is in progress or the previous writer failed. "
        "Do not train from this directory.\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _create_encoder(
    tactile_encoder: str,
    *,
    device_name: str,
    preprocess_profile: str | None,
    task: str,
) -> tuple[T3LargeTactileEncoder, torch.device]:
    device = torch.device(device_name)
    encoder = T3LargeTactileEncoder(
        tactile_encoder,
        preprocess_profile=preprocess_profile,
        task=task,
    )
    encoder.requires_grad_(False)
    encoder = encoder.to(device=device, dtype=torch.bfloat16).eval()
    return encoder, device


def main() -> None:
    args = _parse_args()
    if args.episode_end < args.episode_start:
        raise ValueError("episode-end must be >= episode-start")
    if args.batch_size <= 0 or args.pool_grid <= 0 or args.validate_samples < 0:
        raise ValueError(
            "batch-size/pool-grid must be positive and validate-samples nonnegative"
        )
    if args.validate_only and args.overwrite:
        raise ValueError("--validate-only and --overwrite are mutually exclusive")
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    preprocess = resolve_t3_preprocess_profile(
        getattr(args, "tactile_preprocess_profile", None),
        task=args.task,
    )
    tactile_datasets = preprocess.source_fields
    task_root = output_root / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    episode_paths: dict[int, Path] = {}
    episode_lengths: dict[int, int] = {}
    for episode in range(args.episode_start, args.episode_end + 1):
        episode_path = data_root / args.task / "clean" / f"{episode}.hdf5"
        if not episode_path.is_file():
            raise FileNotFoundError(episode_path)
        episode_lengths[episode] = _inspect_episode(episode_path, tactile_datasets)
        episode_paths[episode] = episode_path

    expected_manifest = build_t3_world_cache_manifest(
        task=args.task,
        episode_lengths=episode_lengths,
        pool_grid=args.pool_grid,
        checkpoint_dir=args.tactile_encoder,
        episode_paths=episode_paths,
        preprocess_profile=getattr(args, "tactile_preprocess_profile", None),
    )
    manifest_path = task_root / T3_WORLD_CACHE_MANIFEST
    incomplete_path = task_root / T3_WORLD_CACHE_INCOMPLETE

    if args.validate_only:
        # A metadata-only validation deliberately avoids instantiating T3 or
        # loading its PyTorch weights. Shape/dtype, every cached value, and all
        # source/cache hashes are still checked below.
        existing_manifest = load_t3_world_cache_manifest(output_root, args.task)
        validate_t3_world_cache_manifest(
            existing_manifest,
            expected_manifest,
            manifest_path=manifest_path,
        )
        validate_t3_world_cache_sources(
            existing_manifest,
            episode_paths,
            verify_sha256=True,
        )
        encoder: T3LargeTactileEncoder | None = None
        device: torch.device | None = None
        if args.validate_samples > 0:
            encoder, device = _create_encoder(
                args.tactile_encoder,
                device_name=args.device,
                preprocess_profile=getattr(
                    args,
                    "tactile_preprocess_profile",
                    None,
                ),
                task=args.task,
            )
        for episode, episode_path in episode_paths.items():
            output_path = task_root / "clean" / f"{episode}.npy"
            if not output_path.is_file():
                raise FileNotFoundError(output_path)
            _validate_episode(
                encoder,
                episode_path,
                output_path,
                pool_grid=args.pool_grid,
                device=device,
                sample_count=args.validate_samples,
                tactile_datasets=tactile_datasets,
            )
        validate_t3_world_cache_files(
            existing_manifest,
            output_root,
            args.task,
            verify_sha256=True,
            # _validate_episode performed the complete finite scan.
            verify_finite=False,
        )
        print(f"[manifest verified] {manifest_path}", flush=True)
        return

    with exclusive_t3_world_cache_lock(task_root):
        if incomplete_path.exists() and not args.overwrite:
            raise RuntimeError(
                f"T3 cache generation did not finish cleanly: {incomplete_path}. "
                "Regenerate it with --overwrite."
            )

        existing_manifest = None
        if not args.overwrite:
            if manifest_path.exists():
                # The writer lock prevents concurrent mutation. Read directly
                # because the transaction marker is installed immediately
                # after this read and intentionally blocks normal consumers.
                existing_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                validate_t3_world_cache_manifest(
                    existing_manifest,
                    expected_manifest,
                    manifest_path=manifest_path,
                )
                validate_t3_world_cache_sources(
                    existing_manifest,
                    episode_paths,
                    verify_sha256=False,
                )
            elif any((task_root / "clean").glob("*.npy")):
                raise RuntimeError(
                    f"Unversioned T3 cache files exist under {task_root}; "
                    "regenerate them with --overwrite."
                )

        _write_incomplete_marker(incomplete_path)
        succeeded = False
        try:
            encoder, device = _create_encoder(
                args.tactile_encoder,
                device_name=args.device,
                preprocess_profile=getattr(
                    args,
                    "tactile_preprocess_profile",
                    None,
                ),
                task=args.task,
            )
            cache_paths: dict[int, Path] = {}
            for episode, episode_path in episode_paths.items():
                output_path = task_root / "clean" / f"{episode}.npy"
                cache_paths[episode] = output_path
                _cache_episode(
                    encoder,
                    episode_path,
                    output_path,
                    batch_size=args.batch_size,
                    pool_grid=args.pool_grid,
                    device=device,
                    overwrite=args.overwrite,
                    tactile_datasets=tactile_datasets,
                )
                _validate_episode(
                    encoder,
                    episode_path,
                    output_path,
                    pool_grid=args.pool_grid,
                    device=device,
                    sample_count=args.validate_samples,
                    tactile_datasets=tactile_datasets,
                )

            if existing_manifest is None:
                manifest = build_t3_world_cache_manifest(
                    task=args.task,
                    episode_lengths=episode_lengths,
                    pool_grid=args.pool_grid,
                    checkpoint_dir=args.tactile_encoder,
                    episode_paths=episode_paths,
                    include_source_sha256=True,
                    cache_paths=cache_paths,
                    preprocess_profile=getattr(
                        args,
                        "tactile_preprocess_profile",
                        None,
                    ),
                )
                manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
                manifest["generator"] = str(Path(__file__).resolve())
                _write_json_atomic(manifest_path, manifest)
                validate_t3_world_cache_sources(
                    manifest,
                    episode_paths,
                    verify_sha256=True,
                )
                validate_t3_world_cache_files(
                    manifest,
                    output_root,
                    args.task,
                    verify_sha256=True,
                    # _validate_episode performed the complete finite scan.
                    verify_finite=False,
                )
                print(f"[manifest] {manifest_path}", flush=True)
            else:
                validate_t3_world_cache_files(
                    existing_manifest,
                    output_root,
                    args.task,
                    verify_sha256=False,
                    # _validate_episode performed the complete finite scan.
                    verify_finite=False,
                )
                print(f"[manifest verified] {manifest_path}", flush=True)
            succeeded = True
        finally:
            # A failed or killed writer leaves the marker behind, while flock
            # itself is crash-safe and is released automatically by the kernel.
            if succeeded:
                incomplete_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
