#!/usr/bin/env python3
"""Fast startup validation for an immutable frozen-T3 cache.

This companion validator checks the exact generation contract and all recorded
source/cache file stats without re-reading the HDF5/NPY payloads. Use
cache_t3_latents.py --validate-only for a full checksum and finite-value scan.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlinf.data.datasets.openpi_pytorch.univtac.univtac_sft_dataset import (
    TASK_PROMPTS,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_cache_manifest import (
    T3_WORLD_CACHE_MANIFEST,
    build_t3_world_cache_manifest,
    load_t3_world_cache_manifest,
    validate_t3_world_cache_files,
    validate_t3_world_cache_manifest,
    validate_t3_world_cache_sources,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_preprocess_profile import (
    T3_PREPROCESS_PROFILES,
    resolve_t3_preprocess_profile,
)
from scripts.univtac.cache_t3_latents import _inspect_episode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_PROMPTS), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tactile-encoder", required=True)
    parser.add_argument(
        "--tactile-preprocess-profile",
        choices=T3_PREPROCESS_PROFILES,
        required=True,
    )
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=49)
    parser.add_argument("--pool-grid", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.episode_end < args.episode_start:
        raise ValueError("episode-end must be >= episode-start")
    if args.pool_grid <= 0:
        raise ValueError("pool-grid must be positive")

    data_root = Path(args.data_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    task_root = output_root / args.task
    episode_paths: dict[int, Path] = {}
    episode_lengths: dict[int, int] = {}
    tactile_datasets = resolve_t3_preprocess_profile(
        getattr(args, "tactile_preprocess_profile", None),
        task=args.task,
    ).source_fields
    for episode in range(args.episode_start, args.episode_end + 1):
        episode_path = data_root / args.task / "clean" / f"{episode}.hdf5"
        if not episode_path.is_file():
            raise FileNotFoundError(episode_path)
        episode_paths[episode] = episode_path
        episode_lengths[episode] = _inspect_episode(episode_path, tactile_datasets)

    expected_manifest = build_t3_world_cache_manifest(
        task=args.task,
        episode_lengths=episode_lengths,
        pool_grid=args.pool_grid,
        checkpoint_dir=args.tactile_encoder,
        episode_paths=episode_paths,
        preprocess_profile=getattr(args, "tactile_preprocess_profile", None),
    )
    existing_manifest = load_t3_world_cache_manifest(output_root, args.task)
    manifest_path = task_root / T3_WORLD_CACHE_MANIFEST
    validate_t3_world_cache_manifest(
        existing_manifest,
        expected_manifest,
        manifest_path=manifest_path,
        runtime_policy="consumer",
    )
    validate_t3_world_cache_sources(
        existing_manifest,
        episode_paths,
        verify_sha256=False,
    )
    validate_t3_world_cache_files(
        existing_manifest,
        output_root,
        args.task,
        verify_sha256=False,
        verify_finite=False,
    )
    print(
        f"[manifest verified] {manifest_path} validation_mode=manifest",
        flush=True,
    )


if __name__ == "__main__":
    main()
