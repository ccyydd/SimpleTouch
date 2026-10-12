# Copyright 2026 The SimpleTac Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Versioned manifests for offline frozen-T3 world-model caches."""

from __future__ import annotations

import ast
import contextlib
import fcntl
import hashlib
import json
import os
import socket
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

import cv2
import numpy as np
import timm
import torch

from .t3_preprocess_profile import (
    T3_LEGACY_PREPROCESS_PROFILE,
    T3_UNIVTAC_RGB_MARKER_PREPROCESS_PROFILE,
    resolve_t3_preprocess_profile,
)
from .t3_tactile_encoder import (
    T3_CACHE_IMPLEMENTATION_VERSION,
    T3_LARGE_WIDTH,
    T3_LEGACY_CACHE_IMPLEMENTATION_VERSION,
    T3_LEGACY_PREPROCESS_VERSION,
    T3_NUM_HEADS,
    T3_PREPROCESS_VERSION,
    T3_SENSOR_DEPTH,
    T3_TRUNK_DEPTH,
)

T3_WORLD_CACHE_SCHEMA_VERSION = 3
T3_WORLD_CACHE_MANIFEST = "manifest.json"
T3_WORLD_CACHE_INCOMPLETE = ".cache_generation_incomplete"
T3_WORLD_CACHE_LOCK = ".cache_generation.lock"

# Strict compatibility whitelist for the immutable cache produced by the
# immediately preceding official-v3 implementation. These hashes are exact,
# not permissive wildcards; only the legacy ImageNet profile may use it.
_LEGACY_V3_ENCODER_SOURCE_SHA256 = (
    "af3df5fd3917708a2f9d5f78acb4d481a614f327eebb1ca3307e49fcf22372f9"
)
_LEGACY_V3_CACHE_GENERATOR_SHA256 = (
    "c20fd2e8e50d7639f4c5aeb89e8d39d2c59e266a264c7bb8f99e838a2d279307"
)
_LEGACY_V3_RGB_DECODER_SHA256 = (
    "71c14b4ee43499026160ffbd4df76ba585162482782cd65dd5f9f4b1528c8c0b"
)
# The immediately preceding schema-v3 generator was hardcoded to rgb_marker.
# The current generator only parameterizes that source field; its marker-mode
# tensors are byte-identical. Keep this exact digest compatible only with the
# immutable marker profile, never with the new raw-rgb profile.
_RGB_MARKER_V1_CACHE_GENERATOR_SHA256 = (
    "09d9650b71ff264075004a4be9f15d77e6ea0183ce28356dd15b064787457a78"
)
# This exact producer revision only removes unused head/wrist stream checks.
# Decoding, T3 encoding, pooling and serialization are unchanged. Bind both ends
# so a later numerical change cannot silently inherit this compatibility rule.
_CAMERA_OPTIONAL_GENERATOR_PAIR = (
    "5984becb0c054254867e5ee991136c3d75716f6eef1a9506f13e9b7998d4442b",
    "5d80fbb704c33dc097dd68f34aaa6ac47eaca98c6e64b331026e9986d316e869",
)


@contextlib.contextmanager
def exclusive_t3_world_cache_lock(task_root: str | Path) -> Iterator[Path]:
    """Hold a crash-safe, non-blocking writer lock for one task cache.

    ``flock`` is released by the kernel even if the generator is killed.  The
    lock file itself intentionally remains in place: unlinking lock files can
    create an inode race in which two writers lock different files with the
    same pathname.  ``T3_WORLD_CACHE_INCOMPLETE`` separately records whether
    the previous writer completed its transaction.
    """
    task_root = Path(task_root).expanduser()
    task_root.mkdir(parents=True, exist_ok=True)
    lock_path = task_root / T3_WORLD_CACHE_LOCK
    file_descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    acquired = False
    try:
        try:
            fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as error:
            os.lseek(file_descriptor, 0, os.SEEK_SET)
            holder = (
                os.read(file_descriptor, 4096).decode("utf-8", errors="replace").strip()
            )
            suffix = f" Current holder: {holder}" if holder else ""
            raise RuntimeError(
                f"Another T3 cache writer holds {lock_path}.{suffix}"
            ) from error

        lock_metadata = {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        encoded = (json.dumps(lock_metadata, sort_keys=True) + "\n").encode("utf-8")
        os.ftruncate(file_descriptor, 0)
        os.lseek(file_descriptor, 0, os.SEEK_SET)
        os.write(file_descriptor, encoded)
        os.fsync(file_descriptor)
        yield lock_path
    finally:
        if acquired:
            fcntl.flock(file_descriptor, fcntl.LOCK_UN)
        os.close(file_descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def t3_checkpoint_sha256(checkpoint_dir: str | Path) -> str:
    """Hash the exact Mini sensor tower and shared trunk checkpoints."""
    checkpoint_dir = Path(checkpoint_dir).expanduser()
    paths = (
        checkpoint_dir / "encoders" / "mini.pth",
        checkpoint_dir / "trunk.pth",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing T3 checkpoint files: {missing}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def t3_encoder_source_sha256() -> str:
    """Hash cache-producing encoder code, independently of the git commit."""
    encoder_source = Path(__file__).with_name("t3_tactile_encoder.py")
    return _sha256_file(encoder_source)


def univtac_rgb_decoder_source_sha256() -> str:
    """Hash only the UniVTAC JPEG decoder implementation.

    The cache consumes decoded tactile frames, so changing channel semantics
    must invalidate every previously generated latent even when the T3 code
    and weights are unchanged.
    """
    dataset_source = (
        Path(__file__).parents[4]
        / "data"
        / "datasets"
        / "openpi_pytorch"
        / "univtac"
        / "univtac_sft_dataset.py"
    )
    source = dataset_source.read_text(encoding="utf-8")
    tree = ast.parse(source)
    decoder = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_decode_rgb"
        ),
        None,
    )
    if decoder is None:
        raise RuntimeError(f"Could not locate _decode_rgb in {dataset_source}")
    decoder_source = ast.get_source_segment(source, decoder)
    if decoder_source is None:
        raise RuntimeError(f"Could not read _decode_rgb source from {dataset_source}")
    return hashlib.sha256(decoder_source.encode("utf-8")).hexdigest()


def t3_cache_generator_source_sha256() -> str:
    """Hash the complete cache generator, including input normalization."""
    generator_source = (
        Path(__file__).parents[4].parent / "scripts" / "univtac" / "cache_t3_latents.py"
    )
    return _sha256_file(generator_source)


def _runtime_versions() -> dict[str, str]:
    """Return cache-relevant numerical dependency versions."""
    return {
        "numpy": str(np.__version__),
        "opencv": str(cv2.__version__),
        "timm": str(timm.__version__),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
    }


def _source_metadata(path: Path, *, include_sha256: bool) -> dict[str, Any]:
    stat = path.stat()
    metadata: dict[str, Any] = {
        "file": path.name,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        metadata["sha256"] = _sha256_file(path)
    return metadata


def cache_file_metadata(path: str | Path) -> dict[str, Any]:
    """Return immutable metadata recorded after one cache file is complete."""
    path = Path(path)
    stat = path.stat()
    return {
        "file": str(Path("clean") / path.name),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def build_t3_world_cache_manifest(
    *,
    task: str,
    episode_lengths: Mapping[int, int],
    pool_grid: int,
    checkpoint_dir: str | Path,
    episode_paths: Mapping[int, str | Path] | None = None,
    include_source_sha256: bool = False,
    cache_paths: Mapping[int, str | Path] | None = None,
    preprocess_profile: str | None = None,
) -> dict[str, Any]:
    """Build the deterministic portion of a complete cache manifest."""
    if pool_grid <= 0:
        raise ValueError(f"pool_grid must be positive, got {pool_grid}")
    episodes: dict[str, dict[str, Any]] = {}
    for episode, length in sorted(episode_lengths.items()):
        metadata: dict[str, Any] = {"frames": int(length)}
        if episode_paths is not None:
            path = Path(episode_paths[episode]).expanduser()
            metadata["source"] = _source_metadata(
                path,
                include_sha256=include_source_sha256,
            )
        if cache_paths is not None:
            metadata["cache"] = cache_file_metadata(cache_paths[episode])
        episodes[str(episode)] = metadata
    resolved_preprocess = resolve_t3_preprocess_profile(
        preprocess_profile,
        task=task,
    )
    return {
        "schema_version": T3_WORLD_CACHE_SCHEMA_VERSION,
        "status": "complete",
        "task": task,
        "episodes": episodes,
        "encoder": {
            "name": "t3_large",
            "sensor_domain": "mini",
            "checkpoint_sha256": t3_checkpoint_sha256(checkpoint_dir),
            "source_sha256": t3_encoder_source_sha256(),
            "implementation_version": T3_CACHE_IMPLEMENTATION_VERSION,
            "width": T3_LARGE_WIDTH,
            "num_heads": T3_NUM_HEADS,
            "sensor_depth": T3_SENSOR_DEPTH,
            "trunk_depth": T3_TRUNK_DEPTH,
        },
        "preprocess": {
            "t3": T3_PREPROCESS_VERSION,
            "profile": resolved_preprocess.as_fingerprint(),
            "univtac_rgb_decoder_sha256": univtac_rgb_decoder_source_sha256(),
            "cache_generator_sha256": t3_cache_generator_source_sha256(),
        },
        "runtime": _runtime_versions(),
        "output": {
            "dtype": "float16",
            "pool_grid": int(pool_grid),
            "tokens_per_sensor": 1 + int(pool_grid) ** 2,
            "token_selection": "cls_plus_spatial",
            "width": T3_LARGE_WIDTH,
            "sensor_order": ["left_gsmini_rgb", "right_gsmini_rgb"],
        },
    }


def _collect_subset_mismatches(
    actual: Any,
    expected: Any,
    *,
    path: str,
) -> list[str]:
    """Compare expected fields recursively while allowing recorded extras."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [path]
        mismatches: list[str] = []
        for key, value in expected.items():
            child = f"{path}.{key}" if path else str(key)
            if key not in actual:
                mismatches.append(child)
            else:
                mismatches.extend(
                    _collect_subset_mismatches(actual[key], value, path=child)
                )
        return mismatches
    return [] if actual == expected else [path]


def load_t3_world_cache_manifest(cache_root: str | Path, task: str) -> dict[str, Any]:
    task_root = Path(cache_root).expanduser() / task
    incomplete = task_root / T3_WORLD_CACHE_INCOMPLETE
    if incomplete.exists():
        raise RuntimeError(
            f"T3 cache generation did not finish cleanly: {incomplete}. "
            "Regenerate it with --overwrite."
        )
    manifest_path = task_root / T3_WORLD_CACHE_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing T3 cache manifest: {manifest_path}. Regenerate the cache "
            "with the current scripts/univtac/cache_t3_latents.py."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_t3_world_cache_manifest(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    runtime_policy: Literal["strict", "consumer"] = "strict",
) -> None:
    """Validate semantics; distinguish immutable readers from cache producers.

    Producer/regeneration checks remain strict by default. A consumer reads
    existing float16 NPY targets, not newly generated latents: the producer's
    numerical-library versions are provenance, not reader requirements. This
    does not assert bitwise equivalence of online T3 on different backends.
    """
    if runtime_policy not in ("strict", "consumer"):
        raise ValueError(f"Unknown T3 cache runtime policy: {runtime_policy!r}")
    if runtime_policy == "consumer":
        producer_runtime = actual.get("runtime")
        if not isinstance(producer_runtime, Mapping) or any(
            not isinstance(producer_runtime.get(key), str)
            or not producer_runtime[key].strip()
            for key in expected["runtime"]
        ):
            raise ValueError(
                f"Missing or invalid T3 runtime provenance: {manifest_path}"
            )
        runtime_differences = _collect_subset_mismatches(
            producer_runtime, expected["runtime"], path="runtime"
        )
        # Reuse all strict semantic/legacy checks without mutating either
        # manifest or rewriting the cache's original producer provenance.
        consumer_expected = dict(expected)
        consumer_expected["runtime"] = dict(producer_runtime)
        validate_t3_world_cache_manifest(
            actual, consumer_expected, manifest_path=manifest_path
        )
        if runtime_differences:
            warnings.warn(
                "T3 immutable cache consumer accepted producer/runtime differences "
                f"{runtime_differences} at {manifest_path}; "
                f"producer={dict(producer_runtime)}, "
                f"consumer={dict(expected['runtime'])}. Semantic contract unchanged.",
                UserWarning,
                stacklevel=2,
            )
        return
    if (
        actual.get("schema_version") == T3_WORLD_CACHE_SCHEMA_VERSION
        and expected.get("schema_version") == T3_WORLD_CACHE_SCHEMA_VERSION
        and (
            actual.get("preprocess", {}).get("cache_generator_sha256"),
            expected.get("preprocess", {}).get("cache_generator_sha256"),
        )
        == _CAMERA_OPTIONAL_GENERATOR_PAIR
    ):
        compatible_expected = dict(expected)
        compatible_expected["preprocess"] = dict(expected["preprocess"])
        compatible_expected["preprocess"]["cache_generator_sha256"] = (
            _CAMERA_OPTIONAL_GENERATOR_PAIR[0]
        )
        validate_t3_world_cache_manifest(
            actual, compatible_expected, manifest_path=manifest_path
        )
        return
    expected_profile = (
        expected.get("preprocess", {}).get("profile", {}).get("profile_id")
    )
    if (
        actual.get("schema_version") == T3_WORLD_CACHE_SCHEMA_VERSION
        and expected.get("schema_version") == T3_WORLD_CACHE_SCHEMA_VERSION
        and expected_profile == T3_UNIVTAC_RGB_MARKER_PREPROCESS_PROFILE
        and actual.get("preprocess", {}).get("cache_generator_sha256")
        == _RGB_MARKER_V1_CACHE_GENERATOR_SHA256
    ):
        compatible_expected = dict(expected)
        compatible_preprocess = dict(expected["preprocess"])
        compatible_preprocess["cache_generator_sha256"] = (
            _RGB_MARKER_V1_CACHE_GENERATOR_SHA256
        )
        compatible_expected["preprocess"] = compatible_preprocess
        compatible_required = {
            key: compatible_expected[key]
            for key in (
                "schema_version",
                "status",
                "task",
                "episodes",
                "encoder",
                "preprocess",
                "runtime",
                "output",
            )
        }
        compatible_mismatches = _collect_subset_mismatches(
            actual,
            compatible_required,
            path="",
        )
        if not compatible_mismatches:
            return
    if (
        actual.get("schema_version") == 2
        and expected.get("schema_version") == T3_WORLD_CACHE_SCHEMA_VERSION
        and expected_profile == T3_LEGACY_PREPROCESS_PROFILE
    ):
        legacy_encoder = dict(expected["encoder"])
        legacy_encoder["source_sha256"] = _LEGACY_V3_ENCODER_SOURCE_SHA256
        legacy_encoder["implementation_version"] = (
            T3_LEGACY_CACHE_IMPLEMENTATION_VERSION
        )
        legacy_required = {
            "schema_version": 2,
            "status": expected["status"],
            "task": expected["task"],
            "episodes": expected["episodes"],
            "encoder": legacy_encoder,
            "preprocess": {
                "t3": T3_LEGACY_PREPROCESS_VERSION,
                "univtac_rgb_decoder_sha256": _LEGACY_V3_RGB_DECODER_SHA256,
                "cache_generator_sha256": _LEGACY_V3_CACHE_GENERATOR_SHA256,
            },
            "runtime": expected["runtime"],
            "output": expected["output"],
        }
        legacy_mismatches = _collect_subset_mismatches(
            actual,
            legacy_required,
            path="",
        )
        if not legacy_mismatches:
            return
        raise ValueError(
            f"Stale or incompatible legacy T3 cache manifest {manifest_path}; "
            "the schema-v2 compatibility whitelist matched the legacy profile "
            "but not the exact official-v3 provenance/semantics: "
            f"{sorted(set(legacy_mismatches))}."
        )
    required = {
        key: expected[key]
        for key in (
            "schema_version",
            "status",
            "task",
            "episodes",
            "encoder",
            "preprocess",
            "runtime",
            "output",
        )
    }
    mismatches = _collect_subset_mismatches(actual, required, path="")
    if mismatches:
        raise ValueError(
            f"Stale or incompatible T3 cache manifest {manifest_path}; "
            f"mismatched fields: {sorted(set(mismatches))}. Regenerate with "
            "scripts/univtac/cache_t3_latents.py --overwrite."
        )


def validate_t3_world_cache_files(
    manifest: Mapping[str, Any],
    cache_root: str | Path,
    task: str,
    *,
    verify_sha256: bool,
    verify_finite: bool,
) -> None:
    """Validate cache files against manifest metadata.

    Dataset startup can use the inexpensive stat checks. Offline
    ``--validate-only`` additionally enables full SHA256 and finite scans.
    """
    task_root = Path(cache_root).expanduser() / task
    for episode, episode_metadata in manifest.get("episodes", {}).items():
        cache_metadata = episode_metadata.get("cache")
        if not isinstance(cache_metadata, Mapping):
            raise ValueError(
                f"Manifest is missing cache metadata for episode {episode}; "
                "regenerate with --overwrite."
            )
        relative_path = Path(str(cache_metadata.get("file", "")))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Unsafe cache path for episode {episode}: {relative_path}"
            )
        cache_path = task_root / relative_path
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        stat = cache_path.stat()
        for key, value in (
            ("size_bytes", int(stat.st_size)),
            ("mtime_ns", int(stat.st_mtime_ns)),
        ):
            if int(cache_metadata.get(key, -1)) != value:
                raise ValueError(
                    f"Cache metadata mismatch for {cache_path}: {key}={value}, "
                    f"manifest={cache_metadata.get(key)}"
                )
        if verify_sha256:
            digest = _sha256_file(cache_path)
            if digest != cache_metadata.get("sha256"):
                raise ValueError(
                    f"Cache checksum mismatch for {cache_path}: {digest} != "
                    f"{cache_metadata.get('sha256')}"
                )
        if verify_finite:
            cached = np.load(cache_path, mmap_mode="r")
            for start in range(0, int(cached.shape[0]), 32):
                if not np.isfinite(cached[start : start + 32]).all():
                    raise ValueError(f"Cache contains non-finite values: {cache_path}")


def validate_t3_world_cache_sources(
    manifest: Mapping[str, Any],
    episode_paths: Mapping[int, str | Path],
    *,
    verify_sha256: bool,
) -> None:
    """Ensure the source HDF5 episodes still match cache generation inputs."""
    episodes = manifest.get("episodes", {})
    for episode, path_value in episode_paths.items():
        path = Path(path_value).expanduser()
        source = episodes.get(str(episode), {}).get("source")
        if not isinstance(source, Mapping):
            raise ValueError(
                f"Manifest is missing source metadata for episode {episode}; "
                "regenerate with --overwrite."
            )
        stat = path.stat()
        for key, value in (
            ("size_bytes", int(stat.st_size)),
            ("mtime_ns", int(stat.st_mtime_ns)),
        ):
            if int(source.get(key, -1)) != value:
                raise ValueError(
                    f"Source metadata mismatch for {path}: {key}={value}, "
                    f"manifest={source.get(key)}"
                )
        if verify_sha256:
            digest = _sha256_file(path)
            if digest != source.get("sha256"):
                raise ValueError(
                    f"Source checksum mismatch for {path}: {digest} != "
                    f"{source.get('sha256')}"
                )
