#!/usr/bin/env python3
"""Recompute UniVTAC state/action statistics with baseline chunk alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import stat
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlinf.data.datasets.openpi_pytorch.univtac import (  # noqa: E402
    univtac_sft_dataset as _dataset_contract,
)
from rlinf.data.datasets.openpi_pytorch.univtac.recorded_command import (  # noqa: E402
    episode_action_dataset,
)

TASK_PROMPTS = _dataset_contract.TASK_PROMPTS
UNIVTAC_ACTION_ALIGNMENT = _dataset_contract.UNIVTAC_ACTION_ALIGNMENT
UNIVTAC_TERMINAL_HANDLING = _dataset_contract.UNIVTAC_TERMINAL_HANDLING


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_PROMPTS), default="insert_hole")
    parser.add_argument(
        "--data-root",
        required=True,
    )
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=49)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the task-specific normalization JSON.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Recompute and verify metadata, source hashes, and every statistic "
            "without rewriting the target file"
        ),
    )
    parser.add_argument(
        "--verify-mode",
        choices=("full", "attested", "auto"),
        default="full",
        help=(
            "full recomputes all source hashes and statistics; attested checks "
            "a previously generated immutable-source attestation; auto tries "
            "attested validation and falls back to full"
        ),
    )
    parser.add_argument(
        "--write-attestation",
        action="store_true",
        help=(
            "Atomically write an attestation after a successful full validation; "
            "with --verify-mode=auto this happens only after a full fallback"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _episode_samples(joints: np.ndarray, horizon: int, targets: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return q[t] states and FTP-1/N0 mix targets at t...t+horizon-1."""
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 2 or joints.shape[1] < 8 or joints.shape[0] == 0:
        raise ValueError(f"Expected non-empty [T, >=8] joints, got {joints.shape}")
    if horizon <= 0:
        raise ValueError(f"action horizon must be positive, got {horizon}")
    states = joints[:, :8].copy()
    targets = joints if targets is None else np.asarray(targets, dtype=np.float32)
    if targets.shape != joints.shape or not np.isfinite(targets).all():
        raise ValueError("Action targets must match the measured-state shape and be finite")
    offsets = np.arange(horizon, dtype=np.int64)
    indices = np.minimum(
        np.arange(joints.shape[0], dtype=np.int64)[:, None] + offsets[None, :],
        joints.shape[0] - 1,
    )
    actions = targets[indices, :8].copy()
    actions[..., :7] -= states[:, None, :7]
    return states, actions.reshape(-1, 8)


def _statistics(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8 or values.shape[0] == 0:
        raise ValueError(f"Expected non-empty [N, 8] values, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Normalization inputs contain non-finite values")
    q01, q99 = np.quantile(values, [0.01, 0.99], axis=0)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_STATISTIC_KEYS = ("mean", "std", "q01", "q99")
_ACTION_SEMANTICS = "arm_delta_from_q_t_gripper_absolute"
_ATTESTATION_SCHEMA_VERSION = 1
_ATTESTATION_SUFFIX = ".attestation-v1.json"
_NORM_SOURCE_KEYS = {"episode", "frames", "size_bytes", "mtime_ns", "sha256"}
_ATTESTED_SOURCE_KEYS = {
    "episode",
    "resolved_path",
    "filesystem_identity",
    "inode",
    "size_bytes",
    "mtime_ns",
    "ctime_ns",
    "frames",
    "sha256",
}
_FILESYSTEM_IDENTITY_COMMON_KEYS = {
    "kind",
    "fs_type",
    "mount_source",
    "mount_root",
    "relative_path",
}
_LOCAL_FILESYSTEM_IDENTITY_KEYS = {
    *_FILESYSTEM_IDENTITY_COMMON_KEYS,
    "device",
    "host_boot_id",
}
_NFS_TYPES = {"nfs", "nfs4"}
_MOUNTINFO_PATH = Path("/proc/self/mountinfo")
_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_MOUNTINFO_ESCAPE = re.compile(r"\\([0-7]{3})")
_ATTESTATION_KEYS = {
    "schema_version",
    "created_at_utc",
    "task",
    "episode_start",
    "episode_end",
    "action_horizon",
    "action_alignment",
    "terminal_handling",
    "action_semantics",
    "norm_stats_resolved_path",
    "norm_stats_sha256",
    "source_code_sha256",
    "episodes",
}
_VALIDATOR_SOURCE_FILES = {
    "scripts/univtac/recompute_norm_stats.py": Path(__file__).resolve(),
    ("rlinf/data/datasets/openpi_pytorch/univtac/univtac_sft_dataset.py"): Path(
        _dataset_contract.__file__
    ).resolve(),
}


def _attestation_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}{_ATTESTATION_SUFFIX}")


def _source_code_sha256() -> dict[str, str]:
    return {
        relative_path: _sha256_file(source_path)
        for relative_path, source_path in _VALIDATOR_SOURCE_FILES.items()
    }


def _validate_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _validate_json_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _decode_mountinfo_field(value: str) -> str:
    return _MOUNTINFO_ESCAPE.sub(
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _read_mountinfo() -> list[tuple[Path, str, str, str]]:
    """Return mount point, filesystem type, source, and filesystem root."""
    mounts: list[tuple[Path, str, str, str]] = []
    for line_number, line in enumerate(
        _MOUNTINFO_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError as error:
            raise ValueError(
                f"Malformed mountinfo line {line_number}: missing separator"
            ) from error
        if separator < 6 or len(fields) < separator + 4:
            raise ValueError(f"Malformed mountinfo line {line_number}")
        mount_root = _decode_mountinfo_field(fields[3])
        mount_point = Path(_decode_mountinfo_field(fields[4]))
        fs_type = _decode_mountinfo_field(fields[separator + 1])
        mount_source = _decode_mountinfo_field(fields[separator + 2])
        if not mount_point.is_absolute() or not mount_root.startswith("/"):
            raise ValueError(
                f"Malformed mountinfo paths on line {line_number}: "
                f"root={mount_root!r}, mount_point={str(mount_point)!r}"
            )
        mounts.append((mount_point, fs_type, mount_source, mount_root))
    if not mounts:
        raise ValueError(f"No mounts found in {_MOUNTINFO_PATH}")
    return mounts


def _normalize_nfs_source(source: str) -> str:
    if source.startswith("["):
        delimiter = source.find("]:")
        if delimiter < 0:
            raise ValueError(f"Malformed bracketed NFS source: {source!r}")
        server = source[1:delimiter]
        export = source[delimiter + 2 :]
        normalized_server = f"[{server.lower().rstrip('.')}]"
    else:
        server, separator, export = source.partition(":")
        if not separator:
            raise ValueError(f"Malformed NFS source: {source!r}")
        normalized_server = server.lower().rstrip(".")
    if not server or not normalized_server or not export.startswith("/"):
        raise ValueError(f"Malformed NFS source: {source!r}")
    return f"{normalized_server}:{posixpath.normpath(export)}"


def _filesystem_identity(
    path: Path,
    *,
    device: int,
    mounts: list[tuple[Path, str, str, str]] | None = None,
) -> dict[str, int | str]:
    """Build a mount-stable NFS identity and a host-bound local identity."""
    resolved = path.resolve(strict=True)
    candidate_mounts: list[tuple[Path, str, str, str, Path]] = []
    for mount_point, fs_type, mount_source, mount_root in mounts or _read_mountinfo():
        try:
            relative = resolved.relative_to(mount_point)
        except ValueError:
            continue
        candidate_mounts.append(
            (mount_point, fs_type, mount_source, mount_root, relative)
        )
    if not candidate_mounts:
        raise ValueError(f"No mountinfo entry contains source path: {resolved}")
    mount_point, fs_type, mount_source, mount_root, relative = max(
        candidate_mounts,
        key=lambda candidate: len(candidate[0].parts),
    )
    del mount_point
    relative_path = relative.as_posix()
    normalized_root = posixpath.normpath(mount_root)
    if fs_type in _NFS_TYPES:
        return {
            "kind": "nfs_export",
            "fs_type": "nfs",
            "mount_source": _normalize_nfs_source(mount_source),
            "mount_root": normalized_root,
            "relative_path": relative_path,
        }

    boot_id = _BOOT_ID_PATH.read_text(encoding="utf-8").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", boot_id):
        raise ValueError(f"Invalid local host boot ID from {_BOOT_ID_PATH}")
    return {
        "kind": "local_mount",
        "fs_type": fs_type,
        "mount_source": mount_source,
        "mount_root": normalized_root,
        "relative_path": relative_path,
        "device": device,
        "host_boot_id": boot_id,
    }


def _validate_filesystem_identity(value: object, *, episode: int) -> None:
    if not isinstance(value, dict):
        raise ValueError(
            f"Attested episode {episode} filesystem_identity must be an object"
        )
    kind = value.get("kind")
    expected_keys = (
        _FILESYSTEM_IDENTITY_COMMON_KEYS
        if kind == "nfs_export"
        else _LOCAL_FILESYSTEM_IDENTITY_KEYS
        if kind == "local_mount"
        else None
    )
    if expected_keys is None or set(value) != expected_keys:
        raise ValueError(
            f"Attested episode {episode} filesystem_identity schema mismatch: "
            f"kind={kind!r}, keys={sorted(value)}"
        )
    for field in ("fs_type", "mount_source", "mount_root", "relative_path"):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(
                f"Attested episode {episode} filesystem_identity.{field} "
                "must be a non-empty string"
            )
    mount_root = value["mount_root"]
    relative_path = value["relative_path"]
    if not mount_root.startswith("/") or posixpath.normpath(mount_root) != mount_root:
        raise ValueError(
            f"Attested episode {episode} mount_root is not canonical"
        )
    if (
        relative_path.startswith("/")
        or posixpath.normpath(relative_path) != relative_path
        or relative_path == ".."
        or relative_path.startswith("../")
    ):
        raise ValueError(
            f"Attested episode {episode} relative_path is not canonical"
        )
    if kind == "nfs_export":
        if value["fs_type"] != "nfs":
            raise ValueError(
                f"Attested episode {episode} NFS identity has invalid fs_type"
            )
        if _normalize_nfs_source(value["mount_source"]) != value["mount_source"]:
            raise ValueError(
                f"Attested episode {episode} NFS source is not canonical"
            )
    else:
        _validate_json_integer(
            value["device"],
            name=f"Attested episode {episode} filesystem_identity.device",
        )
        if not isinstance(value["host_boot_id"], str) or not re.fullmatch(
            r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
            value["host_boot_id"],
        ):
            raise ValueError(
                f"Attested episode {episode} local host_boot_id is invalid"
            )


def _validate_statistic_group_structure(
    actual: object,
    *,
    name: str,
) -> dict[str, np.ndarray]:
    if not isinstance(actual, dict):
        raise ValueError(f"Normalization group {name!r} must be a JSON object")
    if set(actual) != set(_STATISTIC_KEYS):
        raise ValueError(
            f"Normalization group {name!r} requires exactly "
            f"{list(_STATISTIC_KEYS)}, got {sorted(actual)}"
        )
    arrays: dict[str, np.ndarray] = {}
    for key in _STATISTIC_KEYS:
        try:
            array = np.asarray(actual[key], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Normalization {name}.{key} must be a numeric array"
            ) from error
        if array.shape != (8,):
            raise ValueError(
                f"Normalization {name}.{key} must have shape (8,), got "
                f"{array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(
                f"Normalization {name}.{key} contains non-finite values"
            )
        arrays[key] = array
    if np.any(arrays["std"] < 0):
        raise ValueError(f"Normalization {name}.std must be nonnegative")
    if np.any(arrays["q01"] > arrays["q99"]):
        raise ValueError(f"Normalization {name} has q01 greater than q99")
    return arrays


def _validate_statistic_group(
    actual: object,
    expected: dict[str, list[float]],
    *,
    name: str,
) -> None:
    arrays = _validate_statistic_group_structure(actual, name=name)
    for key in _STATISTIC_KEYS:
        array = arrays[key]
        expected_array = np.asarray(expected[key], dtype=np.float64)
        if not np.array_equal(array, expected_array):
            max_error = float(np.max(np.abs(array - expected_array)))
            raise ValueError(
                f"Normalization {name}.{key} does not match the source data; "
                f"max_abs_error={max_error:.12g}"
            )


def _validate_norm_stats_structure(
    document: object,
    *,
    expected_task: str,
    expected_horizon: int,
    expected_episode_start: int,
    expected_episode_end: int,
) -> list[dict[str, int | str]]:
    """Validate the self-describing norm contract without reading payloads."""
    if not isinstance(document, dict):
        raise ValueError("Normalization stats document must be a JSON object")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Normalization stats document is missing metadata")
    expected_metadata = {
        "task": expected_task,
        "action_horizon": expected_horizon,
        "action_alignment": UNIVTAC_ACTION_ALIGNMENT,
        "terminal_handling": UNIVTAC_TERMINAL_HANDLING,
        "action_semantics": _ACTION_SEMANTICS,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Normalization metadata mismatch: {mismatches}")

    sources = metadata.get("episodes")
    if not isinstance(sources, list):
        raise ValueError("Normalization metadata episodes must be a JSON list")
    expected_episodes = list(range(expected_episode_start, expected_episode_end + 1))
    actual_episodes: list[int] = []
    validated_sources: list[dict[str, int | str]] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or set(source) != _NORM_SOURCE_KEYS:
            keys = sorted(source) if isinstance(source, dict) else None
            raise ValueError(
                "Normalization metadata episode source requires exactly "
                f"{sorted(_NORM_SOURCE_KEYS)}, got {keys} at index {index}"
            )
        episode = source["episode"]
        frames = source["frames"]
        size_bytes = source["size_bytes"]
        mtime_ns = source["mtime_ns"]
        if not isinstance(episode, int) or isinstance(episode, bool):
            raise ValueError(f"Normalization source episode must be int: {episode!r}")
        if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
            raise ValueError(
                f"Normalization source frames must be a positive int: {frames!r}"
            )
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes <= 0
        ):
            raise ValueError(
                "Normalization source size_bytes must be a positive int: "
                f"{size_bytes!r}"
            )
        if not isinstance(mtime_ns, int) or isinstance(mtime_ns, bool):
            raise ValueError(f"Normalization source mtime_ns must be int: {mtime_ns!r}")
        _validate_sha256(
            source["sha256"],
            name=f"Normalization source episode {episode} sha256",
        )
        actual_episodes.append(episode)
        validated_sources.append(source)
    if actual_episodes != expected_episodes:
        raise ValueError(
            "Normalization metadata episode range mismatch: "
            f"actual={actual_episodes}, expected={expected_episodes}"
        )

    norm_stats = document.get("norm_stats")
    if not isinstance(norm_stats, dict) or set(norm_stats) != {"state", "actions"}:
        keys = sorted(norm_stats) if isinstance(norm_stats, dict) else None
        raise ValueError(
            "Normalization stats require exactly state/actions groups, "
            f"got {keys}"
        )
    for name in ("state", "actions"):
        _validate_statistic_group_structure(norm_stats[name], name=name)
    return validated_sources


def _validate_norm_stats_document(
    document: object,
    *,
    expected_task: str,
    expected_horizon: int,
    expected_sources: list[dict[str, int | str]],
    expected_statistics: dict[str, dict[str, list[float]]],
) -> None:
    """Strictly bind one stats file to its data, semantics, and exact values."""
    if not isinstance(document, dict):
        raise ValueError("Normalization stats document must be a JSON object")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Normalization stats document is missing metadata")
    expected_metadata = {
        "task": expected_task,
        "episodes": expected_sources,
        "action_horizon": expected_horizon,
        "action_alignment": UNIVTAC_ACTION_ALIGNMENT,
        "terminal_handling": UNIVTAC_TERMINAL_HANDLING,
        "action_semantics": _ACTION_SEMANTICS,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Normalization metadata mismatch: {mismatches}")

    norm_stats = document.get("norm_stats")
    if not isinstance(norm_stats, dict) or set(norm_stats) != {"state", "actions"}:
        keys = sorted(norm_stats) if isinstance(norm_stats, dict) else None
        raise ValueError(
            "Normalization stats require exactly state/actions groups, "
            f"got {keys}"
        )
    for name in ("state", "actions"):
        _validate_statistic_group(
            norm_stats[name],
            expected_statistics[name],
            name=name,
        )


def _collect_expected(
    *,
    data_root: Path,
    task: str,
    episode_start: int,
    episode_end: int,
    action_horizon: int,
    collect_attested_sources: bool = False,
) -> tuple[
    dict[str, dict[str, list[float]]],
    list[dict[str, int | str]],
    list[dict[str, object]],
]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    sources: list[dict[str, int | str]] = []
    attested_sources: list[dict[str, object]] = []
    for episode in range(episode_start, episode_end + 1):
        path = data_root / task / "clean" / f"{episode}.hdf5"
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved_path = path.resolve(strict=True)
        stat_before = resolved_path.stat()
        if not stat.S_ISREG(stat_before.st_mode):
            raise ValueError(f"Episode source is not a regular file: {resolved_path}")
        with h5py.File(resolved_path, "r") as file:
            if "embodiment/joint" not in file:
                raise KeyError(f"Episode {resolved_path} is missing embodiment/joint")
            joint = file["embodiment/joint"]
            if joint.ndim != 2 or joint.shape[1] < 8 or joint.shape[0] == 0:
                raise ValueError(
                    f"Episode {resolved_path} requires non-empty joint shape [T, >=8], "
                    f"got {joint.shape}"
                )
            joints = np.asarray(joint[:, :8], dtype=np.float32)
            targets = np.asarray(episode_action_dataset(file)[:, :8], dtype=np.float32)
        episode_states, episode_actions = _episode_samples(
            joints,
            action_horizon,
            targets,
        )
        states.append(episode_states)
        actions.append(episode_actions)
        digest = _sha256_file(resolved_path)
        stat_after = resolved_path.stat()
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(stat_before, field) != getattr(stat_after, field)
            for field in identity_fields
        ):
            raise RuntimeError(
                f"Episode changed while normalization statistics were being "
                f"computed: {resolved_path}"
            )
        sources.append(
            {
                "episode": episode,
                "frames": int(joints.shape[0]),
                "size_bytes": int(stat_after.st_size),
                "mtime_ns": int(stat_after.st_mtime_ns),
                "sha256": digest,
            }
        )
        if collect_attested_sources:
            attested_sources.append(
                {
                    "episode": episode,
                    "resolved_path": str(resolved_path),
                    "filesystem_identity": _filesystem_identity(
                        resolved_path,
                        device=int(stat_after.st_dev),
                    ),
                    "inode": int(stat_after.st_ino),
                    "size_bytes": int(stat_after.st_size),
                    "mtime_ns": int(stat_after.st_mtime_ns),
                    "ctime_ns": int(stat_after.st_ctime_ns),
                    "frames": int(joints.shape[0]),
                    "sha256": digest,
                }
            )
    statistics = {
        "state": _statistics(np.concatenate(states, axis=0)),
        "actions": _statistics(np.concatenate(actions, axis=0)),
    }
    return statistics, sources, attested_sources


def _attestation_payload(
    *,
    output: Path,
    task: str,
    episode_start: int,
    episode_end: int,
    action_horizon: int,
    attested_sources: list[dict[str, object]],
    norm_stats_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": _ATTESTATION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "episode_start": episode_start,
        "episode_end": episode_end,
        "action_horizon": action_horizon,
        "action_alignment": UNIVTAC_ACTION_ALIGNMENT,
        "terminal_handling": UNIVTAC_TERMINAL_HANDLING,
        "action_semantics": _ACTION_SEMANTICS,
        "norm_stats_resolved_path": str(output.resolve(strict=True)),
        "norm_stats_sha256": norm_stats_sha256,
        "source_code_sha256": _source_code_sha256(),
        "episodes": attested_sources,
    }


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path, *, description: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is not valid JSON: {path}") from error


def _load_json_with_sha256(
    path: Path,
    *,
    description: str,
) -> tuple[object, str]:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field) != getattr(after, field) for field in identity_fields
    ):
        raise RuntimeError(f"{description} changed while it was read: {path}")
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is not valid JSON: {path}") from error
    return document, hashlib.sha256(payload).hexdigest()


def _validate_attested_source(
    source: object,
    *,
    expected_episode: int,
    expected_path: Path,
    norm_source: dict[str, int | str],
) -> None:
    if not isinstance(source, dict) or set(source) != _ATTESTED_SOURCE_KEYS:
        keys = sorted(source) if isinstance(source, dict) else None
        raise ValueError(
            "Attested episode source requires exactly "
            f"{sorted(_ATTESTED_SOURCE_KEYS)}, got {keys}"
        )
    episode = _validate_json_integer(
        source["episode"],
        name="Attested source episode",
    )
    for field in ("inode", "mtime_ns", "ctime_ns"):
        _validate_json_integer(
            source[field],
            name=f"Attested episode {episode} {field}",
        )
    _validate_json_integer(
        source["size_bytes"],
        name=f"Attested episode {episode} size_bytes",
        minimum=1,
    )
    _validate_json_integer(
        source["frames"],
        name=f"Attested episode {episode} frames",
        minimum=1,
    )
    if not isinstance(source["resolved_path"], str):
        raise ValueError(f"Attested episode {episode} resolved_path must be a string")
    _validate_filesystem_identity(
        source["filesystem_identity"],
        episode=episode,
    )
    expected_resolved = expected_path.resolve(strict=True)
    if episode != expected_episode:
        raise ValueError(
            "Attested episode order mismatch: "
            f"actual={source['episode']!r}, expected={expected_episode}"
        )
    if source["resolved_path"] != str(expected_resolved):
        raise ValueError(
            "Attested episode resolved path mismatch: "
            f"actual={source['resolved_path']!r}, expected={str(expected_resolved)!r}"
        )

    file_stat = expected_resolved.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"Attested episode source is not regular: {expected_resolved}")
    identity = {
        "inode": int(file_stat.st_ino),
        "size_bytes": int(file_stat.st_size),
        "mtime_ns": int(file_stat.st_mtime_ns),
        "ctime_ns": int(file_stat.st_ctime_ns),
    }
    mismatches = {
        key: (source.get(key), value)
        for key, value in identity.items()
        if source.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Attested episode {expected_episode} file identity mismatch: {mismatches}"
        )
    current_filesystem_identity = _filesystem_identity(
        expected_resolved,
        device=int(file_stat.st_dev),
    )
    if source["filesystem_identity"] != current_filesystem_identity:
        raise ValueError(
            f"Attested episode {expected_episode} filesystem identity mismatch: "
            f"actual={current_filesystem_identity}, "
            f"expected={source['filesystem_identity']}"
        )

    with h5py.File(expected_resolved, "r") as file:
        if "embodiment/joint" not in file:
            raise KeyError(f"Episode {expected_resolved} is missing embodiment/joint")
        joint = file["embodiment/joint"]
        if joint.ndim != 2 or joint.shape[1] < 8 or joint.shape[0] == 0:
            raise ValueError(
                f"Episode {expected_resolved} requires non-empty joint shape "
                f"[T, >=8], got {joint.shape}"
            )
        frames = int(joint.shape[0])
    file_stat_after = expected_resolved.stat()
    if any(
        getattr(file_stat, field) != getattr(file_stat_after, field)
        for field in (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    ):
        raise RuntimeError(
            f"Episode changed during attested validation: {expected_resolved}"
        )
    if source["frames"] != frames:
        raise ValueError(
            f"Attested episode {expected_episode} frame mismatch: "
            f"actual={frames}, expected={source['frames']!r}"
        )

    digest = _validate_sha256(
        source["sha256"],
        name=f"Attested episode {expected_episode} sha256",
    )
    expected_norm_source = {
        "episode": expected_episode,
        "frames": frames,
        "size_bytes": identity["size_bytes"],
        "mtime_ns": identity["mtime_ns"],
        "sha256": digest,
    }
    if norm_source != expected_norm_source:
        raise ValueError(
            f"Normalization metadata episode {expected_episode} does not match "
            "the attested source"
        )


def _validate_attestation(
    *,
    output: Path,
    data_root: Path,
    task: str,
    episode_start: int,
    episode_end: int,
    action_horizon: int,
) -> None:
    document, actual_norm_sha256 = _load_json_with_sha256(
        output,
        description="Normalization stats",
    )
    norm_sources = _validate_norm_stats_structure(
        document,
        expected_task=task,
        expected_horizon=action_horizon,
        expected_episode_start=episode_start,
        expected_episode_end=episode_end,
    )

    attestation_path = _attestation_path(output)
    if not attestation_path.is_file():
        raise FileNotFoundError(
            f"Normalization attestation is missing: {attestation_path}"
        )
    attestation = _load_json(
        attestation_path,
        description="Normalization attestation",
    )
    if not isinstance(attestation, dict) or set(attestation) != _ATTESTATION_KEYS:
        keys = sorted(attestation) if isinstance(attestation, dict) else None
        raise ValueError(
            "Normalization attestation requires exactly "
            f"{sorted(_ATTESTATION_KEYS)}, got {keys}"
        )
    if not isinstance(attestation["created_at_utc"], str):
        raise ValueError("Normalization attestation created_at_utc must be a string")
    try:
        created_at = datetime.fromisoformat(attestation["created_at_utc"])
    except ValueError as error:
        raise ValueError(
            "Normalization attestation created_at_utc must be ISO-8601"
        ) from error
    if created_at.tzinfo is None:
        raise ValueError(
            "Normalization attestation created_at_utc must include a timezone"
        )
    for field, minimum in (
        ("schema_version", 1),
        ("episode_start", 0),
        ("episode_end", 0),
        ("action_horizon", 1),
    ):
        _validate_json_integer(
            attestation[field],
            name=f"Normalization attestation {field}",
            minimum=minimum,
        )
    expected_contract = {
        "schema_version": _ATTESTATION_SCHEMA_VERSION,
        "task": task,
        "episode_start": episode_start,
        "episode_end": episode_end,
        "action_horizon": action_horizon,
        "action_alignment": UNIVTAC_ACTION_ALIGNMENT,
        "terminal_handling": UNIVTAC_TERMINAL_HANDLING,
        "action_semantics": _ACTION_SEMANTICS,
        "norm_stats_resolved_path": str(output.resolve(strict=True)),
        "source_code_sha256": _source_code_sha256(),
    }
    mismatches = {
        key: (attestation.get(key), value)
        for key, value in expected_contract.items()
        if attestation.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Normalization attestation contract mismatch: {mismatches}")
    expected_norm_sha256 = _validate_sha256(
        attestation["norm_stats_sha256"],
        name="Attested norm_stats_sha256",
    )
    if actual_norm_sha256 != expected_norm_sha256:
        raise ValueError(
            "Normalization stats SHA256 does not match its attestation: "
            f"actual={actual_norm_sha256}, expected={expected_norm_sha256}"
        )

    attested_sources = attestation["episodes"]
    if not isinstance(attested_sources, list) or len(attested_sources) != len(
        norm_sources
    ):
        actual_count = (
            len(attested_sources) if isinstance(attested_sources, list) else None
        )
        raise ValueError(
            "Normalization attestation episode count mismatch: "
            f"actual={actual_count}, expected={len(norm_sources)}"
        )
    for offset, (source, norm_source) in enumerate(
        zip(attested_sources, norm_sources, strict=True)
    ):
        episode = episode_start + offset
        _validate_attested_source(
            source,
            expected_episode=episode,
            expected_path=data_root / task / "clean" / f"{episode}.hdf5",
            norm_source=norm_source,
        )


def main() -> None:
    args = _parse_args()
    if args.episode_end < args.episode_start:
        raise ValueError("episode-end must be >= episode-start")
    if args.validate_only and args.overwrite:
        raise ValueError("--validate-only and --overwrite are mutually exclusive")
    if args.write_attestation and not args.validate_only:
        raise ValueError("--write-attestation requires --validate-only")
    if args.write_attestation and args.verify_mode not in {"full", "auto"}:
        raise ValueError("--write-attestation requires --verify-mode=full or auto")
    data_root = Path(args.data_root).expanduser()
    output = Path(args.output).expanduser()
    if args.validate_only and not output.is_file():
        raise FileNotFoundError(output)
    if not args.validate_only and output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")

    if args.validate_only:
        validation_started = time.monotonic()
        if args.verify_mode in {"attested", "auto"}:
            try:
                _validate_attestation(
                    output=output,
                    data_root=data_root,
                    task=args.task,
                    episode_start=args.episode_start,
                    episode_end=args.episode_end,
                    action_horizon=args.action_horizon,
                )
            except Exception as error:  # noqa: BLE001 - auto must fail closed.
                if args.verify_mode == "attested":
                    raise
                reason = " ".join(str(error).splitlines())
                print(
                    "[simpletac-norm-attestation] status=fallback "
                    f"from=attested to=full reason={reason!r}",
                    flush=True,
                )
            else:
                elapsed = time.monotonic() - validation_started
                print(
                    f"[verified] validation_mode=attested task={args.task} "
                    f"episodes={args.episode_start}-{args.episode_end} "
                    f"horizon={args.action_horizon} output={output} "
                    f"elapsed_seconds={elapsed:.3f}",
                    flush=True,
                )
                return

        statistics, sources, attested_sources = _collect_expected(
            data_root=data_root,
            task=args.task,
            episode_start=args.episode_start,
            episode_end=args.episode_end,
            action_horizon=args.action_horizon,
            collect_attested_sources=args.write_attestation,
        )
        document, norm_stats_sha256 = _load_json_with_sha256(
            output,
            description="Normalization stats",
        )
        _validate_norm_stats_document(
            document,
            expected_task=args.task,
            expected_horizon=args.action_horizon,
            expected_sources=sources,
            expected_statistics=statistics,
        )
        attestation_path: Path | None = None
        if args.write_attestation:
            attestation_path = _attestation_path(output)
            if _sha256_file(output) != norm_stats_sha256:
                raise RuntimeError(
                    "Normalization stats changed before attestation was written"
                )
            _atomic_write_json(
                attestation_path,
                _attestation_payload(
                    output=output,
                    task=args.task,
                    episode_start=args.episode_start,
                    episode_end=args.episode_end,
                    action_horizon=args.action_horizon,
                    attested_sources=attested_sources,
                    norm_stats_sha256=norm_stats_sha256,
                ),
            )
        elapsed = time.monotonic() - validation_started
        attestation_message = (
            f" attestation={attestation_path}" if attestation_path is not None else ""
        )
        print(
            f"[verified] validation_mode=full task={args.task} "
            f"episodes={args.episode_start}-{args.episode_end} "
            f"horizon={args.action_horizon} output={output} "
            f"elapsed_seconds={elapsed:.3f}{attestation_message}",
            flush=True,
        )
        return

    statistics, sources, _ = _collect_expected(
        data_root=data_root,
        task=args.task,
        episode_start=args.episode_start,
        episode_end=args.episode_end,
        action_horizon=args.action_horizon,
    )
    payload = {
        "norm_stats": statistics,
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": args.task,
            "episodes": sources,
            "action_horizon": args.action_horizon,
            "action_alignment": UNIVTAC_ACTION_ALIGNMENT,
            "terminal_handling": UNIVTAC_TERMINAL_HANDLING,
            "action_semantics": _ACTION_SEMANTICS,
        },
    }
    _validate_norm_stats_document(
        payload,
        expected_task=args.task,
        expected_horizon=args.action_horizon,
        expected_sources=sources,
        expected_statistics=statistics,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        f"[done] task={args.task} episodes={args.episode_start}-{args.episode_end} "
        f"horizon={args.action_horizon} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
