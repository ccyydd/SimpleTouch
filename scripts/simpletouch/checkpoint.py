#!/usr/bin/env python3
"""Strict, portable validation for SimpleTouch training checkpoints.

The functions in this module are intentionally side-effect free except for
``write_success_marker``. The SFT worker writes the marker after the actor checkpoint,
rank-local loader/RNG states, and bundled normalization statistics are durable.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rlinf.utils.tactile_tail_supervision import resolve_tail_supervision

SUCCESS_MARKER_NAME = "_SUCCESS"
FINGERPRINT_SCHEMA_VERSION = 4
# Large weight/shard files use presence and size; small metadata keeps SHA-256.
SUCCESS_MARKER_SCHEMA_VERSION = 4
INFERENCE_FINGERPRINT_SCHEMA_VERSION = 1
_FULL_WEIGHTS_RELATIVE_PATH = "actor/model_state_dict/full_weights.pt"
INFERENCE_REQUIRED_PATHS = frozenset(
    {_FULL_WEIGHTS_RELATIVE_PATH, "config.yaml", "norm_stats.json"}
)
_DCP_SHARD_PREFIX = "actor/dcp_checkpoint/"
IMPLEMENTATION_CONTRACT = "simpletouch_univtac_release_v1"
_MISSING = object()
_GLOBAL_STEP_RE = re.compile(r"^global_step_(\d+)$")
_NORM_STATS_ROOT = Path(
    os.environ.get("SIMPLETOUCH_NORM_STATS_ROOT", "assets/norm_stats")
)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest, rejecting missing/non-files."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Required fingerprint input is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_file_uses_size_only_completion(relative_path: str) -> bool:
    """Return whether schema-v4 deliberately omits this large payload digest."""
    return relative_path == _FULL_WEIGHTS_RELATIVE_PATH or (
        relative_path.startswith(_DCP_SHARD_PREFIX)
        and relative_path.endswith(".distcp")
    )


def canonical_json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _norm_stats_path(task: str, action_horizon: int) -> Path:
    variant = f"simpletac_pi05_univtac_t0_{task}"
    if action_horizon != 50:
        variant = f"simpletac_pi05_univtac_t0_h{action_horizon}_{task}"
    return _NORM_STATS_ROOT / variant / f"univtac_{task}" / "norm_stats.json"


def _resolve_public_training_dataset(
    dataset_profile: str, task: str, action_horizon: int
) -> tuple[Path, int, int, Path, str]:
    """Resolve only the UniVTAC dataset supported by this release."""
    from rlinf.data.datasets.openpi_pytorch.univtac.univtac_sft_dataset import (
        UNIVTAC_CAMERA_TYPES,
    )

    if dataset_profile != "univtac" or task not in UNIVTAC_CAMERA_TYPES:
        raise ValueError("Only the six UniVTAC tasks are supported")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    return (
        Path(os.environ.get("SIMPLETOUCH_DATA_ROOT", "data/UniVTAC")),
        0,
        49,
        _norm_stats_path(task, action_horizon),
        "simpletac_univtac",
    )


def _select(config: Any, dotted_key: str, default: Any = _MISSING) -> Any:
    """Select from a plain mapping or an OmegaConf object."""
    if isinstance(config, Mapping):
        current: Any = config
        for component in dotted_key.split("."):
            if not isinstance(current, Mapping) or component not in current:
                if default is _MISSING:
                    raise KeyError(f"Missing resume fingerprint field: {dotted_key}")
                return default
            current = current[component]
        return current

    try:
        from omegaconf import OmegaConf

        value = OmegaConf.select(config, dotted_key, default=_MISSING)
    except ImportError as error:
        raise TypeError(
            "Config must be a mapping when OmegaConf is unavailable"
        ) from error
    if value is _MISSING:
        if default is _MISSING:
            raise KeyError(f"Missing resume fingerprint field: {dotted_key}")
        return default
    return value


def _normalized_path(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve())


def _resolved_tactile_expert(config: Any) -> dict[str, Any]:
    import dataclasses
    from rlinf.models.embodiment.openpi_pytorch import resolve_pi0_config

    resolved = resolve_pi0_config(_select(config, "actor.model"))
    contract = {
        name.removeprefix("tactile_expert_"): value
        for name, value in dataclasses.asdict(resolved).items()
        if name.startswith("tactile_expert_")
    }
    contract["future_tail_mode"] = resolved.tactile_expert_future_tail_mode
    return contract


def _resolved_t3_preprocess(config: Any) -> dict[str, Any]:
    """Resolve the explicitly configured T3 RGB preprocessing."""
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_preprocess_profile import (
        resolve_t3_preprocess_profile,
    )

    profile_id = _select(
        config,
        "actor.model.openpi.tactile_preprocess_profile",
        None,
    )
    task = _select(
        config,
        "actor.model.openpi.tactile_preprocess_task",
        _select(config, "data.task"),
    )
    resolved = resolve_t3_preprocess_profile(
        None if profile_id is None else str(profile_id),
        task=None if task is None else str(task),
    )
    data_task = str(_select(config, "data.task"))
    if resolved.task is not None and resolved.task != data_task:
        raise ValueError(
            "tactile_preprocess_task must match data.task: "
            f"preprocess={resolved.task!r}, data={data_task!r}"
        )
    return resolved.as_fingerprint()


def _manifest_path(cache_root: Any, task: str) -> Path:
    return Path(str(cache_root)).expanduser() / task / "manifest.json"


def _training_code_hashes() -> dict[str, str]:
    """Bind a resume to the exact data/model/training implementation."""
    repo_root = Path(__file__).resolve().parents[2]
    recursive_roots = (
        repo_root / "rlinf" / "data" / "datasets" / "openpi_pytorch" / "univtac",
        repo_root / "rlinf" / "models" / "embodiment" / "openpi_pytorch",
        repo_root / "rlinf" / "hybrid_engines" / "fsdp" / "strategy",
    )
    explicit_files = (
        repo_root / "rlinf" / "runners" / "sft_runner.py",
        repo_root / "rlinf" / "workers" / "sft" / "fsdp_sft_worker.py",
        repo_root / "rlinf" / "workers" / "sft" / "fsdp_vla_sft_worker.py",
        repo_root / "rlinf" / "hybrid_engines" / "fsdp" / "fsdp_model_manager.py",
        repo_root / "rlinf" / "utils" / "ckpt_convertor" / "openpi" / "_core.py",
        repo_root / "scripts" / "univtac" / "serve_policy.py",
        repo_root / "rlinf" / "utils" / "tactile_tail_supervision.py",
    )
    files = set(explicit_files)
    for root in recursive_roots:
        files.update(root.rglob("*.py"))
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Training implementation files are missing: {missing}")
    return {
        str(path.relative_to(repo_root)): sha256_file(path) for path in sorted(files)
    }


def _runtime_versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "h5py",
        "safetensors",
        "omegaconf",
    )
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _training_source_digests() -> dict[str, str]:
    """Hash code that can change SimpleTac data/model/loss semantics.

    The contract string documents intended semantics; source digests also
    catch an implementation edit that forgot to bump that manual version.
    Repository-relative keys allow the worktree itself to move safely.
    """
    repo_root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "rlinf/data/datasets/openpi_pytorch/univtac/univtac_sft_dataset.py",
        "rlinf/data/datasets/openpi_pytorch/univtac/univtac_sft_data_loader.py",
        "rlinf/models/embodiment/openpi_pytorch/__init__.py",
        "rlinf/models/embodiment/openpi_pytorch/sft_action_model.py",
        "rlinf/models/embodiment/openpi_pytorch/pi0_model/model.py",
        "rlinf/models/embodiment/openpi_pytorch/pi0_model/pi0.py",
        "rlinf/models/embodiment/openpi_pytorch/pi0_model/gemma.py",
        "rlinf/models/embodiment/openpi_pytorch/pi0_model/siglip.py",
        "rlinf/models/embodiment/openpi_pytorch/pi0_model/t3_tactile_encoder.py",
        "rlinf/models/embodiment/openpi_pytorch/pi0_model/t3_preprocess_profile.py",
        "rlinf/utils/ckpt_convertor/openpi/_core.py",
        "scripts/univtac/serve_policy.py",
        "rlinf/utils/tactile_tail_supervision.py",
    )
    return {
        relative_path: sha256_file(repo_root / relative_path)
        for relative_path in relative_paths
    }


def infer_world_size(config: Any) -> int:
    """Infer the actor world size from RLinf's component placement string."""
    placements = _select(config, "cluster.component_placement")
    if not isinstance(placements, Mapping):
        raise TypeError(
            "cluster.component_placement must be a mapping, got "
            f"{type(placements).__name__}"
        )
    actor_placement = next(
        (
            value
            for component_names, value in placements.items()
            if "actor" in str(component_names).split(",")
        ),
        None,
    )
    if actor_placement is None:
        raise KeyError("No actor entry in cluster.component_placement")
    devices: set[int] = set()
    for item in str(actor_placement).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid actor device range: {item}")
            devices.update(range(start, end + 1))
        else:
            devices.add(int(item))
    if not devices:
        raise ValueError(f"Empty actor device placement: {actor_placement!r}")
    return len(devices)


def build_training_fingerprint(
    config: Any, *, world_size: int, norm_stats_path: str | Path | None = None
) -> dict[str, Any]:
    """Build the canonical training-semantics fingerprint.

    Logging destinations, experiment names, ``resume_dir``, and total training
    steps are deliberately excluded. Total steps are validated separately and
    may stay unchanged or increase. Everything that can change model/data/loss
    semantics is included here.
    """
    task = str(_select(config, "data.task"))
    configured_norm_stats = (
        norm_stats_path
        if norm_stats_path is not None
        else _select(config, "data.norm_stats_path")
    )
    cache_root = _select(config, "data.tactile_world_cache_path", None)
    future_cache_root = _select(config, "data.tactile_expert_future_cache_path", None)
    tactile_expert = _resolved_tactile_expert(config)
    needs_t3_preprocess = tactile_expert["enabled"]
    if cache_root:
        cache = {
            "mode": "cached",
            "manifest_sha256": sha256_file(_manifest_path(cache_root, task)),
        }
    else:
        cache = {"mode": "online", "manifest_sha256": None}
    micro_batch = int(_select(config, "actor.micro_batch_size"))
    global_batch = int(_select(config, "actor.global_batch_size"))
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    denominator = micro_batch * world_size
    if global_batch % denominator:
        raise ValueError(
            f"global batch is not divisible by micro batch * world size: {global_batch} vs {micro_batch} * {world_size}"
        )
    tokenizer_path = _select(config, "data.tokenizer_path")
    fingerprint = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "implementation_contract": IMPLEMENTATION_CONTRACT,
        "implementation_files": _training_code_hashes(),
        "runtime_versions": _runtime_versions(),
        "implementation_sources": _training_source_digests(),
        "data": {
            "task": task,
            "camera_type": str(_select(config, "data.camera_type")),
            "train_data_paths": _normalized_path(
                _select(config, "data.train_data_paths")
            ),
            "episode_start": int(_select(config, "data.episode_start")),
            "episode_end": int(_select(config, "data.episode_end")),
            "normalization": str(_select(config, "data.normalization")),
            "norm_stats_sha256": sha256_file(configured_norm_stats),
            "tokenizer_path": _normalized_path(tokenizer_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "num_workers": int(_select(config, "data.num_workers")),
            "action_loss_masking": str(
                _select(
                    config,
                    "data.action_loss_masking",
                    resolve_tail_supervision(
                        _select(
                            config,
                            "actor.model.openpi.tactile_expert_tail_supervision_mode",
                            None,
                        ),
                        _select(
                            config, "actor.model.openpi.tactile_expert_wam_mode", "fast"
                        ),
                    )[0],
                )
            ),
            "tactile_world_cache": cache,
        },
        "model": {
            "model_type": str(_select(config, "actor.model.model_type")),
            "model_path": _normalized_path(_select(config, "actor.model.model_path")),
            "precision": str(_select(config, "actor.model.precision")),
            "num_action_chunks": int(_select(config, "actor.model.num_action_chunks")),
            "action_dim": int(_select(config, "actor.model.action_dim")),
            "openpi": {
                "task": str(_select(config, "actor.model.openpi.task")),
                "config_name": str(_select(config, "actor.model.openpi.config_name")),
                "action_chunk": int(_select(config, "actor.model.openpi.action_chunk")),
                "action_env_dim": int(
                    _select(config, "actor.model.openpi.action_env_dim")
                ),
                "model_action_dim": int(
                    _select(config, "actor.model.openpi.model_action_dim")
                ),
                "paligemma_variant": str(
                    _select(config, "actor.model.openpi.paligemma_variant")
                ),
                "action_expert_variant": str(
                    _select(config, "actor.model.openpi.action_expert_variant")
                ),
                "max_token_len": int(
                    _select(config, "actor.model.openpi.max_token_len")
                ),
                "discrete_state_input": bool(
                    _select(config, "actor.model.openpi.discrete_state_input")
                ),
                "tactile_encoder_path": _normalized_path(
                    _select(config, "actor.model.openpi.tactile_encoder_path")
                ),
                "tactile_encoder_frozen": bool(
                    _select(config, "actor.model.openpi.tactile_encoder_frozen")
                ),
                "tactile_preprocess": _resolved_t3_preprocess(config)
                if needs_t3_preprocess
                else None,
                "tactile_expert": tactile_expert,
            },
        },
        "optimizer": {
            "lr": float(_select(config, "actor.optim.lr")),
            "adam_beta1": float(_select(config, "actor.optim.adam_beta1")),
            "adam_beta2": float(_select(config, "actor.optim.adam_beta2")),
            "adam_eps": float(_select(config, "actor.optim.adam_eps")),
            "weight_decay": float(_select(config, "actor.optim.weight_decay")),
            "clip_grad": float(_select(config, "actor.optim.clip_grad")),
            "lr_warmup_steps": int(_select(config, "actor.optim.lr_warmup_steps")),
            "lr_scheduler": str(_select(config, "actor.optim.lr_scheduler")),
            "num_cycles": float(_select(config, "actor.optim.num_cycles")),
            "min_lr": float(_select(config, "actor.optim.min_lr")),
        },
        "batching": {
            "micro_batch_size": micro_batch,
            "global_batch_size": global_batch,
            "world_size": int(world_size),
            "gradient_accumulation": global_batch // denominator,
            "seed": int(_select(config, "actor.seed")),
        },
        "fsdp": {
            "strategy": str(_select(config, "actor.fsdp_config.strategy")),
            "sharding_strategy": str(
                _select(config, "actor.fsdp_config.sharding_strategy")
            ),
            "use_orig_params": bool(
                _select(config, "actor.fsdp_config.use_orig_params")
            ),
            "gradient_checkpointing": bool(
                _select(config, "actor.fsdp_config.gradient_checkpointing")
            ),
            "gradient_checkpointing_use_reentrant": bool(
                _select(
                    config, "actor.fsdp_config.gradient_checkpointing_use_reentrant"
                )
            ),
            "mixed_precision": {
                "param_dtype": str(
                    _select(config, "actor.fsdp_config.mixed_precision.param_dtype")
                ),
                "reduce_dtype": str(
                    _select(config, "actor.fsdp_config.mixed_precision.reduce_dtype")
                ),
                "buffer_dtype": str(
                    _select(config, "actor.fsdp_config.mixed_precision.buffer_dtype")
                ),
            },
        },
        "runner": {
            "max_epochs": int(_select(config, "runner.max_epochs")),
            "val_check_interval": int(_select(config, "runner.val_check_interval")),
            "save_interval": int(_select(config, "runner.save_interval")),
        },
    }
    if future_cache_root:
        fingerprint["data"]["tactile_expert_future_cache"] = {
            "mode": "cached",
            "manifest_sha256": sha256_file(_manifest_path(future_cache_root, task)),
        }
    training_backend = str(
        _select(config, "actor.model.openpi.training_attention_backend", "eager")
    )
    if training_backend == "auto":
        training_backend = (
            "sdpa"
            if _select(config, "actor.model.openpi.tactile_expert_wam_mode", "fast")
            == "idm"
            else "eager"
        )
    if training_backend not in {"eager", "sdpa"}:
        raise ValueError(f"Invalid training attention backend: {training_backend!r}")
    fingerprint["model"]["openpi"]["training_attention_backend"] = training_backend
    return fingerprint


def fingerprint_differences(
    expected: Any,
    actual: Any,
    *,
    prefix: str = "",
) -> dict[str, tuple[Any, Any]]:
    """Return leaf-level differences as ``path -> (expected, actual)``."""
    differences: dict[str, tuple[Any, Any]] = {}
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                differences[path] = ("<missing>", actual[key])
            elif key not in actual:
                differences[path] = (expected[key], "<missing>")
            else:
                differences.update(
                    fingerprint_differences(
                        expected[key],
                        actual[key],
                        prefix=path,
                    )
                )
        return differences
    if expected != actual:
        differences[prefix or "<root>"] = (expected, actual)
    return differences


def _required_checkpoint_files(
    checkpoint_root: Path,
) -> dict[str, dict[str, int | str]]:
    """Capture every required resumable-checkpoint file and its final size."""
    required = [
        checkpoint_root / "actor" / "data.pt",
        checkpoint_root / "actor" / "rng.pt",
        checkpoint_root / "actor" / "model_state_dict" / "full_weights.pt",
        checkpoint_root / "actor" / "dcp_checkpoint" / ".metadata",
        checkpoint_root / "config.yaml",
        checkpoint_root / "norm_stats.json",
    ]
    distcp_files = sorted(
        (checkpoint_root / "actor" / "dcp_checkpoint").glob("*.distcp")
    )
    if not distcp_files:
        raise FileNotFoundError(
            "Checkpoint contains no DCP shard files under "
            f"{checkpoint_root / 'actor' / 'dcp_checkpoint'}"
        )
    required.extend(distcp_files)
    result: dict[str, dict[str, int | str]] = {}
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete checkpoint, missing: {path}")
        size = int(path.stat().st_size)
        if size <= 0:
            raise ValueError(f"Incomplete checkpoint, empty file: {path}")
        relative_path = str(path.relative_to(checkpoint_root))
        metadata: dict[str, int | str] = {"size_bytes": size}
        if not checkpoint_file_uses_size_only_completion(relative_path):
            metadata["sha256"] = sha256_file(path)
        result[relative_path] = metadata
    return result


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, sort_keys=True, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def remove_success_marker(checkpoint_root: str | Path) -> None:
    """Remove a stale completion marker before overwriting a checkpoint."""
    marker = Path(checkpoint_root) / SUCCESS_MARKER_NAME
    try:
        marker.unlink()
    except FileNotFoundError:
        pass


def write_success_marker(
    checkpoint_root: str | Path,
    config: Any,
    *,
    step: int,
    world_size: int,
) -> dict[str, Any]:
    """Atomically mark a fully written SimpleTac actor checkpoint."""
    checkpoint_root = Path(checkpoint_root)
    bundled_stats = checkpoint_root / "norm_stats.json"
    fingerprint = build_training_fingerprint(
        config,
        world_size=world_size,
        norm_stats_path=bundled_stats,
    )
    payload = {
        "schema_version": SUCCESS_MARKER_SCHEMA_VERSION,
        "step": int(step),
        "training_fingerprint": fingerprint,
        "training_fingerprint_sha256": canonical_json_digest(fingerprint),
        "required_files": _required_checkpoint_files(checkpoint_root),
    }
    _atomic_write_json(checkpoint_root / SUCCESS_MARKER_NAME, payload)
    return payload


def completion_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Read inference metadata without treating an export as a new training run."""
    key = (
        "inference_fingerprint"
        if payload.get("bundle_type") == "inference"
        else "training_fingerprint"
    )
    return payload.get(key, {})


def _load_success_marker(checkpoint_root: Path) -> dict[str, Any]:
    marker_path = checkpoint_root / SUCCESS_MARKER_NAME
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint is incomplete: missing {marker_path}"
        )
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid checkpoint success marker: {marker_path}") from error
    marker_schema_version = payload.get("schema_version")
    if (
        type(marker_schema_version) is not int
        or marker_schema_version != SUCCESS_MARKER_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported checkpoint success-marker schema: {marker_schema_version!r}"
        )
    inference = payload.get("bundle_type") == "inference"
    fingerprint_kind = "inference" if inference else "training"
    fingerprint = completion_metadata(payload)
    if not isinstance(fingerprint, dict):
        raise ValueError(f"Checkpoint success marker has no {fingerprint_kind} fingerprint")
    expected_schema = (
        INFERENCE_FINGERPRINT_SCHEMA_VERSION
        if inference
        else FINGERPRINT_SCHEMA_VERSION
    )
    if fingerprint.get("schema_version") != expected_schema:
        raise ValueError(f"Unsupported {fingerprint_kind} fingerprint schema")
    recorded_digest = payload.get(
        "inference_fingerprint_sha256" if inference else "training_fingerprint_sha256"
    )
    actual_digest = canonical_json_digest(fingerprint)
    if recorded_digest != actual_digest:
        raise ValueError(
            "Checkpoint success-marker fingerprint digest mismatch: "
            f"recorded={recorded_digest}, actual={actual_digest}"
        )
    return payload


def _validate_recorded_files(
    checkpoint_root: Path,
    recorded_files: Any,
    *,
    relative_paths: set[str] | None = None,
) -> None:
    if not isinstance(recorded_files, Mapping) or not recorded_files:
        raise ValueError("Checkpoint success marker has no required-file manifest")
    checkpoint_root = checkpoint_root.resolve()
    selected_files = recorded_files
    if relative_paths is not None:
        missing_records = sorted(relative_paths - set(recorded_files))
        if missing_records:
            raise ValueError(
                "Checkpoint success marker does not record required files: "
                f"{missing_records}"
            )
        selected_files = {path: recorded_files[path] for path in sorted(relative_paths)}
    for relative_path, expected_metadata in selected_files.items():
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(
                "Checkpoint required-file path must be a non-empty string, "
                f"got {relative_path!r}"
            )
        if not isinstance(expected_metadata, Mapping):
            raise ValueError(
                "Checkpoint required-file metadata must be an object, got "
                f"{expected_metadata!r} for {relative_path!r}"
            )
        size_only = checkpoint_file_uses_size_only_completion(relative_path)
        expected_fields = {"size_bytes"} if size_only else {"size_bytes", "sha256"}
        if set(expected_metadata) != expected_fields:
            raise ValueError(
                "Checkpoint required-file metadata must contain exactly "
                f"{sorted(expected_fields)} for {relative_path!r}"
            )
        expected_size = expected_metadata["size_bytes"]
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
        ):
            raise ValueError(
                f"Invalid recorded size for {relative_path!r}: {expected_size!r}"
            )
        expected_digest = expected_metadata.get("sha256")
        if not size_only:
            if not isinstance(expected_digest, str) or not re.fullmatch(
                r"[0-9a-f]{64}", expected_digest
            ):
                raise ValueError(
                    "Invalid recorded SHA-256 for "
                    f"{relative_path!r}: {expected_digest!r}"
                )
        path = (checkpoint_root / relative_path).resolve()
        try:
            path.relative_to(checkpoint_root)
        except ValueError as error:
            raise ValueError(
                f"Checkpoint required-file path escapes its root: {relative_path!r}"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint file is missing: {path}")
        actual_size = int(path.stat().st_size)
        if actual_size != expected_size:
            raise ValueError(
                "Resume checkpoint file size mismatch: "
                f"{path}: marker={expected_size}, actual={actual_size}"
            )
        if not size_only:
            actual_digest = sha256_file(path)
            if actual_digest != expected_digest:
                raise ValueError(
                    "Resume checkpoint file SHA-256 mismatch: "
                    f"{path}: marker={expected_digest}, actual={actual_digest}"
                )


def find_global_step_checkpoint_root(path: str | Path) -> Path | None:
    """Return the enclosing ``global_step_N`` directory, if one exists."""
    candidate = Path(path).expanduser()
    if not candidate.is_dir():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if _GLOBAL_STEP_RE.fullmatch(directory.name):
            return directory.resolve()
    return None


def validate_checkpoint_completion(
    checkpoint_root: str | Path,
    *,
    required_paths: set[str] | None = None,
    verify_mode: str = "full",
) -> dict[str, Any]:
    """Validate the atomic completion marker and recorded checkpoint files.

    Completion markers have no digests for full weights or DCP shards:
    both modes validate those large
    payloads by presence/size while retaining SHA-256 checks for small recovery
    metadata. Actual DCP/weight loading remains responsible for large-payload
    structural validation.

    Evaluation/conversion callers that pass ``required_paths`` continue to use
    the default full verification for the exact files they consume.
    """
    if verify_mode not in {"fast", "full"}:
        raise ValueError(f"verify_mode must be 'fast' or 'full', got {verify_mode!r}")
    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    step = checkpoint_step(checkpoint_root)
    payload = _load_success_marker(checkpoint_root)
    if payload.get("bundle_type") == "inference" and (
        required_paths is None or set(required_paths) != set(INFERENCE_REQUIRED_PATHS)
    ):
        raise ValueError("Inference-only bundles cannot resume training")
    if payload.get("step") != step:
        raise ValueError(
            "Checkpoint directory/marker step mismatch: "
            f"directory={step}, marker={payload.get('step')!r}"
        )
    _validate_recorded_files(
        checkpoint_root,
        payload.get("required_files"),
        relative_paths=required_paths,
    )
    return payload


def checkpoint_step(checkpoint_root: str | Path) -> int:
    checkpoint_root = Path(checkpoint_root)
    match = _GLOBAL_STEP_RE.fullmatch(checkpoint_root.name)
    if match is None:
        raise ValueError(
            f"resume_dir must end in global_step_<N>, got {checkpoint_root}"
        )
    return int(match.group(1))


def validate_resume_checkpoint(
    checkpoint_root: str | Path,
    current_config: Any,
    *,
    current_world_size: int,
    requested_total_steps: int,
    verify_mode: str = "fast",
) -> dict[str, Any]:
    """Validate completeness and exact semantic compatibility before launch."""
    from omegaconf import OmegaConf

    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    step = checkpoint_step(checkpoint_root)
    current_runner_steps = int(_select(current_config, "runner.max_steps"))
    current_scheduler_steps = int(
        _select(current_config, "actor.optim.total_training_steps")
    )
    if (
        current_runner_steps != requested_total_steps
        or current_scheduler_steps != requested_total_steps
    ):
        raise ValueError(
            "Requested total steps must match both current runner and scheduler "
            "configuration: "
            f"requested={requested_total_steps}, "
            f"runner={current_runner_steps}, "
            f"scheduler={current_scheduler_steps}"
        )
    if requested_total_steps <= step:
        raise ValueError(
            "Resume requires requested total steps to be greater than the "
            f"checkpoint step: requested={requested_total_steps}, step={step}"
        )

    payload = validate_checkpoint_completion(
        checkpoint_root,
        verify_mode=verify_mode,
    )

    config_path = checkpoint_root / "config.yaml"
    bundled_stats = checkpoint_root / "norm_stats.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint is incomplete: missing {config_path}"
        )
    if not bundled_stats.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint is incomplete: missing {bundled_stats}"
        )
    try:
        saved_config = OmegaConf.load(config_path)
    except Exception as error:
        raise ValueError(f"Invalid resume config: {config_path}") from error

    current_fingerprint = build_training_fingerprint(
        current_config, world_size=current_world_size
    )
    differences = fingerprint_differences(
        payload["training_fingerprint"], current_fingerprint
    )
    if differences:
        raise ValueError(
            f"Resume configuration or implementation differs from the checkpoint: {differences}"
        )

    saved_runner_steps = int(_select(saved_config, "runner.max_steps"))
    saved_scheduler_steps = int(
        _select(saved_config, "actor.optim.total_training_steps")
    )
    if (
        requested_total_steps < saved_runner_steps
        or requested_total_steps < saved_scheduler_steps
    ):
        raise ValueError(
            "Resume may keep or increase total training steps, but may not "
            "decrease them: "
            f"requested={requested_total_steps}, "
            f"saved_runner={saved_runner_steps}, "
            f"saved_scheduler={saved_scheduler_steps}"
        )
    return payload
