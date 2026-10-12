#!/usr/bin/env python3
"""Serve an RLinf-aligned SimpleTac checkpoint to the UniVTAC plugin."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import re
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import safetensors.torch
import torch
import websockets
from omegaconf import DictConfig, OmegaConf
from openpi_client import msgpack_numpy

from rlinf.data.datasets.openpi_pytorch.univtac import (
    TASK_PROMPTS,
    UNIVTAC_CAMERA_TYPES,
    UniVTACTransform,
    collate_univtac_observations,
)
from rlinf.data.datasets.openpi_pytorch.univtac.univtac_sft_dataset import (
    apply_head_image_mode,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_preprocess_profile import (
    resolve_t3_preprocess_profile,
)
from rlinf.utils.ckpt_convertor.openpi._core import (
    as_state_dict,
    strip_wrapper_prefix,
)
from scripts.simpletouch.checkpoint import (
    SUCCESS_MARKER_NAME,
    completion_metadata,
    find_global_step_checkpoint_root,
    validate_checkpoint_completion,
)

LOGGER = logging.getLogger("simpletac.server")
_FULL_WEIGHTS_CANDIDATES = (
    "actor/model_state_dict/full_weights.pt",
    "model_state_dict/full_weights.pt",
    "full_weights.pt",
)


def _is_tactile_expert_state_key(key: str) -> bool:
    """Identify the integrated Gemma expert at ModuleList slot two."""
    if key.startswith("tactile_expert_"):
        return True
    if key.startswith("llm.final_norms.2."):
        return True
    if not key.startswith("llm.layers."):
        return False
    return any(
        marker in key
        for marker in (
            ".attn.q_proj.2.",
            ".attn.k_proj.2.",
            ".attn.v_proj.2.",
            ".attn.o_proj.2.",
            ".pre_attention_norms.2.",
            ".pre_ffw_norms.2.",
            ".mlps.2.",
        )
    )


_CONFIG_SEMANTIC_KEYS: dict[str, tuple[str, ...]] = {
    "task": ("data.task", "task"),
    "action_horizon": (
        "actor.model.num_action_chunks",
        "actor.model.openpi.action_chunk",
        "action_horizon",
    ),
    "model_action_dim": ("actor.model.openpi.model_action_dim", "action_dim"),
    "env_action_dim": ("actor.model.action_dim", "env_action_dim"),
    "max_token_len": ("actor.model.openpi.max_token_len", "max_token_len"),
    "pi05": ("actor.model.openpi.pi05", "pi05"),
    "discrete_state_input": (
        "actor.model.openpi.discrete_state_input",
        "discrete_state_input",
    ),
    "normalization": ("data.normalization", "normalization"),
    "camera_type": ("data.camera_type", "camera_type"),
    "tactile_preprocess_profile": (
        "actor.model.openpi.tactile_preprocess_profile",
        "tactile_preprocess_profile",
    ),
    "tactile_preprocess_task": (
        "actor.model.openpi.tactile_preprocess_task",
        "tactile_preprocess_task",
    ),
    "tactile_preprocess": (
        "actor.model.openpi.tactile_preprocess",
        "tactile_preprocess",
    ),
    "action_loss_masking": ("data.action_loss_masking", "action_loss_masking"),
    "tactile_expert_enabled": (
        "actor.model.openpi.tactile_expert_enabled",
        "tactile_expert_enabled",
    ),
    "tactile_expert_width": (
        "actor.model.openpi.tactile_expert_width",
        "tactile_expert_width",
    ),
    "tactile_expert_depth": (
        "actor.model.openpi.tactile_expert_depth",
        "tactile_expert_depth",
    ),
    "tactile_expert_mlp_dim": (
        "actor.model.openpi.tactile_expert_mlp_dim",
        "tactile_expert_mlp_dim",
    ),
    "tactile_expert_num_heads": (
        "actor.model.openpi.tactile_expert_num_heads",
        "tactile_expert_num_heads",
    ),
    "tactile_expert_num_kv_heads": (
        "actor.model.openpi.tactile_expert_num_kv_heads",
        "tactile_expert_num_kv_heads",
    ),
    "tactile_expert_head_dim": (
        "actor.model.openpi.tactile_expert_head_dim",
        "tactile_expert_head_dim",
    ),
    "tactile_expert_init_mode": (
        "actor.model.openpi.tactile_expert_init_mode",
        "tactile_expert_init_mode",
    ),
    "tactile_expert_wam_mode": (
        "actor.model.openpi.tactile_expert_wam_mode",
        "tactile_expert_wam_mode",
    ),
    "tactile_expert_tail_supervision_mode": (
        "actor.model.openpi.tactile_expert_tail_supervision_mode",
        "tactile_expert_tail_supervision_mode",
    ),
    "tactile_expert_idm_condition_noise_prob": (
        "actor.model.openpi.tactile_expert_idm_condition_noise_prob",
        "tactile_expert_idm_condition_noise_prob",
    ),
    "tactile_expert_future_attention_backend": (
        "actor.model.openpi.tactile_expert_future_attention_backend",
        "tactile_expert_future_attention_backend",
    ),
    "tactile_expert_future_prediction_mode": (
        "actor.model.openpi.tactile_expert_future_prediction_mode",
        "tactile_expert_future_prediction_mode",
    ),
    "tactile_expert_future_horizon": (
        "actor.model.openpi.tactile_expert_future_horizon",
        "tactile_expert_future_horizon",
    ),
    "tactile_expert_future_stride": (
        "actor.model.openpi.tactile_expert_future_stride",
        "tactile_expert_future_stride",
    ),
    "tactile_expert_future_pool_grid": (
        "actor.model.openpi.tactile_expert_future_pool_grid",
        "tactile_expert_future_pool_grid",
    ),
    "tactile_expert_future_normalization": (
        "actor.model.openpi.tactile_expert_future_normalization",
        "tactile_expert_future_normalization",
    ),
    "tactile_expert_future_loss_weight": (
        "actor.model.openpi.tactile_expert_future_loss_weight",
        "tactile_expert_future_loss_weight",
    ),
    "paligemma_variant": ("actor.model.openpi.paligemma_variant", "paligemma_variant"),
    "action_expert_variant": (
        "actor.model.openpi.action_expert_variant",
        "action_expert_variant",
    ),
    "norm_stats_sha256": ("data.norm_stats_sha256", "norm_stats_sha256"),
    "tokenizer_sha256": ("data.tokenizer_sha256", "tokenizer_sha256"),
}
_SIMPLETAC_CONTRACT = {
    "action_horizon": 50,
    "model_action_dim": 32,
    "env_action_dim": 8,
    "max_token_len": 200,
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "pi05": True,
    "discrete_state_input": True,
    "normalization": "quantile",
}


def _resolve_checkpoint(path: Path) -> tuple[str, Path]:
    if path.is_file():
        kind = "safetensors" if path.suffix == ".safetensors" else "rlinf"
        return kind, path
    safetensors_path = path / "model.safetensors"
    if safetensors_path.is_file():
        return "safetensors", safetensors_path
    for relative_path in _FULL_WEIGHTS_CANDIDATES:
        candidate = path / relative_path
        if candidate.is_file():
            return "rlinf", candidate
    raise FileNotFoundError(
        f"No model.safetensors or full_weights.pt found under {path}"
    )


def _load_state_dict(checkpoint: Path) -> tuple[dict[str, torch.Tensor], Path]:
    kind, weights_path = _resolve_checkpoint(checkpoint)
    if kind == "safetensors":
        state_dict = safetensors.torch.load_file(str(weights_path), device="cpu")
    else:
        payload = torch.load(
            str(weights_path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        state_dict = strip_wrapper_prefix(as_state_dict(payload))
    return state_dict, weights_path


def _validate_rlinf_checkpoint_completion(
    weights_path: Path,
    *,
    verify_mode: str = "full",
) -> dict[str, Any] | None:
    """Reject interrupted RLinf global-step checkpoints before loading weights."""
    checkpoint_root = find_global_step_checkpoint_root(weights_path)
    if checkpoint_root is None:
        return None
    marker = checkpoint_root / SUCCESS_MARKER_NAME
    if not marker.is_file():
        raise FileNotFoundError(
            f"RLinf global-step checkpoint has no atomic completion marker: {marker}"
        )
    required_paths = {
        str(weights_path.resolve().relative_to(checkpoint_root)),
        "config.yaml",
        "norm_stats.json",
    }
    return validate_checkpoint_completion(
        checkpoint_root,
        required_paths=required_paths,
        verify_mode=verify_mode,
    )


def _validate_inference_asset(
    path: Path,
    *,
    name: str,
    config_digest: Any | None,
    marker_digest: Any | None,
) -> str:
    """Bind an inference asset to config/marker SHA-256 metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    expected = [
        digest for digest in (config_digest, marker_digest) if digest is not None
    ]
    for digest in expected:
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid {name} SHA-256 metadata: {digest!r}")
    if len(set(expected)) > 1:
        raise ValueError(
            f"Checkpoint config and completion marker disagree on {name} SHA-256: "
            f"{expected}"
        )
    if not expected:
        raise ValueError(f"Checkpoint has no SHA-256 metadata for {name}")
    actual = _sha256_file(path)
    if actual != expected[0]:
        raise ValueError(
            f"{name} SHA-256 mismatch: checkpoint={expected[0]}, "
            f"actual={actual}, path={path}"
        )
    return actual


def _load_checkpoint_config(
    weights_path: Path,
) -> tuple[DictConfig | None, Path | None]:
    """Load the nearest config, preferring the format matching the weights."""
    preferred_names = (
        ("config.json", "config.yaml")
        if weights_path.suffix == ".safetensors"
        else ("config.yaml", "config.json")
    )
    directory = weights_path if weights_path.is_dir() else weights_path.parent
    for _ in range(8):
        present = {
            config_name: directory / config_name
            for config_name in preferred_names
            if (directory / config_name).is_file()
        }
        if present:
            loaded: dict[str, DictConfig] = {}
            for config_name, config_path in present.items():
                try:
                    loaded[config_name] = OmegaConf.load(config_path)
                except Exception as error:
                    raise RuntimeError(
                        f"Failed to load checkpoint config {config_path}"
                    ) from error
            if len(loaded) == 2:
                yaml_config = loaded["config.yaml"]
                json_config = loaded["config.json"]
                conflicts: dict[str, tuple[Any, Any]] = {}
                for name, keys in _CONFIG_SEMANTIC_KEYS.items():
                    yaml_value = _config_value(yaml_config, keys)
                    json_value = _config_value(json_config, keys)
                    if (
                        yaml_value is not None
                        and json_value is not None
                        and (
                            yaml_value != json_value
                            or type(yaml_value) is not type(json_value)
                        )
                    ):
                        conflicts[name] = (yaml_value, json_value)
                if conflicts:
                    raise ValueError(
                        "Conflicting config.yaml/config.json checkpoint metadata "
                        f"under {directory}: {conflicts}"
                    )
            selected_name = next(name for name in preferred_names if name in loaded)
            return loaded[selected_name], present[selected_name]
        if directory.parent == directory:
            break
        directory = directory.parent
    return None, None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_value(
    config: DictConfig | None,
    keys: str | tuple[str, ...],
) -> Any | None:
    if config is None:
        return None
    if isinstance(keys, str):
        keys = (keys,)
    for key in keys:
        value = OmegaConf.select(config, key, default=None)
        if value is not None:
            return value
    return None


def _resolve_norm_stats_path(
    *,
    weights_path: Path,
    checkpoint_config: DictConfig | None,
    config_path: Path | None,
    override: str | None,
) -> Path:
    """Prefer checkpoint-bundled stats and reject inconsistent overrides."""
    bundled_candidates: list[Path] = []
    if config_path is not None:
        bundled_candidates.append(config_path.parent / "norm_stats.json")
    bundled_candidates.append(weights_path.parent / "norm_stats.json")
    configured = _config_value(
        checkpoint_config,
        ("data.norm_stats_path", "norm_stats_path"),
    )
    configured_path: Path | None = None
    if configured is not None:
        configured_path = Path(str(configured)).expanduser()
        if not configured_path.is_absolute() and config_path is not None:
            configured_path = config_path.parent / configured_path

    existing_bundled: list[Path] = []
    seen: set[Path] = set()
    for path in bundled_candidates:
        resolved = path.resolve()
        if resolved not in seen and path.is_file():
            seen.add(resolved)
            existing_bundled.append(path)
    if len(existing_bundled) > 1:
        reference = existing_bundled[0]
        for candidate in existing_bundled[1:]:
            if _sha256_file(candidate) != _sha256_file(reference):
                raise ValueError(
                    "Multiple bundled normalization files disagree: "
                    f"{reference} versus {candidate}"
                )

    selected = existing_bundled[0] if existing_bundled else None
    if configured_path is not None and configured_path.is_file():
        if selected is not None and _sha256_file(configured_path) != _sha256_file(
            selected
        ):
            raise ValueError(
                "Bundled normalization stats differ from checkpoint config: "
                f"bundled={selected}, configured={configured_path}"
            )
        if selected is None:
            selected = configured_path
    override_path = Path(override).expanduser() if override else None
    if override_path is not None and not override_path.is_file():
        raise FileNotFoundError(
            f"Normalization stats override not found: {override_path}"
        )
    if selected is None:
        if override_path is None:
            raise FileNotFoundError(
                "No checkpoint-bundled or configured norm_stats.json was found; "
                "pass --norm-stats explicitly."
            )
        return override_path
    if override_path is not None and _sha256_file(override_path) != _sha256_file(
        selected
    ):
        raise ValueError(
            "Normalization stats override differs from checkpoint metadata: "
            f"override={override_path}, checkpoint={selected}"
        )
    return selected


def _validate_simpletac_contract(
    config: DictConfig | None,
) -> dict[str, Any]:
    """Resolve and strictly validate non-shape SimpleTac inference semantics."""
    if config is None:
        raise ValueError("Checkpoint model/data metadata is required")

    resolved: dict[str, Any] = {}
    missing: list[str] = []
    for name in _SIMPLETAC_CONTRACT.keys():
        value = _config_value(config, _CONFIG_SEMANTIC_KEYS[name])
        if value is None:
            missing.append(name)
        else:
            resolved[name] = value
    if missing:
        raise ValueError(
            f"Checkpoint is missing required SimpleTac contract metadata: {missing}"
        )

    invalid_types = {
        name: type(resolved[name]).__name__
        for name, expected in _SIMPLETAC_CONTRACT.items()
        if (
            (isinstance(expected, bool) and type(resolved[name]) is not bool)
            or (
                isinstance(expected, int)
                and not isinstance(expected, bool)
                and (
                    not isinstance(resolved[name], int)
                    or isinstance(resolved[name], bool)
                )
            )
            or (isinstance(expected, str) and not isinstance(resolved[name], str))
        )
    }
    if invalid_types:
        raise TypeError(
            f"SimpleTac checkpoint contract metadata has invalid types: {invalid_types}"
        )
    action_horizon_values = {
        key: _config_value(config, key)
        for key in _CONFIG_SEMANTIC_KEYS["action_horizon"]
        if _config_value(config, key) is not None
    }
    if len(set(action_horizon_values.values())) > 1:
        raise ValueError(
            "SimpleTac checkpoint has inconsistent action horizon metadata: "
            f"{action_horizon_values}"
        )
    if resolved["action_horizon"] <= 0:
        raise ValueError(
            "SimpleTac checkpoint action_horizon must be positive, got "
            f"{resolved['action_horizon']}"
        )
    mismatches = {
        name: (resolved[name], expected)
        for name, expected in _SIMPLETAC_CONTRACT.items()
        if name != "action_horizon" and resolved[name] != expected
    }
    if mismatches:
        raise ValueError(f"SimpleTac checkpoint contract mismatch: {mismatches}")
    return resolved


def _resolve_checkpoint_task(
    config: DictConfig | None,
    requested_task: str,
) -> str:
    """Require checkpoint task metadata to match the requested UniVTAC task."""
    configured_task = _config_value(config, ("data.task", "task"))
    if configured_task is None:
        raise ValueError("Checkpoint task metadata is required")
    if not isinstance(configured_task, str) or not configured_task.strip():
        raise TypeError(
            "Checkpoint task metadata must be a non-empty string, got "
            f"{configured_task!r}"
        )
    configured_task = configured_task.strip()
    if configured_task != requested_task:
        raise ValueError(
            "Checkpoint task mismatch: checkpoint was trained for "
            f"{configured_task!r}, but evaluation requested {requested_task!r}"
        )
    return configured_task


def _resolve_tactile_mode(
    state_dict: dict[str, torch.Tensor],
    mode: str,
    *,
    configured_requires_tactile: bool | None = None,
) -> tuple[bool, bool]:
    """Infer whether any deployed policy path requires live touch."""
    detected = any(_is_tactile_expert_state_key(key) for key in state_dict)
    required = (
        detected
        if configured_requires_tactile is None
        else bool(configured_requires_tactile)
    )
    if mode == "auto":
        return required, detected

    requested = mode == "true"
    if requested != required:
        raise ValueError(
            f"Tactile mode mismatch: checkpoint requires={required}, "
            f"additive_weights_detected={detected}, "
            f"requested={requested}. Omit the override or use --tactile-mode auto."
        )
    return requested, detected


def _resolve_tactile_expert(state_dict, config):
    import dataclasses

    resolved = _current_model_config(config)
    detected = any(_is_tactile_expert_state_key(key) for key in state_dict)
    if resolved.tactile_expert_enabled != detected:
        raise ValueError(
            "Tactile Expert mismatch between checkpoint config and weights"
        )
    contract = {
        name.removeprefix("tactile_expert_"): value
        for name, value in dataclasses.asdict(resolved).items()
        if name.startswith("tactile_expert_")
    }
    contract["future_tail_mode"] = resolved.tactile_expert_future_tail_mode
    return contract


def _move_observation(observation: Observation, device: torch.device) -> Observation:
    def move(value: Any) -> Any:
        return value.to(device) if isinstance(value, torch.Tensor) else value

    return Observation(
        images={key: move(value) for key, value in observation.images.items()},
        image_masks={
            key: move(value) for key, value in observation.image_masks.items()
        },
        state=move(observation.state),
        tactile_images=(
            {key: move(value) for key, value in observation.tactile_images.items()}
            if observation.tactile_images is not None
            else None
        ),
        tactile_masks=(
            {key: move(value) for key, value in observation.tactile_masks.items()}
            if observation.tactile_masks is not None
            else None
        ),
        tactile_future_images=(
            {
                key: move(value)
                for key, value in observation.tactile_future_images.items()
            }
            if observation.tactile_future_images is not None
            else None
        ),
        tactile_future_masks=(
            {
                key: move(value)
                for key, value in observation.tactile_future_masks.items()
            }
            if observation.tactile_future_masks is not None
            else None
        ),
        tactile_future_target_valid=move(observation.tactile_future_target_valid),
        tactile_world_latents=move(observation.tactile_world_latents),
        tactile_world_future_latents=move(observation.tactile_world_future_latents),
        tokenized_prompt=move(observation.tokenized_prompt),
        tokenized_prompt_mask=move(observation.tokenized_prompt_mask),
        token_ar_mask=move(observation.token_ar_mask),
        token_loss_mask=move(observation.token_loss_mask),
    )


def _current_model_config(config):
    from rlinf.models.embodiment.openpi_pytorch import resolve_pi0_config

    if config is None:
        raise ValueError("Checkpoint configuration is required")
    if OmegaConf.select(config, "actor.model") is not None:
        return resolve_pi0_config(config.actor.model)
    values = OmegaConf.to_container(config, resolve=True)
    for key in (
        "task",
        "camera_type",
        "normalization",
        "norm_stats_path",
        "norm_stats_sha256",
        "tokenizer_path",
        "tokenizer_sha256",
        "tactile_preprocess",
        "env_action_dim",
    ):
        values.pop(key, None)
    return Pi0Config(**values)


class SimpleTacPolicy:
    """UniVTAC input/output adapter around RLinf's aligned Pi0.5 model."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._device = torch.device(args.device)
        self._num_steps = args.num_steps
        _, weights_path = _resolve_checkpoint(Path(args.checkpoint))
        completion_payload = _validate_rlinf_checkpoint_completion(
            weights_path,
            verify_mode=getattr(args, "checkpoint_verify_mode", "full"),
        )
        checkpoint_config, config_path = _load_checkpoint_config(weights_path)
        _current_model_config(checkpoint_config)
        self._task = _resolve_checkpoint_task(
            checkpoint_config,
            args.task,
        )
        self._contract = _validate_simpletac_contract(
            checkpoint_config,
        )
        configured_camera_type = _config_value(
            checkpoint_config,
            _CONFIG_SEMANTIC_KEYS["camera_type"],
        )
        expected_camera_type = UNIVTAC_CAMERA_TYPES[self._task]
        if configured_camera_type != expected_camera_type:
            raise ValueError(
                f"Checkpoint camera_type must be {expected_camera_type!r} for {self._task}"
            )
        self._camera_type = configured_camera_type
        self._norm_stats_path = _resolve_norm_stats_path(
            weights_path=weights_path,
            checkpoint_config=checkpoint_config,
            config_path=config_path,
            override=args.norm_stats,
        )
        configured_t3_profile = _config_value(
            checkpoint_config,
            _CONFIG_SEMANTIC_KEYS["tactile_preprocess_profile"],
        )
        configured_t3_task = _config_value(
            checkpoint_config,
            _CONFIG_SEMANTIC_KEYS["tactile_preprocess_task"],
        )
        configured_tactile_expert = _config_value(
            checkpoint_config, _CONFIG_SEMANTIC_KEYS["tactile_expert_enabled"]
        )
        resolve_configured_t3 = configured_tactile_expert is True
        if resolve_configured_t3 and configured_t3_profile is None:
            raise ValueError(
                "Tactile Expert requires an explicit T3 preprocess profile"
            )
        resolved_t3_profile = resolve_t3_preprocess_profile(
            (
                str(configured_t3_profile)
                if resolve_configured_t3 and configured_t3_profile is not None
                else None
            ),
            task=(
                (self._task if configured_t3_task is None else str(configured_t3_task))
                if resolve_configured_t3
                else None
            ),
        )
        self._tactile_preprocess_profile = resolved_t3_profile.profile_id
        self._tactile_preprocess_task = resolved_t3_profile.task
        if configured_tactile_expert is True and resolved_t3_profile.field != "rgb":
            raise ValueError(
                "Tactile Expert checkpoint requires marker-free RGB T3 input, "
                f"got preprocess field {resolved_t3_profile.field!r}"
            )
        if (
            self._tactile_preprocess_task is not None
            and self._tactile_preprocess_task != self._task
        ):
            raise ValueError(
                "Checkpoint T3 preprocess task mismatch: "
                f"checkpoint={self._tactile_preprocess_task!r}, "
                f"requested={self._task!r}"
            )
        self._tactile_preprocess_fingerprint = resolved_t3_profile.as_fingerprint()
        recorded_t3_profiles: list[tuple[str, Any]] = []
        config_t3_fingerprint = _config_value(
            checkpoint_config,
            _CONFIG_SEMANTIC_KEYS["tactile_preprocess"],
        )
        if resolve_configured_t3 and config_t3_fingerprint is not None:
            recorded_t3_profiles.append(
                (
                    "checkpoint config",
                    OmegaConf.to_container(config_t3_fingerprint, resolve=True)
                    if OmegaConf.is_config(config_t3_fingerprint)
                    else config_t3_fingerprint,
                )
            )
        marker_openpi = (
            completion_metadata(completion_payload).get("model", {}).get("openpi", {})
            if completion_payload is not None
            else {}
        )
        marker_t3_fingerprint = (
            marker_openpi.get("tactile_preprocess")
            if isinstance(marker_openpi, dict)
            else None
        )
        if resolve_configured_t3 and marker_t3_fingerprint is not None:
            recorded_t3_profiles.append(
                ("checkpoint _SUCCESS marker", marker_t3_fingerprint)
            )
        for source, recorded_profile in recorded_t3_profiles:
            if recorded_profile != self._tactile_preprocess_fingerprint:
                raise ValueError(
                    f"Resolved T3 preprocess differs from {source}: "
                    f"recorded={recorded_profile}, "
                    f"resolved={self._tactile_preprocess_fingerprint}"
                )
        has_t3_profile_proof = bool(recorded_t3_profiles)
        marker_data = (
            completion_metadata(completion_payload).get("data", {})
            if completion_payload is not None
            else {}
        )
        self._norm_stats_sha256 = _validate_inference_asset(
            self._norm_stats_path,
            name="normalization stats",
            config_digest=_config_value(
                checkpoint_config,
                _CONFIG_SEMANTIC_KEYS["norm_stats_sha256"],
            ),
            marker_digest=marker_data.get("norm_stats_sha256"),
        )
        self._tokenizer_path = Path(args.tokenizer).expanduser()
        self._tokenizer_sha256 = _validate_inference_asset(
            self._tokenizer_path,
            name="PaliGemma tokenizer",
            config_digest=_config_value(
                checkpoint_config,
                _CONFIG_SEMANTIC_KEYS["tokenizer_sha256"],
            ),
            marker_digest=marker_data.get("tokenizer_sha256"),
        )
        # Multi-gigabyte tensors are loaded only after inexpensive task and
        # normalization metadata checks have succeeded.
        state_dict, loaded_weights_path = _load_state_dict(weights_path)
        if loaded_weights_path != weights_path:
            raise RuntimeError(
                "Checkpoint resolution changed unexpectedly: "
                f"{weights_path} -> {loaded_weights_path}"
            )
        self._tactile_expert_contract = _resolve_tactile_expert(
            state_dict,
            checkpoint_config,
        )
        self._use_tactile_expert = bool(self._tactile_expert_contract["enabled"])
        policy_requires_tactile = self._use_tactile_expert
        self._use_tactile, detected_tactile = _resolve_tactile_mode(
            state_dict,
            args.tactile_mode,
            configured_requires_tactile=policy_requires_tactile,
        )
        if self._use_tactile != policy_requires_tactile:
            raise ValueError(
                "Requested tactile sensor mode disagrees with deployed policy "
                "paths: sensor="
                f"{self._use_tactile}, "
                f"tactile_expert={self._use_tactile_expert}"
            )
        if self._use_tactile_expert and not has_t3_profile_proof:
            raise ValueError(
                "A task-specific T3 checkpoint must embed its complete resolved "
                "preprocess fingerprint in config.json or the _SUCCESS marker"
            )
        self._max_token_len = self._contract["max_token_len"]
        LOGGER.info(
            "Checkpoint tactile mode: sensor_detected=%s sensor_selected=%s expert=%s config=%s",
            detected_tactile,
            self._use_tactile,
            self._use_tactile_expert,
            config_path,
        )
        config = Pi0Config(
            pi05=self._contract["pi05"],
            discrete_state_input=self._contract["discrete_state_input"],
            action_dim=self._contract["model_action_dim"],
            action_horizon=self._contract["action_horizon"],
            max_token_len=self._max_token_len,
            paligemma_variant=self._contract["paligemma_variant"],
            action_expert_variant=self._contract["action_expert_variant"],
            dtype="bfloat16",
            # The complete evaluation checkpoint is loaded strictly below;
            # avoid filling a 432M-parameter TE with disposable random values.
            defer_pretrained_weight_init=True,
            tactile_encoder_path=(
                args.tactile_encoder if self._use_tactile_expert else ""
            ),
            tactile_preprocess_profile=self._tactile_preprocess_profile,
            tactile_preprocess_task=self._tactile_preprocess_task,
            tactile_expert_enabled=self._use_tactile_expert,
            tactile_expert_wam_mode=self._tactile_expert_contract["wam_mode"],
            tactile_expert_tail_supervision_mode=self._tactile_expert_contract[
                "tail_supervision_mode"
            ],
            tactile_expert_idm_condition_noise_prob=self._tactile_expert_contract[
                "idm_condition_noise_prob"
            ],
            tactile_expert_future_attention_backend=self._tactile_expert_contract[
                "future_attention_backend"
            ],
            tactile_expert_width=int(self._tactile_expert_contract["width"]),
            tactile_expert_depth=int(self._tactile_expert_contract["depth"]),
            tactile_expert_mlp_dim=int(self._tactile_expert_contract["mlp_dim"]),
            tactile_expert_num_heads=int(self._tactile_expert_contract["num_heads"]),
            tactile_expert_num_kv_heads=int(
                self._tactile_expert_contract["num_kv_heads"]
            ),
            tactile_expert_head_dim=int(self._tactile_expert_contract["head_dim"]),
            tactile_expert_init_mode=str(self._tactile_expert_contract["init_mode"]),
            tactile_expert_future_prediction_mode=str(
                self._tactile_expert_contract["future_prediction_mode"]
            ),
            tactile_expert_future_horizon=int(
                self._tactile_expert_contract["future_horizon"]
            ),
            tactile_expert_future_stride=int(
                self._tactile_expert_contract["future_stride"]
            ),
            tactile_expert_future_pool_grid=int(
                self._tactile_expert_contract["future_pool_grid"]
            ),
            tactile_expert_future_normalization=str(
                self._tactile_expert_contract["future_normalization"]
            ),
            tactile_expert_future_loss_weight=float(
                self._tactile_expert_contract["future_loss_weight"]
            ),
        )
        model = config.create()
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint/model mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        del state_dict
        self._model = model.to(device=self._device, dtype=torch.bfloat16).eval()
        self._model.requires_grad_(False)
        self._model.gradient_checkpointing_disable()
        self._transform = UniVTACTransform(
            norm_stats_path=str(self._norm_stats_path),
            normalization=self._contract["normalization"],
            tokenizer_path=str(self._tokenizer_path),
            max_token_len=self._max_token_len,
            task=self._task,
            action_horizon=self._contract["action_horizon"],
            use_tactile=self._use_tactile,
            camera_type=self._camera_type,
        )
        self.metadata = {
            "checkpoint": str(weights_path),
            "norm_stats": str(self._norm_stats_path),
            "norm_stats_sha256": self._norm_stats_sha256,
            "tokenizer_sha256": self._tokenizer_sha256,
            "task": self._task,
            "camera_type": self._camera_type,
            "use_tactile": self._use_tactile,
            "tactile_preprocess": self._tactile_preprocess_fingerprint,
            "tactile_expert": dict(self._tactile_expert_contract),
            "max_token_len": self._max_token_len,
            "pi05": self._contract["pi05"],
            "discrete_state_input": self._contract["discrete_state_input"],
            "action_horizon": self._contract["action_horizon"],
            "action_dim": self._contract["env_action_dim"],
            "model_action_dim": self._contract["model_action_dim"],
            "num_steps": args.num_steps,
        }
        LOGGER.info("Loaded %s", weights_path)

    @torch.inference_mode()
    def infer_batch(
        self,
        raws: Sequence[dict[str, Any]],
        *,
        noise: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        """Predict absolute-qpos chunks; optional noise supports reproducible tests."""
        if not raws:
            raise ValueError("Cannot infer an empty SimpleTac batch")
        raw_states: list[np.ndarray] = []
        transformed_items: list[dict[str, Any]] = []
        for raw in raws:
            raw_state = np.asarray(raw["observation/state"], dtype=np.float32).reshape(
                -1
            )
            if raw_state.shape != (8,):
                raise ValueError(f"Expected an 8D state, got {raw_state.shape}")
            prompt = str(raw.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("A non-empty task prompt is required")
            required = []
            if self._camera_type in {"head", "all"}:
                required.append("observation/head_rgb")
            if self._camera_type in {"wrist", "all"}:
                required.append("observation/wrist_rgb")
            missing = [key for key in required if key not in raw]
            if self._use_tactile:
                tactile_keys = (
                    "observation/left_tactile_rgb",
                    "observation/right_tactile_rgb",
                )
                missing.extend(key for key in tactile_keys if key not in raw)
            if missing:
                raise KeyError(f"Missing SimpleTac observation keys: {missing}")
            raw_states.append(raw_state)
            policy_raw = dict(raw)
            if self._camera_type == "wrist":
                policy_raw["observation/head_rgb"] = np.zeros_like(
                    np.asarray(raw["observation/wrist_rgb"], dtype=np.uint8)
                )
            else:
                policy_raw["observation/head_rgb"] = apply_head_image_mode(
                    np.asarray(raw["observation/head_rgb"], dtype=np.uint8),
                    task=self._task,
                    mode="sensor",
                )
            transformed_items.append(self._transform(policy_raw))

        observation = collate_univtac_observations(transformed_items)
        observation = _move_observation(observation, self._device)
        model_noise: torch.Tensor | None = None
        if noise is not None:
            model_noise = torch.as_tensor(noise, dtype=torch.float32)
            expected_noise_shape = (
                len(raws),
                self._contract["action_horizon"],
                self._contract["model_action_dim"],
            )
            if tuple(model_noise.shape) != expected_noise_shape:
                raise ValueError(
                    f"Explicit SimpleTac noise has shape {tuple(model_noise.shape)}, "
                    f"expected {expected_noise_shape}"
                )
            if not torch.isfinite(model_noise).all():
                raise ValueError("Explicit SimpleTac noise contains non-finite values")
            model_noise = model_noise.to(self._device)
        normalized_actions = self._model.sample_actions(
            observation,
            num_steps=self._num_steps,
            noise=model_noise,
        )
        normalized_actions = normalized_actions.float().cpu().numpy()
        actions = np.stack(
            [
                self._transform.unnormalize_actions(
                    normalized_actions[index], raw_state
                )
                for index, raw_state in enumerate(raw_states)
            ]
        )
        expected_shape = (
            len(raws),
            self._contract["action_horizon"],
            self._contract["env_action_dim"],
        )
        if actions.shape != expected_shape or not np.all(np.isfinite(actions)):
            raise RuntimeError(
                f"Invalid sampled action batch: shape={actions.shape}, "
                f"finite={np.all(np.isfinite(actions))}"
            )
        return actions.astype(np.float32, copy=False)

    @torch.inference_mode()
    def infer(self, raw: dict[str, Any]) -> dict[str, np.ndarray]:
        """Predict one absolute-qpos action chunk from a live observation."""
        return {"actions": self.infer_batch([raw])[0]}


class WebsocketPolicyServer:
    """MessagePack websocket server compatible with the vendored client."""

    def __init__(self, policy: SimpleTacPolicy, host: str, port: int) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        # PyTorch/CUDA inference is synchronous. Keep it off the asyncio event
        # loop so websocket control frames are still serviced during a slow
        # first request, while serializing access to the single policy model.
        self._inference_lock = asyncio.Lock()

    async def _handler(self, websocket) -> None:
        LOGGER.info("Connection opened from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._policy.metadata))
        try:
            async for message in websocket:
                started = time.monotonic()
                try:
                    observation = msgpack_numpy.unpackb(message)
                    async with self._inference_lock:
                        response = await asyncio.to_thread(
                            self._policy.infer, observation
                        )
                    response["server_timing"] = {
                        "infer_ms": (time.monotonic() - started) * 1000.0
                    }
                    await websocket.send(packer.pack(response))
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(code=1011, reason="Policy inference failed")
                    raise
        except websockets.ConnectionClosed:
            LOGGER.info("Connection closed from %s", websocket.remote_address)

    async def run(self) -> None:
        """Serve until interrupted."""
        async with websockets.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            # A cold T3 + pi0.5 request can exceed websockets' 20-second
            # default. Keep pings enabled, but don't treat inference latency
            # as a dead peer on this local policy/simulator connection.
            ping_timeout=None,
        ) as server:
            LOGGER.info("Serving SimpleTac on ws://%s:%d", self._host, self._port)
            await server.serve_forever()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=sorted(TASK_PROMPTS), default="insert_hole")
    parser.add_argument(
        "--norm-stats",
        default=None,
        help=(
            "Optional override. Checkpoint-bundled/configured stats are used "
            "by default, and a differing override is rejected."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get(
            "SIMPLETOUCH_TOKENIZER_PATH", "./assets/paligemma_tokenizer.model"
        ),
    )
    parser.add_argument(
        "--tactile-encoder",
        default=os.environ.get("SIMPLETOUCH_T3_PATH", "./assets/t3_large"),
    )
    parser.add_argument(
        "--tactile-mode",
        choices=("auto", "true", "false"),
        default="auto",
        help="Infer tactile support from checkpoint keys by default.",
    )
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 1 <= args.num_steps <= 100:
        raise ValueError(f"num_steps must be in [1, 100], got {args.num_steps}")
    policy = SimpleTacPolicy(args)
    asyncio.run(WebsocketPolicyServer(policy, args.host, args.port).run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
