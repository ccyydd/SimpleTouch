# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Build the UniVTAC Pi0.5 policy and its independent Tactile Expert."""

from __future__ import annotations

import os
import pathlib
import time
from typing import Any

from rlinf.config import torch_dtype_from_precision
from rlinf.utils.logging import get_logger

logger = get_logger()


def _is_tactile_expert_state_key(key: str) -> bool:
    if key.startswith(
        ("tactile_expert_", "tactile_future_", "future_tactile_", "llm.final_norms.2.")
    ):
        return True
    return key.startswith("llm.layers.") and any(
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


def _is_tactile_state_key(key: str) -> bool:
    return key.startswith("tactile_encoder.") or _is_tactile_expert_state_key(key)


def _load_openpi_checkpoint(
    model: Any,
    state_dict: dict[str, Any],
    *,
    use_tactile_modules: bool,
    use_tactile_expert: bool = False,
    allow_missing_tactile_expert_from_verified_base: bool = False,
) -> None:
    if not use_tactile_modules or any(_is_tactile_state_key(k) for k in state_dict):
        model.load_state_dict(state_dict, strict=True)
        return
    if use_tactile_expert and not allow_missing_tactile_expert_from_verified_base:
        raise RuntimeError(
            "Missing Tactile Expert weights are allowed only in the verified Pi0.5 base"
        )
    expected_missing = {k for k in model.state_dict() if _is_tactile_state_key(k)}
    if not expected_missing:
        raise RuntimeError("Tactile policy contains no tactile parameters")
    incompatible = model.load_state_dict(state_dict, strict=False)
    if (
        set(incompatible.missing_keys) != expected_missing
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Base checkpoint must contain every non-tactile parameter and no unexpected parameters"
        )
    if use_tactile_expert:
        model.initialize_tactile_expert_from_base_checkpoint()


def resolve_pi0_config(cfg: Any):
    """Resolve only the current release model schema; unknown fields are errors."""
    from omegaconf import OmegaConf
    from .pi0_model.pi0_config import Pi0Config

    model_cfg = cfg.openpi
    values = OmegaConf.to_container(model_cfg, resolve=True)
    task = values.pop("task", "sft")
    if task not in ("sft", "eval"):
        raise ValueError("SimpleTouch supports task='sft' or 'eval'")
    for key in ("config_name", "action_chunk", "action_env_dim"):
        values.pop(key, None)
    values.update(
        pi05=True,
        action_horizon=int(cfg.num_action_chunks),
        action_dim=int(values.pop("model_action_dim", 32)),
        dtype="bfloat16",
        defer_pretrained_weight_init=True,
    )
    return Pi0Config(**values)


def get_model(cfg: Any, torch_dtype: Any = None) -> Any:
    import safetensors.torch

    from .sft_action_model import OpenPiPytorchSFTActionModel
    from .utils.model_builders import _build_eval_model

    pi0_config = resolve_pi0_config(cfg)
    model_path = pathlib.Path(cfg.model_path).expanduser()
    weights_path = model_path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    base = os.environ.get("SIMPLETOUCH_BASE_MODEL_PATH")
    verified_base = (
        bool(base) and model_path.resolve() == pathlib.Path(base).expanduser().resolve()
    )
    started = time.monotonic()
    model = pi0_config.create()
    logger.info("Model constructed in %.3fs", time.monotonic() - started)
    started = time.monotonic()
    state = safetensors.torch.load_file(str(weights_path), device="cpu")
    _load_openpi_checkpoint(
        model,
        state,
        use_tactile_modules=pi0_config.tactile_expert_enabled,
        use_tactile_expert=pi0_config.tactile_expert_enabled,
        allow_missing_tactile_expert_from_verified_base=verified_base,
    )
    logger.info(
        "Checkpoint loaded in %.3fs from %s", time.monotonic() - started, weights_path
    )
    target_dtype = (
        torch_dtype
        if torch_dtype is not None
        else torch_dtype_from_precision(cfg.precision)
    )
    if target_dtype is not None:
        model = model.to(target_dtype)
    common = dict(num_steps=int(cfg.num_steps), action_env_dim=int(cfg.action_dim))
    if cfg.openpi.get("task", "sft") == "sft":
        return OpenPiPytorchSFTActionModel(model, **common)
    return _build_eval_model(
        cfg, cfg.openpi, model, action_chunk=int(cfg.num_action_chunks), **common
    )
