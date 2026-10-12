# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Pi0.5 and the layer-aligned SimpleTouch Tactile Expert."""

from __future__ import annotations

import dataclasses
import math

import torch

from rlinf.utils.tactile_tail_supervision import resolve_tail_supervision

from . import gemma, model
from .t3_preprocess_profile import resolve_t3_preprocess_profile


@dataclasses.dataclass
class Pi0Config(model.BaseModelConfig):
    dtype: str = "bfloat16"
    defer_pretrained_weight_init: bool = False
    paligemma_variant: gemma.Variant = "gemma_2b"
    action_expert_variant: gemma.Variant = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    pi05: bool = False
    discrete_state_input: bool | None = None

    tactile_encoder_path: str = ""
    tactile_encoder_frozen: bool = True
    tactile_preprocess_profile: str | None = None
    tactile_preprocess_task: str | None = None
    tactile_expert_enabled: bool = False
    tactile_expert_wam_mode: str = "fast"
    tactile_expert_init_mode: str = "from_action_expert"
    tactile_expert_idm_condition_noise_prob: float = 0.5
    tactile_expert_future_prediction_mode: str = "single_horizon_full"
    tactile_expert_future_stride: int = 5
    tactile_expert_future_pool_grid: int = 6
    tactile_expert_future_normalization: str = "t3_reference_rms"
    tactile_expert_future_loss_weight: float = 0.0
    tactile_expert_tail_supervision_mode: str | None = None
    tactile_expert_future_attention_backend: str = "eager"
    training_attention_backend: str = "eager"

    # Persist resolved shapes in checkpoints, but infer them from the backbone
    # when training a new model. They are not independent architecture knobs.
    tactile_expert_width: int | None = None
    tactile_expert_depth: int | None = None
    tactile_expert_mlp_dim: int | None = None
    tactile_expert_num_heads: int | None = None
    tactile_expert_num_kv_heads: int | None = None
    tactile_expert_head_dim: int | None = None
    tactile_expert_future_horizon: int | None = None

    @property
    def tactile_expert_future_tail_mode(self):
        return resolve_tail_supervision(
            self.tactile_expert_tail_supervision_mode, self.tactile_expert_wam_mode
        )[1]

    def __post_init__(self):
        if self.pi05 and self.max_token_len == 48:
            self.max_token_len = 200
        if self.discrete_state_input is None:
            self.discrete_state_input = self.pi05
        ae = gemma.get_config(self.action_expert_variant)
        vlm = gemma.get_config(self.paligemma_variant)
        for name in (
            "width",
            "depth",
            "mlp_dim",
            "num_heads",
            "num_kv_heads",
            "head_dim",
        ):
            key = "tactile_expert_" + name
            expected = getattr(ae, name)
            if getattr(self, key) is None:
                setattr(self, key, expected)
            if self.tactile_expert_enabled and getattr(self, key) != expected:
                raise ValueError(f"{key} must match the Action Expert ({expected})")
        if self.tactile_expert_future_horizon is None:
            self.tactile_expert_future_horizon = self.action_horizon
        if self.tactile_expert_enabled:
            if not self.pi05 or self.action_expert_variant != "gemma_300m":
                raise ValueError("Tactile Expert requires pi05 and gemma_300m")
            for name in ("depth", "num_heads", "num_kv_heads", "head_dim"):
                if getattr(ae, name) != getattr(vlm, name):
                    raise ValueError(
                        "Tactile Expert attention geometry must match both Gemma streams"
                    )
            if self.tactile_expert_future_horizon != self.action_horizon:
                raise ValueError(
                    "tactile_expert_future_horizon must match action_horizon"
                )
        choices = {
            "tactile_expert_wam_mode": ("fast", "joint", "idm"),
            "tactile_expert_init_mode": ("random", "from_action_expert"),
            "tactile_expert_future_prediction_mode": (
                "single_horizon_full",
                "multi_horizon_pooled",
            ),
            "tactile_expert_future_normalization": ("none", "t3_reference_rms"),
            "tactile_expert_future_attention_backend": ("eager", "sdpa"),
            "training_attention_backend": ("auto", "eager", "sdpa"),
        }
        for name, options in choices.items():
            if getattr(self, name) not in options:
                raise ValueError(f"{name} must be one of {options}")
        resolve_tail_supervision(
            self.tactile_expert_tail_supervision_mode, self.tactile_expert_wam_mode
        )
        if self.tactile_expert_wam_mode != "fast" and not (
            self.tactile_expert_enabled and self.tactile_expert_future_loss_weight > 0
        ):
            raise ValueError(
                "Joint/IDM require an enabled Tactile Expert and positive future loss"
            )
        for name in (
            "tactile_expert_future_horizon",
            "tactile_expert_future_stride",
            "tactile_expert_future_pool_grid",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.tactile_expert_future_pool_grid > 14:
            raise ValueError("tactile_expert_future_pool_grid cannot exceed 14")
        if self.tactile_expert_future_prediction_mode == "multi_horizon_pooled" and (
            self.tactile_expert_future_horizon % self.tactile_expert_future_stride
        ):
            raise ValueError(
                "multi_horizon_pooled requires future_horizon divisible by future_stride"
            )
        if not 0 <= self.tactile_expert_idm_condition_noise_prob <= 1:
            raise ValueError(
                "tactile_expert_idm_condition_noise_prob must be in [0, 1]"
            )
        weight = self.tactile_expert_future_loss_weight
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                "tactile_expert_future_loss_weight must be finite and nonnegative"
            )
        profile = resolve_t3_preprocess_profile(
            self.tactile_preprocess_profile if self.tactile_expert_enabled else None,
            task=self.tactile_preprocess_task if self.tactile_expert_enabled else None,
        )
        if self.tactile_expert_enabled and profile.field != "rgb":
            raise ValueError("Tactile Expert requires marker-free RGB T3 input")
        self.tactile_preprocess_profile = profile.profile_id
        self.tactile_preprocess_task = profile.task

    def create(self, **kwargs):
        from .pi0 import Pi0

        return Pi0(self)

    def fake_obs(self, batch_size=1):
        obs = super().fake_obs(batch_size)
        if self.tactile_expert_enabled:
            image = torch.ones(batch_size, *model.IMAGE_RESOLUTION, 3)
            mask = torch.ones(batch_size, dtype=torch.bool)
            obs.tactile_images = {
                key: image for key in ("left_gsmini_rgb", "right_gsmini_rgb")
            }
            obs.tactile_masks = {key: mask for key in obs.tactile_images}
            if self.tactile_expert_future_loss_weight > 0:
                multi = (
                    self.tactile_expert_future_prediction_mode == "multi_horizon_pooled"
                )
                steps = (
                    self.tactile_expert_future_horizon
                    // self.tactile_expert_future_stride
                )
                obs.tactile_future_images = {
                    key: value[:, None].expand(-1, steps, *value.shape[1:])
                    if multi
                    else value
                    for key, value in obs.tactile_images.items()
                }
                obs.tactile_future_masks = {
                    key: mask[:, None].expand(-1, steps) if multi else mask
                    for key in obs.tactile_images
                }
                obs.tactile_future_target_valid = torch.ones(
                    (batch_size, steps) if multi else (batch_size,), dtype=torch.bool
                )
        return obs
