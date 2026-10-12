# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base model classes for PyTorch models, aligned with JAX models/model.py."""

from __future__ import annotations

import abc
import dataclasses
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF

logger = logging.getLogger("openpi")

# Image keys expected by the model
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize images to target size with padding to preserve aspect ratio.

    Args:
        images: (..., H, W, C) float32 tensor in [-1, 1] or uint8 in [0, 255]
        height: target height
        width: target width
        mode: interpolation mode

    Returns:
        Resized and padded tensor
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    # (B, H, W, C) -> (B, C, H, W)
    images = images.permute(0, 3, 1, 2)
    B, C, cur_h, cur_w = images.shape

    ratio = max(cur_w / width, cur_h / height)
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)

    resized = tF.interpolate(
        images, size=(resized_h, resized_w), mode=mode, align_corners=False
    )

    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
    elif images.dtype in (torch.float32, torch.float16, torch.bfloat16):
        resized = resized.clamp(-1.0, 1.0)

    pad_h0, rem_h = divmod(height - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(width - resized_w, 2)
    pad_w1 = pad_w0 + rem_w

    fill_val = 0 if images.dtype == torch.uint8 else -1.0
    padded = tF.pad(
        resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=fill_val
    )

    # (B, C, H, W) -> (B, H, W, C)
    padded = padded.permute(0, 2, 3, 1)

    if squeeze_batch:
        padded = padded.squeeze(0)
    return padded


@dataclasses.dataclass
class Observation:
    """Holds observations, i.e., inputs to the model. PyTorch-compatible version."""

    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    # Optional SimpleTac GelSight Mini inputs. The model consumes the two keys
    # ``left_gsmini_rgb`` and ``right_gsmini_rgb`` in that fixed order.
    tactile_images: dict[str, torch.Tensor] | None = None
    tactile_masks: dict[str, torch.Tensor] | None = None
    # Training-only ground-truth frame(s). Single-horizon mode stores t+H; the
    # multi-horizon mode stores [t+s, ..., t+H]. Tensors repeat the final frame
    # past the episode end. ``tactile_future_target_valid`` gates loss only in
    # mask-tail modes; explicit tail modes never use it to gate attention.
    tactile_future_images: dict[str, torch.Tensor] | None = None
    # Optional frozen-T3 cache in left/right order. Current latents have shape
    # [B, 2, 197, 1024]; future latents may include a horizon axis and pooled tokens.
    tactile_world_latents: torch.Tensor | None = None
    tactile_world_future_latents: torch.Tensor | None = None
    tokenized_prompt: torch.Tensor | None = None
    tokenized_prompt_mask: torch.Tensor | None = None
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None
    tactile_future_masks: dict[str, torch.Tensor] | None = None
    # Whether each future target occurs before the episode boundary. Mask-tail
    # modes exclude invalid targets from loss; repeat-tail modes supervise them.
    tactile_future_target_valid: torch.Tensor | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        """Convert a nested dict to an Observation."""
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError(
                "tokenized_prompt and tokenized_prompt_mask must be provided together."
            )

        images = data["image"]
        image_masks = data["image_mask"]

        # Convert uint8 images to [-1, 1] float32
        for key in images:
            if images[key].dtype == torch.uint8:
                images[key] = images[key].to(torch.float32) / 255.0 * 2.0 - 1.0
            elif images[key].dtype == np.uint8:
                images[key] = (
                    torch.from_numpy(images[key].astype(np.float32)) / 255.0 * 2.0 - 1.0
                )

        tactile_images = data.get("tactile")
        if tactile_images is not None:
            for key in tactile_images:
                if isinstance(tactile_images[key], np.ndarray):
                    tactile_images[key] = torch.from_numpy(tactile_images[key])
                if tactile_images[key].dtype == torch.uint8:
                    tactile_images[key] = (
                        tactile_images[key].to(torch.float32) / 255.0 * 2.0 - 1.0
                    )

        tactile_future_images = data.get("tactile_future")
        if tactile_future_images is not None:
            for key in tactile_future_images:
                if isinstance(tactile_future_images[key], np.ndarray):
                    tactile_future_images[key] = torch.from_numpy(
                        tactile_future_images[key]
                    )
                if tactile_future_images[key].dtype == torch.uint8:
                    tactile_future_images[key] = (
                        tactile_future_images[key].to(torch.float32) / 255.0 * 2.0 - 1.0
                    )
        tactile_future_masks = data.get("tactile_future_mask")
        if tactile_future_masks is not None:
            tactile_future_masks = {
                key: torch.as_tensor(value, dtype=torch.bool)
                for key, value in tactile_future_masks.items()
            }
        tactile_future_target_valid = data.get("tactile_future_target_valid")
        if tactile_future_target_valid is not None:
            tactile_future_target_valid = torch.as_tensor(
                tactile_future_target_valid,
                dtype=torch.bool,
            )
        tactile_world_latents = data.get("tactile_world_latents")
        if isinstance(tactile_world_latents, np.ndarray):
            tactile_world_latents = torch.from_numpy(tactile_world_latents)
        tactile_world_future_latents = data.get("tactile_world_future_latents")
        if isinstance(tactile_world_future_latents, np.ndarray):
            tactile_world_future_latents = torch.from_numpy(
                tactile_world_future_latents
            )

        return cls(
            images=images,
            image_masks=image_masks,
            state=data["state"],
            tactile_images=tactile_images,
            tactile_masks=data.get("tactile_mask"),
            tactile_future_images=tactile_future_images,
            tactile_future_masks=tactile_future_masks,
            tactile_future_target_valid=tactile_future_target_valid,
            tactile_world_latents=tactile_world_latents,
            tactile_world_future_latents=tactile_world_future_latents,
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        tactile_images = result.pop("tactile_images")
        tactile_masks = result.pop("tactile_masks")
        tactile_future_images = result.pop("tactile_future_images")
        tactile_future_masks = result.pop("tactile_future_masks")
        if tactile_images is not None:
            result["tactile"] = tactile_images
        if tactile_masks is not None:
            result["tactile_mask"] = tactile_masks
        if tactile_future_images is not None:
            result["tactile_future"] = tactile_future_images
        if tactile_future_masks is not None:
            result["tactile_future_mask"] = tactile_future_masks
        if "actions" in result:
            del result["actions"]
        return result


def _tensor_to_dtype(t: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    """Cast a tensor to dtype, but skip integer tensors (e.g. token indices)."""
    if t is None:
        return None
    if t.dtype in (torch.long, torch.int, torch.int32, torch.int64, torch.bool):
        return t
    return t.to(dtype=dtype)


def _observation_to_dtype(obs: Observation, dtype: torch.dtype) -> Observation:
    """Cast all float tensors in an Observation to the target dtype.

    Used to ensure inputs match FSDP2 MixedPrecisionPolicy parameter dtype,
    since cast_forward_inputs cannot reach tensors nested inside dataclasses.

    Images are kept in float32 — SigLIPViT runs stem in float32 internally (matching JAX).
    """
    return Observation(
        images=obs.images,  # Leave images in float32 for SigLIPViT stem
        image_masks={k: _tensor_to_dtype(v, dtype) for k, v in obs.image_masks.items()},
        state=obs.state.to(dtype=dtype),
        # T3 casts its input to the patch-embedding parameter dtype itself.
        tactile_images=obs.tactile_images,
        tactile_masks=(
            {k: _tensor_to_dtype(v, dtype) for k, v in obs.tactile_masks.items()}
            if obs.tactile_masks is not None
            else None
        ),
        tactile_future_images=obs.tactile_future_images,
        tactile_future_masks=(
            {
                k: _tensor_to_dtype(v, torch.bool)
                for k, v in obs.tactile_future_masks.items()
            }
            if obs.tactile_future_masks is not None
            else None
        ),
        tactile_future_target_valid=_tensor_to_dtype(
            obs.tactile_future_target_valid, torch.bool
        ),
        tactile_world_latents=_tensor_to_dtype(obs.tactile_world_latents, dtype),
        tactile_world_future_latents=_tensor_to_dtype(
            obs.tactile_world_future_latents, dtype
        ),
        tokenized_prompt=_tensor_to_dtype(obs.tokenized_prompt, dtype),
        tokenized_prompt_mask=_tensor_to_dtype(obs.tokenized_prompt_mask, dtype),
        token_ar_mask=_tensor_to_dtype(obs.token_ar_mask, dtype),
        token_loss_mask=_tensor_to_dtype(obs.token_loss_mask, dtype),
    )


def _augmax_geometric_transform(
    images: torch.Tensor,
    crop_offsets: torch.Tensor,
    angles_degrees: torch.Tensor,
) -> torch.Tensor:
    """Apply OpenPI's fused augmax crop/resize/rotate resampling.

    ``images`` is BCHW in [0, 1]. Augmax composes the three coordinate
    transforms and performs one bilinear, zero-padded resample; doing three
    torchvision operations changes both interpolation and boundary values.
    """
    batch_size, _, height, width = images.shape
    if crop_offsets.shape != (batch_size, 2):
        raise ValueError(
            f"crop_offsets must have shape {(batch_size, 2)}, got "
            f"{tuple(crop_offsets.shape)}"
        )
    if angles_degrees.shape != (batch_size,):
        raise ValueError(
            f"angles_degrees must have shape {(batch_size,)}, got "
            f"{tuple(angles_degrees.shape)}"
        )
    crop_height = int(height * 0.95)
    crop_width = int(width * 0.95)
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError(f"Images are too small for a 95% crop: {(height, width)}")

    coordinate_dtype = images.dtype
    y = torch.arange(height, device=images.device, dtype=coordinate_dtype)
    x = torch.arange(width, device=images.device, dtype=coordinate_dtype)
    y, x = torch.meshgrid(y, x, indexing="ij")
    y = y - (height / 2.0 - 0.5)
    x = x - (width / 2.0 - 0.5)
    y = y.unsqueeze(0)
    x = x.unsqueeze(0)

    theta = torch.deg2rad(
        angles_degrees.to(device=images.device, dtype=coordinate_dtype)
    )
    cosine = torch.cos(theta)[:, None, None]
    sine = torch.sin(theta)[:, None, None]
    rotated_y = cosine * y + sine * x
    rotated_x = -sine * y + cosine * x
    sampled_y = (
        rotated_y * (crop_height / height)
        + crop_offsets[:, 0].to(images.device, coordinate_dtype)[:, None, None]
        + (height / 2.0 - 0.5)
    )
    sampled_x = (
        rotated_x * (crop_width / width)
        + crop_offsets[:, 1].to(images.device, coordinate_dtype)[:, None, None]
        + (width / 2.0 - 0.5)
    )
    # align_corners=False maps pixel center x to (2*x+1)/width-1.
    grid = torch.stack(
        (
            (2.0 * sampled_x + 1.0) / width - 1.0,
            (2.0 * sampled_y + 1.0) / height - 1.0,
        ),
        dim=-1,
    )
    return tF.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def _augmax_color_transform(
    images: torch.Tensor,
    brightness: torch.Tensor,
    contrast: torch.Tensor,
    hue_shift: torch.Tensor,
) -> torch.Tensor:
    """Port augmax 0.4.1 ColorJitter's RGB->HSV pixel transform.

    The locked augmax release samples saturation but accidentally discards its
    return value. We intentionally preserve that behavior for JAX/PyTorch
    parity instead of silently defining a different training recipe.
    """
    batch_size = images.shape[0]
    parameter_shape = (batch_size, 1, 1, 1)
    brightness = brightness.to(images.device, images.dtype).reshape(parameter_shape)
    contrast = contrast.to(images.device, images.dtype).reshape(parameter_shape)
    hue_shift = hue_shift.to(images.device, images.dtype).reshape(parameter_shape)

    value, argmax = torch.max(images, dim=1, keepdim=True)
    minimum = torch.min(images, dim=1, keepdim=True).values
    value_range = value - minimum
    safe_range = torch.where(
        value_range == 0, torch.ones_like(value_range), value_range
    )
    second_index = torch.remainder(argmax + 1, 3)
    third_index = torch.remainder(argmax + 2, 3)
    second = torch.gather(images, 1, second_index)
    third = torch.gather(images, 1, third_index)
    hue = (2.0 * argmax.to(images.dtype) + (second - third) / safe_range) / 6.0
    hue = torch.where(value_range == 0, torch.zeros_like(hue), hue)
    safe_value = torch.where(value == 0, torch.ones_like(value), value)
    saturation = torch.where(
        value == 0, torch.zeros_like(value), value_range / safe_value
    )

    value = torch.where(
        brightness < 0,
        value * (1.0 + brightness),
        value * (1.0 - brightness) + brightness,
    )
    slant = torch.tan((contrast + 1.0) * (torch.pi / 4.0))
    boundary_denominator = 2.0 * (1.0 - slant.square())
    # augmax's analytic expression is 0/0 at exactly neutral contrast. JAX
    # reaches the identity middle branch because comparisons with NaN are
    # false; use the equivalent explicit [0, 1] boundaries without NaNs.
    neutral_contrast = contrast == 0
    safe_denominator = torch.where(
        neutral_contrast,
        torch.ones_like(boundary_denominator),
        boundary_denominator,
    )
    first_boundary = (slant - slant.square()) / safe_denominator
    first_boundary = torch.where(
        neutral_contrast,
        torch.zeros_like(first_boundary),
        first_boundary,
    )
    second_boundary = 1.0 - first_boundary
    value = torch.where(
        value < first_boundary,
        value / slant,
        torch.where(
            value > second_boundary,
            value / slant + 1.0 - 1.0 / slant,
            slant * (value - 0.5) + 0.5,
        ),
    )
    hue = hue + hue_shift

    channel_offsets = torch.tensor(
        (5.0, 3.0, 1.0),
        device=images.device,
        dtype=images.dtype,
    ).reshape(1, 3, 1, 1)
    k = torch.remainder(channel_offsets + hue * 6.0, 6.0)
    ramp = torch.clamp(torch.minimum(k, 4.0 - k), min=0.0, max=1.0)
    return value - value * saturation * ramp


def _apply_openpi_augmentation(
    images: torch.Tensor,
    *,
    apply_geometry: bool,
    rng: torch.Generator | None,
    color_random: torch.Tensor | None = None,
    color_apply: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample and apply the OpenPI/augmax 0.4.1 augmentation distribution."""
    batch_size, _, height, width = images.shape
    if apply_geometry:
        crop_height = int(height * 0.95)
        crop_width = int(width * 0.95)
        crop_random = torch.rand(batch_size, 2, generator=rng)
        crop_limits = torch.tensor(
            ((height - crop_height) / 2.0, (width - crop_width) / 2.0)
        )
        crop_offsets = (crop_random * 2.0 - 1.0) * crop_limits
        angles = torch.rand(batch_size, generator=rng) * 10.0 - 5.0
        images = _augmax_geometric_transform(images, crop_offsets, angles)

    # augmax samples all four strengths even though its 0.4.1 saturation
    # branch is a no-op. Keep the draw so the declared distribution is clear.
    if color_random is None:
        color_random = torch.rand(batch_size, 4, generator=rng) * 2.0 - 1.0
    elif color_random.shape != (batch_size, 4):
        raise ValueError(
            f"color_random must have shape {(batch_size, 4)}, got "
            f"{tuple(color_random.shape)}"
        )
    transformed = _augmax_color_transform(
        images,
        brightness=color_random[:, 0] * 0.3,
        contrast=color_random[:, 1] * 0.4,
        hue_shift=color_random[:, 2] * 0.1,
    )
    if color_apply is None:
        color_apply = torch.rand(batch_size, generator=rng) < 0.5
    elif color_apply.shape != (batch_size,):
        raise ValueError(
            f"color_apply must have shape {(batch_size,)}, got "
            f"{tuple(color_apply.shape)}"
        )
    do_apply = color_apply.to(device=images.device, dtype=torch.bool)
    return torch.where(do_apply[:, None, None, None], transformed, images)


def preprocess_observation(
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    rng: torch.Generator | None = None,
    augmentation_exempt_keys: Sequence[str] = (),
) -> Observation:
    """Preprocess observations with optional image augmentations.

    For training, applies random crop, rotate, and color jitter augmentations.
    Resizes images to the target resolution with padding. Keys listed in
    ``augmentation_exempt_keys`` are resized but never randomly augmented;
    this is used when the two fixed wrist slots carry tactile images rather
    than natural camera views.
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(
            f"images dict missing keys: expected {image_keys}, got {list(observation.images)}"
        )
    augmentation_exempt_keys = frozenset(augmentation_exempt_keys)
    unknown_exempt_keys = augmentation_exempt_keys.difference(image_keys)
    if unknown_exempt_keys:
        raise ValueError(
            "augmentation_exempt_keys must be selected from image_keys; got "
            f"unknown keys {sorted(unknown_exempt_keys)}"
        )

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    shared_wrist_color: tuple[torch.Tensor, torch.Tensor] | None = None
    for key in image_keys:
        image = observation.images[key]

        # Standardize to (B, H, W, C) format if input is (B, C, H, W)
        if image.shape[-1] <= 4 and image.shape[1] > 4:
            pass  # already (B, H, W, C)
        elif image.shape[1] <= 4:
            image = image.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        if image.shape[1:3] != image_resolution:
            image = resize_with_pad_torch(image, *image_resolution)

        if train and key not in augmentation_exempt_keys:
            # Convert from [-1, 1] to [0, 1] for augmentations
            image = image / 2.0 + 0.5
            color_random = None
            color_apply = None
            if "wrist" in key:
                if shared_wrist_color is None:
                    shared_wrist_color = (
                        torch.rand(image.shape[0], 4, generator=rng) * 2.0 - 1.0,
                        torch.rand(image.shape[0], generator=rng) < 0.5,
                    )
                color_random, color_apply = shared_wrist_color
            image = _apply_openpi_augmentation(
                image.permute(0, 3, 1, 2),
                apply_geometry="wrist" not in key,
                rng=rng,
                color_random=color_random,
                color_apply=color_apply,
            ).permute(0, 2, 3, 1)

            # Back to [-1, 1]
            image = image * 2.0 - 1.0

        out_images[key] = image

    # Build masks
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool)
        else:
            mask = observation.image_masks[key]
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask)
            out_masks[key] = mask.bool()

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tactile_images=observation.tactile_images,
        tactile_masks=observation.tactile_masks,
        tactile_future_images=observation.tactile_future_images,
        tactile_future_masks=observation.tactile_future_masks,
        tactile_future_target_valid=observation.tactile_future_target_valid,
        tactile_world_latents=observation.tactile_world_latents,
        tactile_world_future_latents=observation.tactile_world_future_latents,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models."""

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @abc.abstractmethod
    def create(self, **kwargs) -> BaseModel:
        """Create a new model, initializing parameters."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        """Create fake observations for testing."""
        return Observation(
            images={
                k: torch.ones(batch_size, *IMAGE_RESOLUTION, 3) for k in IMAGE_KEYS
            },
            image_masks={
                k: torch.ones(batch_size, dtype=torch.bool) for k in IMAGE_KEYS
            },
            state=torch.ones(batch_size, self.action_dim),
            tokenized_prompt=torch.ones(
                batch_size, self.max_token_len, dtype=torch.long
            ),
            tokenized_prompt_mask=torch.ones(
                batch_size, self.max_token_len, dtype=torch.bool
            ),
        )

    def fake_act(self, batch_size: int = 1) -> torch.Tensor:
        """Create fake actions for testing."""
        return torch.ones(batch_size, self.action_horizon, self.action_dim)


class BaseModel(nn.Module, abc.ABC):
    """Base class for all model implementations."""

    def __init__(self, action_dim: int, action_horizon: int, max_token_len: int):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_token_len = max_token_len

    @abc.abstractmethod
    def compute_loss(
        self,
        observation: Observation,
        actions: torch.Tensor,
        *,
        train: bool = False,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Compute the loss for a batch of observations and actions."""

    @abc.abstractmethod
    def sample_actions(
        self,
        observation: Observation,
        *,
        num_steps: int = 10,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample actions given an observation."""

    def forward(
        self,
        observation: Observation,
        actions: torch.Tensor,
        *,
        train: bool = True,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Default forward pass computes the loss."""
        return self.compute_loss(observation, actions, train=train, rng=rng)
