# Copyright 2026 The SimpleTac Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""T3-large encoder for SimpleTouch tactile observations and future targets.

UniVTAC exposes two GelSight Mini pads. They share the official T3-large
``mini`` sensor encoder and shared trunk. The caller can select either the
final CLS token or all 197 final tokens from each pad. This module itself
does not contain a projection or task-specific adapter.
"""

from __future__ import annotations

import pathlib
from functools import partial

import timm.models.vision_transformer as timm_vit
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from .t3_preprocess_profile import (
    T3PreprocessProfile,
    resolve_t3_preprocess_profile,
)

T3_LARGE_WIDTH = 1024
T3_IMAGE_SIZE = 224
T3_RESIZE_SIZE = 256
T3_PATCH_SIZE = 16
T3_TOKENS_PER_SENSOR = (T3_IMAGE_SIZE // T3_PATCH_SIZE) ** 2 + 1
T3_SENSOR_DEPTH = 6
T3_TRUNK_DEPTH = 9
# T3 Table 1 specifies eight heads for the 1024-wide large model.
T3_NUM_HEADS = 8
T3_INPUT_MEAN = (0.485, 0.456, 0.406)
T3_INPUT_STD = (0.229, 0.224, 0.225)
T3_LEGACY_CACHE_IMPLEMENTATION_VERSION = (
    "t3_large_mini_official_eval_v3_univtac_rgb_order"
)
T3_LEGACY_PREPROCESS_VERSION = "rgb01_resize_shorter256_centercrop224_imagenet_norm_v1"
# Bump this value whenever a semantic change alters frozen-T3 cache values.
# Cache manifests additionally record a SHA256 of this source file, so a stale
# cache is rejected even if this constant is accidentally left unchanged.
T3_CACHE_IMPLEMENTATION_VERSION = (
    "t3_large_mini_deterministic_eval_v4_profiled_rgb_order"
)
T3_PREPROCESS_VERSION = "resolved_t3_preprocess_profile_v1"


class _T3MiniSensorEncoder(nn.Module):
    """T3-large sensor-specific ViT stem, matching ``encoders/mini.pth``."""

    def __init__(self) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = timm_vit.PatchEmbed(
            img_size=T3_IMAGE_SIZE,
            patch_size=T3_PATCH_SIZE,
            in_chans=3,
            embed_dim=T3_LARGE_WIDTH,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, T3_LARGE_WIDTH))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, T3_LARGE_WIDTH))
        self.pos_drop = nn.Dropout(p=0.0)
        self.blocks = nn.ModuleList(
            [
                timm_vit.Block(
                    T3_LARGE_WIDTH,
                    T3_NUM_HEADS,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(T3_SENSOR_DEPTH)
            ]
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images)
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = self.pos_drop(torch.cat([cls, tokens], dim=1) + self.pos_embed)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class _T3SharedTrunk(nn.Module):
    """T3-large shared transformer trunk, matching ``trunk.pth``."""

    def __init__(self) -> None:
        super().__init__()
        # The official sensor encoder uses eps=1e-6, whereas TransformerTrunk
        # uses PyTorch LayerNorm's default eps=1e-5.
        norm_layer = nn.LayerNorm
        self.blocks = nn.ModuleList(
            [
                timm_vit.Block(
                    T3_LARGE_WIDTH,
                    T3_NUM_HEADS,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(T3_TRUNK_DEPTH)
            ]
        )
        self.norm = norm_layer(T3_LARGE_WIDTH)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


def _load_exact(module: nn.Module, path: pathlib.Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing T3-large checkpoint: {path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    incompatible = module.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"T3-large checkpoint mismatch for {path}: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


class T3LargeTactileEncoder(nn.Module):
    """Encode left/right 224x224 RGB tactile images into T3-large tokens."""

    def __init__(
        self,
        checkpoint_dir: str | pathlib.Path,
        *,
        preprocess_profile: str | None = None,
        task: str | None = None,
    ) -> None:
        super().__init__()
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        # Plain immutable metadata deliberately creates no persistent state-dict
        # key, so old T3-containing checkpoints still load strictly.
        self.preprocess_profile = resolve_t3_preprocess_profile(
            preprocess_profile,
            task=task,
        )
        self.sensor_encoder = _T3MiniSensorEncoder()
        self.shared_trunk = _T3SharedTrunk()
        self.gradient_checkpointing = False
        self.gradient_checkpointing_use_reentrant = False
        _load_exact(self.sensor_encoder, checkpoint_dir / "encoders" / "mini.pth")
        _load_exact(self.shared_trunk, checkpoint_dir / "trunk.pth")

    def gradient_checkpointing_enable(self, *, use_reentrant: bool = False) -> None:
        self.gradient_checkpointing = True
        self.gradient_checkpointing_use_reentrant = use_reentrant

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def output_reference_rms(self) -> float:
        """Return the frozen trunk's data-independent output RMS reference.

        The final LayerNorm presents unit-RMS activations to its learned affine
        transform.  Consequently ``sqrt(mean(weight**2 + bias**2))`` is the
        natural global scale of T3's published output coordinate system.  It
        is fixed by the checkpoint, independent of batch composition, and can
        therefore be used as an invertible diffusion-target scale factor.
        """
        norm = self.shared_trunk.norm
        with torch.no_grad():
            reference_rms = (
                (
                    norm.weight.detach().float().square()
                    + norm.bias.detach().float().square()
                )
                .mean()
                .sqrt()
            )
        value = float(reference_rms.item())
        if not torch.isfinite(reference_rms).item() or value <= 0.0:
            raise ValueError(
                f"T3 final LayerNorm produced an invalid output RMS reference: {value}"
            )
        return value

    def forward(
        self, tactile_images: torch.Tensor, *, return_all_tokens: bool = False
    ) -> torch.Tensor:
        """Return one CLS token or all 197 tokens for each tactile sensor."""
        if (
            tactile_images.ndim != 5
            or tactile_images.shape[1] != 2
            or tactile_images.shape[2] != 3
        ):
            raise ValueError(
                "Expected tactile_images shaped [B, 2, 3, H, W] in left/right "
                "order, "
                f"got {tuple(tactile_images.shape)}"
            )
        batch_size = tactile_images.shape[0]
        images = tactile_images.reshape(
            batch_size * 2,
            3,
            tactile_images.shape[-2],
            tactile_images.shape[-1],
        )
        images = self.preprocess(images, profile=self.preprocess_profile)
        dtype = self.sensor_encoder.patch_embed.proj.weight.dtype
        images = images.to(dtype=dtype)
        tokens = self.sensor_encoder.patch_embed(images)
        cls = self.sensor_encoder.cls_token.expand(images.shape[0], -1, -1)
        tokens = self.sensor_encoder.pos_drop(
            torch.cat([cls, tokens], dim=1) + self.sensor_encoder.pos_embed
        )
        for block in self.sensor_encoder.blocks:
            if self.gradient_checkpointing and self.training:
                tokens = torch.utils.checkpoint.checkpoint(
                    block,
                    tokens,
                    use_reentrant=self.gradient_checkpointing_use_reentrant,
                )
            else:
                tokens = block(tokens)
        for block in self.shared_trunk.blocks:
            if self.gradient_checkpointing and self.training:
                tokens = torch.utils.checkpoint.checkpoint(
                    block,
                    tokens,
                    use_reentrant=self.gradient_checkpointing_use_reentrant,
                )
            else:
                tokens = block(tokens)
        tokens = self.shared_trunk.norm(tokens)
        tokens = tokens.reshape(batch_size, 2, T3_TOKENS_PER_SENSOR, T3_LARGE_WIDTH)
        if return_all_tokens:
            return tokens
        return tokens[:, :, 0, :]

    @staticmethod
    def preprocess(
        images: torch.Tensor,
        *,
        profile: T3PreprocessProfile | str | None = None,
        task: str | None = None,
    ) -> torch.Tensor:
        """Apply deterministic evaluation geometry and resolved RGB z-score.

        ``Observation.from_dict`` supplies RGB tensors in ``[-1, 1]``. T3 was
        pretrained with RGB converted to ``[0, 1]``. Missing profile metadata
        intentionally preserves the pre-v11 ImageNet-normalized behavior.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"T3 preprocessing expects [N, 3, H, W], got {tuple(images.shape)}"
            )
        resolved_profile = (
            profile
            if isinstance(profile, T3PreprocessProfile)
            else resolve_t3_preprocess_profile(profile, task=task)
        )
        images = (
            images.float()
            .add(1.0)
            .mul(0.5)
            .clamp(
                resolved_profile.input_range[0],
                resolved_profile.input_range[1],
            )
        )
        height, width = images.shape[-2:]
        scale = resolved_profile.resize_shorter / min(height, width)
        resized_height = max(
            resolved_profile.crop_height,
            int(round(height * scale)),
        )
        resized_width = max(
            resolved_profile.crop_width,
            int(round(width * scale)),
        )
        images = F.interpolate(
            images,
            size=(resized_height, resized_width),
            mode=resolved_profile.interpolation,
            align_corners=resolved_profile.align_corners,
            antialias=resolved_profile.antialias,
        )
        top = (resized_height - resolved_profile.crop_height) // 2
        left = (resized_width - resolved_profile.crop_width) // 2
        images = images[
            :,
            :,
            top : top + resolved_profile.crop_height,
            left : left + resolved_profile.crop_width,
        ]
        mean = images.new_tensor(resolved_profile.mean).view(1, 3, 1, 1)
        std = images.new_tensor(resolved_profile.std).view(1, 3, 1, 1)
        return (images - mean) / std

    def encode_world_tokens(
        self,
        tactile_images: torch.Tensor,
        *,
        pool_grid: int = 14,
    ) -> torch.Tensor:
        """Return CLS plus spatial tokens for the frozen world model.

        The default ``pool_grid=14`` preserves T3-large's complete 14x14 patch
        grid and therefore returns all 197 tokens per sensor without pooling.
        Smaller grids remain available as an explicit efficiency ablation.
        """
        if pool_grid <= 0:
            raise ValueError(f"pool_grid must be positive, got {pool_grid}")
        tokens = self(tactile_images, return_all_tokens=True)
        return pool_t3_spatial_tokens(tokens, pool_grid=pool_grid)


def pool_t3_spatial_tokens(
    tokens: torch.Tensor,
    *,
    pool_grid: int,
) -> torch.Tensor:
    """Keep T3 CLS and adaptively average-pool its square patch grid.

    This function accepts both online T3 output and cached frozen-T3 tokens.
    Keeping it outside the encoder lets a full 14x14 cache serve the
    ``pooled_3x3`` world-model mode without regenerating cache files.
    """
    if tokens.ndim != 4:
        raise ValueError(
            "Expected T3 tokens shaped [B, sensors, tokens, width], got "
            f"{tuple(tokens.shape)}"
        )
    if pool_grid <= 0:
        raise ValueError(f"pool_grid must be positive, got {pool_grid}")
    cls = tokens[:, :, :1]
    spatial = tokens[:, :, 1:]
    batch_size, num_sensors, num_patches, width = spatial.shape
    side = int(num_patches**0.5)
    if side * side != num_patches:
        raise ValueError(f"T3 patch count must be square, got {num_patches}")
    if pool_grid > side:
        raise ValueError(
            "Cannot recover a finer T3 grid from pooled tokens: "
            f"requested {pool_grid}x{pool_grid}, source is {side}x{side}"
        )
    if pool_grid == side:
        return tokens
    grid = spatial.reshape(batch_size * num_sensors, side, side, width)
    grid = grid.permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid, pool_grid)
    pooled = pooled.flatten(2).transpose(1, 2)
    pooled = pooled.reshape(
        batch_size,
        num_sensors,
        pool_grid * pool_grid,
        width,
    )
    return torch.cat([cls, pooled], dim=2)
