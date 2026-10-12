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

"""Pi0 model for PyTorch, aligned with JAX models/pi0.py.

Flow matching model for continuous action generation.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
from collections.abc import Iterator

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import gemma, model, siglip
from .pi0_config import Pi0Config
from .t3_tactile_encoder import (
    T3_LARGE_WIDTH,
    T3_TOKENS_PER_SENSOR,
    T3LargeTactileEncoder,
    pool_t3_spatial_tokens,
)
from .utils import _str_to_dtype

logger = logging.getLogger("openpi")


@contextlib.contextmanager
def _defer_torch_parameter_initialization(enabled: bool) -> Iterator[None]:
    """Skip disposable ``torch.nn.init`` fills during checkpoint-backed builds.

    PyTorch module constructors initialize parameters even when a checkpoint is
    loaded immediately afterwards.  On an eight-rank CPU-constrained worker,
    independently filling the complete Pi0.5, SigLIP, T3, and third-expert
    tensors can dominate startup.  This narrowly scoped context retains tensor
    allocation while turning only in-place initialization helpers into no-ops.
    Callers must subsequently load every affected pretrained tensor.
    """
    if not enabled:
        yield
        return

    function_names = (
        "constant_",
        "dirac_",
        "eye_",
        "kaiming_normal_",
        "kaiming_uniform_",
        "normal_",
        "ones_",
        "orthogonal_",
        "sparse_",
        "trunc_normal_",
        "uniform_",
        "xavier_normal_",
        "xavier_uniform_",
        "zeros_",
    )
    originals = {
        name: getattr(nn.init, name)
        for name in function_names
        if hasattr(nn.init, name)
    }

    def passthrough(
        tensor: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        del args, kwargs
        return tensor

    try:
        for name in originals:
            setattr(nn.init, name, passthrough)
        yield
    finally:
        for name, function in originals.items():
            setattr(nn.init, name, function)


def make_attn_mask(input_mask: torch.Tensor, mask_ar: torch.Tensor) -> torch.Tensor:
    """Create attention mask from input mask and autoregressive mask.

    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs.

    Args:
        input_mask: bool[B, N] - true if token is valid
        mask_ar: bool[N] - true where next token starts a new autoregressive block
    """
    mask_ar = mask_ar.expand(input_mask.shape[0], -1)
    cumsum = torch.cumsum(mask_ar.int(), dim=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return torch.logical_and(attn_mask, valid_mask)


def posemb_sincos(
    pos: torch.Tensor,
    embedding_dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """Sine-cosine positional embedding for scalar positions.

    Args:
        pos: (B,) float positions
        embedding_dim: output dimension (must be even)

    Returns:
        (B, embedding_dim) positional embedding
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = torch.linspace(
        0.0, 1.0, embedding_dim // 2, device=pos.device, dtype=torch.float32
    )
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = torch.einsum("i,j->ij", pos.float(), 1.0 / period * 2 * torch.pi)
    # Match JAX which keeps posemb in float32. However, PT Linear does not support
    # mixed float32/bf16 matmul, so cast back to the model's embed_dtype.
    # The caller should upcast to float32 if needed for high-precision ops.
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1).to(
        pos.dtype
    )


class Pi0(model.BaseModel):
    """Pi0 flow matching model for continuous action generation."""

    def __init__(self, config: Pi0Config):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.embed_dtype = _str_to_dtype(config.dtype)
        self._config = config
        self.tactile_encoder_frozen = config.tactile_encoder_frozen
        self.tactile_expert_enabled = bool(config.tactile_expert_enabled)
        self.tactile_expert_wam_mode = config.tactile_expert_wam_mode
        self.tactile_expert_init_mode = str(config.tactile_expert_init_mode)
        self.tactile_expert_future_prediction_mode = str(
            config.tactile_expert_future_prediction_mode
        )
        self.tactile_expert_future_horizon = int(config.tactile_expert_future_horizon)
        self.tactile_expert_future_stride = int(config.tactile_expert_future_stride)
        self.tactile_expert_future_pool_grid = int(
            config.tactile_expert_future_pool_grid
        )
        self.tactile_expert_future_normalization = str(
            config.tactile_expert_future_normalization
        )
        self.tactile_expert_future_attention_backend = str(
            config.tactile_expert_future_attention_backend
        )

        paligemma_config = gemma.get_config(config.paligemma_variant)
        action_expert_config = gemma.get_config(config.action_expert_variant)

        expert_configs = [paligemma_config, action_expert_config]
        # The tactile branch is a real third expert: it owns all
        # projections, norms, and FFNs while sharing only the attention graph
        # with PaliGemma and the Action Expert.  Its 18 layer slots line up with
        # the Action Expert so the clean tactile pass contributes one distinct
        # K/V block at every layer.
        if self.tactile_expert_enabled:
            tactile_expert_config = gemma.Config(
                width=int(config.tactile_expert_width),
                depth=int(config.tactile_expert_depth),
                mlp_dim=int(config.tactile_expert_mlp_dim),
                num_heads=int(config.tactile_expert_num_heads),
                num_kv_heads=int(config.tactile_expert_num_kv_heads),
                head_dim=int(config.tactile_expert_head_dim),
            )
            expert_configs.append(tactile_expert_config)

        # Expert 0 (PaliGemma) uses regular RMSNorm. The two flow experts use
        # independent adaRMS modulation, hence independent diffusion times.
        adarms = [False, config.pi05]
        if self.tactile_expert_enabled:
            adarms.append(True)
        training_attention_backend = config.training_attention_backend
        if training_attention_backend == "auto":
            training_attention_backend = (
                "sdpa" if self.tactile_expert_wam_mode == "idm" else "eager"
            )
        with _defer_torch_parameter_initialization(config.defer_pretrained_weight_init):
            self.llm = gemma.Module(
                configs=expert_configs,
                embed_dtype=config.dtype,
                adarms=adarms,
                use_gradient_checkpointing=False,
                training_attention_backend=training_attention_backend,
            )

            # SigLIP vision encoder
            self.img = siglip.SigLIPViT(
                variant="So400m/14",
                pool_type="none",
                num_classes=paligemma_config.width,
                use_gradient_checkpointing=False,
                dtype_mm=config.dtype,
            )
        logger.info("Pi0.5 Gemma/SigLIP structures allocated")

        action_expert_width = action_expert_config.width
        if self.tactile_expert_enabled:
            if not config.tactile_encoder_path:
                raise ValueError(
                    "tactile_encoder_path is required when an online tactile "
                    "policy input is enabled"
                )
            # Encode current touch online once for the Tactile Expert.
            with _defer_torch_parameter_initialization(
                config.defer_pretrained_weight_init
            ):
                self.tactile_encoder = T3LargeTactileEncoder(
                    config.tactile_encoder_path,
                    preprocess_profile=config.tactile_preprocess_profile,
                    task=config.tactile_preprocess_task,
                )
            logger.info("Primary T3 tactile encoder loaded")
            if self.tactile_encoder_frozen:
                self.tactile_encoder.requires_grad_(False)
                self.tactile_encoder.eval()
            if self.tactile_expert_enabled:
                # Store a Python scalar before distributed wrapping. Unlike
                # batch RMS, this value is deterministic across ranks/runs and
                # does not add checkpoint state or change current-touch T3.
                self.tactile_expert_future_reference_rms = (
                    self.tactile_encoder.output_reference_rms()
                )
        if self.tactile_expert_enabled:
            tactile_width = tactile_expert_config.width
            # Checkpoint-backed construction fills every tensor below either
            # from the complete checkpoint or via one of the explicit TE
            # initializers after loading the verified Pi0.5 base.
            with _defer_torch_parameter_initialization(
                config.defer_pretrained_weight_init
            ):
                self.tactile_expert_in_proj = nn.Linear(
                    T3_LARGE_WIDTH,
                    tactile_width,
                )
                self.tactile_expert_out_proj = nn.Linear(
                    tactile_width,
                    T3_LARGE_WIDTH,
                )
                self.tactile_expert_sensor_embedding = nn.Embedding(
                    2,
                    tactile_width,
                )
                self.tactile_expert_time_mlp_in = nn.Linear(
                    tactile_width,
                    tactile_width,
                )
                self.tactile_expert_time_mlp_out = nn.Linear(
                    tactile_width,
                    tactile_width,
                )
        self.action_dim = config.action_dim

        # Action input projection
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_width)

        if config.pi05:
            self.time_mlp_in = nn.Linear(action_expert_width, action_expert_width)
            self.time_mlp_out = nn.Linear(action_expert_width, action_expert_width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_width)
            self.action_time_mlp_in = nn.Linear(
                2 * action_expert_width, action_expert_width
            )
            self.action_time_mlp_out = nn.Linear(
                action_expert_width, action_expert_width
            )

        # Action output projection
        self.action_out_proj = nn.Linear(action_expert_width, config.action_dim)

        # Optional PointNet

        self._init_weights()
        if self.tactile_expert_enabled:
            tactile_parameter_count = self._tactile_expert_parameter_count()
            future_tokens = 2 * T3_TOKENS_PER_SENSOR
            if self.tactile_expert_future_prediction_mode == "multi_horizon_pooled":
                future_steps = (
                    self.tactile_expert_future_horizon
                    // self.tactile_expert_future_stride
                )
                future_tokens = (
                    future_steps * 2 * (1 + self.tactile_expert_future_pool_grid**2)
                )
            logger.info(
                "Tactile Expert enabled: wam_mode=%s params=%d current_tokens=%d "
                "future_mode=%s future_tokens=%d layers=%d init=%s "
                "preserve_t3_width=%s "
                "future_normalization=%s future_reference_rms=%.8f "
                "future_attention_backend=%s "
                "gradient_checkpointing=%s",
                self.tactile_expert_wam_mode,
                tactile_parameter_count,
                2 * T3_TOKENS_PER_SENSOR,
                self.tactile_expert_future_prediction_mode,
                future_tokens,
                len(self.llm.layers),
                self.tactile_expert_init_mode,
                True,
                self.tactile_expert_future_normalization,
                self.tactile_expert_future_reference_rms,
                self.tactile_expert_future_attention_backend,
                self.llm.gradient_checkpointing,
            )

    def _tactile_expert_parameter_count(self) -> int:
        """Count parameters owned exclusively by the third expert path."""
        if not self.tactile_expert_enabled:
            return 0
        modules: list[nn.Module] = [
            self.tactile_expert_in_proj,
            self.tactile_expert_out_proj,
            self.tactile_expert_sensor_embedding,
            self.tactile_expert_time_mlp_in,
            self.tactile_expert_time_mlp_out,
            self.llm.final_norms[2],
        ]
        for block in self.llm.layers:
            modules.extend(
                [
                    block.attn.q_proj[2],
                    block.attn.k_proj[2],
                    block.attn.v_proj[2],
                    block.attn.o_proj[2],
                    block.pre_attention_norms[2],
                    block.pre_ffw_norms[2],
                    block.mlps[2],
                ]
            )
        unique_parameters = {
            id(parameter): parameter
            for expert_module in modules
            if expert_module is not None
            for parameter in expert_module.parameters()
        }
        return sum(parameter.numel() for parameter in unique_parameters.values())

    def _init_weights(self):
        """Initialize projection weights."""
        nn.init.normal_(self.action_in_proj.weight, std=0.02)
        nn.init.zeros_(self.action_in_proj.bias)
        nn.init.normal_(self.action_out_proj.weight, std=0.02)
        nn.init.zeros_(self.action_out_proj.bias)

        if self.pi05:
            nn.init.normal_(self.time_mlp_in.weight, std=0.02)
            nn.init.zeros_(self.time_mlp_in.bias)
            nn.init.normal_(self.time_mlp_out.weight, std=0.02)
            nn.init.zeros_(self.time_mlp_out.bias)
        else:
            nn.init.normal_(self.state_proj.weight, std=0.02)
            nn.init.zeros_(self.state_proj.bias)
            nn.init.normal_(self.action_time_mlp_in.weight, std=0.02)
            nn.init.zeros_(self.action_time_mlp_in.bias)
            nn.init.normal_(self.action_time_mlp_out.weight, std=0.02)
            nn.init.zeros_(self.action_time_mlp_out.bias)

        if self.tactile_expert_enabled and (
            not self._config.defer_pretrained_weight_init
        ):
            nn.init.eye_(self.tactile_expert_in_proj.weight)
            nn.init.zeros_(self.tactile_expert_in_proj.bias)
            nn.init.normal_(self.tactile_expert_out_proj.weight, std=0.02)
            nn.init.zeros_(self.tactile_expert_out_proj.bias)
            # Start without inventing sensor offsets on top of frozen T3
            # content. The embedding remains trainable from the first step.
            nn.init.zeros_(self.tactile_expert_sensor_embedding.weight)
            nn.init.normal_(self.tactile_expert_time_mlp_in.weight, std=0.02)
            nn.init.zeros_(self.tactile_expert_time_mlp_in.bias)
            nn.init.normal_(self.tactile_expert_time_mlp_out.weight, std=0.02)
            nn.init.zeros_(self.tactile_expert_time_mlp_out.bias)

    @staticmethod
    def _copy_expert_parameter(
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Copy the identically sized Action-Expert tensor without interpolation."""
        if source.shape != target.shape:
            raise ValueError(f"AE/TE shape mismatch: {source.shape} != {target.shape}")
        return source.detach().to(device=target.device, dtype=target.dtype)

    def initialize_tactile_expert_from_action_expert(self) -> None:
        """Bootstrap expert 2 from the loaded base Action Expert (expert 1).

        The checkpoint loader calls this exactly when loading a verified base
        checkpoint that has no tactile-expert state.  A trained tactile
        checkpoint is always loaded strictly and never reinitialized.
        """
        if not self.tactile_expert_enabled:
            return

        # These interface tensors have no Action-Expert counterpart. Reset
        # them deterministically because checkpoint-backed construction defers
        # initialization of the transformer body on every distributed rank.
        generator = self._new_tactile_expert_init_generator()
        self._initialize_tactile_expert_interface(generator=generator)

        def copy_parameter(target: torch.Tensor, source: torch.Tensor) -> None:
            with torch.no_grad():
                target.copy_(self._copy_expert_parameter(source, target))

        for block in self.llm.layers:
            for projections in (
                block.attn.q_proj,
                block.attn.k_proj,
                block.attn.v_proj,
                block.attn.o_proj,
            ):
                source_projection = projections[1]
                target_projection = projections[2]
                if source_projection is None or target_projection is None:
                    if source_projection is not target_projection:
                        raise RuntimeError(
                            "Action and tactile experts use incompatible attention "
                            "projection layouts"
                        )
                    continue
                copy_parameter(target_projection.weight, source_projection.weight)

            for norms in (block.pre_attention_norms, block.pre_ffw_norms):
                source_norm = norms[1]
                target_norm = norms[2]
                if not source_norm.adaptive or not target_norm.adaptive:
                    raise RuntimeError(
                        "Action-to-tactile initialization requires adaptive RMSNorm"
                    )
                copy_parameter(
                    target_norm.ada_modulation.weight,
                    source_norm.ada_modulation.weight,
                )
                copy_parameter(
                    target_norm.ada_modulation.bias,
                    source_norm.ada_modulation.bias,
                )

            source_mlp = block.mlps[1]
            target_mlp = block.mlps[2]
            if not isinstance(source_mlp, gemma.FeedForward) or not isinstance(
                target_mlp, gemma.FeedForward
            ):
                raise RuntimeError(
                    "Action-to-tactile initialization requires Gemma FeedForward modules"
                )
            copy_parameter(target_mlp.w_gating, source_mlp.w_gating)
            copy_parameter(target_mlp.w_linear, source_mlp.w_linear)

        source_final_norm = self.llm.final_norms[1]
        target_final_norm = self.llm.final_norms[2]
        copy_parameter(
            target_final_norm.ada_modulation.weight,
            source_final_norm.ada_modulation.weight,
        )
        copy_parameter(
            target_final_norm.ada_modulation.bias,
            source_final_norm.ada_modulation.bias,
        )
        copy_parameter(self.tactile_expert_time_mlp_in.weight, self.time_mlp_in.weight)
        copy_parameter(self.tactile_expert_time_mlp_in.bias, self.time_mlp_in.bias)
        copy_parameter(
            self.tactile_expert_time_mlp_out.weight,
            self.time_mlp_out.weight,
        )
        copy_parameter(self.tactile_expert_time_mlp_out.bias, self.time_mlp_out.bias)

    def _new_tactile_expert_init_generator(self) -> torch.Generator:
        """Return a local seeded generator without changing process RNG state."""
        device = self.tactile_expert_in_proj.weight.device
        if device.type == "meta":
            raise RuntimeError("Cannot initialize Tactile Expert meta tensors")
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        return generator

    def _initialize_tactile_expert_interface(
        self,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        """Initialize T3 adapters and future-flow projections reproducibly."""
        nn.init.eye_(self.tactile_expert_in_proj.weight)
        nn.init.zeros_(self.tactile_expert_in_proj.bias)
        nn.init.normal_(
            self.tactile_expert_out_proj.weight,
            std=0.02,
            generator=generator,
        )
        nn.init.zeros_(self.tactile_expert_out_proj.bias)
        nn.init.zeros_(self.tactile_expert_sensor_embedding.weight)
        nn.init.normal_(
            self.tactile_expert_time_mlp_in.weight,
            std=0.02,
            generator=generator,
        )
        nn.init.zeros_(self.tactile_expert_time_mlp_in.bias)
        nn.init.normal_(
            self.tactile_expert_time_mlp_out.weight,
            std=0.02,
            generator=generator,
        )
        nn.init.zeros_(self.tactile_expert_time_mlp_out.bias)

    def initialize_tactile_expert_random(self) -> None:
        """Initialize expert 2 independently with native Pi0/Gemma rules.

        A private fixed-seed generator gives every distributed rank identical
        new weights without changing any process CPU/CUDA RNG stream. This is
        a full random expert initialization; only the
        same-width T3 adapter is deliberately identity initialized so it does
        not destroy the frozen representation before the first Transformer.
        """
        if not self.tactile_expert_enabled:
            return
        generator = self._new_tactile_expert_init_generator()
        self._initialize_tactile_expert_interface(generator=generator)
        tactile_config = self.llm.configs[2]
        for block in self.llm.layers:
            attention = block.attn
            nn.init.normal_(
                attention.q_proj[2].weight,
                std=1.0 / tactile_config.width**0.5,
                generator=generator,
            )
            if attention.k_proj[2] is not None:
                nn.init.normal_(
                    attention.k_proj[2].weight,
                    std=1.0 / tactile_config.width**0.5,
                    generator=generator,
                )
                nn.init.normal_(
                    attention.v_proj[2].weight,
                    std=1.0 / tactile_config.width**0.5,
                    generator=generator,
                )
            nn.init.normal_(
                attention.o_proj[2].weight,
                std=1.0 / (tactile_config.num_heads * tactile_config.head_dim) ** 0.5,
                generator=generator,
            )
            for norms in (block.pre_attention_norms, block.pre_ffw_norms):
                nn.init.zeros_(norms[2].ada_modulation.weight)
                nn.init.zeros_(norms[2].ada_modulation.bias)
            tactile_mlp = block.mlps[2]
            if not isinstance(tactile_mlp, gemma.FeedForward):
                raise RuntimeError(
                    "Tactile Expert initialization requires Gemma FeedForward modules"
                )
            nn.init.normal_(
                tactile_mlp.w_gating[0],
                std=1.0 / tactile_mlp.features**0.5,
                generator=generator,
            )
            nn.init.normal_(
                tactile_mlp.w_gating[1],
                std=1.0 / tactile_mlp.features**0.5,
                generator=generator,
            )
            nn.init.normal_(
                tactile_mlp.w_linear,
                std=1.0 / tactile_mlp.hidden_dim**0.5,
                generator=generator,
            )
        final_norm = self.llm.final_norms[2]
        nn.init.zeros_(final_norm.ada_modulation.weight)
        nn.init.zeros_(final_norm.ada_modulation.bias)

    def initialize_tactile_expert_from_base_checkpoint(self) -> None:
        """Apply the configured initialization after loading the Pi0.5 base."""
        if self.tactile_expert_init_mode == "random":
            self.initialize_tactile_expert_random()
        elif self.tactile_expert_init_mode == "from_action_expert":
            self.initialize_tactile_expert_from_action_expert()
        else:
            raise RuntimeError(
                "Unsupported Tactile Expert initialization: "
                f"{self.tactile_expert_init_mode!r}"
            )

    @staticmethod
    def _stack_tactile_images(
        tactile_images: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Standardize current or multi-horizon left/right GelSight frames.

        Returns ``[B, 2, 3, H, W]`` for one endpoint and
        ``[B, K, 2, 3, H, W]`` for a sequence of future endpoints.
        """
        if tactile_images is None:
            raise ValueError("Tactile-enabled SimpleTac requires tactile images")
        tactile_keys = ("left_gsmini_rgb", "right_gsmini_rgb")
        missing = [key for key in tactile_keys if key not in tactile_images]
        if missing:
            raise ValueError(f"Missing tactile image keys: {missing}")

        images = []
        for key in tactile_keys:
            image = tactile_images[key]
            if image.ndim not in (4, 5):
                raise ValueError(
                    f"Tactile image {key!r} must be rank 4 or 5, got {image.shape}"
                )
            if image.ndim == 4:
                if image.shape[-1] == 3:
                    image = image.permute(0, 3, 1, 2)
                elif image.shape[1] != 3:
                    raise ValueError(
                        f"Tactile image {key!r} must be NHWC or NCHW RGB, "
                        f"got {image.shape}"
                    )
            elif image.shape[-1] == 3:
                image = image.permute(0, 1, 4, 2, 3)
            elif image.shape[2] != 3:
                raise ValueError(
                    f"Multi-horizon tactile image {key!r} must be BKHWC or "
                    f"BKCHW RGB, got {image.shape}"
                )
            images.append(image)
        sensor_dim = 1 if images[0].ndim == 4 else 2
        return torch.stack(images, dim=sensor_dim)

    def encode_policy_tactile(
        self,
        obs: model.Observation,
    ) -> torch.Tensor | None:
        """Encode current left/right touch once for all online policy paths."""
        if not getattr(self, "tactile_expert_enabled", False):
            return None
        tactile_images = self._stack_tactile_images(obs.tactile_images)
        if self.tactile_encoder_frozen:
            with torch.no_grad():
                return self.tactile_encoder(
                    tactile_images,
                    return_all_tokens=True,
                )
        return self.tactile_encoder(
            tactile_images,
            return_all_tokens=True,
        )

    def encode_future_tactile(
        self,
        obs: model.Observation,
    ) -> torch.Tensor | None:
        """Encode training-only future tactile target(s) with the frozen T3."""
        if not self.tactile_expert_enabled:
            return None
        if obs.tactile_future_images is None:
            return None
        tactile_images = self._stack_tactile_images(obs.tactile_future_images)
        # The new recipe requires an immutable T3 coordinate system for both
        # endpoints.  Keeping this unconditional makes an accidental config or
        # train/eval-mode regression fail at construction rather than silently
        # changing the target during optimization.
        if not self.tactile_encoder_frozen:
            raise RuntimeError("Tactile Expert targets require a frozen T3 encoder")
        if tactile_images.ndim == 6:
            return self._encode_multi_horizon_future_tactile(tactile_images)
        with torch.no_grad():
            encoded = self.tactile_encoder(
                tactile_images,
                return_all_tokens=True,
            )
        return self._select_tactile_expert_future_tokens(encoded)

    def _encode_multi_horizon_future_tactile(
        self,
        future_images: torch.Tensor,
    ) -> torch.Tensor:
        """Encode every future endpoint in one throughput-oriented T3 batch."""
        if future_images.ndim != 6:
            raise ValueError(
                "Multi-horizon tactile images must be [B, K, 2, 3, H, W], "
                f"got {tuple(future_images.shape)}"
            )
        batch_size, future_steps = future_images.shape[:2]
        with torch.no_grad():
            encoded = self.tactile_encoder(
                future_images.flatten(0, 1),
                return_all_tokens=True,
            )
        encoded = encoded.reshape(batch_size, future_steps, *encoded.shape[1:])
        return self._select_tactile_expert_future_tokens(encoded)

    def encode_tactile_expert_endpoints(
        self,
        obs: model.Observation,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode current touch; trainable T3 uses immutable cached targets."""
        if not self.tactile_expert_enabled:
            raise RuntimeError("Tactile Expert endpoint encoding requires TE enabled")
        cached_current = getattr(obs, "tactile_world_latents", None)
        cached_future = getattr(obs, "tactile_world_future_latents", None)
        if not self.tactile_encoder_frozen:
            if cached_current is not None:
                raise ValueError("Trainable T3 requires online current tactile images")
            if cached_future is None:
                raise ValueError(
                    "Trainable T3 requires immutable cached future targets"
                )
        current = None
        if cached_current is not None:
            if cached_current.ndim != 4 or tuple(cached_current.shape[1:]) != (
                2,
                T3_TOKENS_PER_SENSOR,
                T3_LARGE_WIDTH,
            ):
                raise ValueError(
                    "Cached current T3 latents must be [B, 2, 197, D], got "
                    f"{tuple(cached_current.shape)}"
                )
            current = cached_current.to(dtype=self.embed_dtype)
        if cached_future is not None:
            prediction_mode = getattr(
                self,
                "tactile_expert_future_prediction_mode",
                "single_horizon_full",
            )
            if prediction_mode == "single_horizon_full":
                valid_shape = cached_future.ndim == 4 and tuple(
                    cached_future.shape[-3:]
                ) == (2, T3_TOKENS_PER_SENSOR, T3_LARGE_WIDTH)
                expected = "[B, 2, 197, 1024]"
            elif prediction_mode == "multi_horizon_pooled":
                pooled_tokens = (
                    1 + getattr(self, "tactile_expert_future_pool_grid", 6) ** 2
                )
                valid_shape = (
                    cached_future.ndim == 5
                    and cached_future.shape[-3] == 2
                    and cached_future.shape[-2] in {T3_TOKENS_PER_SENSOR, pooled_tokens}
                    and cached_future.shape[-1] == T3_LARGE_WIDTH
                )
                expected = (
                    "[B, K, 2, 197, 1024] or already-pooled "
                    f"[B, K, 2, {pooled_tokens}, 1024]"
                )
            else:
                raise RuntimeError(
                    f"Unsupported future prediction mode: {prediction_mode!r}"
                )
            if not valid_shape:
                raise ValueError(
                    "Cached future T3 latents have the wrong layout: expected "
                    f"{expected}, got {tuple(cached_future.shape)}"
                )
            cached_future = cached_future.detach().to(dtype=self.embed_dtype)
        if current is not None:
            if cached_future is not None:
                return current, self._select_tactile_expert_future_tokens(cached_future)
            if obs.tactile_future_images is None:
                return current, None
            return current, self.encode_future_tactile(obs)

        current_images = self._stack_tactile_images(obs.tactile_images)
        if cached_future is not None or obs.tactile_future_images is None:
            current = self.encode_policy_tactile(obs)
            future = (
                None
                if cached_future is None
                else self._select_tactile_expert_future_tokens(cached_future)
            )
            return current, future

        future_images = self._stack_tactile_images(obs.tactile_future_images)
        single_future = future_images.ndim == current_images.ndim
        future_geometry = (
            future_images.shape[1:] if single_future else future_images.shape[2:]
        )
        if current_images.shape[1:] != future_geometry:
            # Dataset frames normally share geometry. Keep the API robust to a
            # custom loader without forcing a resize outside the immutable T3
            # preprocessing profile.
            current = self.encode_policy_tactile(obs)
            if current is None:
                raise RuntimeError("Tactile Expert failed to encode current touch")
            return current, self.encode_future_tactile(obs)
        if not single_future:
            batch_size, future_steps = future_images.shape[:2]
            # T3 is frozen, and H20 memory is ample for the complete endpoint
            # batch. Combining current+t1...tK avoids K+1 serial encoder calls
            # and lets the accelerator process all sensor images at once.
            all_images = torch.cat(
                [current_images.unsqueeze(1), future_images],
                dim=1,
            )
            with torch.no_grad():
                endpoints = self.tactile_encoder(
                    all_images.flatten(0, 1),
                    return_all_tokens=True,
                )
            endpoints = endpoints.reshape(
                batch_size,
                future_steps + 1,
                *endpoints.shape[1:],
            )
            current = endpoints[:, 0]
            future = self._select_tactile_expert_future_tokens(endpoints[:, 1:])
            return current, future
        batch_size = current_images.shape[0]
        with torch.no_grad():
            endpoints = self.tactile_encoder(
                torch.cat([current_images, future_images], dim=0),
                return_all_tokens=True,
            )
        future = endpoints[batch_size:]
        return endpoints[:batch_size], self._select_tactile_expert_future_tokens(future)

    def _select_tactile_expert_future_tokens(
        self,
        future_tactile: torch.Tensor,
    ) -> torch.Tensor:
        """Apply only the configured training-target spatial compression."""
        mode = getattr(
            self,
            "tactile_expert_future_prediction_mode",
            "single_horizon_full",
        )
        if mode == "single_horizon_full":
            if future_tactile.ndim != 4:
                raise ValueError(
                    "single_horizon_full expects [B, 2, 197, D], got "
                    f"{tuple(future_tactile.shape)}"
                )
            return future_tactile
        if mode != "multi_horizon_pooled":
            raise RuntimeError(f"Unsupported future prediction mode: {mode!r}")
        if future_tactile.ndim != 5:
            raise ValueError(
                "multi_horizon_pooled expects [B, K, 2, T, D], got "
                f"{tuple(future_tactile.shape)}"
            )
        pooled_tokens = 1 + getattr(self, "tactile_expert_future_pool_grid", 6) ** 2
        if future_tactile.shape[-3] != 2 or future_tactile.shape[-1] != T3_LARGE_WIDTH:
            raise ValueError(
                "multi_horizon_pooled requires two T3 sensors with width "
                f"{T3_LARGE_WIDTH}, got {tuple(future_tactile.shape)}"
            )
        if future_tactile.shape[-2] == pooled_tokens:
            # Dedicated future caches persist the exact training target. Do
            # not pool it again or perturb its values on the GPU.
            return future_tactile
        if future_tactile.shape[-2] != T3_TOKENS_PER_SENSOR:
            raise ValueError(
                "multi_horizon_pooled requires either complete 197-token T3 "
                f"grids or {pooled_tokens} pre-pooled tokens, got "
                f"{tuple(future_tactile.shape)}"
            )
        batch_size, future_steps = future_tactile.shape[:2]
        pooled = pool_t3_spatial_tokens(
            future_tactile.flatten(0, 1),
            pool_grid=getattr(self, "tactile_expert_future_pool_grid", 6),
        )
        return pooled.reshape(batch_size, future_steps, *pooled.shape[1:])

    def normalize_tactile_expert_future(
        self,
        future_tactile: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize only a generated future-T3 endpoint for flow matching."""
        mode = self.tactile_expert_future_normalization
        if mode == "none":
            return future_tactile
        if mode != "t3_reference_rms":
            raise RuntimeError(
                f"Unsupported Tactile Expert future normalization: {mode!r}"
            )
        reference_rms = self.tactile_expert_future_reference_rms
        if not math.isfinite(reference_rms) or reference_rms <= 0.0:
            raise RuntimeError(
                "Tactile Expert future reference RMS must be finite and positive, "
                f"got {reference_rms}"
            )
        return future_tactile / reference_rms

    def denormalize_tactile_expert_future(
        self,
        normalized_future_tactile: torch.Tensor,
    ) -> torch.Tensor:
        """Map a generated future endpoint back to the raw frozen-T3 scale."""
        mode = self.tactile_expert_future_normalization
        if mode == "none":
            return normalized_future_tactile
        if mode != "t3_reference_rms":
            raise RuntimeError(
                f"Unsupported Tactile Expert future normalization: {mode!r}"
            )
        return normalized_future_tactile * self.tactile_expert_future_reference_rms

    def _embed_tactile_expert_tokens(
        self,
        tactile_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Adapt full left/right T3 tokens without duplicating T3 positions."""
        expected = (2, T3_TOKENS_PER_SENSOR, T3_LARGE_WIDTH)
        if tactile_latents.ndim != 4 or tuple(tactile_latents.shape[1:]) != expected:
            raise ValueError(
                "Tactile Expert requires full T3 latents shaped "
                f"[B, {expected[0]}, {expected[1]}, {expected[2]}], got "
                f"{tuple(tactile_latents.shape)}"
            )
        projection = self.tactile_expert_in_proj
        sensor_table = self.tactile_expert_sensor_embedding
        tokens = projection(tactile_latents)
        sensor_ids = torch.arange(2, device=tokens.device)
        sensor_embedding = sensor_table(sensor_ids).to(tokens.dtype)
        tokens = tokens + sensor_embedding[None, :, None, :]
        return tokens.flatten(1, 2)

    def _embed_tactile_expert_future_tokens(
        self,
        tactile_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Embed one full endpoint or a pooled multi-horizon trajectory."""
        prediction_mode = getattr(
            self,
            "tactile_expert_future_prediction_mode",
            "single_horizon_full",
        )
        if prediction_mode == "single_horizon_full":
            return self._embed_tactile_expert_tokens(tactile_latents)
        expected_tokens = (
            1
            + getattr(
                self,
                "tactile_expert_future_pool_grid",
                6,
            )
            ** 2
        )
        expected_tail = (2, expected_tokens, T3_LARGE_WIDTH)
        if (
            tactile_latents.ndim != 5
            or tuple(tactile_latents.shape[2:]) != expected_tail
        ):
            raise ValueError(
                "Pooled multi-horizon T3 latents must be shaped "
                f"[B, K, {expected_tail[0]}, {expected_tail[1]}, "
                f"{expected_tail[2]}], got {tuple(tactile_latents.shape)}"
            )
        tokens = self.tactile_expert_in_proj(tactile_latents)
        sensor_ids = torch.arange(2, device=tokens.device)
        sensor_embedding = self.tactile_expert_sensor_embedding(sensor_ids).to(
            tokens.dtype
        )
        tokens = tokens + sensor_embedding[None, None, :, None, :]
        return tokens.flatten(1, 3)

    def _embed_tactile_expert_time(self, time: torch.Tensor) -> torch.Tensor:
        """Return the tactile flow time used by this expert's adaRMS layers."""
        width = self.tactile_expert_time_mlp_in.in_features
        time_embedding = posemb_sincos(time.to(self.embed_dtype), width)
        time_embedding = self.tactile_expert_time_mlp_in(time_embedding)
        time_embedding = F.silu(time_embedding)
        return F.silu(self.tactile_expert_time_mlp_out(time_embedding))

    def embed_prefix(
        self,
        obs: model.Observation,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed the vision-language prefix.

        Returns:
            tokens: (B, S, emb_dim) embedded tokens
            input_mask: (B, S) mask of valid tokens
            ar_mask: (S,) autoregressive mask (all False for prefix)
        """
        tokens = []
        input_mask = []
        ar_mask = []

        # Pi0.5 assigns fixed semantic/position slots to these three views.
        # Never derive their order from a caller's dict insertion order: doing
        # so would silently move head/left/right content to different Gemma
        # positions. Reject incomplete or extended image dictionaries instead
        # of encoding an ambiguous prefix.
        expected_image_keys = set(model.IMAGE_KEYS)
        actual_image_keys = set(obs.images)
        actual_mask_keys = set(obs.image_masks)
        missing_images = expected_image_keys - actual_image_keys
        extra_images = actual_image_keys - expected_image_keys
        missing_masks = expected_image_keys - actual_mask_keys
        extra_masks = actual_mask_keys - expected_image_keys
        if missing_images or extra_images or missing_masks or extra_masks:
            raise ValueError(
                "Pi0.5 image prefix must contain exactly the canonical "
                f"IMAGE_KEYS={model.IMAGE_KEYS}; missing_images="
                f"{sorted(missing_images)}, extra_images={sorted(extra_images)}, "
                f"missing_masks={sorted(missing_masks)}, "
                f"extra_masks={sorted(extra_masks)}"
            )

        # Embed images through SigLIP in canonical head/left/right order.
        for name in model.IMAGE_KEYS:
            image_tokens, _ = self.img(obs.images[name])  # (B, num_patches, width)
            tokens.append(image_tokens)

            # Image tokens use bidirectional attention
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        # Add language tokens
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.llm.embed(obs.tokenized_prompt)
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)
        ar_mask = torch.tensor(ar_mask, device=tokens.device)
        return tokens, input_mask, ar_mask

    def _embed_pi05_time(self, timestep: torch.Tensor) -> torch.Tensor:
        """Return Pi0.5's adaRMS conditioning for a flow timestep."""
        time_emb = posemb_sincos(
            timestep.to(self.embed_dtype),
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        )
        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        return F.silu(time_emb)

    def embed_suffix(
        self,
        obs: model.Observation,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Embed the suffix (state + noisy actions + time embedding).

        Args:
            obs: observation
            noisy_actions: (B, action_horizon, action_dim)
            timestep: (B,) float timestep values

        Returns:
            tokens: (B, S, emb_dim)
            input_mask: (B, S)
            ar_mask: (S,)
            adarms_cond: (B, emb_dim) or None
        """
        input_mask = []
        tokens = []

        B = noisy_actions.shape[0]

        # Cast to embed_dtype to ensure consistency between training and inference
        # (during training FSDP2 casts forward inputs, but inference may pass float32).
        noisy_actions = noisy_actions.to(self.embed_dtype)
        timestep = timestep.to(self.embed_dtype)

        if not self.pi05:
            # Add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(
                torch.ones(B, 1, dtype=torch.bool, device=state_token.device)
            )

        # Embed actions
        action_tokens = self.action_in_proj(noisy_actions)

        if self.pi05:
            # Time MLP for adaRMS conditioning
            action_expert_tokens = action_tokens
            adarms_cond = self._embed_pi05_time(timestep)
        else:
            # Mix timestep + action through MLP
            time_emb = posemb_sincos(
                timestep,
                self.action_in_proj.out_features,
                min_period=4e-3,
                max_period=4.0,
            )
            time_tokens = einops.repeat(
                time_emb, "b emb -> b s emb", s=self.action_horizon
            )
            action_time_tokens = torch.cat([action_tokens, time_tokens], dim=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = F.silu(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(
            torch.ones(
                action_expert_tokens.shape[:2],
                dtype=torch.bool,
                device=action_expert_tokens.device,
            )
        )

        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)

        # Build ar_mask with correct length matching input_mask.shape[1]
        ar_mask = torch.zeros(
            input_mask.shape[1], dtype=torch.bool, device=tokens.device
        )
        if not self.pi05:
            ar_mask[:2] = True
        else:
            ar_mask[0] = True

        return tokens, input_mask, ar_mask, adarms_cond

    def compute_loss(
        self,
        observation: model.Observation,
        actions: torch.Tensor,
        *,
        train: bool = False,
        rng: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        tactile_future_noise: torch.Tensor | None = None,
        tactile_future_time: torch.Tensor | None = None,
        return_auxiliary_losses: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute flow matching loss.

        Returns:
            loss: (B, action_horizon) per-timestep MSE loss
        """
        B = actions.shape[0]
        device = actions.device

        # Preprocess first (requries float32 for image ops),
        # then cast to model dtype for FSDP2 mixed precision compatibility.
        preprocess_kwargs = {"train": train, "rng": rng}
        observation = model.preprocess_observation(observation, **preprocess_kwargs)

        embed_dtype = self.embed_dtype
        observation = model._observation_to_dtype(observation, embed_dtype)
        actions = actions.to(dtype=embed_dtype)
        dtype = actions.dtype

        # Sample noise and time (or use provided values for reproducibility)
        if noise is None:
            noise = torch.randn(
                actions.shape, device=device, dtype=dtype, generator=rng
            )
        else:
            noise = noise.to(dtype=dtype)
        if time is None:
            time = (
                torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
                .sample((B,))
                .to(device=device, dtype=dtype)
            )
            time = time * 0.999 + 0.001
        else:
            time = time.to(dtype=dtype)
        time_expanded = time[:, None, None]

        # Flow matching interpolation
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        future_target: torch.Tensor | None = None
        wam_mode = getattr(self, "tactile_expert_wam_mode", "fast")
        tactile_future_loss_enabled = bool(
            (train or wam_mode != "fast")
            and getattr(self, "tactile_expert_enabled", False)
            and float(self._config.tactile_expert_future_loss_weight) > 0.0
        )
        if tactile_future_loss_enabled:
            encoded_tactile, future_target = self.encode_tactile_expert_endpoints(
                observation
            )
        else:
            encoded_tactile = self.encode_policy_tactile(observation)
        auxiliary_losses: dict[str, torch.Tensor] = {}
        if getattr(self, "tactile_expert_enabled", False):
            if encoded_tactile is None:
                raise RuntimeError("Tactile Expert requires current T3 output")
            if (
                tactile_future_loss_enabled
                and observation.tactile_future_target_valid is None
            ):
                raise ValueError(
                    "Tactile Expert training requires explicit future-target "
                    "validity; terminal placeholders must never be supervised"
                )
            vl_context, vl_prefix_mask, context_kv_cache = self.build_prefix_cache(
                observation
            )
            context_mask = vl_prefix_mask
            context_mask, context_kv_cache = self.build_tactile_expert_cache(
                observation,
                encoded_tactile,
                context_mask,
                context_kv_cache,
            )
            if wam_mode == "fast":
                suffix_out = self.run_suffix(
                    observation, x_t, time, context_kv_cache, context_mask
                )

            if tactile_future_loss_enabled:
                if future_target is None:
                    raise ValueError(
                        "A positive Tactile Expert future loss requires its "
                        "configured real tactile target frames"
                    )
                raw_future_target = future_target.to(dtype=embed_dtype)
                future_target = self.normalize_tactile_expert_future(raw_future_target)
                if tactile_future_noise is None:
                    tactile_future_noise = torch.randn(
                        future_target.shape,
                        device=future_target.device,
                        dtype=future_target.dtype,
                        generator=rng,
                    )
                else:
                    tactile_future_noise = tactile_future_noise.to(
                        device=future_target.device,
                        dtype=future_target.dtype,
                    )
                    if tactile_future_noise.shape != future_target.shape:
                        raise ValueError(
                            "tactile_future_noise must match the T3 target: "
                            f"noise={tuple(tactile_future_noise.shape)}, "
                            f"target={tuple(future_target.shape)}"
                        )
                if tactile_future_time is None:
                    tactile_future_time = (
                        torch.distributions.Beta(
                            torch.tensor(1.5),
                            torch.tensor(1.0),
                        )
                        .sample((B,))
                        .to(device=device, dtype=dtype)
                    )
                    tactile_future_time = tactile_future_time * 0.999 + 0.001
                else:
                    tactile_future_time = tactile_future_time.to(
                        device=device,
                        dtype=dtype,
                    )
                    if tactile_future_time.shape != (B,):
                        raise ValueError(
                            "tactile_future_time must have shape "
                            f"{(B,)}, got {tuple(tactile_future_time.shape)}"
                        )
                expanded_future_time = tactile_future_time.reshape(
                    B,
                    *((1,) * (future_target.ndim - 1)),
                )
                noisy_future = (
                    expanded_future_time * tactile_future_noise
                    + (1 - expanded_future_time) * future_target
                )
                future_velocity_target = tactile_future_noise - future_target
                future_result = self.run_tactile_future_suffix(
                    observation,
                    noisy_future,
                    tactile_future_time,
                    context_kv_cache,
                    context_mask,
                    **({"return_kv_cache": True} if wam_mode == "joint" else {}),
                )
                future_velocity, future_token_mask = future_result[:2]
                # Attention and RoPE must not depend on episode-tail labels.
                # The historical null mode retains its original Fast masks.
                future_loss_mask = future_token_mask
                if (
                    getattr(self._config, "tactile_expert_tail_supervision_mode", None)
                    is not None
                ):
                    future_loss_mask = self._tactile_expert_token_mask(
                        observation,
                        future=True,
                        device=future_velocity.device,
                        batch_size=B,
                        purpose="loss",
                    )
                if wam_mode != "fast":
                    if wam_mode == "joint":
                        # Official FastWAMJoint is one-way: actions read the
                        # noisy future, but future queries never read actions.
                        action_cache = future_result[2]
                    else:
                        # Separate teacher-forced condition branch. It must
                        # not read the noisy future-loss branch's K/V cache.
                        condition, condition_time = self._idm_training_condition(
                            future_target, rng=rng
                        )
                        _, _, action_cache = self.run_tactile_future_suffix(
                            observation,
                            condition,
                            condition_time,
                            context_kv_cache,
                            context_mask,
                            return_kv_cache=True,
                        )
                    suffix_out = self.run_suffix(
                        observation,
                        x_t,
                        time,
                        action_cache,
                        torch.cat([context_mask, future_token_mask], dim=1),
                    )
                flattened_prediction = future_velocity.flatten(1, -2)
                flattened_target = future_velocity_target.flatten(1, -2)
                flattened_absolute_target = future_target.flatten(1, -2)
                flattened_raw_absolute_target = raw_future_target.flatten(1, -2)
                if future_token_mask.shape != flattened_prediction.shape[:2]:
                    raise RuntimeError(
                        "Future tactile mask/prediction mismatch: "
                        f"mask={tuple(future_token_mask.shape)}, "
                        f"prediction={tuple(flattened_prediction.shape)}"
                    )
                valid_prediction = flattened_prediction[future_loss_mask]
                valid_target = flattened_target[future_loss_mask]
                valid_absolute_target = flattened_absolute_target[future_loss_mask]
                valid_raw_absolute_target = flattened_raw_absolute_target[
                    future_loss_mask
                ]
                if valid_prediction.numel() == 0:
                    future_loss = future_velocity.sum() * 0.0
                elif (
                    getattr(
                        self,
                        "tactile_expert_future_prediction_mode",
                        "single_horizon_full",
                    )
                    == "multi_horizon_pooled"
                ):
                    future_loss = self._multi_horizon_tactile_flow_loss(
                        future_velocity,
                        future_velocity_target,
                        future_loss_mask,
                    )
                else:
                    future_loss = F.mse_loss(
                        valid_prediction.float(),
                        valid_target.float(),
                    )
                auxiliary_losses = {
                    "tactile_expert_future_loss": future_loss,
                    "tactile_expert_future_valid_fraction": future_loss_mask.float().mean(),
                    "tactile_expert_future_target_rms": valid_absolute_target.float()
                    .square()
                    .mean()
                    .sqrt()
                    if valid_absolute_target.numel()
                    else future_loss.detach(),
                    "tactile_expert_future_raw_target_rms": (
                        valid_raw_absolute_target.float().square().mean().sqrt()
                    )
                    if valid_raw_absolute_target.numel()
                    else future_loss.detach(),
                    "tactile_expert_future_reference_rms": future_loss.new_tensor(
                        self.tactile_expert_future_reference_rms
                    ),
                }
        else:
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time
            )
            input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
            ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = torch.cumsum(input_mask.int(), dim=1) - 1
            (vl_context, suffix_out), _ = self.llm(
                [prefix_tokens, suffix_tokens],
                positions=positions,
                mask=attn_mask,
                adarms_cond=[None, adarms_cond],
            )
            vl_prefix_mask = prefix_mask

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = torch.mean(torch.square(v_t - u_t), dim=-1)
        if return_auxiliary_losses:
            return action_loss, auxiliary_losses
        return action_loss

    def build_prefix_cache(
        self,
        observation: model.Observation,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Embed prefix tokens and run one LLM pass to build the KV cache.

        The caller is responsible for preprocessing the observation (image
        resize/pad, mask defaults) — this method only consumes the prepared
        observation so it can be shared between the eval Euler sampler and the
        RL train-time forward where the observation has already been built.

        Returns:
            prefix_out:  (B, prefix_len, paligemma_width) paligemma-side hidden states.
            prefix_mask: (B, prefix_len) bool mask of valid prefix positions.
            kv_cache:    per-layer KV cache to feed into subsequent suffix passes.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = torch.cumsum(prefix_mask.int(), dim=1) - 1
        outputs, kv_cache = self.llm(
            [prefix_tokens, None],
            positions=positions,
            mask=prefix_attn_mask,
        )
        return outputs[0], prefix_mask, kv_cache

    def _tactile_expert_token_mask(
        self,
        observation: model.Observation,
        *,
        future: bool,
        device: torch.device,
        batch_size: int,
        purpose: str = "loss",
    ) -> torch.Tensor:
        """Expand sensor validity; terminal validity belongs to loss only."""
        if purpose not in ("loss", "attention"):
            raise ValueError(f"Unknown tactile mask purpose: {purpose}")
        tail_mode = getattr(
            getattr(self, "_config", None), "tactile_expert_tail_supervision_mode", None
        )
        if future:
            mask_dict = getattr(observation, "tactile_future_masks", None)
            target_valid = getattr(observation, "tactile_future_target_valid", None)
            if tail_mode is not None and (
                purpose == "attention" or tail_mode == "repeat"
            ):
                target_valid = None
        else:
            mask_dict = observation.tactile_masks
            target_valid = None
        keys = ("left_gsmini_rgb", "right_gsmini_rgb")
        prediction_mode = getattr(
            self,
            "tactile_expert_future_prediction_mode",
            "single_horizon_full",
        )
        multi_horizon = future and prediction_mode == "multi_horizon_pooled"
        future_steps = (
            self.tactile_expert_future_horizon // self.tactile_expert_future_stride
            if multi_horizon
            else None
        )
        if mask_dict is None:
            mask_shape = (
                (batch_size, future_steps, 2) if multi_horizon else (batch_size, 2)
            )
            sensor_mask = torch.ones(*mask_shape, dtype=torch.bool, device=device)
        else:
            missing = [key for key in keys if key not in mask_dict]
            if missing:
                raise ValueError(f"Missing tactile mask keys: {missing}")
            sensor_dim = 2 if multi_horizon else 1
            sensor_mask = torch.stack(
                [mask_dict[key].to(device=device, dtype=torch.bool) for key in keys],
                dim=sensor_dim,
            )
            expected_sensor_shape = (
                (batch_size, future_steps, 2) if multi_horizon else (batch_size, 2)
            )
            if sensor_mask.shape != expected_sensor_shape:
                raise ValueError(
                    "Tactile sensor masks have the wrong shape: expected "
                    f"{expected_sensor_shape}, got {tuple(sensor_mask.shape)}"
                )
        if target_valid is not None:
            target_valid = target_valid.to(device=device, dtype=torch.bool)
            expected_valid_shape = (
                (batch_size, future_steps) if multi_horizon else (batch_size,)
            )
            if target_valid.shape != expected_valid_shape:
                raise ValueError(
                    "tactile_future_target_valid must have shape "
                    f"{expected_valid_shape}, got {tuple(target_valid.shape)}"
                )
            sensor_mask = sensor_mask & target_valid.unsqueeze(-1)
        tokens_per_sensor = (
            1 + getattr(self, "tactile_expert_future_pool_grid", 6) ** 2
            if multi_horizon
            else T3_TOKENS_PER_SENSOR
        )
        return (
            sensor_mask.unsqueeze(-1)
            .expand(*sensor_mask.shape, tokens_per_sensor)
            .reshape(batch_size, -1)
        )

    def _multi_horizon_tactile_flow_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        flat_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Average token MSE over all valid future horizons.

        This mirrors video-flow training: adding more supervised frames changes
        target coverage rather than multiplying the auxiliary loss scale.
        """
        if prediction.shape != target.shape or prediction.ndim != 5:
            raise ValueError(
                "Multi-horizon tactile flow tensors must share "
                f"[B, K, 2, T, D], got {tuple(prediction.shape)} and "
                f"{tuple(target.shape)}"
            )
        batch_size, future_steps, sensors, tokens_per_sensor = prediction.shape[:4]
        token_mask = flat_token_mask.reshape(
            batch_size,
            future_steps,
            sensors,
            tokens_per_sensor,
        )
        token_loss = (prediction.float() - target.float()).square().mean(dim=-1)
        valid_count = token_mask.sum(dim=(2, 3))
        horizon_loss = (token_loss * token_mask).sum(
            dim=(2, 3)
        ) / valid_count.clamp_min(1)
        horizon_valid = valid_count > 0
        if not horizon_valid.any():
            return prediction.sum() * 0.0
        return horizon_loss[horizon_valid].mean()

    def build_tactile_expert_cache(
        self,
        observation: model.Observation,
        encoded_tactile: torch.Tensor,
        prefix_mask: torch.Tensor,
        prefix_kv_cache: tuple,
    ) -> tuple[torch.Tensor, tuple]:
        """Prefill clean current touch once and return its layerwise K/V cache.

        Current tactile queries are deliberately isolated from PaliGemma.  The
        returned cache still concatenates prefix and tactile K/V at each layer,
        so downstream action queries can read both sources without either
        source being modified by a noisy action or future-touch branch.
        """
        if not self.tactile_expert_enabled:
            raise RuntimeError("Tactile Expert is disabled")
        tactile_tokens = self._embed_tactile_expert_tokens(encoded_tactile)
        batch_size, tactile_len = tactile_tokens.shape[:2]
        tactile_mask = self._tactile_expert_token_mask(
            observation,
            future=False,
            device=tactile_tokens.device,
            batch_size=batch_size,
        )
        tactile_self = tactile_mask[:, :, None] & tactile_mask[:, None, :]
        tactile_positions = (
            torch.sum(prefix_mask, dim=-1)[:, None]
            + torch.cumsum(tactile_mask.int(), dim=-1)
            - 1
        )
        clean_time = torch.zeros(
            batch_size,
            device=tactile_tokens.device,
            dtype=self.embed_dtype,
        )
        tactile_condition = self._embed_tactile_expert_time(clean_time)
        # Current touch is causally independent of the VLM prefix. Running its
        # self-attention without the (fully masked) prefix columns avoids a
        # large, provably useless attention rectangle. Absolute RoPE positions
        # still place it immediately after the prefix, and the per-layer K/Vs
        # are concatenated below for downstream action/future queries.
        _, tactile_only_kv_cache = self.llm(
            [None, None, tactile_tokens],
            positions=tactile_positions,
            mask=tactile_self,
            adarms_cond=[None, None, tactile_condition],
        )
        if len(prefix_kv_cache) != len(tactile_only_kv_cache):
            raise RuntimeError(
                "Prefix and tactile caches have different layer counts: "
                f"{len(prefix_kv_cache)} vs {len(tactile_only_kv_cache)}"
            )
        tactile_kv_cache = tuple(
            (
                torch.cat([prefix_layer[0], tactile_layer[0]], dim=1),
                torch.cat([prefix_layer[1], tactile_layer[1]], dim=1),
            )
            for prefix_layer, tactile_layer in zip(
                prefix_kv_cache,
                tactile_only_kv_cache,
                strict=True,
            )
        )
        return torch.cat([prefix_mask, tactile_mask], dim=1), tactile_kv_cache

    def run_tactile_future_suffix(
        self,
        observation: model.Observation,
        noisy_future: torch.Tensor,
        future_time: torch.Tensor,
        kv_cache: tuple,
        context_mask: torch.Tensor,
        *,
        return_kv_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, tuple]:
        """Predict future velocity, optionally retaining differentiable K/V.

        The returned cache contains context followed by future tokens. Future
        queries read context and future only, never action tokens.
        """
        future_tokens = self._embed_tactile_expert_future_tokens(noisy_future)
        batch_size, future_len = future_tokens.shape[:2]
        future_mask = self._tactile_expert_token_mask(
            observation,
            future=True,
            device=future_tokens.device,
            batch_size=batch_size,
            purpose="attention",
        )
        future_to_context = future_mask[:, :, None] & context_mask[:, None, :]
        future_self = future_mask[:, :, None] & future_mask[:, None, :]
        attention_mask = torch.cat([future_to_context, future_self], dim=-1)
        positions = (
            torch.sum(context_mask, dim=-1)[:, None]
            + torch.cumsum(future_mask.int(), dim=-1)
            - 1
        )
        tactile_condition = self._embed_tactile_expert_time(future_time)
        expert_index = 2
        streams = [None] * (expert_index + 1)
        conditions = [None] * (expert_index + 1)
        streams[expert_index] = future_tokens
        conditions[expert_index] = tactile_condition
        outputs, future_cache = self.llm(
            streams,
            positions=positions,
            mask=attention_mask,
            kv_cache=kv_cache,
            adarms_cond=conditions,
            attention_backend=getattr(
                self,
                "tactile_expert_future_attention_backend",
                "eager",
            ),
        )
        future_hidden = outputs[expert_index]
        if future_hidden is None:
            raise RuntimeError("Tactile Expert produced no future hidden states")
        future_hidden = future_hidden.reshape(*noisy_future.shape[:-1], -1)
        result = (self.tactile_expert_out_proj(future_hidden), future_mask)
        return (*result, future_cache) if return_kv_cache else result

    def _idm_training_condition(
        self, target: torch.Tensor, *, rng: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher force GT future, independently noised with probability p.

        Keep the local Pi0.5 Beta(1.5, 1) flow schedule rather than importing
        Wan's scheduler. Inverse-CDF sampling makes the supplied RNG effective.
        """
        batch = target.shape[0]
        probability = self._config.tactile_expert_idm_condition_noise_prob
        selected = torch.rand(batch, device=target.device, generator=rng) < probability
        time = torch.rand(batch, device=target.device, generator=rng).pow(1 / 1.5)
        time = ((time * 0.999 + 0.001) * selected).to(target.dtype)
        noise = torch.randn(
            target.shape, device=target.device, dtype=target.dtype, generator=rng
        )
        expanded = time.reshape(batch, *((1,) * (target.ndim - 1)))
        return (1 - expanded) * target + expanded * noise, time

    def _sample_actions_with_future(
        self,
        observation: model.Observation,
        action_noise: torch.Tensor,
        context_mask: torch.Tensor,
        kv_cache: tuple,
        *,
        num_steps: int,
        rng: torch.Generator | None,
        future_noise: torch.Tensor | None,
    ) -> torch.Tensor:
        """Joint: N coupled steps. IDM: N future steps, prefill, N action steps.

        Future latents remain in the normalized model space. No GT future or
        terminal validity is consulted at inference. Current sensor validity
        is propagated to each imagined frame; current T3/VLM caches stay fixed.
        """
        batch = action_noise.shape[0]
        multi = self.tactile_expert_future_prediction_mode == "multi_horizon_pooled"
        frames = self.tactile_expert_future_horizon // self.tactile_expert_future_stride
        shape = (
            (
                batch,
                frames,
                2,
                1 + self.tactile_expert_future_pool_grid**2,
                T3_LARGE_WIDTH,
            )
            if multi
            else (batch, 2, T3_TOKENS_PER_SENSOR, T3_LARGE_WIDTH)
        )
        masks = observation.tactile_masks
        if multi and masks is not None:
            masks = {
                key: value[:, None].expand(-1, frames) for key, value in masks.items()
            }
        observation = dataclasses.replace(
            observation,
            tactile_future_masks=masks,
            tactile_future_target_valid=None,
            tactile_future_images=None,
            tactile_world_future_latents=None,
        )
        if future_noise is None:
            future = torch.randn(
                shape, device=action_noise.device, dtype=torch.float32, generator=rng
            )
        else:
            if tuple(future_noise.shape) != shape:
                raise ValueError(f"tactile_future_noise must have shape {shape}")
            future = future_noise.to(device=action_noise.device, dtype=torch.float32)
        # Keep Euler accumulation in float32, like the action solver; cast
        # only model inputs to BF16 so repeated updates do not round the state.
        action = action_noise
        dt = -1.0 / num_steps
        joint = self.tactile_expert_wam_mode == "joint"
        for step in range(num_steps):
            time = torch.full(
                (batch,),
                1.0 - step / num_steps,
                device=action.device,
                dtype=torch.float32,
            )
            velocity, future_mask, step_cache = self.run_tactile_future_suffix(
                observation,
                future.to(self.embed_dtype),
                time,
                kv_cache,
                context_mask,
                return_kv_cache=True,
            )
            if joint:
                hidden = self.run_suffix(
                    observation,
                    action,
                    time,
                    step_cache,
                    torch.cat([context_mask, future_mask], dim=1),
                )
                action = action + dt * self.velocity_from_suffix(hidden)
            future = future + dt * velocity.float()
        if joint:
            return action
        # Re-encode the FINAL generated endpoint at t=0, not the last t=1/N
        # cache. Hold that cache fixed across all action denoising steps.
        _, future_mask, final_cache = self.run_tactile_future_suffix(
            observation,
            future.to(self.embed_dtype),
            torch.zeros(batch, device=action.device),
            kv_cache,
            context_mask,
            return_kv_cache=True,
        )
        final_mask = torch.cat([context_mask, future_mask], dim=1)
        for step in range(num_steps):
            time = torch.full(
                (batch,),
                1.0 - step / num_steps,
                device=action.device,
                dtype=torch.float32,
            )
            hidden = self.run_suffix(observation, action, time, final_cache, final_mask)
            action = action + dt * self.velocity_from_suffix(hidden)
        return action

    def run_suffix(
        self,
        observation: model.Observation,
        x_t: torch.Tensor,
        t_tensor: torch.Tensor,
        kv_cache: tuple,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One suffix forward pass (action expert) given the prefix KV cache.

        Returns the final action-expert hidden states sliced to the last
        ``action_horizon`` positions: (B, action_horizon, action_expert_width).
        """
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, t_tensor
        )
        expert_tokens = suffix_tokens
        suffix_len = suffix_tokens.shape[1]
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        context_to_suffix_mask = einops.repeat(
            context_mask, "b p -> b s p", s=suffix_len
        )
        full_attn_mask = torch.cat([context_to_suffix_mask, suffix_attn_mask], dim=-1)
        expert_mask = suffix_mask

        suffix_positions = (
            torch.sum(context_mask, dim=-1)[:, None]
            + torch.cumsum(expert_mask.int(), dim=-1)
            - 1
        )
        llm_output = self.llm(
            [None, expert_tokens],
            positions=suffix_positions,
            mask=full_attn_mask,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
            # With only H50 queries and a long IDM cache, eager GQA uses less
            # memory than SDPA's expanded K/V heads on our CUDA/ROCm runtime.
            # Fuse the long context passes, not this short-query action pass.
            attention_backend="eager",
        )
        outputs, _ = llm_output
        return outputs[1][:, -self.action_horizon :]

    def velocity_from_suffix(self, suffix_out_act: torch.Tensor) -> torch.Tensor:
        """Project action-expert hidden states to a velocity prediction v_t."""
        return self.action_out_proj(suffix_out_act)

    def sample_actions(
        self,
        observation: model.Observation,
        *,
        num_steps: int = 10,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        tactile_future_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample actions using Euler ODE solver.

        Args:
            observation: input observation
            num_steps: number of ODE solver steps
            noise: optional initial noise of shape (B, action_horizon, action_dim)
            rng: random generator
            tactile_future_noise: optional initial future noise for Joint/IDM

        Returns:
            actions: (B, action_horizon, action_dim)
        """
        if type(num_steps) is not int or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        preprocess_kwargs = {"train": False}
        observation = model.preprocess_observation(observation, **preprocess_kwargs)

        dt = -1.0 / num_steps
        B = observation.state.shape[0]
        device = observation.state.device

        if noise is None:
            noise = torch.randn(
                B, self.action_horizon, self.action_dim, device=device, generator=rng
            )

        encoded_tactile = self.encode_policy_tactile(observation)
        _, context_mask, kv_cache = self.build_prefix_cache(observation)
        if getattr(self, "tactile_expert_enabled", False):
            if encoded_tactile is None:
                raise RuntimeError("Tactile Expert requires current T3 output")
            context_mask, kv_cache = self.build_tactile_expert_cache(
                observation,
                encoded_tactile,
                context_mask,
                kv_cache,
            )

        if getattr(self, "tactile_expert_wam_mode", "fast") != "fast":
            return self._sample_actions_with_future(
                observation,
                noise,
                context_mask,
                kv_cache,
                num_steps=num_steps,
                rng=rng,
                future_noise=tactile_future_noise,
            )
        x_t = noise
        t = 1.0

        # Euler integration
        while t >= -dt / 2:
            t_tensor = torch.full((B,), t, device=device, dtype=torch.float32)
            suffix_out_act = self.run_suffix(
                observation,
                x_t,
                t_tensor,
                kv_cache,
                context_mask,
            )
            v_t = self.velocity_from_suffix(suffix_out_act)
            x_t = x_t + dt * v_t
            t = t + dt

        return x_t

    def forward(
        self,
        observation: model.Observation,
        actions: torch.Tensor,
        *,
        train: bool = True,
        rng: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        tactile_future_noise: torch.Tensor | None = None,
        tactile_future_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Default forward computes loss."""
        return self.compute_loss(
            observation,
            actions,
            train=train,
            rng=rng,
            noise=noise,
            time=time,
            tactile_future_noise=tactile_future_noise,
            tactile_future_time=tactile_future_time,
        )

    def train(self, mode: bool = True) -> Pi0:
        """Set train mode while keeping configured frozen T3 copies in eval."""
        super().train(mode)
        if (
            getattr(self, "tactile_expert_enabled", False)
        ) and self.tactile_encoder_frozen:
            self.tactile_encoder.eval()
        return self

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict | None = None
    ):
        """Enable gradient checkpointing for memory efficiency.

        Args:
            gradient_checkpointing_kwargs: Optional kwargs forwarded to the activation
                checkpoint. Currently honors ``use_reentrant`` (default ``False``), so
                the FSDP ``gradient_checkpointing_use_reentrant`` setting is respected.
        """
        kwargs = gradient_checkpointing_kwargs or {}
        use_reentrant = kwargs.get("use_reentrant", False)
        self.llm.gradient_checkpointing = True
        self.llm.gradient_checkpointing_use_reentrant = use_reentrant
        self.img.encoder.gradient_checkpointing = True
        self.img.encoder.gradient_checkpointing_use_reentrant = use_reentrant
        if (
            getattr(self, "tactile_expert_enabled", False)
        ) and not self.tactile_encoder_frozen:
            self.tactile_encoder.gradient_checkpointing_enable(
                use_reentrant=use_reentrant
            )

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing (used by the eval / no-recompute path)."""
        self.llm.gradient_checkpointing = False
        self.img.encoder.gradient_checkpointing = False
        if (
            getattr(self, "tactile_expert_enabled", False)
        ) and not self.tactile_encoder_frozen:
            self.tactile_encoder.gradient_checkpointing_disable()
