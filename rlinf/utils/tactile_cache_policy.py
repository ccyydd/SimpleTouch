# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fail-closed training policy for immutable future tactile targets."""

from pathlib import Path
from typing import Any


def require_future_tactile_cache(
    *,
    enabled: bool,
    future_loss_weight: float,
    full_cache: str | Path | None,
    pooled_cache: str | Path | None,
    eval_dataset: bool = False,
) -> None:
    """Reject online future encoding, without changing current-touch inputs.

    Paths must already be resolved (``auto`` is a launcher-only shorthand).
    The cache consumer still validates completion, source identity, encoder,
    preprocessing, token geometry and every episode file before loading data.
    """
    if not enabled or eval_dataset or future_loss_weight <= 0:
        return
    paths = [str(path).strip() for path in (full_cache, pooled_cache) if path]
    if not paths or any(
        path.lower() in {"", "auto", "online", "none"} for path in paths
    ):
        raise ValueError(
            "Future tactile supervision requires a precomputed, validated T3 "
            "cache; online future encoding is forbidden, including on resume. "
            "Run scripts/univtac/cache_t3_latents.py first with the same task, "
            "data root, episode range, T3 weights and preprocessing profile; "
            "then set tactile_expert_future_cache_path (pooled grid 6) or "
            "tactile_world_cache_path (full grid 14). Current touch stays online."
        )


def validate_training_future_cache_config(cfg: Any) -> None:
    """Check direct Hydra launches before any distributed workers are created."""
    from omegaconf import OmegaConf

    require_future_tactile_cache(
        enabled=bool(
            OmegaConf.select(
                cfg, "actor.model.openpi.tactile_expert_enabled", default=False
            )
        ),
        future_loss_weight=float(
            OmegaConf.select(
                cfg, "actor.model.openpi.tactile_expert_future_loss_weight", default=0.0
            )
        ),
        full_cache=OmegaConf.select(cfg, "data.tactile_world_cache_path"),
        pooled_cache=OmegaConf.select(cfg, "data.tactile_expert_future_cache_path"),
    )
