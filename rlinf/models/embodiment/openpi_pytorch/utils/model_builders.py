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

"""Evaluation wrapper for the OpenPI transform pipeline."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _resolve_data_kwargs(cfg):
    """Read the optional ``openpi_data`` override block as a plain dict."""
    from omegaconf import OmegaConf

    data_kwargs = OmegaConf.select(cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)
    return data_kwargs


def _build_eval_model(
    cfg,
    model_cfg,
    model,
    *,
    num_steps,
    action_chunk,
    action_env_dim,
):
    """Build the eval variant: openpi.transforms pipeline, no value head, no
    chain collection, no training-mode forward.

    Produces a bare :class:`OpenPiPytorchEvalActionModel` — same eval path the
    RL subclass uses, but stripped of every PPO-only bit. Selected by
    ``actor.model.openpi.task: eval``.
    """
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch.eval_action_model import (
        OpenPiPytorchEvalActionModel,
    )
    from rlinf.models.embodiment.openpi_pytorch.transforms_pipeline import (
        build_openpi_transforms,
    )

    config_name = str(OmegaConf.select(model_cfg, "config_name", default=""))
    if not config_name:
        raise ValueError(
            "actor.model.openpi.config_name is required for task='eval' "
            "(it selects the upstream openpi TrainConfig, e.g. 'pi05_behavior')."
        )

    input_transforms, output_transforms = build_openpi_transforms(
        cfg.model_path, config_name, data_kwargs=_resolve_data_kwargs(cfg)
    )

    eval_model = OpenPiPytorchEvalActionModel(
        model,
        num_steps=num_steps,
        action_env_dim=action_env_dim,
        action_chunk=action_chunk,
        config_name=config_name,
        state_indices=OmegaConf.select(model_cfg, "state_indices", default=None),
    )
    eval_model.setup_wrappers(input_transforms, output_transforms)
    return eval_model
