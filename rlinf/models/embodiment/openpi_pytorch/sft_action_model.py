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

from __future__ import annotations

from typing import Any

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi_pytorch.openpi_action_model import (
    OpenPiPytorchActionModel,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0


class OpenPiPytorchSFTActionModel(OpenPiPytorchActionModel):
    """SFT variant of :class:`OpenPiPytorchActionModel` (flow-matching loss)."""

    def __init__(self, pi0_model: Pi0, *, num_steps: int, action_env_dim: int):
        super().__init__(pi0_model, num_steps=num_steps, action_env_dim=action_env_dim)

    def tactile_expert_future_loss_weight(self) -> float:
        return float(self.model._config.tactile_expert_future_loss_weight)

    def forward(self, forward_type: ForwardType = ForwardType.SFT, **kwargs):
        """Dispatch — SFT variant only supports :attr:`ForwardType.SFT`."""
        if forward_type != ForwardType.SFT:
            raise NotImplementedError(
                f"{type(self).__name__} only supports ForwardType.SFT; "
                f"got forward_type={forward_type!r}. "
                "Use the RL subclass (actor.model.openpi.task='rl') for PPO."
            )
        return self.sft_forward(**kwargs)

    def sft_forward(self, data: Any) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute action flow loss and the optional future-touch flow loss."""
        observation, actions, action_valid_mask = self._unpack_sft_batch(data)
        observation = self._observation_to_device(observation)
        actions = self._actions_to_device(actions)
        per_timestep_loss, auxiliary_losses = self.model.compute_loss(
            observation,
            actions,
            train=True,
            return_auxiliary_losses=True,
        )
        action_loss = self._masked_action_loss(
            per_timestep_loss,
            action_valid_mask,
        )
        if not auxiliary_losses:
            return action_loss
        loss = action_loss
        graph_losses: dict[str, torch.Tensor] = {
            "_action_loss_for_gradient_diagnostics": action_loss,
        }
        logged_weighted_losses: dict[str, torch.Tensor] = {}
        if "tactile_expert_future_loss" in auxiliary_losses:
            effective_future_weight = self.tactile_expert_future_loss_weight()
            weighted_future_loss = (
                effective_future_weight * auxiliary_losses["tactile_expert_future_loss"]
            )
            loss = loss + weighted_future_loss
            graph_losses["_weighted_tactile_expert_loss_for_gradient_diagnostics"] = (
                weighted_future_loss
            )
            logged_weighted_losses["tactile_expert_future_weighted_loss"] = (
                weighted_future_loss.detach()
            )
            logged_weighted_losses["tactile_expert_future_effective_loss_weight"] = (
                weighted_future_loss.new_tensor(effective_future_weight).detach()
            )
        return {
            "loss": loss,
            # These graph-connected scalars are consumed only by the optional
            # low-frequency FSDP gradient diagnostic. The worker excludes
            # private keys from ordinary scalar logging.
            **graph_losses,
            "action_loss": action_loss.detach(),
            **logged_weighted_losses,
            **{key: value.detach() for key, value in auxiliary_losses.items()},
        }

    def compute_loss(self, data: Any) -> torch.Tensor | dict[str, torch.Tensor]:
        """Alias kept for interface parity with the old action model."""
        return self.sft_forward(data)

    @staticmethod
    def _unpack_sft_batch(data: Any) -> tuple[Any, Any, Any | None]:
        if isinstance(data, (tuple, list)):
            if len(data) not in (2, 3):
                raise ValueError(
                    "SFT batch tuple must be (observation, actions) or "
                    "(observation, actions, action_valid_mask); "
                    f"got length {len(data)}."
                )
            observation, actions = data[:2]
            action_valid_mask = data[2] if len(data) == 3 else None
        elif isinstance(data, dict):
            if "observation" not in data or "actions" not in data:
                raise ValueError(
                    "SFT batch dict must contain 'observation' and 'actions'; "
                    f"got keys {sorted(data)}."
                )
            observation, actions = data["observation"], data["actions"]
            action_valid_mask = data.get("action_valid_mask")
        else:
            raise TypeError(f"Unsupported SFT batch type: {type(data)!r}.")
        if observation is None or actions is None:
            raise ValueError("SFT batch is missing observation or actions.")
        return observation, actions, action_valid_mask

    @staticmethod
    def _masked_action_loss(
        per_timestep_loss: torch.Tensor,
        action_valid_mask: Any | None,
    ) -> torch.Tensor:
        """Average flow loss over recorded action timesteps only."""
        if action_valid_mask is None:
            return per_timestep_loss.mean()
        mask = torch.as_tensor(
            action_valid_mask,
            device=per_timestep_loss.device,
            dtype=torch.bool,
        )
        if mask.shape != per_timestep_loss.shape:
            raise ValueError(
                "action_valid_mask must match per-timestep action loss: "
                f"mask={tuple(mask.shape)}, loss={tuple(per_timestep_loss.shape)}"
            )
        valid = mask.to(per_timestep_loss.dtype)
        return (per_timestep_loss * valid).sum() / valid.sum().clamp_min(1.0)

    def _observation_to_device(self, observation: Any) -> Observation:
        if isinstance(observation, dict):
            observation = Observation.from_dict(observation)
        if not isinstance(observation, Observation):
            raise TypeError(
                f"SFT observation must be an Observation or dict; "
                f"got {type(observation)!r}."
            )
        device = self.device

        def _move(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        return Observation(
            images={k: _move(v) for k, v in observation.images.items()},
            image_masks={k: _move(v) for k, v in observation.image_masks.items()},
            state=_move(observation.state),
            tactile_images=(
                {k: _move(v) for k, v in observation.tactile_images.items()}
                if observation.tactile_images is not None
                else None
            ),
            tactile_masks=(
                {k: _move(v) for k, v in observation.tactile_masks.items()}
                if observation.tactile_masks is not None
                else None
            ),
            tactile_future_images=(
                {k: _move(v) for k, v in observation.tactile_future_images.items()}
                if observation.tactile_future_images is not None
                else None
            ),
            tactile_future_masks=(
                {k: _move(v) for k, v in observation.tactile_future_masks.items()}
                if observation.tactile_future_masks is not None
                else None
            ),
            tactile_future_target_valid=_move(observation.tactile_future_target_valid),
            tactile_world_latents=_move(observation.tactile_world_latents),
            tactile_world_future_latents=_move(
                observation.tactile_world_future_latents
            ),
            tokenized_prompt=_move(observation.tokenized_prompt),
            tokenized_prompt_mask=_move(observation.tokenized_prompt_mask),
            token_ar_mask=_move(observation.token_ar_mask),
            token_loss_mask=_move(observation.token_loss_mask),
        )

    def _actions_to_device(self, actions: Any) -> torch.Tensor:
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        model_action_dim = self.model.action_dim
        if actions.dim() != 3:
            raise ValueError(
                "SFT actions must have shape [B, action_horizon, D]; "
                f"got {tuple(actions.shape)}."
            )
        if actions.shape[-1] == model_action_dim:
            return actions.to(device=self.device, dtype=torch.float32)
        raise ValueError(
            "SFT actions must arrive normalized + padded to the model action "
            f"dim {model_action_dim} (the openpi_pytorch SFT data loader applies the "
            f"openpi transform pipeline before collation); got last dim "
            f"{actions.shape[-1]}."
        )
