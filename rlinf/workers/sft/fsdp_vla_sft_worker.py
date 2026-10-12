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
import math
import os
import shutil
import tempfile
from collections.abc import Callable
from typing import Any

import torch

from rlinf.config import SupportedModel
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

_ACTION_GRADIENT_LOSS_KEY = "_action_loss_for_gradient_diagnostics"
_WORLD_GRADIENT_LOSS_KEY = "_weighted_world_loss_for_gradient_diagnostics"
_TACTILE_EXPERT_GRADIENT_LOSS_KEY = (
    "_weighted_tactile_expert_loss_for_gradient_diagnostics"
)
_GRADIENT_AUXILIARY_NAMES = frozenset({"tactile_world", "tactile_expert"})
_GRADIENT_NORM_EPS = 1.0e-12
_GRADIENT_TRIANGLE_RTOL = 1.0e-3


def gradient_interaction_metrics(
    action_norm: float,
    world_norm: float,
    total_norm: float,
    *,
    auxiliary_name: str = "tactile_world",
) -> dict[str, float]:
    """Recover action/auxiliary gradient alignment from three L2 norms.

    Undefined ratios/cosines are reported as finite zero values together with
    validity flags so an inactive auxiliary loss cannot poison TensorBoard.
    """
    if auxiliary_name not in _GRADIENT_AUXILIARY_NAMES:
        raise ValueError(
            f"Unsupported gradient-diagnostic auxiliary name: {auxiliary_name!r}"
        )
    # Keep the historical ``world_norm`` argument name so existing callers can
    # continue to use it by keyword; it represents the selected auxiliary loss.
    auxiliary_norm = world_norm
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (action_norm, auxiliary_norm, total_norm)
    ):
        raise ValueError(
            "Gradient norms must be finite and nonnegative: "
            f"action={action_norm} auxiliary={auxiliary_norm} total={total_norm}"
        )
    action_auxiliary_dot = 0.5 * (
        total_norm * total_norm
        - action_norm * action_norm
        - auxiliary_norm * auxiliary_norm
    )
    norm_product = action_norm * auxiliary_norm
    norm_sum = action_norm + auxiliary_norm
    triangle_tolerance = _GRADIENT_TRIANGLE_RTOL * max(norm_sum, 1.0)
    triangle_valid = (
        total_norm >= abs(action_norm - auxiliary_norm) - triangle_tolerance
        and total_norm <= norm_sum + triangle_tolerance
    )
    cosine_defined = norm_product > _GRADIENT_NORM_EPS
    raw_cosine = action_auxiliary_dot / norm_product if cosine_defined else 0.0
    cosine = max(-1.0, min(1.0, raw_cosine))
    cosine_valid = cosine_defined and triangle_valid
    ratio_valid = action_norm > _GRADIENT_NORM_EPS
    alignment_name = "world" if auxiliary_name == "tactile_world" else auxiliary_name
    return {
        "action_grad_norm": action_norm,
        f"weighted_{auxiliary_name}_grad_norm": auxiliary_norm,
        "diagnostic_total_grad_norm": total_norm,
        "action_grad_norm_share": (
            action_norm / norm_sum if norm_sum > _GRADIENT_NORM_EPS else 0.0
        ),
        f"weighted_{auxiliary_name}_grad_norm_share": (
            auxiliary_norm / norm_sum if norm_sum > _GRADIENT_NORM_EPS else 0.0
        ),
        f"{auxiliary_name}_to_action_grad_ratio": (
            auxiliary_norm / action_norm if ratio_valid else 0.0
        ),
        f"{auxiliary_name}_to_action_grad_ratio_valid": float(ratio_valid),
        f"action_{alignment_name}_grad_cosine": cosine,
        f"action_{alignment_name}_grad_cosine_valid": float(cosine_valid),
        "gradient_norm_triangle_valid": float(triangle_valid),
        "gradient_interference_factor": (
            total_norm / norm_sum if norm_sum > _GRADIENT_NORM_EPS else 0.0
        ),
    }


def shared_gradient_interaction_metrics(
    action_squares: dict[str, float],
    world_squares: dict[str, float],
    total_squares: dict[str, float],
    *,
    auxiliary_name: str = "tactile_world",
) -> dict[str, float]:
    """Measure alignment only where both losses reach the same parameters."""
    if auxiliary_name not in _GRADIENT_AUXILIARY_NAMES:
        raise ValueError(
            f"Unsupported gradient-diagnostic auxiliary name: {auxiliary_name!r}"
        )
    # As above, preserve the public keyword used by the original world-model
    # diagnostics while allowing the implementation to select another branch.
    auxiliary_squares = world_squares
    action_active = {
        name for name, value in action_squares.items() if value > _GRADIENT_NORM_EPS
    }
    auxiliary_active = {
        name for name, value in auxiliary_squares.items() if value > _GRADIENT_NORM_EPS
    }
    shared_names = action_active & auxiliary_active

    def norm(squares: dict[str, float], names: set[str]) -> float:
        return math.sqrt(sum(squares.get(name, 0.0) for name in names))

    shared_action_norm = norm(action_squares, shared_names)
    shared_auxiliary_norm = norm(auxiliary_squares, shared_names)
    shared_total_norm = norm(total_squares, shared_names)
    shared_metrics = {
        f"shared_{key}": value
        for key, value in gradient_interaction_metrics(
            shared_action_norm,
            shared_auxiliary_norm,
            shared_total_norm,
            auxiliary_name=auxiliary_name,
        ).items()
    }
    action_only_norm = norm(action_squares, action_active - auxiliary_active)
    auxiliary_only_norm = norm(
        auxiliary_squares,
        auxiliary_active - action_active,
    )
    all_action_norm = math.sqrt(sum(action_squares.values()))
    all_auxiliary_norm = math.sqrt(sum(auxiliary_squares.values()))
    shared_metrics.update(
        {
            "shared_gradient_parameter_count": float(len(shared_names)),
            "action_exclusive_grad_norm": action_only_norm,
            f"weighted_{auxiliary_name}_exclusive_grad_norm": auxiliary_only_norm,
            "shared_action_grad_norm_fraction": (
                shared_action_norm / all_action_norm
                if all_action_norm > _GRADIENT_NORM_EPS
                else 0.0
            ),
            f"shared_weighted_{auxiliary_name}_grad_norm_fraction": (
                shared_auxiliary_norm / all_auxiliary_norm
                if all_auxiliary_norm > _GRADIENT_NORM_EPS
                else 0.0
            ),
        }
    )
    return shared_metrics


class FSDPVlaSftWorker(FSDPSftWorker):
    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            == SupportedModel.OPENPI_PYTORCH
        ):
            from rlinf.data.datasets.openpi_pytorch import (
                build_openpi_pytorch_sft_dataloader,
            )

            return build_openpi_pytorch_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            repo_id = resolve_lerobot_repo_id(data_paths)
            if repo_id is None:
                raise ValueError(
                    "OpenPI SFT requires data.train_data_paths to be set to a local "
                    "dataset path or LeRobot repo id."
                )

            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                repo_id=repo_id,
                data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            return data_loader, data_loader.data_config()
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA
        ]:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.DREAMZERO
        ]:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        with self.amp_context:
            output = self.model(forward_type=ForwardType.SFT, data=batch)

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]

        step_metrics = {"loss": loss.detach().item()}
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "loss" or key.startswith("_"):
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        step_metrics[key] = value.detach().item()
                elif isinstance(value, (float, int)):
                    step_metrics[key] = value
        return loss, step_metrics

    def get_gradient_diagnostics(self, batches: list[Any]) -> dict[str, float]:
        """Measure action/auxiliary gradients without changing training state.

        This is deliberately a post-optimizer probe. Three fresh passes use the
        same accumulated global batch, updated parameters, and CPU/CUDA RNG
        states. They recover full-model and shared-parameter gradient alignment
        without retaining another multi-billion-parameter gradient copy.
        """
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            != SupportedModel.OPENPI_PYTORCH
        ):
            return {}
        openpi_cfg = self.cfg.actor.model.openpi
        tactile_expert_enabled = bool(openpi_cfg.get("tactile_expert_enabled", False))
        if tactile_expert_enabled:
            auxiliary_loss_key = _TACTILE_EXPERT_GRADIENT_LOSS_KEY
            auxiliary_name = "tactile_expert"
        else:
            return {}
        if not batches:
            raise ValueError("Gradient diagnostics require at least one micro-batch")

        gradient_device = next(self.model.parameters()).device
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(self.device)

        def restore_rng_state() -> None:
            torch.random.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, self.device)

        def parameter_gradient_squares() -> dict[str, float]:
            names: list[str] = []
            local_squares: list[torch.Tensor] = []
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                names.append(name)
                if parameter.grad is None:
                    local_squares.append(
                        torch.zeros((), device=parameter.device, dtype=torch.float64)
                    )
                else:
                    local_norm = torch.linalg.vector_norm(
                        parameter.grad.detach(),
                        ord=2,
                        dtype=torch.float64,
                    )
                    local_squares.append(local_norm.square())
            if not local_squares:
                return {}
            global_squares = torch.stack(local_squares)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    global_squares,
                    op=torch.distributed.ReduceOp.SUM,
                )
            values = global_squares.cpu().tolist()
            return dict(zip(names, values, strict=True))

        def mean_across_ranks(value: torch.Tensor) -> float:
            reduced = value.detach().to(dtype=torch.float64)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    reduced,
                    op=torch.distributed.ReduceOp.SUM,
                )
                reduced /= torch.distributed.get_world_size()
            return float(reduced)

        def probe(loss_key: str) -> tuple[float, dict[str, float], float]:
            self.optimizer.zero_grad(set_to_none=True)
            restore_rng_state()
            local_loss = torch.zeros((), device=gradient_device, dtype=torch.float64)
            for index, batch in enumerate(batches):
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(index + 1) == len(batches),
                )
                with self.amp_context:
                    output = self.model(forward_type=ForwardType.SFT, data=batch)
                if not isinstance(output, dict) or loss_key not in output:
                    raise RuntimeError(
                        "SimpleTac gradient diagnostics require graph-connected "
                        f"loss key {loss_key!r}; got "
                        f"{sorted(output) if isinstance(output, dict) else type(output)}"
                    )
                diagnostic_loss = output[loss_key]
                if not torch.is_tensor(diagnostic_loss) or diagnostic_loss.numel() != 1:
                    raise RuntimeError(
                        f"Gradient diagnostic loss {loss_key!r} must be scalar"
                    )
                scaled_loss = diagnostic_loss / len(batches)
                local_loss += scaled_loss.detach().to(torch.float64)
                with backward_ctx:
                    scaled_loss.backward()
            squares = parameter_gradient_squares()
            # The existing FSDP implementation returns the global pre-clipping
            # norm. It may clip these temporary gradients, which is harmless
            # because they are cleared before the next probe and never stepped.
            full_norm = float(self._strategy.clip_grad_norm_(model=self.model))
            return full_norm, squares, mean_across_ranks(local_loss)

        try:
            action_norm, action_squares, action_loss = probe(_ACTION_GRADIENT_LOSS_KEY)
            auxiliary_norm, auxiliary_squares, auxiliary_loss = probe(
                auxiliary_loss_key
            )
            total_norm, total_squares, total_loss = probe("loss")
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            restore_rng_state()

        if not all(
            math.isfinite(value) for value in (action_norm, auxiliary_norm, total_norm)
        ):
            raise FloatingPointError(
                "Non-finite SimpleTac gradient diagnostic norms: "
                f"action={action_norm} auxiliary={auxiliary_norm} "
                f"total={total_norm}"
            )
        metrics = gradient_interaction_metrics(
            action_norm,
            auxiliary_norm,
            total_norm,
            auxiliary_name=auxiliary_name,
        )
        metrics.update(
            shared_gradient_interaction_metrics(
                action_squares,
                auxiliary_squares,
                total_squares,
                auxiliary_name=auxiliary_name,
            )
        )
        loss_identity_error = abs(total_loss - action_loss - auxiliary_loss)
        loss_identity_scale = max(
            abs(total_loss),
            abs(action_loss) + abs(auxiliary_loss),
            1.0,
        )
        loss_identity_valid = loss_identity_error <= 1.0e-5 * loss_identity_scale
        metrics.update(
            {
                "gradient_probe_loss_identity_error": loss_identity_error,
                "gradient_probe_loss_identity_valid": float(loss_identity_valid),
                "gradient_probe_is_post_update": 1.0,
                "gradient_probe_micro_batches": float(len(batches)),
            }
        )
        if not loss_identity_valid:
            alignment_name = (
                "world" if auxiliary_name == "tactile_world" else auxiliary_name
            )
            metrics[f"action_{alignment_name}_grad_cosine_valid"] = 0.0
            metrics[f"shared_action_{alignment_name}_grad_cosine_valid"] = 0.0
        return metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        def bundle_norm_stats() -> None:
            norm_stats_path = self.cfg.data.get("norm_stats_path", None)
            if norm_stats_path:
                norm_stats_path = os.path.expanduser(str(norm_stats_path))
                if not os.path.isfile(norm_stats_path):
                    raise FileNotFoundError(
                        f"Cannot bundle missing normalization stats: {norm_stats_path}"
                    )
                self._atomic_copy2(
                    norm_stats_path,
                    os.path.join(
                        os.path.dirname(save_path),
                        "norm_stats.json",
                    ),
                )

        # Rank-zero-only filesystem errors must be reported to every rank
        # before anybody enters the next collective. Otherwise rank 0 exits
        # while the remaining ranks hang forever in all_gather_object.
        self._run_rank_zero_io("bundle normalization statistics", bundle_norm_stats)

        if self._data_loader_supports_checkpointing(self.data_loader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            self._run_rank_zero_io(
                "save dataloader state",
                lambda: self._atomic_torch_save(
                    all_states,
                    os.path.join(save_path, "data.pt"),
                ),
            )

            rng_state = get_rng_state()
            all_rng_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_rng_states, rng_state)
            self._run_rank_zero_io(
                "save RNG state",
                lambda: self._atomic_torch_save(
                    all_rng_states,
                    os.path.join(save_path, "rng.pt"),
                ),
            )

            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)

        if self._data_loader_supports_checkpointing(self.data_loader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            if not isinstance(all_states, (list, tuple)):
                raise TypeError(
                    "Dataloader checkpoint must contain one state per rank, "
                    f"got {type(all_states).__name__}"
                )
            if len(all_states) != self._world_size:
                raise ValueError(
                    "Dataloader checkpoint world size mismatch: "
                    f"checkpoint={len(all_states)}, current={self._world_size}"
                )
            state = all_states[self._rank]
            self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)

            rng_path = os.path.join(load_path, "rng.pt")
            if not os.path.isfile(rng_path):
                raise FileNotFoundError(
                    "Strict dataloader resume also requires rank-local RNG "
                    f"state, but it is missing: {rng_path}"
                )
            all_rng_states = torch.load(rng_path, weights_only=False)
            if not isinstance(all_rng_states, (list, tuple)):
                raise TypeError(
                    "RNG checkpoint must contain one state per rank, "
                    f"got {type(all_rng_states).__name__}"
                )
            if len(all_rng_states) != self._world_size:
                raise ValueError(
                    "RNG checkpoint world size mismatch: "
                    f"checkpoint={len(all_rng_states)}, current={self._world_size}"
                )
            set_rng_state(all_rng_states[self._rank])

            torch.distributed.barrier()

    @staticmethod
    def _data_loader_supports_checkpointing(data_loader: Any) -> bool:
        """Accept torchdata and project-specific stateful loader protocols."""
        return (
            data_loader is not None
            and callable(getattr(data_loader, "state_dict", None))
            and callable(getattr(data_loader, "load_state_dict", None))
        )

    def _run_rank_zero_io(
        self,
        description: str,
        operation: Callable[[], None],
    ) -> None:
        """Run rank-zero I/O and broadcast success/failure to every rank."""
        local_error: BaseException | None = None
        result: list[str | None] = [None]
        if self._rank == 0:
            try:
                operation()
            except BaseException as error:
                local_error = error
                result[0] = f"{type(error).__name__}: {error}"
        torch.distributed.broadcast_object_list(result, src=0)
        if result[0] is not None:
            failure = RuntimeError(
                f"Checkpoint rank-zero I/O failed while trying to {description}: "
                f"{result[0]}"
            )
            if local_error is not None:
                raise failure from local_error
            raise failure

    @staticmethod
    def _atomic_copy2(source: str, destination: str) -> None:
        """Copy a file in-place atomically on the destination filesystem."""
        destination_dir = os.path.dirname(destination)
        os.makedirs(destination_dir, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(destination)}.",
            suffix=".tmp",
            dir=destination_dir,
        )
        os.close(descriptor)
        try:
            shutil.copy2(source, temporary_path)
            with open(temporary_path, "rb") as temporary_file:
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
        except BaseException:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _atomic_torch_save(value: Any, destination: str) -> None:
        """Write a torch payload through a same-directory temporary file."""
        destination_dir = os.path.dirname(destination)
        os.makedirs(destination_dir, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(destination)}.",
            suffix=".tmp",
            dir=destination_dir,
        )
        os.close(descriptor)
        try:
            torch.save(value, temporary_path)
            with open(temporary_path, "rb") as temporary_file:
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
        except BaseException:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.OPENPI_PYTORCH:
            return max(1, len(self.data_loader) // self.gradient_accumulation)
        if model_type == SupportedModel.OPENPI:
            num_batches = len(self._openpi_pytorch_dataloader(self.data_loader))
            return max(1, num_batches // self.gradient_accumulation)
        return super().get_max_steps_per_epoch()

    @staticmethod
    def _openpi_pytorch_dataloader(openpi_dataloader: Any):
        """Unwrap OpenPI `DataLoaderImpl` to the inner PyTorch DataLoader.

        OpenPI torch path:
          DataLoaderImpl._data_loader -> TorchDataLoader
          TorchDataLoader._data_loader / .torch_loader -> torch.utils.data.DataLoader

        """
        torch_data_loader = getattr(openpi_dataloader, "_data_loader", None)
        pytorch_dl = getattr(torch_data_loader, "_data_loader", None) or getattr(
            torch_data_loader, "torch_loader", None
        )
        if pytorch_dl is None:
            raise TypeError(
                "OpenPI dataloader does not expose an inner torch DataLoader; cannot infer steps per epoch from len()."
            )
        return pytorch_dl
