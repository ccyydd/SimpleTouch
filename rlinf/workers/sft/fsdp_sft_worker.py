# Copyright 2025 The RLinf Authors.
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

import logging
import os
import time
from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.data.utils import forward_set_epoch
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.cuda_memory import begin_memory_measurement, memory_metrics
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import clear_memory


def gradient_diagnostics_due(global_step: int, interval: int) -> bool:
    """Return whether a post-update diagnostic probe is due at this step."""
    return interval > 0 and global_step > 0 and global_step % interval == 0


class FSDPSftWorker(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()

        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # set the global batch size, micro batch size, eval batch size and gradient accumulation
        self.global_batch_size = self.cfg.actor.global_batch_size
        self.micro_batch_size = self.cfg.actor.micro_batch_size
        self.eval_batch_size = self.cfg.actor.get("eval_batch_size", 1)
        self.gradient_diagnostics_interval = int(
            self.cfg.actor.get("gradient_diagnostics_interval", 0)
        )
        if self.gradient_diagnostics_interval < 0:
            raise ValueError("actor.gradient_diagnostics_interval must be nonnegative")

        assert (
            self.global_batch_size % (self.micro_batch_size * self._world_size) == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"
        self.gradient_accumulation = (
            self.global_batch_size // self.micro_batch_size // self._world_size
        )

        # if train_data_paths is not set, the code will just eval the model
        if self.cfg.data.get("train_data_paths") is None:
            logging.warning("train_data_paths is not set, will just eval the model")
            assert self.cfg.data.get("val_data_paths") is not None, (
                "train_data_paths is not set, val_data_paths must be set"
            )
            self.data_loader = None
            self.data_iter = None
        else:
            self.data_loader, self.data_config = self.build_dataloader(
                self.cfg.data.train_data_paths, eval_dataset=False
            )
            self.data_iter = iter(self.data_loader)

        if self.cfg.data.get("val_data_paths") is not None:
            self.eval_data_loader, self.eval_data_config = self.build_dataloader(
                self.cfg.data.val_data_paths, eval_dataset=True
            )
        else:
            self.eval_data_loader = None

        self.global_step = 0
        # set the dataloader epoch and data iter offset
        self._data_epoch = 0
        self._data_iter_offset = 0

    def init_worker(self):
        self.setup_model_and_optimizer()

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self):
        model = get_model(self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def set_global_step(self, global_step):
        self.global_step = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def get_max_steps_per_epoch(self):
        if self.data_loader is not None:
            return max(1, len(self.data_loader) // self.gradient_accumulation)
        return 0

    def run_eval(self):
        assert self.eval_data_loader is not None, "eval_data_loader is not set"

        # reset the eval_data_iter
        eval_data_iter = iter(self.eval_data_loader)

        with self.worker_timer():
            eval_step = len(eval_data_iter)
            eval_pbar = tqdm(
                initial=0,
                total=eval_step,
                desc="Evaluate Step",
                dynamic_ncols=True,
            )
            self.model.eval()
            total = eval_step * self.eval_batch_size
            correct = 0

            # get the next batch
            for _ in range(eval_step):
                correct += self.get_eval_model_output(next(eval_data_iter))
                eval_pbar.update(1)

            metrics = {
                "eval_accuracy": float(correct / max(1, total)),
            }
            metrics = all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)
            return metrics

    def run_training(self):
        with self.worker_timer():
            self.model.train()
            memory_before = begin_memory_measurement()

            metrics = {}
            run_gradient_diagnostics = gradient_diagnostics_due(
                self.global_step,
                self.gradient_diagnostics_interval,
            )
            diagnostic_batches = [] if run_gradient_diagnostics else None
            data_wait_seconds = 0.0

            for idx in range(self.gradient_accumulation):
                # set the gradient accumulation backward_ctx
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                )

                data_wait_start = time.perf_counter()
                try:
                    batch = next(self.data_iter)
                    self._data_iter_offset += 1
                except StopIteration:
                    self._data_epoch += 1
                    logging.info(
                        f"[INFO] data_iter exhausted, reset iterator self._data_epoch {self._data_epoch}"
                    )
                    forward_set_epoch(self.data_loader, self._data_epoch)
                    self.data_iter = iter(self.data_loader)
                    batch = next(self.data_iter)
                    self._data_iter_offset = 1
                data_wait_seconds += time.perf_counter() - data_wait_start

                if diagnostic_batches is not None:
                    diagnostic_batches.append(batch)
                loss, step_metrics = self.get_train_model_output(batch)
                append_to_dict(metrics, step_metrics)

                loss = loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

            # in one step do the optimizer step
            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)

            if diagnostic_batches is not None:
                diagnostic_metrics = self.get_gradient_diagnostics(diagnostic_batches)
                if diagnostic_metrics:
                    append_to_dict(metrics, diagnostic_metrics)

            self.lr_scheduler.step()
            lr_value = self.optimizer.param_groups[0]["lr"]
            grad_norm_value = (
                float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            append_to_dict(
                metrics,
                {
                    "learning_rate": lr_value,
                    "grad_norm": grad_norm_value,
                },
            )

            if self.global_step > 0 and self.global_step % 1000 == 0:
                clear_memory()

            append_to_dict(metrics, memory_metrics(memory_before))
            append_to_dict(metrics, {"data_wait_seconds": data_wait_seconds})
            if hasattr(self.data_loader, "_epoch"):
                append_to_dict(
                    metrics,
                    {
                        "data_sampler_epoch": self.data_loader._epoch,
                        "data_batch_in_epoch": self.data_loader._batch_in_epoch,
                    },
                )
            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            return train_metrics

    def get_gradient_diagnostics(self, batches: list[Any]) -> dict[str, float]:
        """Return optional low-frequency gradient metrics for a global batch."""
        del batches
        return {}

    @abstractmethod
    def build_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def get_train_model_output(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_eval_model_output(self, batch: dict[str, Any]):
        raise NotImplementedError
