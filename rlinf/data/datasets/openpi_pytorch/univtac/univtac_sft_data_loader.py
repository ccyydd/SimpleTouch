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

"""UniVTAC SFT loader for RLinf-aligned PyTorch Pi0.5 SimpleTac."""

from __future__ import annotations

import dataclasses
import json
import math
import multiprocessing
import typing
from pathlib import Path

import numpy as np
import sentencepiece
import torch
from torch.utils.data.distributed import DistributedSampler

from rlinf.data.datasets.openpi_pytorch.univtac.univtac_sft_dataset import (
    TACTILE_FUTURE_PREDICTION_MODES,
    UNIVTAC_ACTION_ALIGNMENT,
    UNIVTAC_ACTION_LOSS_MASKING,
    UNIVTAC_ACTION_LOSS_MASKING_MODES,
    UNIVTAC_CAMERA_TYPES,
    UNIVTAC_TERMINAL_HANDLING,
    UniVTACHDF5Dataset,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.utils.logging import get_logger
from rlinf.utils.tactile_tail_supervision import resolve_tail_supervision

logger = get_logger()

_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_TACTILE_KEYS = ("left_gsmini_rgb", "right_gsmini_rgb")
_RAW_DIM = 8
_MODEL_DIM = 32


def _validate_action_loss_masking(mode: str) -> str:
    if mode not in UNIVTAC_ACTION_LOSS_MASKING_MODES:
        raise ValueError(
            f"Unsupported action_loss_masking {mode!r}; expected one of "
            f"{sorted(UNIVTAC_ACTION_LOSS_MASKING_MODES)}"
        )
    return mode


class _PaligemmaTokenizer:
    """Local-file variant of OpenPI's exact Pi0.5 prompt tokenizer."""

    def __init__(self, path: str, max_len: int) -> None:
        tokenizer_path = Path(path).expanduser()
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Missing PaliGemma tokenizer: {tokenizer_path}")
        self._max_len = max_len
        self._tokenizer = sentencepiece.SentencePieceProcessor(
            model_proto=tokenizer_path.read_bytes()
        )

    def tokenize(self, prompt: str, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        bins = np.linspace(-1, 1, 256 + 1)[:-1]
        discretized_state = np.digitize(state, bins=bins) - 1
        state_str = " ".join(map(str, discretized_state))
        full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
        tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        tokens = tokens[: self._max_len]
        mask = [True] * len(tokens)
        padding = self._max_len - len(tokens)
        if padding:
            tokens += [0] * padding
            mask += [False] * padding
        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=np.bool_)


def _load_stats(
    path: str,
    *,
    expected_task: str,
    expected_action_horizon: int,
    expected_episodes: typing.Sequence[int] | typing.Mapping[int, int] | None = None,
    expected_episode_paths: typing.Mapping[int, str | Path] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    stats_path = Path(path).expanduser()
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing UniVTAC norm stats: {stats_path}")
    document = json.loads(stats_path.read_text())
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(
            f"UniVTAC norm stats at {stats_path} have no alignment metadata. "
            "Regenerate them with scripts/univtac/recompute_norm_stats.py."
        )
    expected_metadata = {
        "task": expected_task,
        "action_horizon": expected_action_horizon,
        "action_alignment": UNIVTAC_ACTION_ALIGNMENT,
        "terminal_handling": UNIVTAC_TERMINAL_HANDLING,
        "action_semantics": "arm_delta_from_q_t_gripper_absolute",
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if expected_episodes is not None:
        recorded = metadata.get("episodes")
        if not isinstance(recorded, list) or any(
            not isinstance(item, dict) or "episode" not in item for item in recorded
        ):
            expected_episode_metadata = (
                [
                    {"episode": int(episode), "frames": int(frames)}
                    for episode, frames in expected_episodes.items()
                ]
                if isinstance(expected_episodes, typing.Mapping)
                else list(expected_episodes)
            )
            mismatches["episodes"] = (recorded, expected_episode_metadata)
        else:
            if isinstance(expected_episodes, typing.Mapping):
                recorded_episodes = [
                    {
                        "episode": int(item["episode"]),
                        "frames": int(item.get("frames", -1)),
                    }
                    for item in recorded
                ]
                expected_episode_metadata = [
                    {"episode": int(episode), "frames": int(frames)}
                    for episode, frames in expected_episodes.items()
                ]
            else:
                recorded_episodes = [int(item["episode"]) for item in recorded]
                expected_episode_metadata = list(expected_episodes)
            if recorded_episodes != expected_episode_metadata:
                mismatches["episodes"] = (
                    recorded_episodes,
                    expected_episode_metadata,
                )
    if expected_episode_paths is not None and isinstance(
        metadata.get("episodes"), list
    ):
        recorded_sources = {
            int(item["episode"]): item
            for item in metadata["episodes"]
            if isinstance(item, dict) and "episode" in item
        }
        for episode, path_value in expected_episode_paths.items():
            path_value = Path(path_value)
            source = recorded_sources.get(int(episode), {})
            stat = path_value.stat()
            for field, actual in (
                ("size_bytes", int(stat.st_size)),
                ("mtime_ns", int(stat.st_mtime_ns)),
            ):
                if source.get(field) != actual:
                    mismatches[f"episodes.{episode}.{field}"] = (
                        source.get(field),
                        actual,
                    )
    if mismatches:
        raise ValueError(
            f"UniVTAC norm stats metadata mismatch at {stats_path}: {mismatches}. "
            "Regenerate the task statistics before training or evaluation."
        )
    payload = document.get("norm_stats", document)
    required = {"state", "actions"}
    if not required.issubset(payload):
        raise KeyError(
            f"Norm stats at {stats_path} require {sorted(required)}, "
            f"got {sorted(payload)}"
        )
    result: dict[str, dict[str, np.ndarray]] = {}
    required_fields = ("mean", "std", "q01", "q99")
    for key in sorted(required):
        item = payload[key]
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            raise KeyError(
                f"Norm stats {key!r} at {stats_path} are missing {missing_fields}"
            )
        converted = {
            field: np.asarray(item[field], dtype=np.float32)
            for field in required_fields
        }
        for field, values in converted.items():
            if values.shape != (_RAW_DIM,):
                raise ValueError(
                    f"Norm stats {key}.{field} at {stats_path} must have shape "
                    f"({_RAW_DIM},), got {values.shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Norm stats {key}.{field} at {stats_path} contain non-finite values"
                )
        if np.any(converted["std"] < 0):
            raise ValueError(
                f"Norm stats {key}.std at {stats_path} must be nonnegative"
            )
        if np.any(converted["q99"] < converted["q01"]):
            raise ValueError(f"Norm stats {key} at {stats_path} have q99 below q01")
        result[key] = converted
    return result


def _normalize(
    value: np.ndarray, stats: dict[str, np.ndarray], normalization: str
) -> np.ndarray:
    if normalization == "quantile":
        return (
            (value - stats["q01"][: value.shape[-1]])
            / (stats["q99"][: value.shape[-1]] - stats["q01"][: value.shape[-1]] + 1e-6)
            * 2.0
            - 1.0
        ).astype(np.float32)
    if normalization == "zscore":
        return (
            (value - stats["mean"][: value.shape[-1]])
            / (stats["std"][: value.shape[-1]] + 1e-6)
        ).astype(np.float32)
    raise ValueError(f"Unsupported normalization {normalization!r}")


class _TransformedUniVTACDataset(torch.utils.data.Dataset):
    """Apply FTP-1 action semantics and OpenPI Pi0.5 preprocessing."""

    def __init__(
        self,
        dataset: UniVTACHDF5Dataset,
        *,
        norm_stats_path: str,
        normalization: str,
        tokenizer_path: str,
        max_token_len: int,
        task: str,
        action_horizon: int,
        use_tactile: bool,
        use_tactile_future: bool = False,
        action_loss_masking: str = UNIVTAC_ACTION_LOSS_MASKING,
        camera_type: str | None = None,
        expected_episodes: typing.Sequence[int]
        | typing.Mapping[int, int]
        | None = None,
        expected_episode_paths: typing.Mapping[int, str | Path] | None = None,
    ) -> None:
        self._dataset = dataset
        self._transform = UniVTACTransform(
            norm_stats_path=norm_stats_path,
            normalization=normalization,
            tokenizer_path=tokenizer_path,
            max_token_len=max_token_len,
            task=task,
            action_horizon=action_horizon,
            use_tactile=use_tactile,
            use_tactile_future=use_tactile_future,
            action_loss_masking=action_loss_masking,
            camera_type=camera_type,
            expected_episodes=expected_episodes,
            expected_episode_paths=expected_episode_paths,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, typing.Any]:
        return self._transform(self._dataset[index])


class UniVTACTransform:
    """Shared train/inference transform for FTP-1-style UniVTAC inputs."""

    def __init__(
        self,
        *,
        norm_stats_path: str,
        normalization: str,
        tokenizer_path: str,
        max_token_len: int,
        task: str,
        action_horizon: int,
        use_tactile: bool,
        use_tactile_future: bool = False,
        action_loss_masking: str = UNIVTAC_ACTION_LOSS_MASKING,
        camera_type: str | None = None,
        expected_episodes: typing.Sequence[int]
        | typing.Mapping[int, int]
        | None = None,
        expected_episode_paths: typing.Mapping[int, str | Path] | None = None,
    ) -> None:
        self._stats = _load_stats(
            norm_stats_path,
            expected_task=task,
            expected_action_horizon=action_horizon,
            expected_episodes=expected_episodes,
            expected_episode_paths=expected_episode_paths,
        )
        self._normalization = normalization
        self._tokenizer = _PaligemmaTokenizer(tokenizer_path, max_token_len)
        self._use_tactile = use_tactile
        self._load_current_tactile = self._use_tactile
        if type(use_tactile_future) is not bool:
            raise TypeError(
                f"use_tactile_future must be boolean, got {use_tactile_future!r}"
            )
        self._use_tactile_future = use_tactile_future
        self._action_loss_masking = _validate_action_loss_masking(
            str(action_loss_masking)
        )
        self._camera_type = camera_type or UNIVTAC_CAMERA_TYPES[task]
        if self._camera_type not in {"head", "wrist", "all"}:
            raise ValueError(
                f"Unsupported UniVTAC camera_type {self._camera_type!r}; "
                "expected 'head', 'wrist', or 'all'"
            )

    def __call__(self, raw: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
        """Convert one raw frame to normalized, tokenized Pi0.5 inputs."""
        raw_state = np.asarray(raw["observation/state"], dtype=np.float32)
        raw_actions = raw.get("actions")
        actions = (
            np.asarray(raw_actions, dtype=np.float32).copy()
            if raw_actions is not None
            else None
        )

        # FTP-1 mix action: arm joints are relative to chunk-start qpos while
        # the gripper target remains an absolute joint position.
        if actions is not None:
            actions[..., :7] -= raw_state[None, :7]
        state = _normalize(raw_state, self._stats["state"], self._normalization)
        if actions is not None:
            actions = _normalize(actions, self._stats["actions"], self._normalization)
        tokens, token_mask = self._tokenizer.tokenize(raw["prompt"], state)

        padded_state = np.zeros((_MODEL_DIM,), dtype=np.float32)
        padded_state[:_RAW_DIM] = state

        head = np.asarray(raw["observation/head_rgb"], dtype=np.uint8)
        use_head_camera = self._camera_type in {"head", "all"}
        use_wrist_camera = self._camera_type in {"wrist", "all"}
        wrist = (
            np.asarray(raw["observation/wrist_rgb"], dtype=np.uint8)
            if use_wrist_camera
            else np.zeros_like(head)
        )
        left_tactile = None
        right_tactile = None
        load_current_tactile = getattr(
            self,
            "_load_current_tactile",
            self._use_tactile,
        )
        if load_current_tactile:
            # These arrays are reused by both the image-native VLM path and
            # T3-based consumers. The HDF5 frame is therefore decoded only
            # once in ``UniVTACHDF5Dataset``.
            left_tactile = np.asarray(
                raw["observation/left_tactile_rgb"], dtype=np.uint8
            )
            right_tactile = np.asarray(
                raw["observation/right_tactile_rgb"], dtype=np.uint8
            )
        vlm_left_wrist = wrist
        vlm_right_wrist = np.zeros_like(head)
        vlm_left_mask = np.bool_(use_wrist_camera)
        vlm_right_mask = np.False_
        item: dict[str, typing.Any] = {
            "image": {
                "base_0_rgb": head,
                "left_wrist_0_rgb": vlm_left_wrist,
                "right_wrist_0_rgb": vlm_right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.bool_(use_head_camera),
                "left_wrist_0_rgb": vlm_left_mask,
                "right_wrist_0_rgb": vlm_right_mask,
            },
            "state": padded_state,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
        }
        if actions is not None:
            padded_actions = np.zeros((actions.shape[0], _MODEL_DIM), dtype=np.float32)
            padded_actions[:, :_RAW_DIM] = actions
            item["actions"] = padded_actions
            if self._action_loss_masking == "mask_repeat_last_padding":
                action_valid_mask = np.asarray(
                    raw.get(
                        "action_valid_mask",
                        np.ones(actions.shape[0], dtype=np.bool_),
                    ),
                    dtype=np.bool_,
                )
                if action_valid_mask.shape != actions.shape[:1]:
                    raise ValueError(
                        "action_valid_mask must match the action horizon: "
                        f"mask={action_valid_mask.shape}, actions={actions.shape}"
                    )
                item["action_valid_mask"] = action_valid_mask
        if load_current_tactile:
            assert left_tactile is not None and right_tactile is not None
            item["tactile"] = {
                "left_gsmini_rgb": left_tactile,
                "right_gsmini_rgb": right_tactile,
            }
            item["tactile_mask"] = {
                "left_gsmini_rgb": np.True_,
                "right_gsmini_rgb": np.True_,
            }
        if getattr(self, "_use_tactile_future", False):
            target_valid = np.asarray(raw["observation/tactile_future_target_valid"])
            if target_valid.ndim not in (0, 1) or target_valid.dtype != np.bool_:
                raise TypeError(
                    "observation/tactile_future_target_valid must be a boolean "
                    f"scalar or vector, got shape={target_valid.shape}, "
                    f"dtype={target_valid.dtype}"
                )
            expected_leading_shape = target_valid.shape
            future_latents = raw.get("observation/tactile_world_future_latents")
            if future_latents is None:
                tactile_future = {
                    "left_gsmini_rgb": np.asarray(
                        raw["observation/left_tactile_future_rgb"],
                        dtype=np.uint8,
                    ),
                    "right_gsmini_rgb": np.asarray(
                        raw["observation/right_tactile_future_rgb"],
                        dtype=np.uint8,
                    ),
                }
                for key, value in tactile_future.items():
                    if value.shape[: target_valid.ndim] != expected_leading_shape:
                        raise ValueError(
                            f"Future tactile image {key!r} must start with horizon "
                            f"shape {expected_leading_shape}, got {value.shape}"
                        )
                item["tactile_future"] = tactile_future
            else:
                item["tactile_world_future_latents"] = np.asarray(future_latents)
            tactile_future_mask = {
                "left_gsmini_rgb": np.asarray(
                    raw["observation/left_tactile_future_mask"], dtype=np.bool_
                ),
                "right_gsmini_rgb": np.asarray(
                    raw["observation/right_tactile_future_mask"], dtype=np.bool_
                ),
            }
            for key, value in tactile_future_mask.items():
                if value.shape != expected_leading_shape:
                    raise ValueError(
                        f"Future tactile mask {key!r} must have shape "
                        f"{expected_leading_shape}, got {value.shape}"
                    )
            item["tactile_future_mask"] = tactile_future_mask
            item["tactile_future_target_valid"] = target_valid.astype(
                np.bool_, copy=False
            )
        return item

    def unnormalize_actions(
        self, normalized_actions: np.ndarray, raw_state: np.ndarray
    ) -> np.ndarray:
        """Map normalized 32D mix actions back to absolute 8D joint targets."""
        normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        actions = normalized_actions[..., :_RAW_DIM]
        stats = self._stats["actions"]
        if self._normalization == "quantile":
            actions = (actions + 1.0) / 2.0 * (
                stats["q99"][:_RAW_DIM] - stats["q01"][:_RAW_DIM] + 1e-6
            ) + stats["q01"][:_RAW_DIM]
        elif self._normalization == "zscore":
            actions = (
                actions * (stats["std"][:_RAW_DIM] + 1e-6) + stats["mean"][:_RAW_DIM]
            )
        else:
            raise ValueError(f"Unsupported normalization {self._normalization!r}")
        actions[..., :7] += np.asarray(raw_state, dtype=np.float32)[None, :7]
        return actions.astype(np.float32)


def collate_univtac_sft_items(
    items: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> tuple[Observation, torch.Tensor, torch.Tensor | None]:
    """Collate transformed UniVTAC frames into RLinf's SFT batch."""
    observation = collate_univtac_observations(items)
    actions = torch.from_numpy(np.stack([item["actions"] for item in items]))
    mask_presence = ["action_valid_mask" in item for item in items]
    if any(mask_presence) and not all(mask_presence):
        raise ValueError("A batch cannot mix masked and unmasked action samples")
    action_valid_mask = (
        torch.from_numpy(np.stack([item["action_valid_mask"] for item in items]))
        if all(mask_presence)
        else None
    )
    return observation, actions, action_valid_mask


def collate_univtac_observations(
    items: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> Observation:
    """Collate transformed frames into a model observation for train or eval."""
    if not items:
        raise ValueError("Cannot collate an empty UniVTAC batch")

    batch: dict[str, typing.Any] = {
        "image": {
            key: torch.from_numpy(np.stack([item["image"][key] for item in items]))
            for key in _IMAGE_KEYS
        },
        "image_mask": {
            key: torch.from_numpy(
                np.asarray([item["image_mask"][key] for item in items], dtype=np.bool_)
            )
            for key in _IMAGE_KEYS
        },
        "state": torch.from_numpy(np.stack([item["state"] for item in items])),
        "tokenized_prompt": torch.from_numpy(
            np.stack([item["tokenized_prompt"] for item in items])
        ).long(),
        "tokenized_prompt_mask": torch.from_numpy(
            np.stack([item["tokenized_prompt_mask"] for item in items])
        ),
    }
    if "tactile" in items[0]:
        batch["tactile"] = {
            key: torch.from_numpy(np.stack([item["tactile"][key] for item in items]))
            for key in _TACTILE_KEYS
        }
        batch["tactile_mask"] = {
            key: torch.from_numpy(
                np.asarray(
                    [item["tactile_mask"][key] for item in items], dtype=np.bool_
                )
            )
            for key in _TACTILE_KEYS
        }
    if "tactile_future" in items[0]:
        batch["tactile_future"] = {
            key: torch.from_numpy(
                np.stack([item["tactile_future"][key] for item in items])
            )
            for key in _TACTILE_KEYS
        }
    generic_future_fields = (
        "tactile_future_mask",
        "tactile_future_target_valid",
    )
    generic_future_presence = [
        all(key in item for key in generic_future_fields) for item in items
    ]
    partial_generic_future = [
        index
        for index, item in enumerate(items)
        if any(key in item for key in generic_future_fields)
        and not all(key in item for key in generic_future_fields)
    ]
    if partial_generic_future:
        raise ValueError(
            "Future-tactile items must contain both sensor masks and target "
            f"validity; partial items={partial_generic_future}"
        )
    if any(generic_future_presence) and not all(generic_future_presence):
        raise ValueError(
            "A batch cannot mix future-tactile target and non-target samples"
        )
    if all(generic_future_presence):
        if (
            "tactile_future" not in batch
            and "tactile_world_future_latents" not in items[0]
        ):
            raise ValueError(
                "Future-tactile masks and validity require images or cached latents"
            )
        batch["tactile_future_mask"] = {
            key: torch.from_numpy(
                np.asarray(
                    [item["tactile_future_mask"][key] for item in items],
                    dtype=np.bool_,
                )
            )
            for key in _TACTILE_KEYS
        }
        batch["tactile_future_target_valid"] = torch.from_numpy(
            np.asarray(
                [item["tactile_future_target_valid"] for item in items],
                dtype=np.bool_,
            )
        )
    if "tactile_world_latents" in items[0]:
        batch["tactile_world_latents"] = torch.from_numpy(
            np.stack([item["tactile_world_latents"] for item in items])
        )
    if "tactile_world_future_latents" in items[0]:
        batch["tactile_world_future_latents"] = torch.from_numpy(
            np.stack([item["tactile_world_future_latents"] for item in items])
        )

    return Observation.from_dict(batch)


@dataclasses.dataclass(frozen=True)
class UniVTACSftDataConfig:
    """Resolved metadata for one UniVTAC task loader."""

    task: str
    action_dim: int
    action_horizon: int
    max_token_len: int
    use_tactile: bool
    use_tactile_expert: bool = False
    load_tactile_future_targets: bool = False
    tactile_expert_future_horizon: int | None = None
    tactile_expert_future_prediction_mode: str | None = None
    tactile_expert_future_stride: int | None = None
    tactile_expert_future_pool_grid: int | None = None
    tactile_expert_future_tail_mode: str | None = None
    tactile_expert_tail_supervision_mode: str | None = None
    tactile_expert_future_cache_path: str | None = None
    tactile_preprocess_profile: str | None = None
    camera_type: str = "head"
    head_image_mode: str = "sensor"
    action_loss_masking: str = UNIVTAC_ACTION_LOSS_MASKING


class UniVTACSftDataLoader:
    """Infinite distributed loop over UniVTAC frames with exact cursor state.

    ``DistributedSampler.set_epoch`` determines the sample order. Checkpoints
    therefore save both the sampler epoch and the next batch within that
    epoch. On restore, already-consumed batches are deterministically skipped
    before yielding the next batch. UniVTAC preprocessing itself is
    deterministic; model-side augmentation randomness is restored separately
    by the SFT worker's RNG checkpoint.
    """

    _STATE_VERSION = 2

    def __init__(
        self,
        torch_loader: torch.utils.data.DataLoader,
        data_config: UniVTACSftDataConfig,
        sampler: DistributedSampler,
    ) -> None:
        self._torch_loader = torch_loader
        self._data_config = data_config
        self._sampler = sampler
        self._epoch = 0
        self._batch_in_epoch = 0

    def data_config(self) -> UniVTACSftDataConfig:
        return self._data_config

    def __iter__(
        self,
    ) -> typing.Iterator[tuple[Observation, torch.Tensor, torch.Tensor]]:
        while True:
            epoch = self._epoch
            start_batch = self._batch_in_epoch
            self._sampler.set_epoch(epoch)
            for batch_index, batch in enumerate(self._torch_loader):
                if batch_index < start_batch:
                    continue
                # Advance the durable cursor *before* yielding: a checkpoint
                # taken immediately after the caller consumes this batch must
                # resume from the following batch, not replay this one.
                self._batch_in_epoch = batch_index + 1
                yield batch
            self._epoch = epoch + 1
            self._batch_in_epoch = 0

    def __len__(self) -> int:
        return len(self._torch_loader)

    def state_dict(self) -> dict[str, typing.Any]:
        """Return a rank-local, validation-rich resume cursor."""
        return {
            "version": self._STATE_VERSION,
            "epoch": self._epoch,
            "batch_in_epoch": self._batch_in_epoch,
            "batches_per_epoch": len(self._torch_loader),
            "loader": {
                "dataset_size": len(self._torch_loader.dataset),
                "batch_size": self._torch_loader.batch_size,
                "drop_last": self._torch_loader.drop_last,
            },
            "data_config": dataclasses.asdict(self._data_config),
            "sampler": {
                "num_replicas": self._sampler.num_replicas,
                "rank": self._sampler.rank,
                "seed": self._sampler.seed,
                "shuffle": self._sampler.shuffle,
                "drop_last": self._sampler.drop_last,
            },
        }

    def load_state_dict(self, state: typing.Mapping[str, typing.Any]) -> None:
        """Restore a cursor, rejecting incompatible datasets or rank layouts."""
        if not isinstance(state, typing.Mapping):
            raise TypeError(
                f"UniVTAC loader state must be a mapping, got {type(state).__name__}"
            )
        version = state.get("version")
        if version != self._STATE_VERSION:
            raise ValueError(
                "Unsupported UniVTAC loader state version: "
                f"expected {self._STATE_VERSION}, got {version!r}"
            )

        expected_batches = len(self._torch_loader)
        saved_batches = state.get("batches_per_epoch")
        if saved_batches != expected_batches:
            raise ValueError(
                "Cannot resume UniVTAC loader after its length changed: "
                f"checkpoint={saved_batches}, current={expected_batches}"
            )

        expected_loader = {
            "dataset_size": len(self._torch_loader.dataset),
            "batch_size": self._torch_loader.batch_size,
            "drop_last": self._torch_loader.drop_last,
        }
        if state.get("loader") != expected_loader:
            raise ValueError(
                "Cannot resume UniVTAC loader after its dataset or batch "
                "layout changed: "
                f"checkpoint={state.get('loader')}, current={expected_loader}"
            )

        expected_config = dataclasses.asdict(self._data_config)
        saved_config = state.get("data_config")
        if saved_config != expected_config:
            raise ValueError(
                "Cannot resume UniVTAC loader with different data config: "
                f"checkpoint={saved_config}, current={expected_config}"
            )

        expected_sampler = {
            "num_replicas": self._sampler.num_replicas,
            "rank": self._sampler.rank,
            "seed": self._sampler.seed,
            "shuffle": self._sampler.shuffle,
            "drop_last": self._sampler.drop_last,
        }
        if state.get("sampler") != expected_sampler:
            raise ValueError(
                "Cannot resume UniVTAC loader with different distributed sampler: "
                f"checkpoint={state.get('sampler')}, current={expected_sampler}"
            )

        epoch = state.get("epoch")
        batch_in_epoch = state.get("batch_in_epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError(f"Invalid UniVTAC loader epoch: {epoch!r}")
        if (
            not isinstance(batch_in_epoch, int)
            or isinstance(batch_in_epoch, bool)
            or not 0 <= batch_in_epoch <= expected_batches
        ):
            raise ValueError(
                "Invalid UniVTAC loader batch cursor: "
                f"{batch_in_epoch!r} for {expected_batches} batches"
            )
        self._epoch = epoch
        self._batch_in_epoch = batch_in_epoch


def _resolve_data_path(data_paths: typing.Any) -> str:
    if isinstance(data_paths, str):
        return data_paths
    paths = list(data_paths)
    if len(paths) != 1:
        raise ValueError(f"UniVTAC requires one data root, got {paths}")
    return str(paths[0])


def _should_load_tactile_expert_future_targets(
    *,
    enabled: bool,
    future_loss_weight: float,
    eval_dataset: bool,
) -> bool:
    """Gate every future-touch data path behind actual positive supervision."""
    return enabled and future_loss_weight > 0.0 and not eval_dataset


def build_univtac_sft_dataloader(
    cfg: typing.Any,
    world_size: int,
    rank: int,
    data_paths: typing.Any,
    eval_dataset: bool = False,
) -> tuple[UniVTACSftDataLoader, UniVTACSftDataConfig]:
    """Build the SimpleTac UniVTAC loader for one distributed SFT rank."""
    from omegaconf import OmegaConf

    data_cfg = cfg.data
    openpi_cfg = cfg.actor.model.openpi
    action_horizon = int(cfg.actor.model.num_action_chunks)
    model_action_horizon = int(
        OmegaConf.select(openpi_cfg, "action_chunk", default=action_horizon)
    )
    action_dim = int(openpi_cfg.model_action_dim)
    if action_horizon <= 0 or action_dim != _MODEL_DIM:
        raise ValueError(
            "SimpleTac requires a positive action_horizon and "
            "model_action_dim=32, got "
            f"{action_horizon} and {action_dim}"
        )

    from rlinf.models.embodiment.openpi_pytorch import resolve_pi0_config

    resolve_pi0_config(cfg.actor.model)

    tactile_expert_enabled_value = OmegaConf.select(
        openpi_cfg,
        "tactile_expert_enabled",
        default=False,
    )
    if type(tactile_expert_enabled_value) is not bool:
        raise TypeError(
            "actor.model.openpi.tactile_expert_enabled must be boolean, got "
            f"{tactile_expert_enabled_value!r}"
        )
    use_tactile_expert = tactile_expert_enabled_value
    tactile_expert_future_horizon_value = OmegaConf.select(
        openpi_cfg,
        "tactile_expert_future_horizon",
        default=action_horizon,
    )
    if type(tactile_expert_future_horizon_value) is not int:
        raise TypeError(
            "actor.model.openpi.tactile_expert_future_horizon must be a positive "
            f"integer, got {tactile_expert_future_horizon_value!r}"
        )
    if tactile_expert_future_horizon_value <= 0:
        raise ValueError(
            "actor.model.openpi.tactile_expert_future_horizon must be positive, "
            f"got {tactile_expert_future_horizon_value}"
        )
    tactile_expert_future_horizon = tactile_expert_future_horizon_value
    tactile_expert_future_prediction_mode = str(
        OmegaConf.select(
            openpi_cfg,
            "tactile_expert_future_prediction_mode",
            default="single_horizon_full",
        )
    )
    if tactile_expert_future_prediction_mode not in TACTILE_FUTURE_PREDICTION_MODES:
        raise ValueError(
            "actor.model.openpi.tactile_expert_future_prediction_mode must be "
            f"one of {sorted(TACTILE_FUTURE_PREDICTION_MODES)}, got "
            f"{tactile_expert_future_prediction_mode!r}"
        )
    tactile_expert_future_stride_value = OmegaConf.select(
        openpi_cfg,
        "tactile_expert_future_stride",
        default=5,
    )
    if (
        type(tactile_expert_future_stride_value) is not int
        or tactile_expert_future_stride_value <= 0
    ):
        raise ValueError(
            "actor.model.openpi.tactile_expert_future_stride must be a positive "
            f"integer, got {tactile_expert_future_stride_value!r}"
        )
    tactile_expert_future_stride = tactile_expert_future_stride_value
    tactile_expert_future_pool_grid_value = OmegaConf.select(
        openpi_cfg,
        "tactile_expert_future_pool_grid",
        default=6,
    )
    if (
        type(tactile_expert_future_pool_grid_value) is not int
        or not 1 <= tactile_expert_future_pool_grid_value <= 14
    ):
        raise ValueError(
            "actor.model.openpi.tactile_expert_future_pool_grid must be an "
            f"integer in [1, 14], got {tactile_expert_future_pool_grid_value!r}"
        )
    tactile_expert_future_pool_grid = tactile_expert_future_pool_grid_value
    if use_tactile_expert and (
        tactile_expert_future_horizon != action_horizon
        or tactile_expert_future_horizon != model_action_horizon
    ):
        raise ValueError(
            "The Tactile Expert target must be exactly t+action_horizon: "
            f"future_horizon={tactile_expert_future_horizon}, "
            f"num_action_chunks={action_horizon}, "
            f"openpi.action_chunk={model_action_horizon}"
        )
    if (
        use_tactile_expert
        and tactile_expert_future_prediction_mode == "multi_horizon_pooled"
        and tactile_expert_future_horizon % tactile_expert_future_stride
    ):
        raise ValueError(
            "multi_horizon_pooled requires tactile_expert_future_horizon to be "
            "divisible by tactile_expert_future_stride"
        )
    tactile_expert_future_tail_mode = resolve_tail_supervision(
        OmegaConf.select(
            openpi_cfg, "tactile_expert_tail_supervision_mode", default=None
        ),
        OmegaConf.select(openpi_cfg, "tactile_expert_wam_mode", default="fast"),
    )[1]
    if use_tactile_expert and tactile_expert_future_tail_mode not in ("mask", "repeat"):
        raise ValueError(
            "actor.model.openpi.tactile_expert_future_tail_mode must be mask or repeat, "
            f"got {tactile_expert_future_tail_mode!r}"
        )
    tactile_expert_future_loss_weight = float(
        OmegaConf.select(
            openpi_cfg,
            "tactile_expert_future_loss_weight",
            default=0.0,
        )
    )
    if (
        not math.isfinite(tactile_expert_future_loss_weight)
        or tactile_expert_future_loss_weight < 0.0
    ):
        raise ValueError(
            "actor.model.openpi.tactile_expert_future_loss_weight must be "
            "finite and non-negative"
        )
    # A zero-weight ablation trains the complete current-touch TE through the
    # action loss, but must not pay for target I/O, a second T3 endpoint, or a
    # 394-token future-flow pass whose gradient is multiplied by zero.
    load_tactile_future_targets = _should_load_tactile_expert_future_targets(
        enabled=use_tactile_expert,
        future_loss_weight=tactile_expert_future_loss_weight,
        eval_dataset=eval_dataset,
    )
    # This semantic flag records whether the policy uses touch. Current TE
    # touch stays online for train/inference parity; only future targets may
    # come from a frozen full-token fallback or dedicated pooled cache.
    use_tactile = use_tactile_expert
    cache_value = OmegaConf.select(
        data_cfg,
        "tactile_world_cache_path",
        default=None,
    )
    tactile_world_cache_path = str(cache_value) if cache_value else None
    future_cache_value = OmegaConf.select(
        data_cfg,
        "tactile_expert_future_cache_path",
        default=None,
    )
    configured_tactile_expert_future_cache_path = (
        str(future_cache_value) if future_cache_value else None
    )
    from rlinf.utils.tactile_cache_policy import require_future_tactile_cache

    require_future_tactile_cache(
        enabled=use_tactile_expert,
        future_loss_weight=tactile_expert_future_loss_weight,
        full_cache=tactile_world_cache_path,
        pooled_cache=configured_tactile_expert_future_cache_path,
        eval_dataset=eval_dataset,
    )
    if configured_tactile_expert_future_cache_path is not None:
        if not use_tactile_expert:
            raise ValueError(
                "data.tactile_expert_future_cache_path requires the Tactile Expert"
            )
        if tactile_expert_future_prediction_mode != "multi_horizon_pooled":
            raise ValueError(
                "data.tactile_expert_future_cache_path requires "
                "multi_horizon_pooled prediction"
            )
        if tactile_expert_future_pool_grid != 6:
            raise ValueError(
                "data.tactile_expert_future_cache_path requires "
                "tactile_expert_future_pool_grid=6"
            )
        if not eval_dataset and not load_tactile_future_targets:
            raise ValueError(
                "data.tactile_expert_future_cache_path is training-only and "
                "requires a positive Tactile-Expert future loss"
            )
        if tactile_world_cache_path is not None:
            raise ValueError(
                "Configure only the dedicated pooled future cache; "
                "tactile_world_cache_path remains the full-token fallback"
            )
    tactile_expert_future_cache_path = (
        configured_tactile_expert_future_cache_path
        if load_tactile_future_targets
        else None
    )
    load_current_tactile_images = use_tactile_expert
    head_image_mode = str(
        OmegaConf.select(data_cfg, "head_image_mode", default="sensor")
    )
    action_loss_masking = _validate_action_loss_masking(
        str(
            OmegaConf.select(
                data_cfg,
                "action_loss_masking",
                default=resolve_tail_supervision(
                    OmegaConf.select(
                        openpi_cfg, "tactile_expert_tail_supervision_mode", default=None
                    ),
                    OmegaConf.select(
                        openpi_cfg, "tactile_expert_wam_mode", default="fast"
                    ),
                )[0],
            )
        )
    )
    tail_supervision_mode = OmegaConf.select(
        openpi_cfg, "tactile_expert_tail_supervision_mode", default=None
    )
    if use_tactile_expert:
        resolve_tail_supervision(
            tail_supervision_mode,
            OmegaConf.select(openpi_cfg, "tactile_expert_wam_mode", default="fast"),
            action_loss_masking=action_loss_masking,
            future_tail_mode=tactile_expert_future_tail_mode,
        )
    elif tail_supervision_mode is not None:
        raise ValueError("tail-supervision-mode requires an enabled Tactile Expert")
    task = str(data_cfg.task)
    if task not in UNIVTAC_CAMERA_TYPES:
        raise ValueError(
            f"Unsupported UniVTAC task {task!r}; "
            f"choose from {sorted(UNIVTAC_CAMERA_TYPES)}"
        )
    needs_t3_preprocess = use_tactile_expert
    active_tactile_preprocess_profile: str | None = None
    if needs_t3_preprocess:
        tactile_preprocess_profile_value = OmegaConf.select(
            openpi_cfg,
            "tactile_preprocess_profile",
            default=None,
        )
        if tactile_preprocess_profile_value is None:
            raise ValueError(
                "Tactile Expert requires an explicit T3 preprocess profile"
            )
        active_tactile_preprocess_profile = str(tactile_preprocess_profile_value)
        tactile_preprocess_task_value = OmegaConf.select(
            openpi_cfg,
            "tactile_preprocess_task",
            default=None,
        )
        tactile_preprocess_task = (
            task
            if tactile_preprocess_task_value is None
            else str(tactile_preprocess_task_value)
        )
        if tactile_preprocess_task != task:
            raise ValueError(
                "tactile_preprocess_task must match data.task: "
                f"preprocess={tactile_preprocess_task!r}, data={task!r}"
            )
    expected_camera_type = UNIVTAC_CAMERA_TYPES[task]
    camera_type = str(
        OmegaConf.select(
            data_cfg,
            "camera_type",
            default=expected_camera_type,
        )
    )
    if camera_type != expected_camera_type:
        raise ValueError(
            f"UniVTAC task {task!r} requires camera_type={expected_camera_type!r}, got {camera_type!r}"
        )
    raw_dataset = UniVTACHDF5Dataset(
        data_dir=_resolve_data_path(data_paths),
        task=task,
        action_horizon=action_horizon,
        episode_start=int(data_cfg.episode_start),
        episode_end=int(data_cfg.episode_end),
        use_tactile=load_current_tactile_images,
        use_tactile_future=load_tactile_future_targets,
        tactile_future_horizon=tactile_expert_future_horizon,
        tactile_future_prediction_mode=tactile_expert_future_prediction_mode,
        tactile_future_stride=tactile_expert_future_stride,
        tactile_world_cache_path=tactile_world_cache_path,
        tactile_expert_future_cache_path=tactile_expert_future_cache_path,
        tactile_encoder_path=str(openpi_cfg.tactile_encoder_path),
        tactile_preprocess_profile=active_tactile_preprocess_profile,
        # Full-token fallback caches always contain CLS + 14x14 patches.
        # Future pooled targets have their own independent grid below.
        tactile_world_pool_grid=14,
        tactile_expert_future_pool_grid=tactile_expert_future_pool_grid,
        camera_type=camera_type,
        head_image_mode=head_image_mode,
        norm_stats_path=str(data_cfg.norm_stats_path),
    )
    dataset = _TransformedUniVTACDataset(
        raw_dataset,
        norm_stats_path=str(data_cfg.norm_stats_path),
        normalization=str(data_cfg.normalization),
        tokenizer_path=str(data_cfg.tokenizer_path),
        max_token_len=int(openpi_cfg.max_token_len),
        task=task,
        action_horizon=action_horizon,
        use_tactile=load_current_tactile_images,
        use_tactile_future=load_tactile_future_targets,
        action_loss_masking=action_loss_masking,
        camera_type=camera_type,
        expected_episodes=raw_dataset.episode_lengths,
        expected_episode_paths=raw_dataset.episode_paths,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        seed=int(cfg.actor.seed),
        drop_last=True,
    )
    num_workers = int(data_cfg.num_workers)
    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=(
            int(cfg.actor.eval_batch_size)
            if eval_dataset
            else int(cfg.actor.micro_batch_size)
        ),
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=collate_univtac_sft_items,
        drop_last=True,
        pin_memory=True,
    )
    resolved = UniVTACSftDataConfig(
        task=task,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=int(openpi_cfg.max_token_len),
        use_tactile=use_tactile,
        use_tactile_expert=use_tactile_expert,
        load_tactile_future_targets=load_tactile_future_targets,
        tactile_expert_future_horizon=(
            tactile_expert_future_horizon if use_tactile_expert else None
        ),
        tactile_expert_future_prediction_mode=(
            tactile_expert_future_prediction_mode if use_tactile_expert else None
        ),
        tactile_expert_future_stride=(
            tactile_expert_future_stride if use_tactile_expert else None
        ),
        tactile_expert_future_pool_grid=(
            tactile_expert_future_pool_grid if use_tactile_expert else None
        ),
        tactile_expert_future_tail_mode=(
            tactile_expert_future_tail_mode if use_tactile_expert else None
        ),
        tactile_expert_tail_supervision_mode=tail_supervision_mode,
        tactile_expert_future_cache_path=tactile_expert_future_cache_path,
        tactile_preprocess_profile=active_tactile_preprocess_profile,
        camera_type=camera_type,
        head_image_mode=raw_dataset.head_image_mode,
        action_loss_masking=action_loss_masking,
    )
    logger.info(
        "UniVTAC loader: task=%s episodes=%d-%d batch=%d camera=%s tactile=%s "
        "expert=%s future_targets=%s future_mode=%s horizon=%s stride=%s pool_grid=%s "
        "future_cache=%s action_loss_masking=%s full_cache=%s",
        task,
        int(data_cfg.episode_start),
        int(data_cfg.episode_end),
        loader.batch_size,
        camera_type,
        use_tactile,
        use_tactile_expert,
        load_tactile_future_targets,
        tactile_expert_future_prediction_mode if use_tactile_expert else None,
        tactile_expert_future_horizon if use_tactile_expert else None,
        tactile_expert_future_stride if use_tactile_expert else None,
        tactile_expert_future_pool_grid if use_tactile_expert else None,
        tactile_expert_future_cache_path,
        action_loss_masking,
        tactile_world_cache_path,
    )
    infinite_loader = UniVTACSftDataLoader(loader, resolved, sampler)
    return infinite_loader, resolved
