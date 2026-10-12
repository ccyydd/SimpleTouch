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

"""Direct UniVTAC HDF5 dataset used by SimpleTac.

Each sample consists of the observation at time ``t`` and 50 absolute joint
positions from ``t`` through ``t + 49``. Indices past the episode boundary
repeat the final joint position to preserve a fixed tensor shape. The validated
0827 recipe includes those repeated targets in the action loss; an explicit
ablation can instead mask them. The shared transform stack later converts the
first seven action dimensions to deltas from the observed state at ``t`` while
leaving the gripper target absolute.
"""

import dataclasses
import pathlib
from collections import OrderedDict
from typing import SupportsIndex

import cv2
import h5py
import numpy as np
import torch

from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_cache_manifest import (
    T3_WORLD_CACHE_MANIFEST,
    build_t3_world_cache_manifest,
    load_t3_world_cache_manifest,
    validate_t3_world_cache_files,
    validate_t3_world_cache_manifest,
    validate_t3_world_cache_sources,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_preprocess_profile import (
    resolve_t3_preprocess_profile,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_tactile_encoder import (
    T3_LARGE_WIDTH,
)


TASK_PROMPTS = {
    "grasp_classify": "use grasped tool for tactile sensing to move to target surface.",
    "insert_HDMI": "insert the HDMI to the fixed slot.",
    "insert_hole": "insert the stick to the hole.",
    "insert_tube": "insert the tube to the fixed slot.",
    "lift_can": "grasp the can and lift it vertically without slippage.",
    "lift_bottle": (
        "grasp the bottle and lift it vertically, keeping its final base "
        "within 5 cm of the wall."
    ),
    "pull_out_key": "pull out the key.",
    "put_bottle_in_shelf": "grasp the bottle, then position it into the shelf cavity.",
}

# Canonical UniVTAC camera contract from policy/task_settings.json. Only
# lift_can and insert_tube use both the head and wrist cameras.
UNIVTAC_CAMERA_TYPES = {
    "grasp_classify": "head",
    "insert_HDMI": "head",
    "insert_hole": "head",
    "insert_tube": "all",
    "lift_can": "all",
    "lift_bottle": "head",
    "pull_out_key": "head",
    "put_bottle_in_shelf": "head",
}

HEAD_IMAGE_MODE_SENSOR = "sensor"
HEAD_IMAGE_MODES = frozenset((HEAD_IMAGE_MODE_SENSOR,))

UNIVTAC_ACTION_ALIGNMENT = "state_t_actions_t_to_t_plus_horizon_minus_1"
UNIVTAC_TERMINAL_HANDLING = "repeat_last"
UNIVTAC_ACTION_LOSS_MASKING = "include_repeat_last_padding"
UNIVTAC_ACTION_LOSS_MASKING_MODES = frozenset(
    (UNIVTAC_ACTION_LOSS_MASKING, "mask_repeat_last_padding")
)
TACTILE_FUTURE_PREDICTION_MODES = frozenset(
    ("single_horizon_full", "multi_horizon_pooled")
)


@dataclasses.dataclass(frozen=True)
class SampleIndex:
    episode_path: pathlib.Path
    timestep: int


def _decode_rgb(encoded: np.bytes_ | bytes | np.ndarray) -> np.ndarray:
    """Decode one HDF5 JPEG byte string into RGB uint8 HWC.

    UniVTAC's collector passes Isaac/TacEx RGB arrays directly to
    ``cv2.imencode`` without first converting RGB to BGR.  Consequently,
    ``cv2.imdecode`` restores the original RGB *numeric channel order* even
    though OpenCV conventionally labels that array BGR.  An additional
    ``COLOR_BGR2RGB`` here would therefore swap the original red and blue
    channels and create a train/deploy mismatch.
    """
    if isinstance(encoded, np.ndarray) and encoded.dtype == np.uint8:
        buffer = encoded.reshape(-1)
    else:
        buffer = np.frombuffer(bytes(encoded), dtype=np.uint8)
    image_rgb = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_rgb is None:
        raise ValueError("Failed to decode a UniVTAC JPEG frame")
    return image_rgb


def resolve_head_image_mode(task: str, mode: str) -> str:
    if mode != "sensor":
        raise ValueError("UniVTAC uses the recorded camera image")
    return mode


def apply_head_image_mode(image: np.ndarray, *, task: str, mode: str) -> np.ndarray:
    resolve_head_image_mode(task, mode)
    return np.asarray(image, dtype=np.uint8)


class UniVTACHDF5Dataset(torch.utils.data.Dataset):
    """Random-access, worker-safe view over one task's raw HDF5 episodes."""

    def __init__(
        self,
        data_dir: str | pathlib.Path,
        task: str,
        action_horizon: int,
        *,
        episode_start: int = 0,
        episode_end: int = 49,
        max_open_files: int = 8,
        load_images: bool = True,
        use_tactile: bool = False,
        use_tactile_future: bool = False,
        tactile_future_horizon: int | None = None,
        tactile_future_prediction_mode: str = "single_horizon_full",
        tactile_future_stride: int = 5,
        tactile_world_cache_path: str | pathlib.Path | None = None,
        tactile_expert_future_cache_path: str | pathlib.Path | None = None,
        tactile_encoder_path: str | pathlib.Path | None = None,
        tactile_preprocess_profile: str | None = None,
        tactile_world_pool_grid: int = 14,
        tactile_expert_future_pool_grid: int = 6,
        camera_type: str | None = None,
        head_image_mode: str = HEAD_IMAGE_MODE_SENSOR,
        norm_stats_path: str | pathlib.Path | None = None,
    ):
        if task not in TASK_PROMPTS:
            raise ValueError(
                f"Unsupported UniVTAC task {task!r}; choose from {sorted(TASK_PROMPTS)}"
            )
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        if type(use_tactile_future) is not bool:
            raise TypeError(
                f"use_tactile_future must be boolean, got {use_tactile_future!r}"
            )
        if tactile_future_horizon is None:
            tactile_future_horizon = action_horizon
        if type(tactile_future_horizon) is not int or tactile_future_horizon <= 0:
            raise ValueError(
                "tactile_future_horizon must be a positive integer, got "
                f"{tactile_future_horizon!r}"
            )
        if tactile_future_prediction_mode not in TACTILE_FUTURE_PREDICTION_MODES:
            raise ValueError(
                "Unsupported tactile_future_prediction_mode "
                f"{tactile_future_prediction_mode!r}; expected one of "
                f"{sorted(TACTILE_FUTURE_PREDICTION_MODES)}"
            )
        if type(tactile_future_stride) is not int or tactile_future_stride <= 0:
            raise ValueError(
                "tactile_future_stride must be a positive integer, got "
                f"{tactile_future_stride!r}"
            )
        if tactile_future_prediction_mode == "multi_horizon_pooled":
            if tactile_future_stride > tactile_future_horizon:
                raise ValueError("tactile_future_stride cannot exceed the horizon")
            if tactile_future_horizon % tactile_future_stride:
                raise ValueError(
                    "multi_horizon_pooled requires tactile_future_horizon to be "
                    "divisible by tactile_future_stride"
                )
        if episode_start < 0 or episode_end < episode_start:
            raise ValueError(f"Invalid episode range [{episode_start}, {episode_end}]")
        self.data_dir = pathlib.Path(data_dir).expanduser()
        self.task = task
        self.action_horizon = action_horizon
        self.episode_start = episode_start
        self.episode_end = episode_end
        self.prompt = TASK_PROMPTS[task]
        self.camera_type = camera_type or UNIVTAC_CAMERA_TYPES[task]
        self.head_image_mode = resolve_head_image_mode(task, head_image_mode)
        if self.camera_type not in {"head", "wrist", "all"}:
            raise ValueError(
                f"Unsupported UniVTAC camera_type {self.camera_type!r}; "
                "expected 'head', 'wrist', or 'all'"
            )
        self.max_open_files = max_open_files
        self.load_images = load_images
        self.use_tactile = use_tactile
        self.use_tactile_future = use_tactile_future
        self.tactile_future_horizon = tactile_future_horizon
        self.tactile_future_prediction_mode = tactile_future_prediction_mode
        self.tactile_future_stride = tactile_future_stride
        self.tactile_future_offsets = (
            tuple(
                range(
                    tactile_future_stride,
                    tactile_future_horizon + 1,
                    tactile_future_stride,
                )
            )
            if tactile_future_prediction_mode == "multi_horizon_pooled"
            else (tactile_future_horizon,)
        )
        self.tactile_world_cache_path = (
            pathlib.Path(tactile_world_cache_path).expanduser()
            if tactile_world_cache_path
            else None
        )
        self.tactile_expert_future_cache_path = (
            pathlib.Path(tactile_expert_future_cache_path).expanduser()
            if tactile_expert_future_cache_path
            else None
        )
        if self.tactile_expert_future_cache_path is not None:
            if not self.use_tactile_future:
                raise ValueError(
                    "tactile_expert_future_cache_path requires tactile future targets"
                )
            if self.tactile_future_prediction_mode != "multi_horizon_pooled":
                raise ValueError(
                    "tactile_expert_future_cache_path requires "
                    "multi_horizon_pooled targets"
                )
            if tactile_expert_future_pool_grid != 6:
                raise ValueError(
                    "The dedicated Tactile-Expert future cache is fixed to pool_grid=6"
                )
        self.tactile_future_cache_path = (
            self.tactile_expert_future_cache_path
            if self.tactile_expert_future_cache_path is not None
            else self.tactile_world_cache_path
        )
        self.load_current_tactile = (
            self.use_tactile
            or False
            or (self.use_tactile_future and self.tactile_future_cache_path is None)
        )
        self.tactile_encoder_path = tactile_encoder_path
        resolved_tactile_preprocess = resolve_t3_preprocess_profile(
            tactile_preprocess_profile,
            task=task,
        )
        self.tactile_preprocess_profile = resolved_tactile_preprocess.profile_id
        self.tactile_image_field = resolved_tactile_preprocess.field
        self.tactile_source_fields = resolved_tactile_preprocess.source_fields
        self.tactile_world_pool_grid = int(tactile_world_pool_grid)
        self.tactile_expert_future_pool_grid = int(tactile_expert_future_pool_grid)
        if (
            self.tactile_world_cache_path is not None
            or self.tactile_expert_future_cache_path is not None
        ) and not tactile_encoder_path:
            raise ValueError(
                "A versioned tactile latent cache requires tactile_encoder_path"
            )
        self._files: OrderedDict[pathlib.Path, h5py.File] = OrderedDict()
        self._latent_files: OrderedDict[pathlib.Path, np.ndarray] = OrderedDict()

        clean_dir = self.data_dir / task / "clean"
        missing = [
            clean_dir / f"{episode}.hdf5"
            for episode in range(episode_start, episode_end + 1)
        ]
        missing = [path for path in missing if not path.is_file()]
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            raise FileNotFoundError(
                f"Missing {len(missing)} requested UniVTAC episodes; "
                f"first missing: {preview}"
            )

        self._samples: list[SampleIndex] = []
        self.episode_lengths: dict[int, int] = {}
        self.episode_paths: dict[int, pathlib.Path] = {}
        for episode in range(episode_start, episode_end + 1):
            episode_path = clean_dir / f"{episode}.hdf5"
            with h5py.File(episode_path, "r") as file:
                self._validate_episode(
                    file,
                    episode_path,
                    use_tactile=self.load_current_tactile,
                    use_head_camera=self.camera_type in {"head", "all"},
                    use_wrist_camera=self.camera_type in {"wrist", "all"},
                    tactile_source_fields=self.tactile_source_fields,
                )
                length = int(file["embodiment/joint"].shape[0])
            if self.tactile_world_cache_path is not None:
                cache_path = self._cache_path(episode)
                if not cache_path.is_file():
                    raise FileNotFoundError(
                        f"Missing frozen-T3 latent cache: {cache_path}"
                    )
                cached = np.load(cache_path, mmap_mode="r")
                expected_shape = (
                    length,
                    2,
                    1 + self.tactile_world_pool_grid**2,
                    T3_LARGE_WIDTH,
                )
                if cached.shape != expected_shape or cached.dtype != np.float16:
                    raise ValueError(
                        f"Invalid latent cache {cache_path}: got "
                        f"{cached.shape}/{cached.dtype}, expected "
                        f"{expected_shape}/float16"
                    )
            if self.tactile_expert_future_cache_path is not None:
                cache_path = self._future_cache_path(episode)
                if not cache_path.is_file():
                    raise FileNotFoundError(
                        f"Missing pooled Tactile-Expert future cache: {cache_path}"
                    )
                cached = np.load(cache_path, mmap_mode="r")
                expected_shape = (
                    length,
                    2,
                    1 + self.tactile_expert_future_pool_grid**2,
                    T3_LARGE_WIDTH,
                )
                if cached.shape != expected_shape or cached.dtype != np.float16:
                    raise ValueError(
                        f"Invalid pooled future cache {cache_path}: got "
                        f"{cached.shape}/{cached.dtype}, expected "
                        f"{expected_shape}/float16"
                    )
            self.episode_lengths[episode] = length
            self.episode_paths[episode] = episode_path
            # Every observation frame is retained. Future action suffixes use
            # repeat-last padding at the episode boundary.
            self._samples.extend(
                SampleIndex(episode_path, timestep) for timestep in range(length)
            )

        if not self._samples:
            raise ValueError(
                f"No frames in {clean_dir} episodes {episode_start}-{episode_end}"
            )
        if self.tactile_world_cache_path is not None and (
            False
            or (
                self.use_tactile_future
                and self.tactile_expert_future_cache_path is None
            )
        ):
            expected_manifest = build_t3_world_cache_manifest(
                task=self.task,
                episode_lengths=self.episode_lengths,
                pool_grid=self.tactile_world_pool_grid,
                checkpoint_dir=self.tactile_encoder_path,
                episode_paths=self.episode_paths,
                preprocess_profile=self.tactile_preprocess_profile,
            )
            actual_manifest = load_t3_world_cache_manifest(
                self.tactile_world_cache_path,
                self.task,
            )
            validate_t3_world_cache_manifest(
                actual_manifest,
                expected_manifest,
                manifest_path=(
                    self.tactile_world_cache_path / self.task / T3_WORLD_CACHE_MANIFEST
                ),
                runtime_policy="consumer",
            )
            validate_t3_world_cache_sources(
                actual_manifest,
                self.episode_paths,
                verify_sha256=False,
            )
            validate_t3_world_cache_files(
                actual_manifest,
                self.tactile_world_cache_path,
                self.task,
                verify_sha256=False,
                verify_finite=False,
            )
        if self.tactile_expert_future_cache_path is not None:
            expected_manifest = build_t3_world_cache_manifest(
                task=self.task,
                episode_lengths=self.episode_lengths,
                pool_grid=self.tactile_expert_future_pool_grid,
                checkpoint_dir=self.tactile_encoder_path,
                episode_paths=self.episode_paths,
                preprocess_profile=self.tactile_preprocess_profile,
            )
            actual_manifest = load_t3_world_cache_manifest(
                self.tactile_expert_future_cache_path,
                self.task,
            )
            validate_t3_world_cache_manifest(
                actual_manifest,
                expected_manifest,
                manifest_path=(
                    self.tactile_expert_future_cache_path
                    / self.task
                    / T3_WORLD_CACHE_MANIFEST
                ),
                runtime_policy="consumer",
            )
            validate_t3_world_cache_sources(
                actual_manifest,
                self.episode_paths,
                verify_sha256=False,
            )
            validate_t3_world_cache_files(
                actual_manifest,
                self.tactile_expert_future_cache_path,
                self.task,
                verify_sha256=False,
                verify_finite=False,
            )

    @staticmethod
    def _validate_episode(
        file: h5py.File,
        path: pathlib.Path,
        *,
        use_tactile: bool,
        use_head_camera: bool,
        use_wrist_camera: bool,
        tactile_source_fields: tuple[str, str],
    ) -> None:
        required = ["embodiment/joint"]
        if use_head_camera:
            required.append("observation/head/rgb")
        if use_wrist_camera:
            required.append("observation/wrist/rgb")
        if use_tactile:
            required.extend(tactile_source_fields)
        missing = [key for key in required if key not in file]
        if missing:
            raise KeyError(f"Episode {path} is missing required keys: {missing}")
        lengths = {int(file[key].shape[0]) for key in required}
        if len(lengths) != 1:
            raise ValueError(
                f"Episode {path} has inconsistent stream lengths: {sorted(lengths)}"
            )

    def __len__(self) -> int:
        return len(self._samples)

    def _open(self, path: pathlib.Path) -> h5py.File:
        if path in self._files:
            return self._files.pop(path)
        file = h5py.File(path, "r")
        if len(self._files) >= self.max_open_files:
            _, oldest = self._files.popitem(last=False)
            oldest.close()
        return file

    def _cache_path(self, episode: int) -> pathlib.Path:
        if self.tactile_world_cache_path is None:
            raise RuntimeError("No tactile world cache root is configured")
        return self.tactile_world_cache_path / self.task / "clean" / f"{episode}.npy"

    def _future_cache_path(self, episode: int) -> pathlib.Path:
        if self.tactile_future_cache_path is None:
            raise RuntimeError("No tactile future cache root is configured")
        return self.tactile_future_cache_path / self.task / "clean" / f"{episode}.npy"

    def _open_latent_cache(self, path: pathlib.Path) -> np.ndarray:
        if path in self._latent_files:
            return self._latent_files.pop(path)
        array = np.load(path, mmap_mode="r")
        if len(self._latent_files) >= self.max_open_files:
            self._latent_files.popitem(last=False)
        return array

    def __getitem__(self, index: SupportsIndex) -> dict:
        sample = self._samples[index.__index__()]
        file = self._open(sample.episode_path)
        # Reinsert after reading from the LRU map so the entry is most-recently used.
        self._files[sample.episode_path] = file
        timestep = sample.timestep
        joint = file["embodiment/joint"]
        state = np.asarray(joint[timestep, :8], dtype=np.float32)
        episode_length = int(joint.shape[0])
        # Clamp future frames at the boundary and retain validity for tail supervision.
        tactile_future_target_valid_values = np.asarray(
            [
                timestep + offset < episode_length
                for offset in self.tactile_future_offsets
            ],
            dtype=np.bool_,
        )
        tactile_future_target_valid = (
            tactile_future_target_valid_values
            if self.tactile_future_prediction_mode == "multi_horizon_pooled"
            else np.bool_(tactile_future_target_valid_values[0])
        )
        future_timestep = min(
            timestep + self.tactile_future_horizon,
            episode_length - 1,
        )
        tactile_future_timesteps = tuple(
            min(timestep + offset, episode_length - 1)
            for offset in self.tactile_future_offsets
        )
        num_valid_actions = min(self.action_horizon, episode_length - timestep)
        action_valid_mask = (
            np.arange(self.action_horizon, dtype=np.int64) < num_valid_actions
        )
        action_end = min(timestep + self.action_horizon, episode_length)
        targets = joint
        actions = np.asarray(targets[timestep:action_end, :8], dtype=np.float32)
        if actions.shape[0] < self.action_horizon:
            padding = np.repeat(
                actions[-1:], self.action_horizon - actions.shape[0], axis=0
            )
            actions = np.concatenate([actions, padding], axis=0)
        if actions.shape != (self.action_horizon, 8):
            raise RuntimeError(
                "Invalid action target loaded for "
                f"{sample.episode_path.name} frame {timestep}: {actions.shape}"
            )
        if self.camera_type in {"wrist", "all"}:
            wrist_rgb = (
                _decode_rgb(file["observation/wrist/rgb"][timestep])
                if self.load_images
                else np.zeros((1, 1, 3), np.uint8)
            )
        else:
            wrist_rgb = None
        if self.camera_type in {"head", "all"}:
            head_rgb = (
                _decode_rgb(file["observation/head/rgb"][timestep])
                if self.load_images
                else np.zeros((1, 1, 3), np.uint8)
            )
            if self.load_images:
                head_rgb = apply_head_image_mode(
                    head_rgb,
                    task=self.task,
                    mode=self.head_image_mode,
                )
        else:
            assert wrist_rgb is not None
            head_rgb = np.zeros_like(wrist_rgb)
        if wrist_rgb is None:
            wrist_rgb = np.zeros_like(head_rgb)
        result = {
            "observation/head_rgb": head_rgb,
            "observation/wrist_rgb": wrist_rgb,
            "observation/state": state,
            "actions": actions,
            "action_valid_mask": action_valid_mask,
            "prompt": self.prompt,
        }
        if self.load_current_tactile:
            result["observation/left_tactile_rgb"] = (
                _decode_rgb(file[self.tactile_source_fields[0]][timestep])
                if self.load_images
                else np.zeros((1, 1, 3), np.uint8)
            )
            result["observation/right_tactile_rgb"] = (
                _decode_rgb(file[self.tactile_source_fields[1]][timestep])
                if self.load_images
                else np.zeros((1, 1, 3), np.uint8)
            )
        # A validated frozen-T3 cache is the training-time replacement for
        # online image encoding. In particular, multi-horizon Tactile-Expert
        # training must not decode K future RGB frames only to ignore them in
        # favour of cached latents.
        needs_future_tactile_images = (
            self.use_tactile_future and self.tactile_future_cache_path is None
        ) or (False and self.tactile_world_cache_path is None)
        if needs_future_tactile_images:
            if (
                self.use_tactile_future
                and self.tactile_future_prediction_mode == "multi_horizon_pooled"
            ):
                for output_key, source_field in (
                    (
                        "observation/left_tactile_future_rgb",
                        self.tactile_source_fields[0],
                    ),
                    (
                        "observation/right_tactile_future_rgb",
                        self.tactile_source_fields[1],
                    ),
                ):
                    result[output_key] = np.stack(
                        [
                            _decode_rgb(file[source_field][future_index])
                            if self.load_images
                            else np.zeros((1, 1, 3), np.uint8)
                            for future_index in tactile_future_timesteps
                        ],
                        axis=0,
                    )
            else:
                result["observation/left_tactile_future_rgb"] = (
                    _decode_rgb(file[self.tactile_source_fields[0]][future_timestep])
                    if self.load_images
                    else np.zeros((1, 1, 3), np.uint8)
                )
                result["observation/right_tactile_future_rgb"] = (
                    _decode_rgb(file[self.tactile_source_fields[1]][future_timestep])
                    if self.load_images
                    else np.zeros((1, 1, 3), np.uint8)
                )
        if self.use_tactile_future:
            result["observation/tactile_future_target_valid"] = (
                tactile_future_target_valid
            )
            # Both streams are mandatory and length-validated at construction.
            # These masks describe sensor presence independently from whether
            # each requested future timestep lies inside the episode.
            future_sensor_mask = (
                np.ones(len(self.tactile_future_offsets), dtype=np.bool_)
                if self.tactile_future_prediction_mode == "multi_horizon_pooled"
                else np.True_
            )
            result["observation/left_tactile_future_mask"] = future_sensor_mask
            result["observation/right_tactile_future_mask"] = future_sensor_mask.copy()
        if self.use_tactile_future and self.tactile_future_cache_path is not None:
            episode = int(sample.episode_path.stem)
            cache_path = self._future_cache_path(episode)
            latent_cache = self._open_latent_cache(cache_path)
            self._latent_files[cache_path] = latent_cache
            if self.tactile_future_prediction_mode == "multi_horizon_pooled":
                result["observation/tactile_world_future_latents"] = np.asarray(
                    latent_cache[list(tactile_future_timesteps)]
                )
            else:
                result["observation/tactile_world_future_latents"] = np.asarray(
                    latent_cache[future_timestep]
                )
        return result

    def close(self) -> None:
        files = getattr(self, "_files", None)
        if files is None:
            return
        for file in files.values():
            try:
                file.close()
            except (TypeError, ValueError):
                # h5py modules may be partially torn down during interpreter shutdown.
                pass
        files.clear()
        latent_files = getattr(self, "_latent_files", None)
        if latent_files is not None:
            latent_files.clear()

    def __getstate__(self) -> dict:
        # HDF5 handles cannot cross PyTorch worker process boundaries.
        state = self.__dict__.copy()
        state["_files"] = OrderedDict()
        state["_latent_files"] = OrderedDict()
        return state

    def __del__(self):
        self.close()
