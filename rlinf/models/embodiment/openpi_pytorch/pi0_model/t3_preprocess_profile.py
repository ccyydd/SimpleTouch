# Copyright 2026 The SimpleTac Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Immutable preprocessing profiles for SimpleTac's T3-large inputs.

Profiles are semantic checkpoint/cache metadata, not tunable runtime defaults.
Once a named profile has produced a checkpoint or cache, its values must never
be edited in place.  A corrected statistic requires a new profile identifier.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

T3_LEGACY_PREPROCESS_PROFILE = "legacy_imagenet_v1"
T3_UNIVTAC_RGB_MARKER_PREPROCESS_PROFILE = (
    "univtac_task_rgb_marker_train_ep0_49_r256_cc224_v1"
)
T3_UNIVTAC_RGB_PREPROCESS_PROFILE = "univtac_task_rgb_train_ep0_49_r256_cc224_v1"
T3_PIPER_PLAY_MAHJONG_RGB_PREPROCESS_PROFILE = (
    "piper_play_mahjong_rgb_train_ep0_52_r256_cc224_v1"
)
T3_PIPER_WIPE_BOARD_RGB_PREPROCESS_PROFILE = (
    "piper_wipe_board_rgb_train_ep0_50_r256_cc224_v1"
)
T3_PIPER_INSERT_USB_RGB_PREPROCESS_PROFILE = (
    "piper_insert_usb_rgb_train_ep0_44_r256_cc224_v1"
)
# Backwards-compatible public name used by existing SimpleTac callers/tests.
T3_UNIVTAC_PREPROCESS_PROFILE = T3_UNIVTAC_RGB_MARKER_PREPROCESS_PROFILE
T3_PREPROCESS_PROFILES = (
    T3_LEGACY_PREPROCESS_PROFILE,
    T3_UNIVTAC_RGB_MARKER_PREPROCESS_PROFILE,
    T3_UNIVTAC_RGB_PREPROCESS_PROFILE,
    T3_PIPER_PLAY_MAHJONG_RGB_PREPROCESS_PROFILE,
    T3_PIPER_WIPE_BOARD_RGB_PREPROCESS_PROFILE,
    T3_PIPER_INSERT_USB_RGB_PREPROCESS_PROFILE,
)
T3_TACTILE_IMAGE_FIELDS = ("rgb", "rgb_marker")

_LEGACY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_LEGACY_IMAGENET_STD = (0.229, 0.224, 0.225)

# Frozen from every frame in UniVTAC train episodes 0--49. Left/right
# ``rgb_marker`` images are combined after the exact deterministic
# resize-shorter-256 + center-crop-224 geometry below. Values are population
# RGB mean/std in [0, 1]. Never mutate this published v1 table; corrections
# require a new profile identifier.
_UNIVTAC_RGB_MARKER_PER_TASK_V1: Mapping[
    str,
    tuple[tuple[float, float, float], tuple[float, float, float]],
] = MappingProxyType(
    {
        "grasp_classify": (
            (0.378014710, 0.405331745, 0.492472720),
            (0.185822787, 0.146273801, 0.194425949),
        ),
        "insert_HDMI": (
            (0.382806827, 0.407359146, 0.496201880),
            (0.172693938, 0.147108061, 0.189555412),
        ),
        "insert_hole": (
            (0.378379829, 0.408016123, 0.499337185),
            (0.166710018, 0.147080258, 0.194060091),
        ),
        "insert_tube": (
            (0.376262564, 0.408611430, 0.500525776),
            (0.162469932, 0.147058146, 0.196737707),
        ),
        "lift_bottle": (
            (0.378777323, 0.408556920, 0.501289226),
            (0.161706435, 0.148344051, 0.193962979),
        ),
        "lift_can": (
            (0.387764188, 0.410146077, 0.497412382),
            (0.170540045, 0.149984980, 0.188655513),
        ),
        "pull_out_key": (
            (0.381145579, 0.410094016, 0.500191897),
            (0.166410088, 0.150034992, 0.194261598),
        ),
        "put_bottle_in_shelf": (
            (0.380058044, 0.406550359, 0.497401925),
            (0.168525027, 0.150256413, 0.193289080),
        ),
    }
)

# Frozen with the same population-statistics recipe as VLA-JEPA-vactile:
# every raw ``rgb`` frame from both Mini sensors in train episodes 0--49,
# after resize-shorter-256 + center-crop-224. These values are copied from the
# verified ``t3_large_rgb_tasknorm_v2_h50`` manifests and are immutable.
_UNIVTAC_RGB_PER_TASK_V1: Mapping[
    str,
    tuple[tuple[float, float, float], tuple[float, float, float]],
] = MappingProxyType(
    {
        "insert_hole": (
            (0.43324513609644605, 0.4675375425033244, 0.5727194211050125),
            (0.10764267005854185, 0.02702625455934782, 0.09167305341849787),
        ),
        "insert_tube": (
            (0.43100133642299454, 0.46833287820123615, 0.5742390409058592),
            (0.10068356448726995, 0.024989086147696574, 0.09701842890393236),
        ),
        "lift_bottle": (
            (0.43405058288098036, 0.46819099041717277, 0.5751479797569947),
            (0.09780383759221649, 0.03255449862556858, 0.08976386654111308),
        ),
        "lift_can": (
            (0.44432924200056767, 0.469982685464645, 0.5707982554162997),
            (0.10980917231339365, 0.03794140844605833, 0.07917222285980911),
        ),
        "pull_out_key": (
            (0.4350446229261188, 0.4694781662340552, 0.5723494885254711),
            (0.10469174658747521, 0.03903922940418246, 0.09062368574408111),
        ),
        "put_bottle_in_shelf": (
            (0.43449485377645597, 0.4656133992706186, 0.5697995611593202),
            (0.10975333166657633, 0.04480353232438676, 0.09080015009378732),
        ),
    }
)

# Frozen from both physical GelSight Mini RGB streams in every frame of the
# reviewed Piper Play Mahjong episodes 0--52. The same deterministic
# resize-shorter-256 + center-crop-224 geometry is applied before population
# statistics are accumulated. This is a separate immutable profile because the
# real sensor distribution must not modify UniVTAC's published profiles.
_PIPER_PLAY_MAHJONG_RGB_V1 = (
    (0.3164837272971747, 0.5744430236626081, 0.6253764799452735),
    (0.09154717035092333, 0.04882673339428545, 0.11567809260875943),
)

# Frozen from both GelSight RGB streams in every reviewed converted episode
# of the 2026-09-14 Piper collection. See compute_piper_t3_rgb_stats.py for
# the exact T3 pre-normalization resize/crop and population-statistics recipe.
_PIPER_20260914_RGB_V1: Mapping[
    str,
    tuple[tuple[float, float, float], tuple[float, float, float]],
] = MappingProxyType(
    {
        "wipe_board": (
            (0.31458305234002876, 0.5740330288311017, 0.637230638067784),
            (0.08809275497569191, 0.08048005018992391, 0.12131633433939147),
        ),
        "insert_usb": (
            (0.3151649948943643, 0.5765428998890577, 0.6356718820261036),
            (0.08981792046526255, 0.05417402594094532, 0.12147174718296333),
        ),
    }
)


@dataclasses.dataclass(frozen=True)
class T3PreprocessProfile:
    """Fully resolved and fingerprintable T3 image preprocessing semantics."""

    profile_id: str
    task: str | None
    field: str
    source_fields: tuple[str, str]
    channel_order: str
    input_range: tuple[float, float]
    resize_shorter: int
    crop_height: int
    crop_width: int
    interpolation: str
    align_corners: bool | None
    antialias: bool
    crop: str
    normalization: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    statistics_source: str
    statistics_episodes: tuple[int, int] | None
    statistics_sensors: str
    std_estimator: str

    def __post_init__(self) -> None:
        if self.profile_id not in T3_PREPROCESS_PROFILES:
            raise ValueError(f"Unknown T3 preprocess profile: {self.profile_id!r}")
        if self.field not in T3_TACTILE_IMAGE_FIELDS or self.channel_order != "RGB":
            raise ValueError(
                "SimpleTac T3 profiles must consume rgb or rgb_marker in RGB order"
            )
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("T3 preprocessing mean/std must each have 3 channels")
        values = (*self.mean, *self.std)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("T3 preprocessing mean/std must be finite")
        if not all(float(value) > 0.0 for value in self.std):
            raise ValueError("T3 preprocessing std must be strictly positive")
        if self.resize_shorter <= 0 or self.crop_height <= 0 or self.crop_width <= 0:
            raise ValueError("T3 resize/crop dimensions must be positive")

    def as_fingerprint(self) -> dict[str, Any]:
        """Return canonical JSON-compatible semantics for checkpoints/caches."""
        return {
            "profile_id": self.profile_id,
            "task": self.task,
            "input": {
                "field": self.field,
                "source_fields": list(self.source_fields),
                "channel_order": self.channel_order,
                "range": list(self.input_range),
            },
            "geometry": {
                "resize_shorter": self.resize_shorter,
                "crop": self.crop,
                "crop_size": [self.crop_height, self.crop_width],
                "interpolation": self.interpolation,
                "align_corners": self.align_corners,
                "antialias": self.antialias,
            },
            "normalization": {
                "type": self.normalization,
                "mean": list(self.mean),
                "std": list(self.std),
                "statistics": {
                    "source": self.statistics_source,
                    "episodes": (
                        None
                        if self.statistics_episodes is None
                        else list(self.statistics_episodes)
                    ),
                    "sensors": self.statistics_sensors,
                    "std_estimator": self.std_estimator,
                    "applied_after_geometry": True,
                },
            },
        }


def _validate_triplet(name: str, values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 RGB values, got {values!r}")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def resolve_t3_preprocess_profile(
    profile_id: str | None,
    *,
    task: str | None = None,
) -> T3PreprocessProfile:
    """Resolve a named profile; missing metadata preserves legacy checkpoints.

    ``None`` intentionally means the pre-v11 ImageNet-normalized behavior. New
    configurations must persist a task-specific profile and task explicitly;
    silently interpreting missing metadata as a new profile would change old
    checkpoint inference.
    """
    resolved_id = (
        T3_LEGACY_PREPROCESS_PROFILE if profile_id is None else str(profile_id)
    )
    if resolved_id not in T3_PREPROCESS_PROFILES:
        raise ValueError(
            f"Unsupported T3 preprocess profile {resolved_id!r}; expected one of "
            f"{T3_PREPROCESS_PROFILES}"
        )
    resolved_task: str | None = None
    field = "rgb_marker"
    if resolved_id == T3_LEGACY_PREPROCESS_PROFILE:
        mean, std = _LEGACY_IMAGENET_MEAN, _LEGACY_IMAGENET_STD
        statistics_source = "imagenet_fallback"
        statistics_episodes = None
        statistics_sensors = "not_applicable"
    elif resolved_id == T3_PIPER_PLAY_MAHJONG_RGB_PREPROCESS_PROFILE:
        if task != "play_mahjong":
            raise ValueError(
                f"T3 preprocess profile {resolved_id!r} requires task='play_mahjong'"
            )
        resolved_task = task
        field = "rgb"
        mean, std = _PIPER_PLAY_MAHJONG_RGB_V1
        statistics_source = "piper_real_reviewed_train"
        statistics_episodes = (0, 52)
        statistics_sensors = "left_right_combined"
    elif resolved_id in {
        T3_PIPER_WIPE_BOARD_RGB_PREPROCESS_PROFILE,
        T3_PIPER_INSERT_USB_RGB_PREPROCESS_PROFILE,
    }:
        required_task = (
            "wipe_board"
            if resolved_id == T3_PIPER_WIPE_BOARD_RGB_PREPROCESS_PROFILE
            else "insert_usb"
        )
        if task != required_task:
            raise ValueError(
                f"T3 preprocess profile {resolved_id!r} requires "
                f"task={required_task!r}"
            )
        resolved_task = task
        field = "rgb"
        mean, std = _PIPER_20260914_RGB_V1[required_task]
        statistics_source = "piper_20260914_reviewed_train"
        statistics_episodes = (0, 50 if required_task == "wipe_board" else 44)
        statistics_sensors = "left_right_combined"
    else:
        if not isinstance(task, str) or not task.strip():
            raise ValueError(
                f"T3 preprocess profile {resolved_id!r} requires a UniVTAC task"
            )
        resolved_task = task.strip()
        field = (
            "rgb" if resolved_id == T3_UNIVTAC_RGB_PREPROCESS_PROFILE else "rgb_marker"
        )
        table = (
            _UNIVTAC_RGB_PER_TASK_V1
            if field == "rgb"
            else _UNIVTAC_RGB_MARKER_PER_TASK_V1
        )
        try:
            mean, std = table[resolved_task]
        except KeyError as error:
            raise ValueError(
                f"No frozen T3 {field} statistics for UniVTAC task "
                f"{resolved_task!r} in profile {resolved_id!r}"
            ) from error
        statistics_source = "univtac_train"
        statistics_episodes = (0, 49)
        statistics_sensors = "left_right_combined"

    return T3PreprocessProfile(
        profile_id=resolved_id,
        task=resolved_task,
        field=field,
        source_fields=(
            f"tactile/left_gsmini/{field}",
            f"tactile/right_gsmini/{field}",
        ),
        channel_order="RGB",
        input_range=(0.0, 1.0),
        resize_shorter=256,
        crop_height=224,
        crop_width=224,
        interpolation="bilinear",
        align_corners=False,
        antialias=True,
        crop="center",
        normalization="per_channel_zscore",
        mean=_validate_triplet("mean", mean),
        std=_validate_triplet("std", std),
        statistics_source=statistics_source,
        statistics_episodes=statistics_episodes,
        statistics_sensors=statistics_sensors,
        std_estimator="population",
    )
