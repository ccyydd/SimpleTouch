"""Explicit, fail-closed support for separately recorded Piper arm commands."""

from __future__ import annotations

import h5py
import numpy as np

ACTION_TARGET_RECORDED_COMMAND = "recorded_command"


def episode_action_dataset(
    file: h5py.File, action_target_source: str | None = None
) -> h5py.Dataset:
    """Resolve action targets without ever substituting commands for observations.

    None is used by normalization generation to follow the episode contract.
    Training always passes an explicit action_target_source.
    """
    recorded = file.attrs.get("action_target_source") == ACTION_TARGET_RECORDED_COMMAND
    requested = (
        recorded
        if action_target_source is None
        else (action_target_source == ACTION_TARGET_RECORDED_COMMAND)
    )
    if recorded != requested:
        raise ValueError(
            "Dataset/selected action_target_source mismatch: recorded_command required"
        )
    if not recorded:
        return file["embodiment/joint"]
    if file.attrs.get("piper_dataset_schema") != "piper_recorded_command_v1":
        raise ValueError("Unrecognized recorded-command schema")
    if (
        file.attrs.get("gripper_action_semantics")
        != "measured_state_proxy_not_effective_command"
    ):
        raise ValueError(
            "Unrecognized gripper semantics; do not guess effective commands"
        )
    required = (
        "action/joint",
        "action/recorded_command_valid",
        "provenance/aligned_time",
        "provenance/command_source_time",
        "provenance/command_age_ms",
    )
    if any(key not in file for key in required):
        raise ValueError("Incomplete recorded-command provenance")
    targets = file["action/joint"]
    states = file["embodiment/joint"]
    if targets.shape != states.shape or targets.ndim != 2 or targets.shape[1] != 9:
        raise ValueError(
            "Recorded actions and measured states must have matching [T,9] shapes"
        )
    if not np.isfinite(targets[:]).all():
        raise ValueError("Nonfinite command labels")
    if np.any(targets[:, 6] != 0) or not np.array_equal(targets[:, 7:], states[:, 7:]):
        raise ValueError("Dummy/gripper proxy values do not match declared semantics")
    valid = file["action/recorded_command_valid"][:]
    expected = np.tile(np.array([True] * 6 + [False] * 3), (len(targets), 1))
    if not np.array_equal(valid, expected):
        raise ValueError("Recorded-command validity mask is inconsistent")
    t, ct, age = (file[k][:] for k in required[2:])
    if any(x.shape != (len(targets),) for x in (t, ct, age)) or not all(
        np.isfinite(x).all() for x in (t, ct, age)
    ):
        raise ValueError("Invalid command timestamps")
    if (
        np.any(np.diff(t) <= 0)
        or np.any(ct > t)
        or np.any(age < 0)
        or np.any(age > 100.0001)
    ):
        raise ValueError("Future/stale command labels")
    if not np.allclose(age, (t - ct) * 1000, rtol=0, atol=0.001):
        raise ValueError("Command-age provenance mismatch")
    return targets
