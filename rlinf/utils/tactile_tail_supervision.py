# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Loss-only episode-tail policies; no torch dependency for launch validation."""

TAIL_SUPERVISION_MODES = ("repeat", "mask", "action_repeat_tactile_mask")


def resolve_tail_supervision(
    mode: str | None,
    wam_mode: str = "fast",
    *,
    action_loss_masking: str | None = None,
    future_tail_mode: str | None = None,
) -> tuple[str, str]:
    """Resolve one switch to both losses, rejecting contradictory overrides.

    None preserves historical Fast behavior, including its attention masks.
    Explicit modes keep terminal validity out of attention in every WAM mode.
    """
    if wam_mode not in ("fast", "joint", "idm"):
        raise ValueError("WAM mode must be fast, joint or idm")
    if mode is None:
        if wam_mode != "fast":
            raise ValueError(
                "Joint/IDM require explicit tail-supervision-mode repeat or mask"
            )
        if future_tail_mode not in (None, "mask"):
            raise ValueError(
                "Repeat tactile tails require explicit tail-supervision-mode repeat"
            )
        return action_loss_masking or "include_repeat_last_padding", "mask"
    if mode not in TAIL_SUPERVISION_MODES:
        raise ValueError(f"Invalid tail-supervision-mode {mode!r}")
    if mode == "action_repeat_tactile_mask" and wam_mode != "fast":
        raise ValueError("action_repeat_tactile_mask is restricted to Fast")
    action = (
        "mask_repeat_last_padding" if mode == "mask" else "include_repeat_last_padding"
    )
    tactile = "repeat" if mode == "repeat" else "mask"
    if action_loss_masking is not None and action_loss_masking != action:
        raise ValueError(
            f"tail-supervision-mode {mode} conflicts with action_loss_masking={action_loss_masking}"
        )
    if future_tail_mode is not None and future_tail_mode != tactile:
        raise ValueError(
            f"tail-supervision-mode {mode} conflicts with future_tail_mode={future_tail_mode}"
        )
    return action, tactile
