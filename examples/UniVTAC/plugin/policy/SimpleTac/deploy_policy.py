"""UniVTAC closed-loop adapter for a remote SimpleTac policy server."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from openpi_client import websocket_client_policy

TASK_PROMPTS = {
    "grasp_classify": (
        "use grasped tool for tactile sensing to move to target surface."
    ),
    "insert_HDMI": "insert the HDMI to the fixed slot.",
    "insert_hole": "insert the stick to the hole.",
    "insert_tube": "insert the tube to the fixed slot.",
    "lift_can": "grasp the can and lift it vertically without slippage.",
    "lift_bottle": (
        "grasp the bottle and lift it vertically, keeping its final base "
        "within 5 cm of the wall."
    ),
    "pull_out_key": "pull out the key.",
    "put_bottle_in_shelf": (
        "grasp the bottle, then position it into the shelf cavity."
    ),
}


def _to_rgb_uint8(image: Any) -> np.ndarray:
    """Convert an RGB frame to contiguous HWC uint8 without changing geometry."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Expected a three-channel RGB image, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size == 0 or float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    # Model-specific resize/pad/crop belongs to the policy server. Resizing here
    # would stretch UniVTAC's non-square camera and tactile frames before the
    # server applies its training-time preprocessing a second time.
    return np.ascontiguousarray(image)


class Policy:
    """Execute non-overlapping SimpleTac action chunks."""

    def __init__(self, args: dict[str, Any]):
        self.host = os.environ.get("SIMPLETAC_HOST", str(args.get("host", "127.0.0.1")))
        self.port = int(os.environ.get("SIMPLETAC_PORT", args.get("port", 8000)))
        execution_horizon_setting = str(
            os.environ.get(
                "SIMPLETAC_EXECUTION_HORIZON",
                args.get("execution_horizon", "task"),
            )
        )
        requested_execution_horizon: int | None
        if execution_horizon_setting.lower() in {"auto", "task"}:
            requested_execution_horizon = None
        else:
            try:
                requested_execution_horizon = int(execution_horizon_setting)
            except ValueError as error:
                raise ValueError(
                    "execution_horizon must be task, auto, or a positive integer; "
                    f"got {execution_horizon_setting!r}"
                ) from error
            if requested_execution_horizon <= 0:
                raise ValueError(
                    "execution_horizon must be auto or a positive integer; "
                    f"got {requested_execution_horizon}"
                )
        # Reject stale ablation settings instead of silently changing execution.
        if str(
            os.environ.get(
                "SIMPLETAC_TEMPORAL_ENSEMBLE", args.get("temporal_ensemble", False)
            )
        ).lower() not in {"0", "false", "no"}:
            raise ValueError("Temporal ensemble is no longer supported")
        self.task_name = os.environ.get(
            "SIMPLETAC_TASK",
            str(args.get("task_name", "insert_hole")),
        ).strip()
        if self.task_name not in TASK_PROMPTS:
            raise ValueError(
                "Unsupported SIMPLETAC_TASK/task_name "
                f"{self.task_name!r}; expected one of {sorted(TASK_PROMPTS)}"
            )
        tactile_setting = os.environ.get(
            "SIMPLETAC_USE_TACTILE", str(args.get("use_tactile", "auto"))
        ).lower()
        requested_tactile: bool | None
        if tactile_setting == "auto":
            requested_tactile = None
        elif tactile_setting in {"1", "true", "yes"}:
            requested_tactile = True
        elif tactile_setting in {"0", "false", "no"}:
            requested_tactile = False
        else:
            raise ValueError(
                "SIMPLETAC_USE_TACTILE must be auto, true, or false; "
                f"got {tactile_setting!r}"
            )
        self.client = websocket_client_policy.WebsocketClientPolicy(
            self.host, self.port
        )
        server_metadata = self.client.get_server_metadata()
        server_action_horizon = int(server_metadata.get("action_horizon", 50))
        if server_action_horizon <= 0:
            self.client.close()
            raise ValueError(
                f"Policy server returned invalid action_horizon={server_action_horizon}"
            )
        self.action_horizon = server_action_horizon
        self.execution_horizon = (
            server_action_horizon
            if requested_execution_horizon is None
            else requested_execution_horizon
        )
        if execution_horizon_setting.lower() == "task" and self.task_name in {
            "insert_hole",
            "insert_tube",
        }:
            # Keep short legacy checkpoints usable without changing model output.
            # Explicit `auto` still means the complete server-owned action chunk.
            self.execution_horizon = min(25, server_action_horizon)
        if self.execution_horizon > server_action_horizon:
            self.client.close()
            raise ValueError(
                "execution_horizon exceeds the policy checkpoint horizon: "
                f"requested={self.execution_horizon}, "
                f"checkpoint={server_action_horizon}"
            )
        server_task = server_metadata.get("task")
        if server_task is not None and str(server_task) != self.task_name:
            self.client.close()
            raise ValueError(
                "Task mismatch between UniVTAC client "
                f"({self.task_name}) and policy server ({server_task})"
            )
        server_tactile = bool(server_metadata.get("use_tactile", False))
        self.camera_type = str(server_metadata.get("camera_type", "all"))
        if self.camera_type not in {"head", "all"}:
            self.client.close()
            raise ValueError(
                f"Policy server returned invalid camera_type={self.camera_type!r}"
            )
        self.use_tactile = (
            server_tactile if requested_tactile is None else requested_tactile
        )
        if self.use_tactile != server_tactile:
            self.client.close()
            raise ValueError(
                "Tactile mode mismatch between UniVTAC client "
                f"({self.use_tactile}) and policy server ({server_tactile})"
            )
        tactile_preprocess = server_metadata.get("tactile_preprocess")
        tactile_input = (
            tactile_preprocess.get("input", {})
            if isinstance(tactile_preprocess, dict)
            else {}
        )
        self.tactile_image_field = str(tactile_input.get("field", "rgb"))
        if self.tactile_image_field != "rgb":
            self.client.close()
            raise ValueError(
                "Policy server returned invalid tactile image field "
                f"{self.tactile_image_field!r}"
            )
        self._chunk: np.ndarray | None = None
        self._chunk_index = 0
        self._chunk_end = self.execution_horizon

    def reset(self) -> None:
        """Discard the pending action chunk at the start of an episode."""
        self._chunk = None
        self._chunk_index = 0

    @staticmethod
    def _qpos(observation: dict[str, Any]) -> np.ndarray:
        """Extract one finite absolute 8D qpos observation."""
        qpos = observation["embodiment"]["joint"][:8]
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.detach().cpu().numpy()
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if qpos.shape != (8,) or not np.all(np.isfinite(qpos)):
            raise ValueError(f"Expected finite 8D qpos, got {qpos}")
        return qpos

    def _execute_action(
        self,
        task: Any,
        *,
        current_state: np.ndarray,
        action: np.ndarray,
    ) -> None:
        """Execute one absolute joint target."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (8,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Expected finite 8D action, got {action}")
        task.take_action(
            torch.as_tensor(action, device=task.device, dtype=torch.float32),
            action_type="qpos",
        )

    def _infer(self, task: Any, observation: dict[str, Any]) -> np.ndarray:
        qpos = self._qpos(observation)

        prompt = str(getattr(task, "instruction", "")).strip()
        if not prompt or prompt.lower() == "empty":
            prompt = TASK_PROMPTS[self.task_name]
        request = {
            "observation/head_rgb": _to_rgb_uint8(
                observation["observation"]["head"]["rgb"]
            ),
            "observation/state": qpos,
            "prompt": prompt,
        }
        if self.camera_type == "all":
            request["observation/wrist_rgb"] = _to_rgb_uint8(
                observation["observation"]["wrist"]["rgb"]
            )
        if self.use_tactile:
            tactile = observation.get("tactile", {})
            left_key = "left_gsmini" if "left_gsmini" in tactile else "left_tactile"
            right_key = "right_gsmini" if "right_gsmini" in tactile else "right_tactile"
            missing = [key for key in (left_key, right_key) if key not in tactile]
            if missing:
                raise KeyError(f"Missing live UniVTAC tactile sensors: {missing}")
            request["observation/left_tactile_rgb"] = _to_rgb_uint8(
                tactile[left_key][self.tactile_image_field]
            )
            request["observation/right_tactile_rgb"] = _to_rgb_uint8(
                tactile[right_key][self.tactile_image_field]
            )
        response = self.client.infer(request)
        chunk = np.asarray(response["actions"], dtype=np.float32)
        expected_shape = (self.action_horizon, 8)
        if chunk.shape != expected_shape:
            raise ValueError(
                f"Expected a {list(expected_shape)} action chunk, got {chunk.shape}"
            )
        if not np.all(np.isfinite(chunk)):
            raise ValueError("SimpleTac returned non-finite actions")
        return chunk

    def _eval_cached_chunk(
        self,
        task: Any,
        observation: dict[str, Any],
        current_state: np.ndarray,
    ) -> None:
        if self._chunk is None or self._chunk_index >= self._chunk_end:
            self._chunk = self._infer(task, observation)
            self._chunk_index = 0
        action = self._chunk[self._chunk_index]
        self._chunk_index += 1
        self._execute_action(task, current_state=current_state, action=action)

    def eval(self, task: Any, observation: dict[str, Any]) -> None:
        """Query when needed, then execute one absolute-qpos target."""
        current_state = self._qpos(observation)
        self._eval_cached_chunk(task, observation, current_state)

    def close(self) -> None:
        """Close the policy-server connection."""
        self.client.close()
