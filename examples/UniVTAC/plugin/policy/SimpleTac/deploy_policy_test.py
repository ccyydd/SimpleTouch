"""Tests for the UniVTAC SimpleTac client adapter."""

import numpy as np
import pytest
import torch

from examples.UniVTAC.plugin.policy.SimpleTac import deploy_policy


class _FakeClient:
    def __init__(
        self,
        *,
        use_tactile=False,
        camera_type="head",
        action_horizon=50,
        call_offset=0.0,
        tactile_image_field="rgb",
    ):
        self.calls = 0
        self.closed = False
        self.requests = []
        self.metadata = {
            "use_tactile": use_tactile,
            "camera_type": camera_type,
            "action_horizon": action_horizon,
            "tactile_preprocess": {"input": {"field": tactile_image_field}},
        }
        self.call_offset = float(call_offset)

    def get_server_metadata(self):
        return self.metadata

    def infer(self, request):
        self.calls += 1
        self.requests.append(request)
        action_horizon = self.metadata["action_horizon"]
        actions = np.arange(action_horizon * 8, dtype=np.float32).reshape(
            action_horizon, 8
        )
        actions = actions + self.call_offset * (self.calls - 1)
        return {"actions": actions}

    def close(self):
        self.closed = True


class _FakeTask:
    instruction = "insert the stick to the hole."
    device = torch.device("cpu")

    def __init__(self):
        self.actions = []

    def take_action(self, action, *, action_type):
        assert action_type == "qpos"
        self.actions.append(action.cpu().numpy())


def _observation():
    return {
        "embodiment": {"joint": torch.zeros(8)},
        "observation": {
            "head": {"rgb": torch.zeros(3, 12, 20)},
            "wrist": {"rgb": torch.zeros(14, 22, 3)},
        },
    }


def test_deploy_executes_chunk_zero_and_replans(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(
        deploy_policy.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda *_: client,
    )
    policy = deploy_policy.Policy({"execution_horizon": 2, "task_name": "insert_hole"})
    task = _FakeTask()
    observation = _observation()

    policy.eval(task, observation)
    policy.eval(task, observation)
    policy.eval(task, observation)

    assert client.calls == 2
    np.testing.assert_array_equal(task.actions[0], np.arange(0, 8, dtype=np.float32))
    np.testing.assert_array_equal(task.actions[1], np.arange(8, 16, dtype=np.float32))
    np.testing.assert_array_equal(task.actions[2], np.arange(0, 8, dtype=np.float32))


@pytest.mark.parametrize("task_name", ["insert_hole", "insert_tube"])
def test_deploy_task_default_executes_25_of_50_steps(monkeypatch, task_name):
    client = _FakeClient()
    monkeypatch.setattr(
        deploy_policy.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda *_: client,
    )
    policy = deploy_policy.Policy({"task_name": task_name})
    task = _FakeTask()
    observation = _observation()

    for _ in range(26):
        policy.eval(task, observation)

    assert policy.action_horizon == 50
    assert policy.execution_horizon == 25
    assert client.calls == 2
    np.testing.assert_array_equal(
        task.actions[24], np.arange(192, 200, dtype=np.float32)
    )
    np.testing.assert_array_equal(task.actions[25], np.arange(0, 8, dtype=np.float32))


@pytest.mark.parametrize(
    ("task_name", "setting", "expected"),
    [
        ("insert_hole", "task", 25),
        ("insert_tube", "task", 25),
        ("put_bottle_in_shelf", "task", 50),
        ("pull_out_key", "task", 50),
        ("insert_hole", "auto", 50),
        ("insert_tube", 50, 50),
        ("put_bottle_in_shelf", 25, 25),
    ],
)
def test_deploy_task_horizon_defaults_and_explicit_overrides(
    monkeypatch, task_name, setting, expected
):
    client = _FakeClient()
    monkeypatch.setattr(
        deploy_policy.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda *_: client,
    )
    policy = deploy_policy.Policy(
        {"task_name": task_name, "execution_horizon": setting}
    )
    assert policy.action_horizon == 50
    assert policy.execution_horizon == expected


def test_deploy_environment_horizon_overrides_task_yaml(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(
        deploy_policy.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda *_: client,
    )
    monkeypatch.setenv("SIMPLETAC_EXECUTION_HORIZON", "50")
    policy = deploy_policy.Policy(
        {"task_name": "insert_hole", "execution_horizon": "task"}
    )
    assert policy.execution_horizon == 50


@pytest.mark.parametrize("setting", ["task", "auto"])
def test_deploy_auto_uses_short_checkpoint_horizon(monkeypatch, setting):
    client = _FakeClient(action_horizon=8)
    monkeypatch.setattr(
        deploy_policy.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda *_: client,
    )
    policy = deploy_policy.Policy(
        {"task_name": "insert_hole", "execution_horizon": setting}
    )
    task = _FakeTask()
    observation = _observation()

    for _ in range(9):
        policy.eval(task, observation)

    assert policy.execution_horizon == 8
    assert client.calls == 2
    np.testing.assert_array_equal(task.actions[7], np.arange(56, 64, dtype=np.float32))
    np.testing.assert_array_equal(task.actions[8], np.arange(0, 8, dtype=np.float32))


def test_deploy_rejects_execution_beyond_checkpoint_horizon(monkeypatch):
    client = _FakeClient(action_horizon=8)
    monkeypatch.setattr(
        deploy_policy.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda *_: client,
    )

    with pytest.raises(ValueError, match="exceeds the policy checkpoint horizon"):
        deploy_policy.Policy({"execution_horizon": 50, "task_name": "insert_hole"})

    assert client.closed is True


def test_deploy_forwards_two_tactile_frames(monkeypatch):
    client = _FakeClient(use_tactile=True)
    monkeypatch.setattr(
        deploy_policy.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda *_: client,
    )
    monkeypatch.setenv("SIMPLETAC_USE_TACTILE", "true")
    policy = deploy_policy.Policy({"execution_horizon": 2, "task_name": "insert_hole"})
    task = _FakeTask()
    observation = _observation()
    observation["tactile"] = {
        "left_tactile": {"rgb": torch.full((16, 16, 3), 0.25)},
        "right_tactile": {"rgb": torch.full((16, 16, 3), 0.75)},
    }

    policy.eval(task, observation)

    request = client.requests[0]
    assert request["observation/head_rgb"].shape == (12, 20, 3)
    assert "observation/wrist_rgb" not in request
    assert request["observation/left_tactile_rgb"].shape == (16, 16, 3)
    assert int(request["observation/left_tactile_rgb"].mean()) == 63
    assert int(request["observation/right_tactile_rgb"].mean()) == 191


def test_deploy_uses_checkpoint_bound_rgb_tactile_stream(monkeypatch):
    client = _FakeClient(use_tactile=True, tactile_image_field="rgb")
    monkeypatch.setattr(
        deploy_policy.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda *_: client,
    )
    policy = deploy_policy.Policy({"task_name": "insert_hole"})
    observation = _observation()
    observation["tactile"] = {
        "left_gsmini": {
            "rgb": torch.full((8, 8, 3), 17),
            "rgb_marker": torch.full((8, 8, 3), 99),
        },
        "right_gsmini": {
            "rgb": torch.full((8, 8, 3), 23),
            "rgb_marker": torch.full((8, 8, 3), 101),
        },
    }

    policy.eval(_FakeTask(), observation)

    assert policy.tactile_image_field == "rgb"
    assert int(client.requests[0]["observation/left_tactile_rgb"].mean()) == 17
    assert int(client.requests[0]["observation/right_tactile_rgb"].mean()) == 23


def test_deploy_forwards_wrist_only_when_server_requests_all_cameras(monkeypatch):
    client = _FakeClient(camera_type="all")
    monkeypatch.setattr(
        deploy_policy.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda *_: client,
    )
    policy = deploy_policy.Policy({"task_name": "insert_tube"})

    policy.eval(_FakeTask(), _observation())

    assert client.requests[0]["observation/wrist_rgb"].shape == (14, 22, 3)


def test_to_rgb_uint8_preserves_non_square_geometry_and_layout():
    chw = torch.full((3, 24, 40), 0.5)

    converted = deploy_policy._to_rgb_uint8(chw)

    assert converted.shape == (24, 40, 3)
    assert converted.dtype == np.uint8
    assert converted.flags.c_contiguous
    assert int(converted.mean()) == 127


def test_deploy_auto_detects_tactile_from_server(monkeypatch):
    client = _FakeClient(use_tactile=True)
    monkeypatch.setattr(
        deploy_policy.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda *_: client,
    )
    policy = deploy_policy.Policy({"execution_horizon": 2, "task_name": "insert_hole"})

    assert policy.use_tactile is True


def test_deploy_rejects_explicit_tactile_mismatch(monkeypatch):
    client = _FakeClient(use_tactile=True)
    monkeypatch.setattr(
        deploy_policy.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda *_: client,
    )
    monkeypatch.setenv("SIMPLETAC_USE_TACTILE", "false")

    with pytest.raises(ValueError, match="Tactile mode mismatch"):
        deploy_policy.Policy({"execution_horizon": 2, "task_name": "insert_hole"})

    assert client.closed is True


def test_deploy_rejects_server_task_mismatch(monkeypatch):
    client = _FakeClient()
    client.metadata["task"] = "put_bottle_in_shelf"
    monkeypatch.setattr(
        deploy_policy.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda *_: client,
    )

    with pytest.raises(ValueError, match="Task mismatch"):
        deploy_policy.Policy({"task_name": "insert_hole"})

    assert client.closed is True


def test_deploy_task_environment_overrides_static_yaml(monkeypatch):
    client = _FakeClient()
    client.metadata["task"] = "put_bottle_in_shelf"
    monkeypatch.setattr(
        deploy_policy.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda *_: client,
    )
    monkeypatch.setenv("SIMPLETAC_TASK", "put_bottle_in_shelf")

    policy = deploy_policy.Policy({"task_name": "insert_hole"})

    assert policy.task_name == "put_bottle_in_shelf"
    assert client.closed is False


def test_deploy_rejects_removed_temporal_ensemble(monkeypatch):
    monkeypatch.setenv("SIMPLETAC_TEMPORAL_ENSEMBLE", "true")
    with pytest.raises(ValueError, match="no longer supported"):
        deploy_policy.Policy({"task_name": "insert_hole"})
