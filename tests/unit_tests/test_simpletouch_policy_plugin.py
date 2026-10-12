"""CPU-only deployment smoke test; no simulator or policy server required."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
    "task_name,horizon",
    [("insert_hole", 25), ("insert_tube", 25), ("put_bottle_in_shelf", 50)],
)
def test_plugin_query_cadence_and_reset(monkeypatch, task_name, horizon):
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/UniVTAC/plugin/policy/SimpleTac/deploy_policy.py"
    )
    spec = importlib.util.spec_from_file_location("simpletouch_plugin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Client:
        def __init__(self, *args):
            self.requests = []

        def get_server_metadata(self):
            return {
                "task": task_name,
                "camera_type": "head",
                "use_tactile": True,
                "action_horizon": 50,
                "tactile_preprocess": {"input": {"field": "rgb"}},
            }

        def infer(self, request):
            self.requests.append(request)
            return {
                "actions": np.arange(50, dtype=np.float32)[:, None].repeat(8, axis=1)
            }

        def close(self):
            pass

    monkeypatch.setattr(module.websocket_client_policy, "WebsocketClientPolicy", Client)
    for key in (
        "SIMPLETAC_TASK",
        "SIMPLETAC_EXECUTION_HORIZON",
        "SIMPLETAC_TEMPORAL_ENSEMBLE",
        "SIMPLETAC_USE_TACTILE",
    ):
        monkeypatch.delenv(key, raising=False)
    policy = module.Policy({"task_name": task_name})
    executed = []
    task = SimpleNamespace(
        device="cpu",
        instruction="",
        take_action=lambda action, **kw: executed.append(action.clone()),
    )
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 201
    observation = {
        "embodiment": {"joint": np.zeros(8)},
        "observation": {"head": {"rgb": rgb}},
        "tactile": {key: {"rgb": rgb} for key in ("left_gsmini", "right_gsmini")},
    }
    policy.reset()
    for _ in range(horizon + 1):
        policy.eval(task, observation)
    assert len(policy.client.requests) == 2
    assert torch.equal(executed[horizon - 1], torch.full((8,), float(horizon - 1)))
    assert torch.equal(executed[horizon], torch.zeros(8))
    request = policy.client.requests[0]
    assert not any("transition" in key for key in request)
    np.testing.assert_array_equal(request["observation/left_tactile_rgb"], rgb)
    policy.reset()
    policy.eval(task, observation)
    assert len(policy.client.requests) == 3
    policy.close()
