"""Unit tests for SimpleTac's UniVTAC alignment and action semantics."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from openpi_client import msgpack_numpy
from torch import nn

from rlinf.data.datasets.openpi_pytorch.univtac import (
    UniVTACHDF5Dataset,
    UniVTACTransform,
)
from rlinf.data.datasets.openpi_pytorch.univtac.univtac_sft_data_loader import (
    _should_load_tactile_expert_future_targets,
    collate_univtac_sft_items,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model import gemma
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_cache_manifest import (
    build_t3_world_cache_manifest,
    validate_t3_world_cache_manifest,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_preprocess_profile import (
    T3_UNIVTAC_PREPROCESS_PROFILE,
    T3_UNIVTAC_RGB_PREPROCESS_PROFILE,
    resolve_t3_preprocess_profile,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_tactile_encoder import (
    T3_INPUT_MEAN,
    T3_INPUT_STD,
    T3_NUM_HEADS,
    T3LargeTactileEncoder,
)
from rlinf.runners.sft_runner import SFTRunner
from scripts.univtac.serve_policy import (
    WebsocketPolicyServer,
    _load_checkpoint_config,
    _resolve_norm_stats_path,
    _validate_simpletac_contract,
)


@pytest.mark.parametrize(
    ("weight", "eval_dataset", "expected"),
    ((0.0, False, False), (0.001, False, True), (0.001, True, False)),
)
def test_future_tactile_data_loading_requires_positive_training_loss(
    weight,
    eval_dataset,
    expected,
):
    assert (
        _should_load_tactile_expert_future_targets(
            enabled=True,
            future_loss_weight=weight,
            eval_dataset=eval_dataset,
        )
        is expected
    )
    assert not _should_load_tactile_expert_future_targets(
        enabled=False,
        future_loss_weight=weight,
        eval_dataset=eval_dataset,
    )


def test_websocket_server_keeps_event_loop_responsive_during_inference():
    request = msgpack_numpy.packb({"observation": np.asarray([1.0])})
    main_thread = threading.get_ident()

    class _Policy:
        metadata = {"ready": True}
        inference_thread: int | None = None

        def infer(self, observation):
            assert observation["observation"].tolist() == [1.0]
            self.inference_thread = threading.get_ident()
            return {"actions": np.asarray([[0.0]], dtype=np.float32)}

    class _Websocket:
        remote_address = ("127.0.0.1", 12345)

        def __init__(self):
            self.messages = [request]
            self.sent = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.messages:
                raise StopAsyncIteration
            return self.messages.pop(0)

        async def send(self, message):
            self.sent.append(message)

    policy = _Policy()
    websocket = _Websocket()
    server = WebsocketPolicyServer(policy, "127.0.0.1", 18066)
    asyncio.run(server._handler(websocket))

    assert policy.inference_thread is not None
    assert policy.inference_thread != main_thread
    assert len(websocket.sent) == 2


def test_websocket_server_disables_inference_keepalive_timeout(monkeypatch):
    captured: dict[str, object] = {}

    class _RunningServer:
        async def serve_forever(self):
            return None

    class _ServeContext:
        async def __aenter__(self):
            return _RunningServer()

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    def _serve(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return _ServeContext()

    monkeypatch.setattr("scripts.univtac.serve_policy.websockets.serve", _serve)
    server = WebsocketPolicyServer(SimpleNamespace(metadata={}), "127.0.0.1", 18066)
    asyncio.run(server.run())

    assert captured["ping_timeout"] is None


class _FakeTactileEncoder(nn.Module):
    """Return recognizable left/right tokens without constructing T3-large."""

    def __init__(self) -> None:
        super().__init__()
        self.last_return_all_tokens: bool | None = None

    def forward(
        self, tactile_images: torch.Tensor, *, return_all_tokens: bool = False
    ) -> torch.Tensor:
        self.last_return_all_tokens = return_all_tokens
        batch_size = tactile_images.shape[0]
        if return_all_tokens:
            values = torch.tensor([10.0, 20.0]).view(1, 2, 1, 1)
            return values.expand(batch_size, 2, 197, 4).clone()
        values = torch.tensor([10.0, 20.0]).view(1, 2, 1)
        return values.expand(batch_size, 2, 4).clone()


class _FakeTokenizer:
    def tokenize(self, prompt: str, state: np.ndarray):
        assert prompt
        assert state.shape == (8,)
        return np.asarray([1, 2, 0]), np.asarray([True, True, False])


def _write_episode(path: Path, length: int = 3) -> np.ndarray:
    path.parent.mkdir(parents=True)
    joints = np.arange(length * 9, dtype=np.float32).reshape(length, 9) / 100.0
    image = np.empty((24, 32, 3), dtype=np.uint8)
    image[...] = (240, 80, 20)
    # Match the UniVTAC collector exactly: sensor RGB is passed directly to
    # OpenCV without an RGB-to-BGR conversion before JPEG encoding.
    encoded_ok, encoded = cv2.imencode(".jpg", image)
    assert encoded_ok
    encoded_bytes = encoded.tobytes()
    frames = np.asarray([encoded_bytes] * length, dtype=f"S{len(encoded_bytes)}")
    raw_rgb = np.empty((24, 32, 3), dtype=np.uint8)
    raw_rgb[...] = (20, 40, 200)
    encoded_rgb_ok, encoded_rgb = cv2.imencode(".jpg", raw_rgb)
    assert encoded_rgb_ok
    encoded_rgb_bytes = encoded_rgb.tobytes()
    rgb_frames = np.asarray(
        [encoded_rgb_bytes] * length, dtype=f"S{len(encoded_rgb_bytes)}"
    )
    with h5py.File(path, "w") as file:
        file.create_dataset("embodiment/joint", data=joints)
        file.create_dataset("observation/head/rgb", data=frames)
        file.create_dataset("observation/wrist/rgb", data=frames)
        file.create_dataset("tactile/left_gsmini/rgb_marker", data=frames)
        file.create_dataset("tactile/right_gsmini/rgb_marker", data=frames)
        file.create_dataset("tactile/left_gsmini/rgb", data=rgb_frames)
        file.create_dataset("tactile/right_gsmini/rgb", data=rgb_frames)
    return joints


def _replace_tactile_rgb_with_timestep_colors(path: Path, length: int) -> None:
    encoded_frames: list[np.ndarray] = []
    for timestep in range(length):
        image = np.full(
            (24, 32, 3),
            (20 + 30 * timestep, 40 + 20 * timestep, 60 + 10 * timestep),
            dtype=np.uint8,
        )
        encoded_ok, encoded = cv2.imencode(".jpg", image)
        assert encoded_ok
        encoded_frames.append(encoded.reshape(-1))
    variable_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(path, "a") as file:
        for key in ("tactile/left_gsmini/rgb", "tactile/right_gsmini/rgb"):
            del file[key]
            dataset = file.create_dataset(key, (length,), dtype=variable_uint8)
            for timestep, encoded in enumerate(encoded_frames):
                dataset[timestep] = encoded


def test_dataset_selects_rgb_tactile_stream_from_preprocess_profile(tmp_path: Path):
    path = tmp_path / "insert_hole" / "clean" / "0.hdf5"
    _write_episode(path, length=1)

    dataset = UniVTACHDF5Dataset(
        tmp_path,
        "insert_hole",
        action_horizon=1,
        episode_start=0,
        episode_end=0,
        use_tactile=True,
        tactile_preprocess_profile=T3_UNIVTAC_RGB_PREPROCESS_PROFILE,
    )

    sample = dataset[0]
    assert dataset.tactile_image_field == "rgb"
    np.testing.assert_allclose(
        sample["observation/left_tactile_rgb"].mean(axis=(0, 1)),
        np.asarray([20, 40, 200]),
        atol=2,
    )


def test_head_only_dataset_does_not_require_or_decode_wrist(tmp_path: Path):
    path = tmp_path / "insert_hole" / "clean" / "0.hdf5"
    _write_episode(path, length=1)
    with h5py.File(path, "a") as file:
        del file["observation/wrist/rgb"]

    dataset = UniVTACHDF5Dataset(
        tmp_path,
        "insert_hole",
        action_horizon=1,
        episode_start=0,
        episode_end=0,
    )
    sample = dataset[0]
    assert sample["observation/wrist_rgb"].shape == sample["observation/head_rgb"].shape
    assert not sample["observation/wrist_rgb"].any()


def test_all_camera_dataset_requires_wrist(tmp_path: Path):
    path = tmp_path / "insert_tube" / "clean" / "0.hdf5"
    _write_episode(path, length=1)
    with h5py.File(path, "a") as file:
        del file["observation/wrist/rgb"]

    with pytest.raises(KeyError, match="observation/wrist/rgb"):
        UniVTACHDF5Dataset(
            tmp_path,
            "insert_tube",
            action_horizon=1,
            episode_start=0,
            episode_end=0,
        )


def _identity_transform(
    *,
    use_tactile: bool = False,
    use_tactile_future: bool = False,
    camera_type: str = "head",
    action_loss_masking: str = "include_repeat_last_padding",
) -> UniVTACTransform:
    transform = UniVTACTransform.__new__(UniVTACTransform)
    transform._stats = {
        "state": {
            "q01": -np.ones(8, dtype=np.float32),
            "q99": np.ones(8, dtype=np.float32),
        },
        "actions": {
            "q01": -np.ones(8, dtype=np.float32),
            "q99": np.ones(8, dtype=np.float32),
        },
    }
    transform._normalization = "quantile"
    transform._tokenizer = _FakeTokenizer()
    transform._use_tactile = use_tactile
    transform._load_current_tactile = use_tactile
    transform._use_tactile_future = use_tactile_future
    transform._action_loss_masking = action_loss_masking
    transform._camera_type = camera_type
    return transform


def test_dataset_uses_current_frame_actions_and_repeat_last_padding(tmp_path: Path):
    joints = _write_episode(tmp_path / "insert_hole" / "clean" / "0.hdf5")
    dataset = UniVTACHDF5Dataset(
        tmp_path,
        "insert_hole",
        action_horizon=5,
        episode_start=0,
        episode_end=0,
        use_tactile=True,
    )

    assert len(dataset) == 3
    sample = dataset[1]
    np.testing.assert_array_equal(sample["observation/state"], joints[1, :8])
    np.testing.assert_array_equal(sample["actions"], joints[[1, 2, 2, 2, 2], :8])
    np.testing.assert_array_equal(
        sample["action_valid_mask"],
        np.array([True, True, False, False, False]),
    )
    assert sample["observation/left_tactile_rgb"].shape == (24, 32, 3)
    assert sample["observation/right_tactile_rgb"].dtype == np.uint8

    first_sample = dataset[0]
    np.testing.assert_array_equal(first_sample["actions"], joints[[0, 1, 2, 2, 2], :8])
    np.testing.assert_array_equal(
        first_sample["action_valid_mask"],
        np.array([True, True, True, False, False]),
    )
    terminal_sample = dataset[2]
    np.testing.assert_array_equal(
        terminal_sample["actions"], joints[[2, 2, 2, 2, 2], :8]
    )
    np.testing.assert_array_equal(
        terminal_sample["action_valid_mask"],
        np.array([True, False, False, False, False]),
    )


def test_dataset_preserves_univtac_collector_rgb_channel_order(tmp_path: Path):
    _write_episode(tmp_path / "insert_hole" / "clean" / "0.hdf5", length=1)
    dataset = UniVTACHDF5Dataset(
        tmp_path,
        "insert_hole",
        action_horizon=1,
        episode_start=0,
        episode_end=0,
        use_tactile=True,
    )

    sample = dataset[0]
    expected_rgb = np.asarray([240.0, 80.0, 20.0])
    for key in (
        "observation/head_rgb",
        "observation/left_tactile_rgb",
        "observation/right_tactile_rgb",
    ):
        decoded_channel_mean = sample[key].mean(axis=(0, 1))
        np.testing.assert_allclose(decoded_channel_mean, expected_rgb, atol=3.0)
    # Insert Hole is head-only in the official UniVTAC task settings.
    assert not sample["observation/wrist_rgb"].any()


def test_tactile_expert_future_target_is_exact_t_plus_h_and_masks_tail(
    tmp_path: Path,
):
    episode_path = tmp_path / "insert_hole" / "clean" / "0.hdf5"
    joints = _write_episode(episode_path, length=4)
    _replace_tactile_rgb_with_timestep_colors(episode_path, length=4)
    dataset = UniVTACHDF5Dataset(
        tmp_path,
        "insert_hole",
        action_horizon=2,
        episode_start=0,
        episode_end=0,
        use_tactile=True,
        use_tactile_future=True,
        tactile_future_horizon=2,
        tactile_preprocess_profile=T3_UNIVTAC_RGB_PREPROCESS_PROFILE,
    )

    valid = dataset[1]
    expected_future = np.asarray([110.0, 100.0, 90.0])
    np.testing.assert_allclose(
        valid["observation/left_tactile_future_rgb"].mean(axis=(0, 1)),
        expected_future,
        atol=2.0,
    )
    assert bool(valid["observation/tactile_future_target_valid"])
    assert bool(valid["observation/left_tactile_future_mask"])
    assert bool(valid["observation/right_tactile_future_mask"])

    tail = dataset[2]
    # The fixed-shape placeholder is the last recorded tactile frame, but it
    # is never a valid t+H training target.
    np.testing.assert_array_equal(
        tail["observation/left_tactile_future_rgb"],
        valid["observation/left_tactile_future_rgb"],
    )
    assert not bool(tail["observation/tactile_future_target_valid"])
    # Action supervision retains the established repeat-last convention.
    np.testing.assert_array_equal(tail["actions"], joints[[2, 3], :8])
    terminal = dataset[3]
    np.testing.assert_array_equal(terminal["actions"], joints[[3, 3], :8])
    assert not bool(terminal["observation/tactile_future_target_valid"])


def test_multi_horizon_future_targets_are_exact_and_mask_each_tail(
    tmp_path: Path,
):
    episode_path = tmp_path / "insert_hole" / "clean" / "0.hdf5"
    _write_episode(episode_path, length=7)
    _replace_tactile_rgb_with_timestep_colors(episode_path, length=7)
    dataset = UniVTACHDF5Dataset(
        tmp_path,
        "insert_hole",
        action_horizon=4,
        episode_start=0,
        episode_end=0,
        use_tactile=True,
        use_tactile_future=True,
        tactile_future_horizon=4,
        tactile_future_prediction_mode="multi_horizon_pooled",
        tactile_future_stride=2,
        tactile_preprocess_profile=T3_UNIVTAC_RGB_PREPROCESS_PROFILE,
    )

    valid = dataset[1]
    future = valid["observation/left_tactile_future_rgb"]
    assert future.shape == (2, 24, 32, 3)
    np.testing.assert_allclose(
        future.mean(axis=(1, 2)),
        np.asarray([[110.0, 100.0, 90.0], [170.0, 140.0, 110.0]]),
        atol=2.0,
    )
    np.testing.assert_array_equal(
        valid["observation/tactile_future_target_valid"],
        np.asarray([True, True]),
    )

    tail = dataset[3]
    np.testing.assert_array_equal(
        tail["observation/tactile_future_target_valid"],
        np.asarray([True, False]),
    )
    # The invalid t+4 slot is shape-preserving only and clamps to the final
    # frame; its validity bit prevents any contribution to the objective.
    np.testing.assert_allclose(
        tail["observation/left_tactile_future_rgb"].mean(axis=(1, 2)),
        np.asarray([[170.0, 140.0, 110.0], [200.0, 160.0, 120.0]]),
        atol=2.0,
    )


def test_multi_horizon_future_transform_and_collate_preserve_time_axis(
    tmp_path: Path,
):
    episode_path = tmp_path / "insert_hole" / "clean" / "0.hdf5"
    _write_episode(episode_path, length=7)
    dataset = UniVTACHDF5Dataset(
        tmp_path,
        "insert_hole",
        action_horizon=4,
        episode_start=0,
        episode_end=0,
        use_tactile=True,
        use_tactile_future=True,
        tactile_future_horizon=4,
        tactile_future_prediction_mode="multi_horizon_pooled",
        tactile_future_stride=2,
    )
    transform = _identity_transform(
        use_tactile=True,
        use_tactile_future=True,
    )

    items = [transform(dataset[1]), transform(dataset[3])]
    observation, _, _ = collate_univtac_sft_items(items)

    assert observation.tactile_future_images["left_gsmini_rgb"].shape == (
        2,
        2,
        24,
        32,
        3,
    )
    torch.testing.assert_close(
        observation.tactile_future_masks["left_gsmini_rgb"],
        torch.ones(2, 2, dtype=torch.bool),
    )
    torch.testing.assert_close(
        observation.tactile_future_target_valid,
        torch.tensor([[True, True], [True, False]]),
    )


@pytest.mark.parametrize("cross_runtime", [False, True])
def test_multi_horizon_tactile_cache_skips_future_rgb_and_preserves_time_axis(
    tmp_path: Path,
    cross_runtime: bool,
):
    episode_path = tmp_path / "insert_hole" / "clean" / "0.hdf5"
    _write_episode(episode_path, length=7)
    cache_root = tmp_path / "cache"
    cache_path = cache_root / "insert_hole" / "clean" / "0.npy"
    cache_path.parent.mkdir(parents=True)
    latents = np.arange(
        7 * 2 * 2 * 1024,
        dtype=np.float16,
    ).reshape(7, 2, 2, 1024)
    np.save(cache_path, latents)
    checkpoint = tmp_path / "t3"
    (checkpoint / "encoders").mkdir(parents=True)
    (checkpoint / "encoders" / "mini.pth").write_bytes(b"mini")
    (checkpoint / "trunk.pth").write_bytes(b"trunk")
    manifest = build_t3_world_cache_manifest(
        task="insert_hole",
        episode_lengths={0: 7},
        pool_grid=1,
        checkpoint_dir=checkpoint,
        episode_paths={0: episode_path},
        include_source_sha256=True,
        cache_paths={0: cache_path},
    )
    if cross_runtime:
        manifest["runtime"].update(torch="2.6.0+rocm6.3.4", torch_cuda="None")
    (cache_root / "insert_hole" / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    dataset = UniVTACHDF5Dataset(
        tmp_path,
        "insert_hole",
        action_horizon=4,
        episode_start=0,
        episode_end=0,
        use_tactile=True,
        use_tactile_future=True,
        tactile_future_horizon=4,
        tactile_future_prediction_mode="multi_horizon_pooled",
        tactile_future_stride=2,
        tactile_world_cache_path=cache_root,
        tactile_encoder_path=checkpoint,
        tactile_world_pool_grid=1,
    )
    transform = _identity_transform(
        use_tactile=True,
        use_tactile_future=True,
    )

    raw = dataset[1]
    assert "observation/left_tactile_rgb" in raw
    assert "observation/right_tactile_rgb" in raw
    assert "observation/left_tactile_future_rgb" not in raw
    items = [transform(raw), transform(dataset[3])]
    observation, _, _ = collate_univtac_sft_items(items)

    assert observation.tactile_images is not None
    assert observation.tactile_future_images is None
    assert observation.tactile_world_latents is None
    assert observation.tactile_world_future_latents.shape == (2, 2, 2, 2, 1024)
    torch.testing.assert_close(
        observation.tactile_world_future_latents[0],
        torch.from_numpy(latents[[3, 5]]),
    )
    torch.testing.assert_close(
        observation.tactile_future_target_valid,
        torch.tensor([[True, True], [True, False]]),
    )


@pytest.mark.parametrize("cross_runtime", [False, True])
def test_dedicated_p6_future_cache_skips_rgb_and_keeps_current_touch_online(
    tmp_path: Path,
    cross_runtime: bool,
):
    episode_path = tmp_path / "insert_hole" / "clean" / "0.hdf5"
    _write_episode(episode_path, length=7)
    cache_root = tmp_path / "future_p6_cache"
    cache_path = cache_root / "insert_hole" / "clean" / "0.npy"
    cache_path.parent.mkdir(parents=True)
    latent_values = np.arange(7 * 2 * 37 * 1024, dtype=np.int64) % 1024
    latents = latent_values.astype(np.float16).reshape(7, 2, 37, 1024)
    np.save(cache_path, latents)
    checkpoint = tmp_path / "t3"
    (checkpoint / "encoders").mkdir(parents=True)
    (checkpoint / "encoders" / "mini.pth").write_bytes(b"mini")
    (checkpoint / "trunk.pth").write_bytes(b"trunk")
    manifest = build_t3_world_cache_manifest(
        task="insert_hole",
        episode_lengths={0: 7},
        pool_grid=6,
        checkpoint_dir=checkpoint,
        episode_paths={0: episode_path},
        include_source_sha256=True,
        cache_paths={0: cache_path},
        preprocess_profile=T3_UNIVTAC_RGB_PREPROCESS_PROFILE,
    )
    if cross_runtime:
        manifest["runtime"].update(torch="2.6.0+rocm6.3.4", torch_cuda="None")
    (cache_root / "insert_hole" / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    dataset = UniVTACHDF5Dataset(
        tmp_path,
        "insert_hole",
        action_horizon=4,
        episode_start=0,
        episode_end=0,
        use_tactile=True,
        use_tactile_future=True,
        tactile_future_horizon=4,
        tactile_future_prediction_mode="multi_horizon_pooled",
        tactile_future_stride=2,
        tactile_expert_future_cache_path=cache_root,
        tactile_encoder_path=checkpoint,
        tactile_preprocess_profile=T3_UNIVTAC_RGB_PREPROCESS_PROFILE,
        tactile_expert_future_pool_grid=6,
    )
    transform = _identity_transform(
        use_tactile=True,
        use_tactile_future=True,
    )

    raw = dataset[1]
    assert "observation/left_tactile_rgb" in raw
    assert "observation/right_tactile_rgb" in raw
    assert "observation/left_tactile_future_rgb" not in raw
    assert "observation/right_tactile_future_rgb" not in raw
    assert "observation/tactile_world_latents" not in raw
    np.testing.assert_array_equal(
        raw["observation/tactile_world_future_latents"],
        latents[[3, 5]],
    )
    observation, _, _ = collate_univtac_sft_items(
        [transform(raw), transform(dataset[3])]
    )

    assert observation.tactile_images is not None
    assert observation.tactile_future_images is None
    assert observation.tactile_world_latents is None
    assert observation.tactile_world_future_latents.shape == (
        2,
        2,
        2,
        37,
        1024,
    )
    torch.testing.assert_close(
        observation.tactile_future_target_valid,
        torch.tensor([[True, True], [True, False]]),
    )


def test_dedicated_future_cache_rejects_non_p6_layout(tmp_path: Path):
    _write_episode(tmp_path / "insert_hole" / "clean" / "0.hdf5", length=2)

    with pytest.raises(ValueError, match="fixed to pool_grid=6"):
        UniVTACHDF5Dataset(
            tmp_path,
            "insert_hole",
            action_horizon=1,
            episode_start=0,
            episode_end=0,
            use_tactile=True,
            use_tactile_future=True,
            tactile_future_horizon=5,
            tactile_future_prediction_mode="multi_horizon_pooled",
            tactile_expert_future_cache_path=tmp_path / "future_cache",
            tactile_expert_future_pool_grid=5,
        )


def test_zero_weight_tactile_expert_cache_keeps_current_touch_online(
    tmp_path: Path,
):
    episode_path = tmp_path / "insert_hole" / "clean" / "0.hdf5"
    _write_episode(episode_path, length=3)
    cache_root = tmp_path / "cache"
    cache_path = cache_root / "insert_hole" / "clean" / "0.npy"
    cache_path.parent.mkdir(parents=True)
    latents = np.arange(
        3 * 2 * 2 * 1024,
        dtype=np.float16,
    ).reshape(3, 2, 2, 1024)
    np.save(cache_path, latents)
    checkpoint = tmp_path / "t3"
    (checkpoint / "encoders").mkdir(parents=True)
    (checkpoint / "encoders" / "mini.pth").write_bytes(b"mini")
    (checkpoint / "trunk.pth").write_bytes(b"trunk")
    manifest = build_t3_world_cache_manifest(
        task="insert_hole",
        episode_lengths={0: 3},
        pool_grid=1,
        checkpoint_dir=checkpoint,
        episode_paths={0: episode_path},
        include_source_sha256=True,
        cache_paths={0: cache_path},
    )
    (cache_root / "insert_hole" / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    dataset = UniVTACHDF5Dataset(
        tmp_path,
        "insert_hole",
        action_horizon=2,
        episode_start=0,
        episode_end=0,
        use_tactile=True,
        use_tactile_future=False,
        tactile_world_cache_path=cache_root,
        tactile_encoder_path=checkpoint,
        tactile_world_pool_grid=1,
    )
    transform = _identity_transform(
        use_tactile=True,
        use_tactile_future=False,
    )

    raw = dataset[1]
    assert "observation/left_tactile_rgb" in raw
    assert "observation/left_tactile_future_rgb" not in raw
    item = transform(raw)
    observation, _, _ = collate_univtac_sft_items([item])

    assert observation.tactile_images is not None
    assert observation.tactile_future_images is None
    assert observation.tactile_world_latents is None
    assert observation.tactile_world_future_latents is None


def test_future_tactile_transform_and_collate_preserve_independent_masks(
    tmp_path: Path,
):
    episode_path = tmp_path / "insert_hole" / "clean" / "0.hdf5"
    _write_episode(episode_path, length=4)
    dataset = UniVTACHDF5Dataset(
        tmp_path,
        "insert_hole",
        action_horizon=2,
        episode_start=0,
        episode_end=0,
        use_tactile=True,
        use_tactile_future=True,
    )
    transform = _identity_transform(
        use_tactile=True,
        use_tactile_future=True,
    )

    items = [transform(dataset[1]), transform(dataset[2])]
    observation, actions, action_valid_mask = collate_univtac_sft_items(items)

    assert actions.shape == (2, 2, 32)
    assert action_valid_mask is None
    assert observation.tactile_future_images is not None
    assert observation.tactile_future_masks is not None
    assert observation.tactile_future_target_valid is not None
    assert observation.tactile_future_images["left_gsmini_rgb"].shape == (
        2,
        24,
        32,
        3,
    )
    assert observation.tactile_future_images["left_gsmini_rgb"].dtype == torch.float32
    torch.testing.assert_close(
        observation.tactile_future_masks["left_gsmini_rgb"],
        torch.tensor([True, True]),
    )
    torch.testing.assert_close(
        observation.tactile_future_target_valid,
        torch.tensor([True, False]),
    )


def test_disabled_future_tactile_path_preserves_legacy_sample_schema(tmp_path: Path):
    episode_path = tmp_path / "insert_hole" / "clean" / "0.hdf5"
    _write_episode(episode_path, length=2)
    dataset = UniVTACHDF5Dataset(
        tmp_path,
        "insert_hole",
        action_horizon=1,
        episode_start=0,
        episode_end=0,
        use_tactile=True,
    )

    sample = dataset[0]

    assert "observation/left_tactile_future_rgb" not in sample
    assert "observation/right_tactile_future_rgb" not in sample
    assert "observation/tactile_future_target_valid" not in sample


def test_cache_manifest_rejects_stale_encoder_code(tmp_path: Path):
    checkpoint = tmp_path / "t3"
    (checkpoint / "encoders").mkdir(parents=True)
    (checkpoint / "encoders" / "mini.pth").write_bytes(b"mini")
    (checkpoint / "trunk.pth").write_bytes(b"trunk")
    expected = build_t3_world_cache_manifest(
        task="insert_hole",
        episode_lengths={0: 4},
        pool_grid=14,
        checkpoint_dir=checkpoint,
    )
    stale = json.loads(json.dumps(expected))
    stale["encoder"]["source_sha256"] = "stale"

    with pytest.raises(ValueError, match="mismatched fields: .*encoder"):
        validate_t3_world_cache_manifest(
            stale,
            expected,
            manifest_path=tmp_path / "manifest.json",
        )


def test_transform_round_trips_ftp1_mix_actions():
    state = np.linspace(-0.2, 0.2, 8, dtype=np.float32)
    absolute_actions = np.stack([state + 0.1, state + 0.2, state + 0.3])
    raw = {
        "observation/head_rgb": np.zeros((16, 20, 3), dtype=np.uint8),
        "observation/wrist_rgb": np.ones((12, 18, 3), dtype=np.uint8),
        "observation/state": state,
        "actions": absolute_actions,
        "action_valid_mask": np.array([True, True, False]),
        "prompt": "insert the stick to the hole.",
    }
    transform = _identity_transform(action_loss_masking="mask_repeat_last_padding")

    transformed = transform(raw)

    assert transformed["state"].shape == (32,)
    assert transformed["actions"].shape == (3, 32)
    # FTP-1/N0 action deltas are always relative to the observation at t. This
    # synthetic input starts at t+1 merely to test a non-zero round trip.
    np.testing.assert_allclose(transformed["actions"][0, :7], 0.1, atol=2e-6)
    np.testing.assert_allclose(
        transformed["actions"][:, 7], absolute_actions[:, 7], atol=2e-6
    )
    np.testing.assert_array_equal(
        transformed["actions"][:, 8:], np.zeros((3, 24), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        transformed["action_valid_mask"],
        np.array([True, True, False]),
    )
    observation, batched_actions, action_valid_mask = collate_univtac_sft_items(
        [transformed, transformed]
    )
    assert observation.state.shape == (2, 32)
    assert batched_actions.shape == (2, 3, 32)
    torch.testing.assert_close(
        action_valid_mask,
        torch.tensor([[True, True, False], [True, True, False]]),
    )
    restored = transform.unnormalize_actions(transformed["actions"], state)
    np.testing.assert_allclose(restored, absolute_actions, atol=2e-6)
    assert transformed["image_mask"]["base_0_rgb"]
    assert not transformed["image_mask"]["left_wrist_0_rgb"]
    assert not transformed["image_mask"]["right_wrist_0_rgb"]


def test_transform_default_includes_repeat_last_padding_in_action_loss():
    state = np.zeros(8, dtype=np.float32)
    raw = {
        "observation/head_rgb": np.zeros((16, 20, 3), dtype=np.uint8),
        "observation/wrist_rgb": np.zeros((12, 18, 3), dtype=np.uint8),
        "observation/state": state,
        "actions": np.stack([state, state]),
        "action_valid_mask": np.array([True, False]),
        "prompt": "insert the stick to the hole.",
    }

    transformed = _identity_transform()(raw)

    assert "action_valid_mask" not in transformed
    _, batched_actions, action_valid_mask = collate_univtac_sft_items(
        [transformed, transformed]
    )
    assert batched_actions.shape == (2, 2, 32)
    assert action_valid_mask is None


def test_transform_uses_wrist_only_for_univtac_all_camera_tasks():
    state = np.zeros(8, dtype=np.float32)
    wrist = np.full((12, 18, 3), 173, dtype=np.uint8)
    raw = {
        "observation/head_rgb": np.zeros((16, 20, 3), dtype=np.uint8),
        "observation/wrist_rgb": wrist,
        "observation/state": state,
        "prompt": "insert the tube to the fixed slot.",
    }

    head_only = _identity_transform(camera_type="head")(raw)
    assert not head_only["image_mask"]["left_wrist_0_rgb"]
    np.testing.assert_array_equal(
        head_only["image"]["left_wrist_0_rgb"],
        np.zeros_like(raw["observation/head_rgb"]),
    )

    all_cameras = _identity_transform(camera_type="all")(raw)
    assert all_cameras["image_mask"]["left_wrist_0_rgb"]
    np.testing.assert_array_equal(all_cameras["image"]["left_wrist_0_rgb"], wrist)


def test_t3_legacy_fallback_uses_published_heads_and_imagenet_normalization():
    # A constant mid-gray RGB frame remains constant after resize/crop, making
    # the exact ImageNet normalization easy to verify without constructing T3.
    images = torch.zeros(2, 3, 240, 320)  # [-1, 1] representation of RGB=0.5
    processed = T3LargeTactileEncoder.preprocess(images)
    expected = torch.tensor(
        [(0.5 - mean) / std for mean, std in zip(T3_INPUT_MEAN, T3_INPUT_STD)]
    ).view(1, 3, 1, 1)

    assert T3_NUM_HEADS == 8
    assert processed.shape == (2, 3, 224, 224)
    assert torch.allclose(processed, expected.expand_as(processed), atol=1e-6)


def test_univtac_t3_profile_has_frozen_per_task_statistics_and_contract():
    expected = {
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

    for task, (mean, std) in expected.items():
        profile = resolve_t3_preprocess_profile(
            T3_UNIVTAC_PREPROCESS_PROFILE,
            task=task,
        )
        assert profile.mean == mean
        assert profile.std == std
        fingerprint = profile.as_fingerprint()
        assert fingerprint["input"]["field"] == "rgb_marker"
        assert fingerprint["input"]["channel_order"] == "RGB"
        assert fingerprint["geometry"] == {
            "resize_shorter": 256,
            "crop": "center",
            "crop_size": [224, 224],
            "interpolation": "bilinear",
            "align_corners": False,
            "antialias": True,
        }
        assert fingerprint["normalization"]["statistics"] == {
            "source": "univtac_train",
            "episodes": [0, 49],
            "sensors": "left_right_combined",
            "std_estimator": "population",
            "applied_after_geometry": True,
        }


def test_t3_preprocess_applies_resolved_task_statistics():
    profile = resolve_t3_preprocess_profile(
        T3_UNIVTAC_PREPROCESS_PROFILE,
        task="insert_hole",
    )
    images = torch.zeros(1, 3, 240, 320)

    processed = T3LargeTactileEncoder.preprocess(images, profile=profile)
    expected = torch.tensor(
        [(0.5 - mean) / std for mean, std in zip(profile.mean, profile.std)]
    ).view(1, 3, 1, 1)

    assert processed.shape == (1, 3, 224, 224)
    assert torch.allclose(processed, expected.expand_as(processed), atol=1e-6)


def test_t3_output_reference_rms_comes_only_from_frozen_final_norm():
    encoder = T3LargeTactileEncoder.__new__(T3LargeTactileEncoder)
    nn.Module.__init__(encoder)
    encoder.shared_trunk = SimpleNamespace(
        norm=SimpleNamespace(
            weight=nn.Parameter(torch.tensor([3.0, 0.0])),
            bias=nn.Parameter(torch.tensor([0.0, 4.0])),
        )
    )

    # sqrt(mean([3^2, 4^2])) = sqrt(12.5); no image/batch statistics enter.
    assert encoder.output_reference_rms() == pytest.approx(12.5**0.5)


def test_t3_rgb_profile_matches_vla_jepa_verified_statistics():
    expected = {
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

    for task, (mean, std) in expected.items():
        profile = resolve_t3_preprocess_profile(
            T3_UNIVTAC_RGB_PREPROCESS_PROFILE,
            task=task,
        )
        assert profile.field == "rgb"
        assert profile.source_fields == (
            "tactile/left_gsmini/rgb",
            "tactile/right_gsmini/rgb",
        )
        assert profile.mean == pytest.approx(mean)
        assert profile.std == pytest.approx(std)


def test_t3_rgb_profile_fails_closed_without_frozen_task_statistics():
    with pytest.raises(ValueError, match="No frozen T3 rgb statistics"):
        resolve_t3_preprocess_profile(
            T3_UNIVTAC_RGB_PREPROCESS_PROFILE,
            task="grasp_classify",
        )


def test_gemma_prefix_cache_remains_differentiable_for_world_conditioning():
    config = gemma.get_config("dummy")
    llm = gemma.Module(
        configs=[config, config],
        embed_dtype="float32",
        adarms=[False, False],
        use_gradient_checkpointing=False,
    )
    prefix = torch.randn(2, 3, config.width, requires_grad=True)
    suffix = torch.randn(2, 4, config.width, requires_grad=True)
    prefix_positions = torch.arange(3).expand(2, -1)
    prefix_mask = torch.ones(2, 3, 3, dtype=torch.bool)
    prefix_output, kv_cache = llm(
        [prefix, None],
        positions=prefix_positions,
        mask=prefix_mask,
    )
    # Mimic the world predictor consuming the last VLM layer while the Action
    # Expert consumes the same differentiable prefix KV cache.
    world_term = prefix_output[0].square().mean()
    suffix_positions = torch.arange(3, 7).expand(2, -1)
    suffix_mask = torch.ones(2, 4, 7, dtype=torch.bool)
    suffix_output, _ = llm(
        [None, suffix],
        positions=suffix_positions,
        mask=suffix_mask,
        kv_cache=kv_cache,
    )

    (world_term + suffix_output[1].square().mean()).backward()

    assert prefix.grad is not None and torch.isfinite(prefix.grad).all()
    assert suffix.grad is not None and torch.isfinite(suffix.grad).all()


def test_gemma_all_masked_query_has_exactly_zero_attention_output():
    config = gemma.get_config("dummy")
    attention = gemma.Attention([config]).eval()
    inputs = torch.randn(1, 2, config.width)
    mask = torch.tensor([[[[False, False], [False, True]]]])

    outputs, _ = attention(
        [inputs],
        positions=torch.arange(2).expand(1, -1),
        attn_mask=mask,
    )

    # A fully masked query is padding, so it must not silently average every
    # value merely because finite masked logits have a uniform softmax.
    assert torch.equal(outputs[0][:, 0], torch.zeros_like(outputs[0][:, 0]))
    assert torch.isfinite(outputs[0]).all()
    assert torch.count_nonzero(outputs[0][:, 1]) > 0


def test_eval_prefers_config_matching_checkpoint_kind_and_rejects_conflicts(
    tmp_path: Path,
):
    checkpoint = tmp_path / "converted"
    checkpoint.mkdir()
    safetensors_path = checkpoint / "model.safetensors"
    safetensors_path.touch()
    nested = {
        "data": {"task": "insert_hole", "normalization": "quantile"},
        "actor": {
            "model": {
                "num_action_chunks": 50,
                "action_dim": 8,
                "openpi": {
                    "config_name": "pi05_univtac",
                    "model_action_dim": 32,
                    "discrete_state_input": True,
                    "pi05": True,
                    "tactile_expert_wam_mode": "fast",
                },
            }
        },
    }
    flat = {
        "task": "insert_hole",
        "action_horizon": 50,
        "action_dim": 32,
        "env_action_dim": 8,
        "pi05": True,
        "discrete_state_input": True,
        "normalization": "quantile",
        "tactile_expert_wam_mode": "fast",
    }
    OmegaConf.save(OmegaConf.create(nested), checkpoint / "config.yaml")
    (checkpoint / "config.json").write_text(json.dumps(flat), encoding="utf-8")

    _, selected_path = _load_checkpoint_config(safetensors_path)
    assert selected_path == checkpoint / "config.json"

    full_weights = checkpoint / "full_weights.pt"
    full_weights.touch()
    _, selected_path = _load_checkpoint_config(full_weights)
    assert selected_path == checkpoint / "config.yaml"

    flat["tactile_expert_wam_mode"] = "joint"
    (checkpoint / "config.json").write_text(json.dumps(flat), encoding="utf-8")
    with pytest.raises(ValueError, match="tactile_expert_wam_mode"):
        _load_checkpoint_config(safetensors_path)

    flat["tactile_expert_wam_mode"] = "fast"
    flat["discrete_state_input"] = False
    (checkpoint / "config.json").write_text(json.dumps(flat), encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting config.yaml/config.json"):
        _load_checkpoint_config(safetensors_path)

    flat["discrete_state_input"] = True
    flat["action_horizon"] = 25
    (checkpoint / "config.json").write_text(json.dumps(flat), encoding="utf-8")
    with pytest.raises(ValueError, match="action_horizon"):
        _load_checkpoint_config(safetensors_path)


def test_eval_prefers_bundled_norm_stats_and_rejects_different_override(
    tmp_path: Path,
):
    checkpoint = tmp_path / "global_step_5000"
    weights_path = checkpoint / "actor" / "model_state_dict" / "full_weights.pt"
    weights_path.parent.mkdir(parents=True)
    weights_path.touch()
    config_path = checkpoint / "config.yaml"
    configured_stats = tmp_path / "configured_stats.json"
    configured_stats.write_text("bundled", encoding="utf-8")
    config = OmegaConf.create({"data": {"norm_stats_path": str(configured_stats)}})
    OmegaConf.save(config, config_path)
    bundled_stats = checkpoint / "norm_stats.json"
    bundled_stats.write_text("bundled", encoding="utf-8")

    selected = _resolve_norm_stats_path(
        weights_path=weights_path,
        checkpoint_config=config,
        config_path=config_path,
        override=None,
    )

    assert selected == bundled_stats
    matching_override = tmp_path / "matching.json"
    matching_override.write_text("bundled", encoding="utf-8")
    assert (
        _resolve_norm_stats_path(
            weights_path=weights_path,
            checkpoint_config=config,
            config_path=config_path,
            override=str(matching_override),
        )
        == bundled_stats
    )
    with pytest.raises(ValueError, match="override differs"):
        different_stats = tmp_path / "different.json"
        different_stats.write_text("different", encoding="utf-8")
        _resolve_norm_stats_path(
            weights_path=weights_path,
            checkpoint_config=config,
            config_path=config_path,
            override=str(different_stats),
        )


def test_eval_resolves_relative_config_norm_stats_and_rejects_bundle_mismatch(
    tmp_path: Path,
):
    checkpoint = tmp_path / "global_step_5000"
    weights_path = checkpoint / "actor" / "model_state_dict" / "full_weights.pt"
    weights_path.parent.mkdir(parents=True)
    weights_path.touch()
    relative_stats = checkpoint / "assets" / "norm_stats.json"
    relative_stats.parent.mkdir()
    relative_stats.write_text("same", encoding="utf-8")
    config_path = checkpoint / "config.yaml"
    config = OmegaConf.create({"data": {"norm_stats_path": "assets/norm_stats.json"}})
    OmegaConf.save(config, config_path)

    selected = _resolve_norm_stats_path(
        weights_path=weights_path,
        checkpoint_config=config,
        config_path=config_path,
        override=None,
    )
    assert selected == relative_stats

    bundled = checkpoint / "norm_stats.json"
    bundled.write_text("different", encoding="utf-8")
    with pytest.raises(ValueError, match="differ from checkpoint config"):
        _resolve_norm_stats_path(
            weights_path=weights_path,
            checkpoint_config=config,
            config_path=config_path,
            override=None,
        )


def test_eval_strictly_validates_simpletac_contract():
    valid = OmegaConf.create(
        {
            "task": "insert_hole",
            "action_horizon": 50,
            "action_dim": 32,
            "env_action_dim": 8,
            "max_token_len": 200,
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "pi05": True,
            "discrete_state_input": True,
            "normalization": "quantile",
        }
    )
    assert _validate_simpletac_contract(
        valid,
    ) == {
        "action_horizon": 50,
        "model_action_dim": 32,
        "env_action_dim": 8,
        "max_token_len": 200,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "pi05": True,
        "discrete_state_input": True,
        "normalization": "quantile",
    }

    horizon_eight = OmegaConf.create(OmegaConf.to_container(valid, resolve=True))
    horizon_eight.action_horizon = 8
    assert (
        _validate_simpletac_contract(
            horizon_eight,
        )["action_horizon"]
        == 8
    )

    invalid = OmegaConf.create(OmegaConf.to_container(valid, resolve=True))
    invalid.action_horizon = 0
    with pytest.raises(ValueError, match="must be positive"):
        _validate_simpletac_contract(invalid)

    inconsistent = OmegaConf.create(
        {
            **OmegaConf.to_container(valid, resolve=True),
            "actor": {
                "model": {
                    "num_action_chunks": 8,
                    "openpi": {"action_chunk": 50},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="inconsistent action horizon"):
        _validate_simpletac_contract(inconsistent)

    del invalid["action_horizon"]
    with pytest.raises(ValueError, match="missing required"):
        _validate_simpletac_contract(invalid)


def test_sft_checkpoint_saves_resolved_config(tmp_path: Path):
    class _SaveHandle:
        def wait(self) -> None:
            return None

    class _Actor:
        def save_checkpoint(self, path: str, step: int) -> _SaveHandle:
            assert Path(path).name == "actor"
            assert step == 5000
            return _SaveHandle()

    runner = SFTRunner.__new__(SFTRunner)
    runner.cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "all_tokens_run",
                }
            },
            "actor": {
                "model": {
                    "openpi": {
                        "tactile_use_all_tokens": True,
                    }
                }
            },
        }
    )
    runner.actor = _Actor()
    runner.global_step = 5000
    runner.early_stop = None

    runner._save_checkpoint()

    saved_config = OmegaConf.load(
        tmp_path / "all_tokens_run" / "checkpoints" / "global_step_5000" / "config.yaml"
    )
    assert saved_config.actor.model.openpi.tactile_use_all_tokens is True


def test_univtac_eval_uses_kit_physical_gpu_index_for_renderer():
    script = (
        Path(__file__).parents[2] / "scripts" / "univtac" / "eval_simpletac.sh"
    ).read_text(encoding="utf-8")

    assert "RENDERER_ACTIVE_GPU_ID=$GPU_ID" in script
    assert "--/renderer/activeGpu=$RENDERER_ACTIVE_GPU_ID" in script
    assert "VULKAN_ACTIVE_GPU_ID=$(" not in script
