"""Focused tests for the layer-aligned future-tactile expert."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from rlinf.models.embodiment.openpi_pytorch.pi0_model import gemma
from rlinf.models.embodiment.openpi_pytorch.pi0_model import model as pi0_model
from rlinf.models.embodiment.openpi_pytorch.pi0_model import pi0 as pi0_module
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_preprocess_profile import (
    T3_UNIVTAC_PREPROCESS_PROFILE,
    T3_UNIVTAC_RGB_PREPROCESS_PROFILE,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_tactile_encoder import (
    T3_LARGE_WIDTH,
    T3_TOKENS_PER_SENSOR,
    pool_t3_spatial_tokens,
)


def _tiny_gemma_config(width: int = 8) -> gemma.Config:
    return gemma.Config(
        width=width,
        depth=1,
        mlp_dim=2 * width,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )


def test_deferred_initialization_skips_fills_and_restores_torch_helpers():
    original_normal = nn.init.normal_
    tensor = torch.full((4,), 7.0)

    with pi0_module._defer_torch_parameter_initialization(True):
        nn.init.normal_(tensor)
        assert nn.init.normal_ is not original_normal

    assert torch.equal(tensor, torch.full((4,), 7.0))
    assert nn.init.normal_ is original_normal
    nn.init.zeros_(tensor)
    assert torch.equal(tensor, torch.zeros(4))


def test_direct_pi0_config_keeps_normal_initialization_by_default():
    assert Pi0Config().defer_pretrained_weight_init is False


def test_future_attention_backend_defaults_to_eager_and_rejects_unknown():
    assert Pi0Config().tactile_expert_future_attention_backend == "eager"
    with pytest.raises(ValueError, match="future_attention_backend"):
        Pi0Config(tactile_expert_future_attention_backend="flash")


@pytest.mark.parametrize("with_cache", [False, True])
def test_gemma_sdpa_matches_eager_gqa_and_keeps_compact_cache(
    monkeypatch,
    with_cache: bool,
):
    torch.manual_seed(11)
    config = _tiny_gemma_config()
    attention = gemma.Attention([config, config, config]).eval()
    batch_size, token_count, cache_length = 2, 3, 4 if with_cache else 0
    tokens = torch.randn(batch_size, token_count, config.width)
    positions = torch.arange(cache_length, cache_length + token_count).expand(
        batch_size,
        -1,
    )
    kv_cache = None
    if with_cache:
        kv_cache = (
            torch.randn(
                batch_size,
                cache_length,
                config.num_kv_heads,
                config.head_dim,
            ),
            torch.randn(
                batch_size,
                cache_length,
                config.num_kv_heads,
                config.head_dim,
            ),
        )
    source_length = cache_length + token_count
    mask = torch.ones(
        batch_size,
        1,
        token_count,
        source_length,
        dtype=torch.bool,
    )
    mask[0, :, 0, -1] = False

    with torch.no_grad():
        eager_output, eager_cache = attention(
            [None, None, tokens],
            positions,
            mask,
            kv_cache,
            attention_backend="eager",
        )

    original_sdpa = gemma.F.scaled_dot_product_attention
    sdpa_shapes: list[tuple[torch.Size, torch.Size, torch.Size]] = []

    def _recording_sdpa(query, key, value, **kwargs):
        sdpa_shapes.append((query.shape, key.shape, value.shape))
        return original_sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(
        gemma.F,
        "scaled_dot_product_attention",
        _recording_sdpa,
    )
    with torch.no_grad():
        sdpa_output, sdpa_cache = attention(
            [None, None, tokens],
            positions,
            mask,
            kv_cache,
            attention_backend="sdpa",
        )

    assert sdpa_shapes == [
        (
            torch.Size((batch_size, config.num_heads, token_count, config.head_dim)),
            torch.Size((batch_size, config.num_heads, source_length, config.head_dim)),
            torch.Size((batch_size, config.num_heads, source_length, config.head_dim)),
        )
    ]
    torch.testing.assert_close(sdpa_output[2], eager_output[2], rtol=1e-5, atol=1e-5)
    assert sdpa_cache[0].shape == (
        batch_size,
        source_length,
        config.num_kv_heads,
        config.head_dim,
    )
    assert sdpa_cache[1].shape == sdpa_cache[0].shape
    torch.testing.assert_close(sdpa_cache[0], eager_cache[0])
    torch.testing.assert_close(sdpa_cache[1], eager_cache[1])


def test_gemma_sdpa_all_masked_rows_are_zero_with_zero_gradients():
    torch.manual_seed(13)
    config = _tiny_gemma_config()
    attention = gemma.Attention([config, config, config]).eval()
    base_tokens = torch.randn(2, 3, config.width)
    positions = torch.arange(3).expand(2, -1)
    mask = torch.ones(2, 1, 3, 3, dtype=torch.bool)
    # One isolated invalid token and one entirely invalid sample exercise both
    # per-sensor padding and terminal future-target padding.
    mask[0, :, 2, :] = False
    mask[0, :, :, 2] = False
    mask[1] = False

    results = {}
    for backend in ("eager", "sdpa"):
        attention.zero_grad(set_to_none=True)
        tokens = base_tokens.clone().requires_grad_(True)
        outputs, _ = attention(
            [None, None, tokens],
            positions,
            mask,
            attention_backend=backend,
        )
        assert torch.isfinite(outputs[2]).all()
        outputs[2].square().sum().backward()
        assert tokens.grad is not None
        assert torch.isfinite(tokens.grad).all()
        results[backend] = (outputs[2].detach(), tokens.grad.detach())

    torch.testing.assert_close(
        results["sdpa"][0],
        results["eager"][0],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        results["sdpa"][1],
        results["eager"][1],
        rtol=2e-5,
        atol=2e-5,
    )
    assert torch.count_nonzero(results["sdpa"][0][0, 2]) == 0
    assert torch.count_nonzero(results["sdpa"][0][1]) == 0
    assert torch.count_nonzero(results["sdpa"][1][0, 2]) == 0
    assert torch.count_nonzero(results["sdpa"][1][1]) == 0


def test_gradient_checkpointing_carries_sdpa_backend(monkeypatch):
    torch.manual_seed(17)
    config = _tiny_gemma_config()
    llm = gemma.Module(
        configs=[config, config, config],
        embed_dtype="float32",
        adarms=[False, False, False],
        use_gradient_checkpointing=True,
    ).train()
    calls = 0
    original_sdpa = gemma.F.scaled_dot_product_attention

    def _recording_sdpa(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_sdpa(*args, **kwargs)

    monkeypatch.setattr(
        gemma.F,
        "scaled_dot_product_attention",
        _recording_sdpa,
    )
    tokens = torch.randn(1, 2, config.width, requires_grad=True)
    outputs, _ = llm(
        [None, None, tokens],
        positions=torch.arange(2).unsqueeze(0),
        mask=torch.ones(1, 2, 2, dtype=torch.bool),
        attention_backend="sdpa",
    )
    assert calls == config.depth
    outputs[2].square().mean().backward()
    assert calls >= config.depth
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_tactile_expert_rejects_unknown_future_vlm_gradient_mode():
    with pytest.raises(TypeError, match="future_vl_gradient_mode"):
        Pi0Config(tactile_expert_future_vl_gradient_mode="unknown")


def test_tactile_expert_requires_marker_free_rgb_preprocessing():
    common = {
        "pi05": True,
        "tactile_expert_enabled": True,
        "tactile_encoder_path": "/unused/t3",
        "tactile_preprocess_task": "insert_hole",
    }
    config = Pi0Config(
        **common,
        tactile_preprocess_profile=T3_UNIVTAC_RGB_PREPROCESS_PROFILE,
    )
    assert config.tactile_preprocess_profile == T3_UNIVTAC_RGB_PREPROCESS_PROFILE

    with pytest.raises(ValueError, match="marker-free RGB"):
        Pi0Config(
            **common,
            tactile_preprocess_profile=T3_UNIVTAC_PREPROCESS_PROFILE,
        )


def test_gemma_trailing_expert_padding_is_exactly_backward_compatible():
    config = _tiny_gemma_config()
    llm = gemma.Module(
        configs=[config, config, config],
        embed_dtype="float32",
        adarms=[False, False, False],
        use_gradient_checkpointing=False,
    ).eval()
    prefix = torch.randn(2, 3, config.width)
    action = torch.randn(2, 2, config.width)
    positions = torch.arange(5).expand(2, -1)
    mask = torch.ones(2, 5, 5, dtype=torch.bool)

    with torch.no_grad():
        padded_outputs, padded_cache = llm(
            [prefix, action],
            positions=positions,
            mask=mask,
        )
        explicit_outputs, explicit_cache = llm(
            [prefix, action, None],
            positions=positions,
            mask=mask,
            adarms_cond=[None, None, None],
        )

    assert len(padded_outputs) == 3
    assert padded_outputs[2] is None
    assert explicit_outputs[2] is None
    torch.testing.assert_close(padded_outputs[0], explicit_outputs[0])
    torch.testing.assert_close(padded_outputs[1], explicit_outputs[1])
    assert len(padded_cache) == config.depth
    for padded_layer, explicit_layer in zip(
        padded_cache,
        explicit_cache,
        strict=True,
    ):
        torch.testing.assert_close(padded_layer[0], explicit_layer[0])
        torch.testing.assert_close(padded_layer[1], explicit_layer[1])


def test_gemma_rejects_more_streams_or_conditions_than_experts():
    config = _tiny_gemma_config()
    llm = gemma.Module(
        configs=[config, config, config],
        embed_dtype="float32",
        adarms=[False, False, False],
        use_gradient_checkpointing=False,
    )
    token = torch.randn(1, 1, config.width)
    positions = torch.zeros(1, 1, dtype=torch.long)
    mask = torch.ones(1, 1, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="more expert streams"):
        llm(
            [token, None, None, None],
            positions=positions,
            mask=mask,
        )
    with pytest.raises(ValueError, match="more adaptive-conditioning streams"):
        llm(
            [token],
            positions=positions,
            mask=mask,
            adarms_cond=[None, None, None, None],
        )


def test_default_tactile_expert_matches_action_expert_parameter_scale():
    # Meta tensors avoid allocating the 1.61 GiB FP32 third-expert payload.
    with torch.device("meta"):
        policy = Pi0.__new__(Pi0)
        nn.Module.__init__(policy)
        policy.tactile_expert_enabled = True
        narrow_context = gemma.Config(
            width=1,
            depth=18,
            mlp_dim=1,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
        tactile_config = gemma.get_config("gemma_300m")
        policy.llm = gemma.Module(
            configs=[narrow_context, narrow_context, tactile_config],
            embed_dtype="float32",
            adarms=[False, True, True],
            use_gradient_checkpointing=False,
        )
        policy.tactile_expert_in_proj = nn.Linear(T3_LARGE_WIDTH, 1024)
        policy.tactile_expert_out_proj = nn.Linear(1024, T3_LARGE_WIDTH)
        policy.tactile_expert_sensor_embedding = nn.Embedding(2, 1024)
        policy.tactile_expert_time_mlp_in = nn.Linear(1024, 1024)
        policy.tactile_expert_time_mlp_out = nn.Linear(1024, 1024)

    assert policy._tactile_expert_parameter_count() == 432_133_120


class _RecordingThreeExpertLlm(nn.Module):
    """Small shape-faithful LLM stand-in for staged-cache tests."""

    def __init__(self, *, depth: int = 2, head_dim: int = 4) -> None:
        super().__init__()
        self.depth = depth
        self.head_dim = head_dim
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        inputs: list[torch.Tensor | None],
        *,
        positions: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: tuple | None = None,
        **kwargs,
    ) -> tuple[list[torch.Tensor | None], tuple]:
        active = next(token for token in inputs if token is not None)
        batch_size, token_count = active.shape[:2]
        self.calls.append(
            {
                "inputs": inputs,
                "positions": positions.clone(),
                "mask": mask.clone(),
                "kv_cache": kv_cache,
                **kwargs,
            }
        )
        layer_cache = tuple(
            (
                torch.full(
                    (batch_size, token_count, 1, self.head_dim),
                    float(layer + 1),
                    device=active.device,
                ),
                torch.full(
                    (batch_size, token_count, 1, self.head_dim),
                    float(-(layer + 1)),
                    device=active.device,
                ),
            )
            for layer in range(self.depth)
        )
        return list(inputs), layer_cache


def _bare_tactile_expert_policy(width: int = 8) -> Pi0:
    policy = Pi0.__new__(Pi0)
    nn.Module.__init__(policy)
    policy.tactile_expert_enabled = True
    policy.embed_dtype = torch.float32
    policy.pi05 = True
    policy.action_horizon = 2
    policy.tactile_expert_in_proj = nn.Linear(T3_LARGE_WIDTH, width)
    policy.tactile_expert_out_proj = nn.Linear(width, T3_LARGE_WIDTH)
    policy.tactile_expert_sensor_embedding = nn.Embedding(2, width)
    policy.tactile_expert_time_mlp_in = nn.Linear(width, width)
    policy.tactile_expert_time_mlp_out = nn.Linear(width, width)
    policy.llm = _RecordingThreeExpertLlm()
    return policy


def test_t3_aligned_input_is_identity_without_duplicate_patch_positions():
    policy = _bare_tactile_expert_policy(width=T3_LARGE_WIDTH)
    with torch.no_grad():
        nn.init.eye_(policy.tactile_expert_in_proj.weight)
        nn.init.zeros_(policy.tactile_expert_in_proj.bias)
        nn.init.zeros_(policy.tactile_expert_sensor_embedding.weight)
    encoded = torch.randn(1, 2, T3_TOKENS_PER_SENSOR, T3_LARGE_WIDTH)

    embedded = policy._embed_tactile_expert_tokens(encoded)

    torch.testing.assert_close(embedded, encoded.flatten(1, 2))


def test_multi_horizon_future_pooling_keeps_each_cls_and_pools_each_frame():
    policy = _bare_tactile_expert_policy()
    policy.tactile_expert_future_prediction_mode = "multi_horizon_pooled"
    policy.tactile_expert_future_pool_grid = 6
    future = torch.randn(2, 3, 2, T3_TOKENS_PER_SENSOR, T3_LARGE_WIDTH)

    selected = policy._select_tactile_expert_future_tokens(future)
    expected = pool_t3_spatial_tokens(
        future.flatten(0, 1),
        pool_grid=6,
    ).reshape(2, 3, 2, 37, T3_LARGE_WIDTH)

    assert selected.shape == (2, 3, 2, 37, T3_LARGE_WIDTH)
    torch.testing.assert_close(selected, expected)
    torch.testing.assert_close(selected[:, :, :, 0], future[:, :, :, 0])


def test_multi_horizon_future_encoder_uses_one_full_t3_batch():
    policy = _bare_tactile_expert_policy()
    policy.tactile_expert_future_prediction_mode = "multi_horizon_pooled"
    policy.tactile_expert_future_pool_grid = 1

    class _RecordingT3(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, images, *, return_all_tokens):
            assert return_all_tokens
            self.batch_sizes.append(images.shape[0])
            endpoint_ids = torch.arange(images.shape[0], dtype=torch.float32)
            return endpoint_ids[:, None, None, None].expand(
                -1,
                2,
                T3_TOKENS_PER_SENSOR,
                T3_LARGE_WIDTH,
            )

    encoder = _RecordingT3()
    policy.tactile_encoder = encoder
    future_images = torch.zeros(1, 5, 2, 3, 8, 8)

    encoded = policy._encode_multi_horizon_future_tactile(future_images)

    assert encoder.batch_sizes == [5]
    assert encoded.shape == (1, 5, 2, 2, T3_LARGE_WIDTH)


def test_multi_horizon_endpoints_combine_current_and_future_in_one_t3_batch():
    policy = _bare_tactile_expert_policy()
    policy.tactile_encoder_frozen = True
    policy.tactile_expert_future_prediction_mode = "multi_horizon_pooled"
    policy.tactile_expert_future_pool_grid = 1

    class _RecordingT3(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, images, *, return_all_tokens):
            assert return_all_tokens
            self.batch_sizes.append(images.shape[0])
            endpoint_ids = torch.arange(images.shape[0], dtype=torch.float32)
            return endpoint_ids[:, None, None, None].expand(
                -1,
                2,
                T3_TOKENS_PER_SENSOR,
                T3_LARGE_WIDTH,
            )

    encoder = _RecordingT3()
    policy.tactile_encoder = encoder
    observation = SimpleNamespace(
        tactile_images={
            "left_gsmini_rgb": torch.zeros(2, 3, 8, 8),
            "right_gsmini_rgb": torch.zeros(2, 3, 8, 8),
        },
        tactile_future_images={
            "left_gsmini_rgb": torch.zeros(2, 5, 3, 8, 8),
            "right_gsmini_rgb": torch.zeros(2, 5, 3, 8, 8),
        },
    )

    current, future = policy.encode_tactile_expert_endpoints(observation)

    assert encoder.batch_sizes == [12]
    assert current.shape == (2, 2, T3_TOKENS_PER_SENSOR, T3_LARGE_WIDTH)
    assert future.shape == (2, 5, 2, 2, T3_LARGE_WIDTH)
    torch.testing.assert_close(current[:, 0, 0, 0], torch.tensor([0.0, 6.0]))
    torch.testing.assert_close(
        future[:, :, 0, 0, 0],
        torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [7.0, 8.0, 9.0, 10.0, 11.0]]),
    )


def test_multi_horizon_endpoints_use_online_current_and_cached_future():
    policy = _bare_tactile_expert_policy()
    policy.tactile_encoder_frozen = True
    policy.tactile_expert_future_prediction_mode = "multi_horizon_pooled"
    policy.tactile_expert_future_pool_grid = 1
    policy.embed_dtype = torch.float32

    class _RecordingT3(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, images, *, return_all_tokens):
            assert return_all_tokens is True
            self.batch_sizes.append(images.shape[0])
            return torch.ones(
                images.shape[0],
                2,
                T3_TOKENS_PER_SENSOR,
                T3_LARGE_WIDTH,
                dtype=images.dtype,
                device=images.device,
            )

    encoder = _RecordingT3()
    policy.tactile_encoder = encoder
    future_cache = torch.randn(
        2,
        5,
        2,
        T3_TOKENS_PER_SENSOR,
        T3_LARGE_WIDTH,
    )
    observation = SimpleNamespace(
        tactile_world_latents=None,
        tactile_world_future_latents=future_cache,
        tactile_images={
            "left_gsmini_rgb": torch.zeros(2, 3, 8, 8),
            "right_gsmini_rgb": torch.zeros(2, 3, 8, 8),
        },
        tactile_future_images=None,
    )

    current, future = policy.encode_tactile_expert_endpoints(observation)

    assert encoder.batch_sizes == [2]
    torch.testing.assert_close(current, torch.ones_like(current))
    expected = pool_t3_spatial_tokens(
        future_cache.flatten(0, 1),
        pool_grid=1,
    ).reshape(2, 5, 2, 2, T3_LARGE_WIDTH)
    torch.testing.assert_close(future, expected)


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize(
    "prediction_mode", ["single_horizon_full", "multi_horizon_pooled"]
)
def test_endpoint_current_gradient_and_immutable_future_target(frozen, prediction_mode):
    policy = _bare_tactile_expert_policy()
    policy.tactile_encoder_frozen = frozen
    policy.tactile_expert_future_prediction_mode = prediction_mode
    policy.tactile_expert_future_pool_grid = 6
    policy.embed_dtype = torch.float32

    class TrainableT3(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, images, *, return_all_tokens):
            return self.weight * torch.ones(images.shape[0], 2, 197, 1024)

    policy.tactile_encoder = TrainableT3()
    target_shape = (
        (1, 10, 2, 197, 1024)
        if prediction_mode == "multi_horizon_pooled"
        else (1, 2, 197, 1024)
    )
    target = torch.ones(target_shape, requires_grad=True)
    obs = SimpleNamespace(
        tactile_world_latents=None,
        tactile_world_future_latents=target,
        tactile_images={
            "left_gsmini_rgb": torch.zeros(1, 3, 8, 8),
            "right_gsmini_rgb": torch.zeros(1, 3, 8, 8),
        },
        tactile_future_images=None,
    )
    current, future = policy.encode_tactile_expert_endpoints(obs)
    assert current.requires_grad is not frozen
    assert not future.requires_grad
    if not frozen:
        current.mean().backward()
        torch.testing.assert_close(policy.tactile_encoder.weight.grad, torch.ones(()))
        assert target.grad is None
        obs.tactile_world_future_latents = None
        with pytest.raises(ValueError, match="immutable cached future"):
            policy.encode_tactile_expert_endpoints(obs)
        obs.tactile_world_latents = target
        with pytest.raises(ValueError, match="online current"):
            policy.encode_tactile_expert_endpoints(obs)


def test_multi_horizon_endpoints_accept_prepooled_future_without_repooling():
    policy = _bare_tactile_expert_policy()
    policy.tactile_encoder_frozen = True
    policy.tactile_expert_future_prediction_mode = "multi_horizon_pooled"
    policy.tactile_expert_future_pool_grid = 6
    policy.embed_dtype = torch.float32

    class _RecordingT3(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, images, *, return_all_tokens):
            assert return_all_tokens is True
            self.batch_sizes.append(images.shape[0])
            return torch.ones(
                images.shape[0],
                2,
                T3_TOKENS_PER_SENSOR,
                T3_LARGE_WIDTH,
                dtype=images.dtype,
                device=images.device,
            )

    encoder = _RecordingT3()
    policy.tactile_encoder = encoder
    future_cache = torch.randn(2, 5, 2, 37, T3_LARGE_WIDTH)
    observation = SimpleNamespace(
        tactile_world_latents=None,
        tactile_world_future_latents=future_cache,
        tactile_images={
            "left_gsmini_rgb": torch.zeros(2, 3, 8, 8),
            "right_gsmini_rgb": torch.zeros(2, 3, 8, 8),
        },
        tactile_future_images=None,
    )

    current, future = policy.encode_tactile_expert_endpoints(observation)

    assert encoder.batch_sizes == [2]
    torch.testing.assert_close(current, torch.ones_like(current))
    assert future.data_ptr() == future_cache.data_ptr()
    torch.testing.assert_close(future, future_cache)


def test_multi_horizon_future_embedding_has_740_tokens_without_new_positions():
    policy = _bare_tactile_expert_policy()
    policy.tactile_expert_future_prediction_mode = "multi_horizon_pooled"
    policy.tactile_expert_future_pool_grid = 6
    future = torch.randn(2, 10, 2, 37, T3_LARGE_WIDTH)

    embedded = policy._embed_tactile_expert_future_tokens(future)

    assert embedded.shape == (2, 740, 8)


def test_multi_horizon_loss_averages_horizons_instead_of_summing_tokens():
    policy = _bare_tactile_expert_policy()
    policy.tactile_expert_future_prediction_mode = "multi_horizon_pooled"
    policy.tactile_expert_future_horizon = 10
    policy.tactile_expert_future_stride = 5
    policy.tactile_expert_future_pool_grid = 1
    observation = SimpleNamespace(
        tactile_future_masks={
            "left_gsmini_rgb": torch.ones(2, 2, dtype=torch.bool),
            "right_gsmini_rgb": torch.ones(2, 2, dtype=torch.bool),
        },
        tactile_future_target_valid=torch.tensor(
            [[True, True], [True, False]],
            dtype=torch.bool,
        ),
    )
    token_mask = policy._tactile_expert_token_mask(
        observation,
        future=True,
        device=torch.device("cpu"),
        batch_size=2,
    )
    target = torch.zeros(2, 2, 2, 2, 1)
    prediction = torch.zeros_like(target)
    prediction[0, 0] = 1.0
    prediction[0, 1] = 3.0
    prediction[1, 0] = 2.0
    prediction[1, 1] = 100.0  # Masked tail must not affect the result.

    loss = policy._multi_horizon_tactile_flow_loss(
        prediction,
        target,
        token_mask,
    )

    assert token_mask.shape == (2, 8)
    assert loss.item() == pytest.approx(14.0 / 3.0)


def _tiny_initializable_tactile_expert() -> Pi0:
    policy = Pi0.__new__(Pi0)
    nn.Module.__init__(policy)
    config = _tiny_gemma_config()
    policy.tactile_expert_enabled = True
    policy.tactile_expert_init_mode = "random"
    policy.llm = gemma.Module(
        configs=[config, config, config],
        embed_dtype="float32",
        adarms=[False, True, True],
        use_gradient_checkpointing=False,
    )
    policy.tactile_expert_in_proj = nn.Linear(T3_LARGE_WIDTH, config.width)
    policy.tactile_expert_out_proj = nn.Linear(config.width, T3_LARGE_WIDTH)
    policy.tactile_expert_sensor_embedding = nn.Embedding(2, config.width)
    policy.tactile_expert_time_mlp_in = nn.Linear(config.width, config.width)
    policy.tactile_expert_time_mlp_out = nn.Linear(config.width, config.width)
    policy.time_mlp_in = nn.Linear(config.width, config.width)
    policy.time_mlp_out = nn.Linear(config.width, config.width)
    return policy


def test_random_tactile_expert_init_is_independent_and_rng_preserving():
    policy = _tiny_initializable_tactile_expert()
    with torch.no_grad():
        policy.llm.layers[0].attn.q_proj[1].weight.fill_(0.25)
    torch.manual_seed(123)
    expected_next = torch.rand(4)
    torch.manual_seed(123)

    policy.initialize_tactile_expert_random()
    actual_next = torch.rand(4)

    torch.testing.assert_close(actual_next, expected_next)
    assert torch.isfinite(policy.llm.layers[0].attn.q_proj[2].weight).all()
    assert not torch.equal(
        policy.llm.layers[0].attn.q_proj[2].weight,
        policy.llm.layers[0].attn.q_proj[1].weight,
    )
    assert (
        torch.count_nonzero(
            policy.llm.layers[0].pre_attention_norms[2].ada_modulation.weight
        )
        == 0
    )
    assert (
        torch.count_nonzero(policy.llm.layers[0].pre_ffw_norms[2].ada_modulation.bias)
        == 0
    )


def test_from_action_expert_init_copies_same_shape_transformer_exactly():
    policy = _tiny_initializable_tactile_expert()
    with torch.no_grad():
        for parameter in policy.llm.layers[0].parameters():
            parameter.copy_(torch.randn_like(parameter))
        policy.time_mlp_in.weight.copy_(torch.randn_like(policy.time_mlp_in.weight))
        policy.time_mlp_in.bias.copy_(torch.randn_like(policy.time_mlp_in.bias))
        policy.time_mlp_out.weight.copy_(torch.randn_like(policy.time_mlp_out.weight))
        policy.time_mlp_out.bias.copy_(torch.randn_like(policy.time_mlp_out.bias))

    policy.initialize_tactile_expert_from_action_expert()

    block = policy.llm.layers[0]
    for projections in (
        block.attn.q_proj,
        block.attn.k_proj,
        block.attn.v_proj,
        block.attn.o_proj,
    ):
        torch.testing.assert_close(projections[2].weight, projections[1].weight)
    torch.testing.assert_close(block.mlps[2].w_gating, block.mlps[1].w_gating)
    torch.testing.assert_close(block.mlps[2].w_linear, block.mlps[1].w_linear)
    torch.testing.assert_close(
        policy.tactile_expert_time_mlp_in.weight,
        policy.time_mlp_in.weight,
    )


def test_tactile_expert_stages_current_action_and_future_as_isolated_streams():
    policy = _bare_tactile_expert_policy()
    policy.tactile_expert_future_attention_backend = "sdpa"
    llm = policy.llm
    batch_size = 2
    prefix_len = 3
    tactile_len = 2 * T3_TOKENS_PER_SENSOR
    observation = SimpleNamespace(
        tactile_masks={
            "left_gsmini_rgb": torch.tensor([True, True]),
            "right_gsmini_rgb": torch.tensor([True, False]),
        },
        tactile_future_masks={
            "left_gsmini_rgb": torch.tensor([True, True]),
            "right_gsmini_rgb": torch.tensor([True, True]),
        },
        tactile_future_target_valid=torch.tensor([True, False]),
    )
    encoded_current = torch.randn(
        batch_size,
        2,
        T3_TOKENS_PER_SENSOR,
        T3_LARGE_WIDTH,
    )
    prefix_mask = torch.tensor([[True, True, True], [True, False, True]])
    prefix_cache = tuple(
        (
            torch.full((batch_size, prefix_len, 1, 4), 20.0 + layer),
            torch.full((batch_size, prefix_len, 1, 4), -20.0 - layer),
        )
        for layer in range(llm.depth)
    )

    context_mask, staged_cache = policy.build_tactile_expert_cache(
        observation,
        encoded_current,
        prefix_mask,
        prefix_cache,
    )

    current_call = llm.calls[-1]
    assert "attention_backend" not in current_call
    assert len(current_call["inputs"]) == 3
    assert current_call["inputs"][0] is None
    assert current_call["inputs"][1] is None
    assert current_call["inputs"][2].shape == (batch_size, tactile_len, 8)
    assert current_call["kv_cache"] is None
    assert current_call["mask"].shape == (batch_size, tactile_len, tactile_len)
    assert current_call["mask"][0].all()
    assert current_call["mask"][1, :T3_TOKENS_PER_SENSOR, :T3_TOKENS_PER_SENSOR].all()
    assert not current_call["mask"][1, T3_TOKENS_PER_SENSOR:].any()
    assert context_mask.shape == (batch_size, prefix_len + tactile_len)
    assert len(staged_cache) == llm.depth
    for layer, (key, value) in enumerate(staged_cache):
        assert key.shape == (batch_size, prefix_len + tactile_len, 1, 4)
        assert value.shape == key.shape
        torch.testing.assert_close(key[:, :prefix_len], prefix_cache[layer][0])
        torch.testing.assert_close(value[:, :prefix_len], prefix_cache[layer][1])

    noisy_future = torch.randn_like(encoded_current)
    future_prediction, future_mask = policy.run_tactile_future_suffix(
        observation,
        noisy_future,
        torch.tensor([0.25, 0.75]),
        staged_cache,
        context_mask,
    )
    future_call = llm.calls[-1]
    assert future_prediction.shape == encoded_current.shape
    assert future_mask.shape == (batch_size, tactile_len)
    assert future_mask[0].all()
    assert not future_mask[1].any()
    assert future_call["inputs"][0] is None
    assert future_call["inputs"][1] is None
    assert future_call["inputs"][2].shape == (batch_size, tactile_len, 8)
    assert future_call["kv_cache"] is staged_cache
    assert future_call["attention_backend"] == "sdpa"
    assert future_call["mask"].shape == (
        batch_size,
        tactile_len,
        prefix_len + 2 * tactile_len,
    )
    assert future_call["mask"][0].all()
    assert not future_call["mask"][1].any()

    action_tokens = torch.randn(batch_size, policy.action_horizon, 8)
    policy.embed_suffix = lambda *_args: (
        action_tokens,
        torch.ones(batch_size, policy.action_horizon, dtype=torch.bool),
        torch.tensor([True, False]),
        torch.randn(batch_size, 8),
    )
    action_output = policy.run_suffix(
        observation,
        torch.zeros(batch_size, policy.action_horizon, 3),
        torch.full((batch_size,), 0.5),
        staged_cache,
        context_mask,
    )
    action_call = llm.calls[-1]
    assert action_call["attention_backend"] == "eager"
    torch.testing.assert_close(action_output, action_tokens)
    assert len(action_call["inputs"]) == 2
    assert action_call["inputs"][0] is None
    assert action_call["inputs"][1] is action_tokens
    assert action_call["kv_cache"] is staged_cache
    # Action and noisy-future tokens never share a transformer invocation;
    # they read the same immutable prefix+current-touch cache independently.
    assert future_call["inputs"][1] is None
    assert action_call["inputs"][-1] is not future_call["inputs"][-1]


def test_all_invalid_future_batch_has_finite_differentiable_zero_loss(monkeypatch):
    policy = Pi0.__new__(Pi0)
    nn.Module.__init__(policy)
    policy.embed_dtype = torch.float32
    policy.tactile_expert_enabled = True
    policy._config = SimpleNamespace(tactile_expert_future_loss_weight=1.0)
    policy.tactile_expert_future_horizon = 50
    policy.tactile_expert_future_normalization = "t3_reference_rms"
    policy.tactile_expert_future_reference_rms = 0.25
    policy.action_horizon = 2
    policy.action_out_proj = nn.Linear(4, 3, bias=False)
    nn.init.zeros_(policy.action_out_proj.weight)
    encoded = torch.zeros(2, 2, T3_TOKENS_PER_SENSOR, T3_LARGE_WIDTH)
    policy.encode_tactile_expert_endpoints = lambda _observation: (
        encoded,
        encoded.clone(),
    )
    prefix_mask = torch.ones(2, 3, dtype=torch.bool)
    staged_mask = torch.ones(2, 3 + 2 * T3_TOKENS_PER_SENSOR, dtype=torch.bool)
    cache = ("cache",)
    policy.build_prefix_cache = lambda _observation: (
        torch.zeros(2, 3, 4),
        prefix_mask,
        cache,
    )
    policy.build_tactile_expert_cache = lambda *_args: (staged_mask, cache)
    seen_times: dict[str, torch.Tensor] = {}

    def _run_action_suffix(_observation, _x_t, action_time, *_args, **_kwargs):
        seen_times["action"] = action_time.detach().clone()
        return torch.zeros(2, 2, 4)

    policy.run_suffix = _run_action_suffix
    future_velocity = torch.ones_like(encoded, requires_grad=True)

    def _run_future_suffix(_observation, _noisy, future_time, *_args, **_kwargs):
        seen_times["future"] = future_time.detach().clone()
        return (
            future_velocity,
            torch.zeros(2, 2 * T3_TOKENS_PER_SENSOR, dtype=torch.bool),
        )

    policy.run_tactile_future_suffix = _run_future_suffix
    monkeypatch.setattr(
        pi0_model,
        "preprocess_observation",
        lambda observation, **_kwargs: observation,
    )
    monkeypatch.setattr(
        pi0_model,
        "_observation_to_dtype",
        lambda observation, _dtype: observation,
    )

    action_loss, auxiliary = policy.compute_loss(
        SimpleNamespace(
            state=torch.zeros(2, 3),
            tactile_future_target_valid=torch.zeros(2, dtype=torch.bool),
        ),
        torch.zeros(2, 2, 3),
        train=True,
        noise=torch.zeros(2, 2, 3),
        time=torch.full((2,), 0.5),
        tactile_future_noise=torch.zeros_like(encoded),
        tactile_future_time=torch.tensor([0.25, 0.75]),
        return_auxiliary_losses=True,
    )

    torch.testing.assert_close(action_loss, torch.zeros_like(action_loss))
    future_loss = auxiliary["tactile_expert_future_loss"]
    assert future_loss.item() == 0.0
    assert torch.isfinite(future_loss)
    future_loss.backward()
    assert future_velocity.grad is not None
    assert torch.count_nonzero(future_velocity.grad) == 0
    assert auxiliary["tactile_expert_future_valid_fraction"].item() == 0.0
    torch.testing.assert_close(seen_times["action"], torch.full((2,), 0.5))
    torch.testing.assert_close(seen_times["future"], torch.tensor([0.25, 0.75]))


def test_future_normalization_is_fixed_invertible_and_does_not_touch_current():
    policy = Pi0.__new__(Pi0)
    nn.Module.__init__(policy)
    policy.tactile_expert_future_normalization = "t3_reference_rms"
    policy.tactile_expert_future_reference_rms = 0.25
    current = torch.tensor([0.5, -0.5])
    future = torch.tensor([0.25, -0.5])
    current_before = current.clone()

    normalized = policy.normalize_tactile_expert_future(future)

    torch.testing.assert_close(normalized, torch.tensor([1.0, -2.0]))
    torch.testing.assert_close(
        policy.denormalize_tactile_expert_future(normalized),
        future,
    )
    torch.testing.assert_close(current, current_before)

    policy.tactile_expert_future_normalization = "none"
    assert policy.normalize_tactile_expert_future(future) is future


def test_compute_loss_normalizes_future_but_caches_raw_current(monkeypatch):
    policy = Pi0.__new__(Pi0)
    nn.Module.__init__(policy)
    policy.embed_dtype = torch.float32
    policy.tactile_expert_enabled = True
    policy.tactile_expert_future_horizon = 2
    policy.tactile_expert_future_normalization = "t3_reference_rms"
    policy.tactile_expert_future_reference_rms = 0.25
    policy._config = SimpleNamespace(tactile_expert_future_loss_weight=0.001)
    policy.action_horizon = 2
    policy.action_out_proj = nn.Linear(4, 3, bias=False)
    nn.init.zeros_(policy.action_out_proj.weight)
    current = torch.full((1, 2, T3_TOKENS_PER_SENSOR, T3_LARGE_WIDTH), 0.5)
    future = torch.full_like(current, 0.25)
    policy.encode_tactile_expert_endpoints = lambda _observation: (current, future)
    context_mask = torch.ones(1, 3, dtype=torch.bool)
    cache = ("cache",)
    policy.build_prefix_cache = lambda _observation: (
        torch.zeros(1, 3, 4),
        context_mask,
        cache,
    )
    seen: dict[str, torch.Tensor] = {}

    def _cache_current(_observation, encoded, mask, incoming_cache):
        seen["current"] = encoded.clone()
        return mask, incoming_cache

    def _future_suffix(_observation, noisy, *_args, **_kwargs):
        seen["noisy_future"] = noisy.clone()
        return torch.zeros_like(noisy), torch.ones(
            1,
            2 * T3_TOKENS_PER_SENSOR,
            dtype=torch.bool,
        )

    policy.build_tactile_expert_cache = _cache_current
    policy.run_suffix = lambda *_args, **_kwargs: torch.zeros(1, 2, 4)
    policy.run_tactile_future_suffix = _future_suffix
    monkeypatch.setattr(
        pi0_model,
        "preprocess_observation",
        lambda observation, **_kwargs: observation,
    )
    monkeypatch.setattr(
        pi0_model,
        "_observation_to_dtype",
        lambda observation, _dtype: observation,
    )

    _, auxiliary = policy.compute_loss(
        SimpleNamespace(
            state=torch.zeros(1, 3),
            tactile_future_target_valid=torch.ones(1, dtype=torch.bool),
        ),
        torch.zeros(1, 2, 3),
        train=True,
        noise=torch.zeros(1, 2, 3),
        time=torch.full((1,), 0.5),
        tactile_future_noise=torch.zeros_like(future),
        tactile_future_time=torch.zeros(1),
        return_auxiliary_losses=True,
    )

    torch.testing.assert_close(seen["current"], current)
    torch.testing.assert_close(seen["noisy_future"], torch.ones_like(future))
    assert auxiliary["tactile_expert_future_loss"].item() == pytest.approx(1.0)
    assert auxiliary["tactile_expert_future_target_rms"].item() == pytest.approx(1.0)
    raw_target_rms = auxiliary["tactile_expert_future_raw_target_rms"]
    assert raw_target_rms.item() == pytest.approx(0.25)


def test_zero_future_weight_skips_future_encoding_and_suffix(monkeypatch):
    policy = Pi0.__new__(Pi0)
    nn.Module.__init__(policy)
    policy.embed_dtype = torch.float32
    policy.tactile_expert_enabled = True
    policy._config = SimpleNamespace(tactile_expert_future_loss_weight=0.0)
    policy.action_horizon = 2
    policy.action_out_proj = nn.Linear(4, 3, bias=False)
    nn.init.zeros_(policy.action_out_proj.weight)
    encoded = torch.zeros(1, 2, T3_TOKENS_PER_SENSOR, T3_LARGE_WIDTH)
    policy.encode_policy_tactile = lambda _observation: encoded
    policy.encode_tactile_expert_endpoints = lambda _observation: (_ for _ in ()).throw(
        AssertionError("future endpoint encoding must be skipped")
    )
    prefix_mask = torch.ones(1, 3, dtype=torch.bool)
    staged_mask = torch.ones(1, 3 + 2 * T3_TOKENS_PER_SENSOR, dtype=torch.bool)
    cache = ("cache",)
    policy.build_prefix_cache = lambda _observation: (
        torch.zeros(1, 3, 4),
        prefix_mask,
        cache,
    )
    policy.build_tactile_expert_cache = lambda *_args: (staged_mask, cache)
    policy.run_suffix = lambda *_args, **_kwargs: torch.zeros(1, 2, 4)
    policy.run_tactile_future_suffix = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("future suffix must be skipped")
    )
    monkeypatch.setattr(
        pi0_model,
        "preprocess_observation",
        lambda observation, **_kwargs: observation,
    )
    monkeypatch.setattr(
        pi0_model,
        "_observation_to_dtype",
        lambda observation, _dtype: observation,
    )

    action_loss, auxiliary = policy.compute_loss(
        SimpleNamespace(state=torch.zeros(1, 3), tactile_future_target_valid=None),
        torch.zeros(1, 2, 3),
        train=True,
        noise=torch.zeros(1, 2, 3),
        time=torch.full((1,), 0.5),
        return_auxiliary_losses=True,
    )

    torch.testing.assert_close(action_loss, torch.zeros_like(action_loss))
    assert auxiliary == {}


def test_inference_encodes_and_prefills_current_touch_once(monkeypatch):
    class _FrozenT3(nn.Module):
        def __init__(self, output: torch.Tensor) -> None:
            super().__init__()
            self.output = output
            self.calls = 0
            self.grad_enabled: list[bool] = []

        def forward(self, images, *, return_all_tokens):
            assert images.shape == (1, 2, 3, 4, 4)
            assert return_all_tokens is True
            self.calls += 1
            self.grad_enabled.append(torch.is_grad_enabled())
            return self.output

    policy = Pi0.__new__(Pi0)
    nn.Module.__init__(policy)
    policy.action_horizon = 2
    policy.action_dim = 3
    policy.tactile_expert_enabled = True
    policy.tactile_encoder_frozen = True
    calls = {"prefix": 0, "current": 0, "suffix": 0}
    encoded = torch.zeros(1, 2, T3_TOKENS_PER_SENSOR, T3_LARGE_WIDTH)
    policy.tactile_encoder = _FrozenT3(encoded).eval()
    cache = ("prefix-cache",)
    staged_cache = ("staged-cache",)
    prefix_mask = torch.ones(1, 2, dtype=torch.bool)
    staged_mask = torch.ones(1, 2 + 2 * T3_TOKENS_PER_SENSOR, dtype=torch.bool)

    def _prefix(_self, _observation):
        calls["prefix"] += 1
        return torch.zeros(1, 2, 4), prefix_mask, cache

    def _current(_self, _observation, current, mask, incoming_cache):
        calls["current"] += 1
        assert current is encoded
        assert mask is prefix_mask
        assert incoming_cache is cache
        return staged_mask, staged_cache

    def _suffix(_self, _observation, x_t, _time, incoming_cache, mask, **_kwargs):
        calls["suffix"] += 1
        assert incoming_cache is staged_cache
        assert mask is staged_mask
        return torch.zeros(x_t.shape[0], x_t.shape[1], 4)

    policy.build_prefix_cache = MethodType(_prefix, policy)
    policy.build_tactile_expert_cache = MethodType(_current, policy)
    policy.run_suffix = MethodType(_suffix, policy)
    policy.velocity_from_suffix = lambda hidden: torch.zeros(
        hidden.shape[0],
        hidden.shape[1],
        policy.action_dim,
    )
    monkeypatch.setattr(
        pi0_module.model,
        "preprocess_observation",
        lambda observation, **_kwargs: observation,
    )

    result = policy.sample_actions(
        SimpleNamespace(
            state=torch.zeros(1, 3),
            tactile_images={
                "left_gsmini_rgb": torch.zeros(1, 3, 4, 4),
                "right_gsmini_rgb": torch.zeros(1, 3, 4, 4),
            },
        ),
        num_steps=10,
        noise=torch.ones(1, 2, 3),
    )

    torch.testing.assert_close(result, torch.ones_like(result))
    assert policy.tactile_encoder.calls == 1
    assert policy.tactile_encoder.grad_enabled == [False]
    assert policy.tactile_encoder.training is False
    assert calls == {"prefix": 1, "current": 1, "suffix": 10}
