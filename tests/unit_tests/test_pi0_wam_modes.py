# Copyright 2026 The RLinf Team.
# SPDX-License-Identifier: Apache-2.0

"""Fast-WAM mode semantics without model checkpoints or a simulator."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rlinf.models.embodiment.openpi_pytorch.pi0_model import gemma
from rlinf.models.embodiment.openpi_pytorch.pi0_model import model as model_module
from rlinf.models.embodiment.openpi_pytorch.pi0_model import pi0 as pi0_module
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_preprocess_profile import (
    T3_UNIVTAC_RGB_PREPROCESS_PROFILE,
)


def bare_policy(mode):
    policy = Pi0.__new__(Pi0)
    nn.Module.__init__(policy)
    policy.tactile_expert_wam_mode = mode
    policy.tactile_expert_enabled = True
    policy.embed_dtype = torch.float32
    policy.action_horizon = 2
    policy.tactile_expert_future_prediction_mode = "single_horizon_full"
    policy.tactile_expert_future_horizon = 10
    policy.tactile_expert_future_stride = 5
    policy.tactile_expert_future_pool_grid = 1
    policy._config = SimpleNamespace(
        tactile_expert_tail_supervision_mode=None if mode == "fast" else "mask",
        tactile_expert_future_loss_weight=0.01,
        tactile_expert_idm_condition_noise_prob=0.0,
    )
    return policy


@pytest.mark.parametrize("mode", ["fast", "joint", "idm"])
def test_mode_config_is_explicit_and_legacy_defaults_to_fast(mode):
    assert Pi0Config().tactile_expert_wam_mode == "fast"
    config = Pi0Config(
        pi05=True,
        tactile_expert_enabled=True,
        tactile_expert_wam_mode=mode,
        tactile_expert_tail_supervision_mode=None if mode == "fast" else "mask",
        tactile_expert_future_loss_weight=0.01,
        tactile_preprocess_profile=T3_UNIVTAC_RGB_PREPROCESS_PROFILE,
        tactile_preprocess_task="insert_hole",
    )
    assert config.tactile_expert_wam_mode == mode


def test_future_time_embedding_accepts_float32_solver_time_in_bfloat16_model():
    policy = bare_policy("joint")
    policy.embed_dtype = torch.bfloat16
    policy.tactile_expert_time_mlp_in = nn.Linear(8, 8).bfloat16()
    policy.tactile_expert_time_mlp_out = nn.Linear(8, 8).bfloat16()
    embedding = policy._embed_tactile_expert_time(torch.tensor([0.5]))
    assert embedding.dtype == torch.bfloat16
    assert torch.isfinite(embedding).all()


@pytest.mark.parametrize(
    "options",
    [
        {"tactile_expert_wam_mode": "unknown"},
        {"tactile_expert_wam_mode": "joint"},
        {
            "tactile_expert_wam_mode": "idm",
            "tactile_expert_future_vl_gradient_mode": "detach",
        },
        {"tactile_expert_idm_condition_noise_prob": float("nan")},
    ],
)
def test_invalid_mode_configs_fail_early(options):
    with pytest.raises((ValueError, TypeError)):
        Pi0Config(**options)


@pytest.mark.parametrize("mode", ["joint", "idm"])
@pytest.mark.parametrize(
    "override",
    [
        {"tactile_expert_enabled": False},
        {"tactile_expert_future_loss_weight": 0.0},
        {"tactile_expert_future_vl_gradient_mode": "detach"},
    ],
)
def test_mode_contract_rejects_disabled_future_or_gradient_isolation(mode, override):
    options = {
        "pi05": True,
        "tactile_expert_enabled": True,
        "tactile_expert_future_loss_weight": 0.01,
        "tactile_preprocess_profile": T3_UNIVTAC_RGB_PREPROCESS_PROFILE,
        "tactile_preprocess_task": "insert_hole",
        "tactile_expert_wam_mode": mode,
        "tactile_expert_tail_supervision_mode": "mask",
    }
    options.update(override)
    with pytest.raises(
        (ValueError, TypeError),
    ):
        Pi0Config(**options)


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_idm_teacher_condition_is_seeded_per_sample(probability):
    policy = bare_policy("idm")
    policy._config.tactile_expert_idm_condition_noise_prob = probability
    target = torch.ones(100, 2, 3, 4)
    result, time = policy._idm_training_condition(
        target, rng=torch.Generator().manual_seed(123)
    )
    again, again_time = policy._idm_training_condition(
        target, rng=torch.Generator().manual_seed(123)
    )
    torch.testing.assert_close(result, again)
    torch.testing.assert_close(time, again_time)
    torch.testing.assert_close(result[time == 0], target[time == 0])
    if probability == 0:
        assert not time.any()
    elif probability == 1:
        assert (time > 0).all()
    else:
        assert 20 < (time > 0).sum() < 80


@pytest.mark.parametrize("mode", ["fast", "joint", "idm"])
def test_loss_routes_future_cache_without_leaking_noisy_branch_to_idm(
    monkeypatch, mode
):
    monkeypatch.setattr(pi0_module, "T3_TOKENS_PER_SENSOR", 3)
    policy = bare_policy(mode)
    policy.tactile_expert_future_normalization = "none"
    policy.tactile_expert_future_reference_rms = 1.0
    policy.action_out_proj = nn.Identity()
    target = torch.ones(1, 2, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    current_cache = object()
    seen = []
    policy.encode_tactile_expert_endpoints = lambda _: (target, target)
    policy.build_prefix_cache = lambda _: (None, mask, current_cache)
    policy.build_tactile_expert_cache = lambda _o, _z, m, c: (m, c)

    def future(_obs, noisy, time, cache, context_mask, **kwargs):
        assert cache is current_cache
        seen.append((noisy.clone(), time.clone()))
        value = noisy.mean()
        result = (torch.zeros_like(noisy), torch.ones(1, 6, dtype=torch.bool))
        return (*result, value) if kwargs.get("return_kv_cache") else result

    def action(_obs, _noisy, _time, cache, context_mask):
        if mode == "fast":
            assert cache is current_cache
            return torch.zeros(1, 2, 4)
        assert context_mask.shape == (1, 9)
        return cache.expand(1, 2, 4)

    policy.run_tactile_future_suffix = future
    policy.run_suffix = action
    monkeypatch.setattr(model_module, "preprocess_observation", lambda obs, **kw: obs)
    monkeypatch.setattr(model_module, "_observation_to_dtype", lambda obs, dtype: obs)
    loss, aux = policy.compute_loss(
        SimpleNamespace(
            state=torch.zeros(1, 4),
            tactile_future_target_valid=torch.ones(1, dtype=torch.bool),
        ),
        torch.zeros(1, 2, 4),
        train=True,
        noise=torch.zeros(1, 2, 4),
        time=torch.tensor([0.5]),
        tactile_future_noise=torch.zeros_like(target),
        tactile_future_time=torch.tensor([0.8]),
        return_auxiliary_losses=True,
    )
    assert len(seen) == (2 if mode == "idm" else 1)
    expected = {"fast": 0.0, "joint": 0.04, "idm": 1.0}[mode]
    torch.testing.assert_close(loss, torch.full_like(loss, expected))
    assert aux["tactile_expert_future_loss"].item() == pytest.approx(1.0)
    if mode == "idm":
        assert seen[1][1].item() == 0
        torch.testing.assert_close(seen[1][0], target)


@pytest.mark.parametrize("mode", ["joint", "idm"])
@pytest.mark.parametrize("multi", [False, True])
def test_sampler_schedule_and_no_ground_truth_future_dependency(mode, multi):
    policy = bare_policy(mode)
    if multi:
        policy.tactile_expert_future_prediction_mode = "multi_horizon_pooled"
    obs = model_module.Observation(
        images={},
        image_masks={},
        state=torch.zeros(1, 4),
        tactile_masks={
            "left_gsmini_rgb": torch.tensor([True]),
            "right_gsmini_rgb": torch.tensor([False]),
        },
        tactile_future_target_valid=torch.tensor([False]),
        tactile_world_future_latents=torch.full((1,), float("nan")),
    )
    shape = (1, 2, 2, 2, 1024) if multi else (1, 2, 197, 1024)
    calls, action_caches = [], []
    current_cache = object()

    def future(observation, latent, time, cache, mask, **kwargs):
        assert cache is current_cache  # Never append a previous noisy suffix.
        assert observation.tactile_world_future_latents is None
        assert observation.tactile_future_target_valid is None
        assert not observation.tactile_future_masks["right_gsmini_rgb"].any()
        calls.append(time.item())
        return (
            torch.ones_like(latent),
            torch.ones(1, latent.numel() // 1024, dtype=torch.bool),
            latent.mean(),
        )

    def action(_obs, noisy, time, cache, mask):
        action_caches.append(cache.item())
        return cache.expand_as(noisy)

    policy.run_tactile_future_suffix = future
    policy.run_suffix = action
    policy.velocity_from_suffix = lambda x: x
    output = policy._sample_actions_with_future(
        obs,
        torch.zeros(1, 2, 4),
        torch.ones(1, 3, dtype=torch.bool),
        current_cache,
        num_steps=2,
        rng=None,
        future_noise=torch.full(shape, 2.0),
    )
    assert calls == ([1.0, 0.5] if mode == "joint" else [1.0, 0.5, 0.0])
    assert action_caches == ([2.0, 1.5] if mode == "joint" else [1.0, 1.0])
    torch.testing.assert_close(
        output, torch.full_like(output, -1.75 if mode == "joint" else -1.0)
    )


def test_staged_one_way_joint_matches_monolithic_outputs_and_gradients():
    torch.manual_seed(123)
    config = gemma.Config(
        width=8, depth=2, mlp_dim=16, num_heads=2, num_kv_heads=1, head_dim=4
    )
    llm = gemma.Module(
        configs=[config] * 3,
        embed_dtype="float32",
        adarms=[False] * 3,
        use_gradient_checkpointing=False,
    )
    action = torch.randn(1, 2, 8, requires_grad=True)
    future = torch.randn(1, 3, 8, requires_grad=True)
    # Logical RoPE order: future then action. Storage order follows experts.
    future_positions = torch.tensor([[0, 1, 2]])
    action_positions = torch.tensor([[3, 4]])
    mask = torch.ones(1, 5, 5, dtype=torch.bool)
    mask[:, 2:, :2] = False  # future cannot read action
    outputs, _ = llm(
        [None, action, future],
        positions=torch.cat([action_positions, future_positions], 1),
        mask=mask,
    )
    total = outputs[1].square().sum() + outputs[2].square().sum()
    expected_grads = torch.autograd.grad(
        total, (action, future, *llm.parameters()), allow_unused=True
    )
    f_out, cache = llm(
        [None, None, future],
        positions=future_positions,
        mask=torch.ones(1, 3, 3, dtype=torch.bool),
    )
    a_out, _ = llm(
        [None, action],
        positions=action_positions,
        mask=torch.ones(1, 2, 5, dtype=torch.bool),
        kv_cache=cache,
    )
    torch.testing.assert_close(a_out[1], outputs[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(f_out[2], outputs[2], atol=1e-5, rtol=1e-5)
    actual_grads = torch.autograd.grad(
        a_out[1].square().sum() + f_out[2].square().sum(),
        (action, future, *llm.parameters()),
        allow_unused=True,
    )
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("mode", ["joint", "idm"])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_transformer_training_backward_and_sampling(
    monkeypatch, mode, multi, dtype
):
    """Shrink widths, not cache/attention/solver logic; include masked sensors."""
    torch.manual_seed(17)
    monkeypatch.setattr(pi0_module, "T3_LARGE_WIDTH", 8)
    monkeypatch.setattr(pi0_module, "T3_TOKENS_PER_SENSOR", 3)
    monkeypatch.setattr(model_module, "preprocess_observation", lambda obs, **kw: obs)
    policy = bare_policy(mode)
    policy.embed_dtype = dtype
    policy.pi05 = True
    policy.action_dim = 3
    policy.tactile_expert_future_normalization = "t3_reference_rms"
    policy.tactile_expert_future_reference_rms = 0.1
    policy.tactile_expert_future_attention_backend = "sdpa" if multi else "eager"
    if multi:
        policy.tactile_expert_future_prediction_mode = "multi_horizon_pooled"
    config = gemma.Config(
        width=8, depth=2, mlp_dim=16, num_heads=2, num_kv_heads=1, head_dim=4
    )
    policy.llm = gemma.Module(
        configs=[config] * 3,
        embed_dtype=str(dtype).removeprefix("torch."),
        adarms=[False, True, True],
        use_gradient_checkpointing=False,
    )
    for name in (
        "tactile_expert_in_proj",
        "tactile_expert_out_proj",
        "tactile_expert_time_mlp_in",
        "tactile_expert_time_mlp_out",
        "time_mlp_in",
        "time_mlp_out",
    ):
        setattr(policy, name, nn.Linear(8, 8))
    policy.tactile_expert_sensor_embedding = nn.Embedding(2, 8)
    policy.action_in_proj = nn.Linear(3, 8)
    policy.action_out_proj = nn.Linear(8, 3)
    policy.to(dtype=dtype)
    # Exercise nonzero residual paths, not just identity through zero gates.
    with torch.no_grad():
        for layer in policy.llm.layers:
            for norms in (layer.pre_attention_norms, layer.pre_ffw_norms):
                for expert in (1, 2):
                    norms[expert].ada_modulation.bias.fill_(0.1)
    current = torch.randn(1, 2, 3, 8, dtype=dtype)
    shape = (1, 2, 2, 2, 8) if multi else (1, 2, 3, 8)
    future = 0.1 * torch.randn(shape, dtype=dtype)
    masks = {
        "left_gsmini_rgb": torch.tensor([True]),
        "right_gsmini_rgb": torch.tensor([False]),
    }
    fm = {k: v[:, None].expand(-1, 2) for k, v in masks.items()} if multi else masks
    obs = model_module.Observation(
        images={},
        image_masks={},
        state=torch.zeros(1, 3),
        tactile_masks=masks,
        tactile_future_masks=fm,
        tactile_future_target_valid=torch.ones(
            (1, 2) if multi else (1,), dtype=torch.bool
        ),
    )
    prefix_mask = torch.ones(1, 2, dtype=torch.bool)
    prefix_cache = tuple(
        (torch.randn(1, 2, 1, 4, dtype=dtype), torch.randn(1, 2, 1, 4, dtype=dtype))
        for _ in range(2)
    )
    policy.build_prefix_cache = lambda *args: (None, prefix_mask, prefix_cache)
    policy.encode_tactile_expert_endpoints = lambda _: (current, future)
    policy.encode_policy_tactile = lambda _: current
    actions = torch.randn(1, 2, 3)

    def losses():
        return policy.compute_loss(
            obs,
            actions,
            train=True,
            noise=torch.zeros(1, 2, 3),
            time=torch.tensor([0.5]),
            tactile_future_noise=torch.zeros_like(future),
            tactile_future_time=torch.tensor([0.5]),
            return_auxiliary_losses=True,
        )

    action_loss, _ = losses()
    # Terminal labels affect supervision only: action forward and future K/V
    # layout must match the nonterminal case in Joint AND IDM.
    obs.tactile_future_target_valid.zero_()
    terminal_action_loss, terminal_aux = losses()
    torch.testing.assert_close(terminal_action_loss, action_loss)
    assert terminal_aux["tactile_expert_future_loss"].item() == 0
    policy._config.tactile_expert_tail_supervision_mode = "repeat"
    repeated_action_loss, repeated_aux = losses()
    torch.testing.assert_close(repeated_action_loss, action_loss)
    assert repeated_aux["tactile_expert_future_loss"].item() > 0
    policy._config.tactile_expert_tail_supervision_mode = "mask"
    obs.tactile_future_target_valid.fill_(True)
    # Avoid retaining multiple compiled graphs during the backward checks.
    del (
        terminal_action_loss,
        terminal_aux,
        repeated_action_loss,
        repeated_aux,
        action_loss,
    )
    action_loss, _ = losses()
    action_loss.mean().backward()
    assert policy.tactile_expert_in_proj.weight.grad.abs().sum() > 0
    assert policy.tactile_expert_out_proj.weight.grad is None
    policy.zero_grad(set_to_none=True)
    # Compiled RoPE may donate buffers. Recompute for each independent backward
    # instead of retaining a graph (the production total-loss uses one backward).
    _, aux = losses()
    aux["tactile_expert_future_loss"].backward()
    assert policy.tactile_expert_out_proj.weight.grad.abs().sum() > 0
    policy.zero_grad(set_to_none=True)
    action_loss, aux = losses()
    (action_loss.mean() + 0.01 * aux["tactile_expert_future_loss"]).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None
    )
    # Remove every GT future field before exercising the public inference API.
    obs.tactile_future_masks = None
    obs.tactile_future_target_valid = None
    output = policy.sample_actions(
        obs,
        num_steps=2,
        noise=torch.zeros(1, 2, 3),
        tactile_future_noise=torch.zeros_like(future),
    )
    assert output.shape == (1, 2, 3)
    assert torch.isfinite(output).all()
