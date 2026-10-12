"""Unsupported experiments fail explicitly; native checkpoint tools remain usable."""

import inspect
import json
from argparse import ArgumentParser

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models import get_model
from rlinf.models.embodiment.openpi_pytorch.pi0_model import gemma
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0
from rlinf.utils.ckpt_convertor.openpi import jax2new
from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict, strip_wrapper_prefix
from scripts.univtac.serve_policy import _current_model_config


def test_release_has_no_distillation_or_gradient_isolation_hooks():
    assert not hasattr(Pi0, "detach_kv_cache_prefix")
    assert not hasattr(gemma, "stop_gradient_through_sequence_prefix")
    for method, removed in (
        (
            Pi0.run_suffix,
            {"return_action_hidden_states", "conditioning_tokens", "conditioning_mask"},
        ),
        (
            gemma.Module.forward,
            {
                "return_expert_hidden_states",
                "expert_hidden_state_index",
                "hidden_state_suffix_length",
                "kv_cache_detach_prefix_length",
            },
        ),
        (Pi0.run_tactile_future_suffix, {"kv_cache_detach_prefix_length"}),
        (Pi0.embed_prefix, {"encoded_tactile"}),
        (Pi0.build_prefix_cache, {"encoded_tactile"}),
    ):
        assert removed.isdisjoint(inspect.signature(method).parameters)


def test_future_tokens_keep_gradients_to_both_context_modalities():
    torch.manual_seed(7)
    config = gemma.Config(
        width=8, depth=1, mlp_dim=16, num_heads=2, num_kv_heads=1, head_dim=4
    )
    llm = gemma.Module(
        configs=[config, config, config],
        embed_dtype="float32",
        adarms=[False, False, False],
        use_gradient_checkpointing=False,
    ).train()
    # The context cache is VL followed by current tactile tokens. Auxiliary
    # prediction must remain connected to both, without the removed detach mode.
    key = torch.randn(1, 5, 1, 4, requires_grad=True)
    value = torch.randn(1, 5, 1, 4, requires_grad=True)
    future = torch.randn(1, 2, 8, requires_grad=True)
    output, cache = llm(
        [None, None, future],
        positions=torch.tensor([[5, 6]]),
        mask=torch.ones(1, 2, 7, dtype=torch.bool),
        kv_cache=((key, value),),
    )
    assert cache[0][0].shape[1] == 7
    output[2].square().mean().backward()
    for tensor in (key, value):
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad[:, :3]) > 0
        assert torch.count_nonzero(tensor.grad[:, 3:]) > 0


def test_lora_rejected_before_model_construction():
    with pytest.raises(ValueError, match="LoRA"):
        get_model(OmegaConf.create({"is_lora": True}))


@pytest.mark.parametrize(
    "variant", ["gemma_2b_lora", "gemma_300m_lora", "gemma_2b_lora_32"]
)
def test_removed_lora_variants(variant):
    with pytest.raises(ValueError):
        gemma.get_config(variant)


@pytest.mark.parametrize(
    "option",
    [
        {"use_tactile": True},
        {"use_tactile_in_action_expert": True},
        {"tactile_vlm_input_mode": "siglip_images"},
        {"tactile_vlm_input_mode": "t3_tokens"},
    ],
)
def test_server_rejects_direct_vlm_and_ae_tactile(option):
    with pytest.raises(TypeError):
        _current_model_config(
            OmegaConf.create(
                {"actor": {"model": {"num_action_chunks": 50, "openpi": option}}}
            )
        )


def test_native_checkpoint_prefix_and_collision_handling():
    tensor = torch.tensor([1.25])
    state = as_state_dict({"state_dict": {"module.model.weight": tensor}})
    stripped = strip_wrapper_prefix(state, cast_dtype=None)
    assert torch.equal(stripped["weight"], tensor)
    with pytest.raises(ValueError, match="duplicate"):
        strip_wrapper_prefix({"model.weight": tensor, "weight": tensor})


def test_base_conversion_still_writes_native_assets(monkeypatch, tmp_path):
    parser = ArgumentParser()
    jax2new.add_arguments(parser)
    norm = tmp_path / "norm.json"
    norm.write_text('{"fixture": true}')
    output = tmp_path / "converted"
    args = parser.parse_args(
        [
            "--input-model",
            str(tmp_path / "jax"),
            "--input-norm-stats",
            str(norm),
            "--output-model",
            str(output),
            "--output-norm-stats",
            str(output / "norm_stats.json"),
        ]
    )
    assert args.action_horizon == 50
    monkeypatch.setattr(jax2new, "_load_jax_params", lambda _: {})
    monkeypatch.setattr(jax2new, "convert_siglip", lambda _: {"image": torch.ones(2)})
    monkeypatch.setattr(jax2new, "convert_llm", lambda *a: {"llm": torch.ones(2)})
    monkeypatch.setattr(
        jax2new, "convert_projections", lambda *a: {"projection": torch.ones(2)}
    )
    jax2new.run(args)
    assert (output / "model.safetensors").is_file()
    assert (output / "norm_stats.json").read_bytes() == norm.read_bytes()
    config = json.loads((output / "config.json").read_text())
    assert config["pi05"] is True
    assert config["action_horizon"] == 50
    assert "pcd" not in config
