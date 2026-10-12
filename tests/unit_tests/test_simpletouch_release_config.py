"""Release configuration, compatibility boundaries and strict checkpoint loading."""

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from rlinf.models.embodiment.openpi_pytorch import _load_openpi_checkpoint, get_model
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config
from scripts.simpletouch.checkpoint import build_training_fingerprint
from scripts.simpletouch import checkpoint as checkpoint_io
from scripts.univtac.serve_policy import (
    _resolve_tactile_expert,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def release_config(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/sft"))
    for variable in ("SIMPLETOUCH_NORM_STATS_PATH", "SIMPLETOUCH_TOKENIZER_PATH"):
        path = tmp_path / variable
        path.write_text("fixture")
        monkeypatch.setenv(variable, str(path))
    with initialize_config_dir(
        version_base="1.1", config_dir=str(ROOT / "examples/sft/config")
    ):
        return compose(
            config_name="simpletac_univtac",
            overrides=["data.tactile_expert_future_cache_path=null"],
        )


@pytest.mark.parametrize("mode", ["fast", "joint", "idm"])
@pytest.mark.parametrize("tail", ["repeat", "mask"])
def test_slim_config_fingerprints_and_resolves_for_inference(
    release_config, mode, tail
):
    cfg = release_config
    cfg.actor.model.openpi.tactile_expert_wam_mode = mode
    cfg.actor.model.openpi.tactile_expert_tail_supervision_mode = tail
    fingerprint = build_training_fingerprint(cfg, world_size=8)
    te = fingerprint["model"]["openpi"]["tactile_expert"]
    assert te["future_tail_mode"] == tail
    expected = (
        "mask_repeat_last_padding" if tail == "mask" else "include_repeat_last_padding"
    )
    assert fingerprint["data"]["action_loss_masking"] == expected
    contract = _resolve_tactile_expert(
        {"tactile_expert_in_proj.weight": torch.empty(0)}, cfg
    )
    assert contract["future_tail_mode"] == tail
    assert contract["wam_mode"] == mode
    assert contract["width"] == 1024


@pytest.mark.parametrize(
    "old_option",
    [
        {"use_tactile_in_action_expert": True},
        {"use_tactile_in_action_expert": False},
        {"use_tactile": True},
        {"use_tactile": False},
        {"tactile_vlm_input_mode": "siglip_images"},
        {"tactile_world_model_enabled": True},
        {"tactile_world_model_enabled": False},
        {"ae_transition_context": "previous_transition"},
        {"tactile_expert_share_action_weights": True},
        {"tactile_expert_input_layout": "legacy_projected"},
        {"tactile_expert_future_vl_gradient_mode": "detach"},
    ],
)
def test_removed_architectures_fail_closed(old_option):
    with pytest.raises(TypeError):
        Pi0Config(**old_option)


@pytest.fixture
def completed_checkpoint(release_config, tmp_path):
    root = tmp_path / "global_step_5000"
    files = (
        "actor/data.pt",
        "actor/rng.pt",
        "actor/model_state_dict/full_weights.pt",
        "actor/dcp_checkpoint/.metadata",
        "actor/dcp_checkpoint/rank0.distcp",
    )
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    OmegaConf.save(release_config, root / "config.yaml", resolve=True)
    (root / "norm_stats.json").write_bytes(
        Path(release_config.data.norm_stats_path).read_bytes()
    )
    checkpoint_io.write_success_marker(root, release_config, step=5000, world_size=8)
    return root


def test_current_checkpoint_completion_and_resume(release_config, completed_checkpoint):
    for mode in ("fast", "full"):
        marker = checkpoint_io.validate_checkpoint_completion(
            completed_checkpoint, verify_mode=mode
        )
        assert (
            "sha256"
            not in marker["required_files"]["actor/model_state_dict/full_weights.pt"]
        )
        assert (
            "sha256"
            not in marker["required_files"]["actor/dcp_checkpoint/rank0.distcp"]
        )
        assert "sha256" in marker["required_files"]["config.yaml"]
        checkpoint_io.validate_resume_checkpoint(
            completed_checkpoint,
            release_config,
            current_world_size=8,
            requested_total_steps=release_config.runner.max_steps,
            verify_mode=mode,
        )


@pytest.mark.parametrize("field", ["marker", "fingerprint"])
def test_checkpoint_does_not_migrate_old_schema(completed_checkpoint, field):
    path = completed_checkpoint / "_SUCCESS"
    payload = json.loads(path.read_text())
    if field == "marker":
        payload["schema_version"] -= 1
    else:
        payload["training_fingerprint"]["schema_version"] -= 1
        payload["training_fingerprint_sha256"] = checkpoint_io.canonical_json_digest(
            payload["training_fingerprint"]
        )
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema"):
        checkpoint_io.validate_checkpoint_completion(completed_checkpoint)


def test_resume_rejects_semantic_change(release_config, completed_checkpoint):
    release_config.actor.optim.lr *= 2
    with pytest.raises(ValueError, match="optimizer.lr"):
        checkpoint_io.validate_resume_checkpoint(
            completed_checkpoint,
            release_config,
            current_world_size=8,
            requested_total_steps=release_config.runner.max_steps,
        )


def test_checkpoint_metadata_corruption_is_not_ignored(completed_checkpoint):
    (completed_checkpoint / "norm_stats.json").write_text("changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        checkpoint_io.validate_checkpoint_completion(
            completed_checkpoint, verify_mode="fast"
        )


class TinyCheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Linear(2, 2)
        self.tactile_encoder = nn.Linear(2, 2)
        self.tactile_expert_in_proj = nn.Linear(2, 2)
        self.initializations = 0

    def initialize_tactile_expert_from_base_checkpoint(self):
        self.initializations += 1


def test_complete_te_checkpoint_is_strict_and_never_reinitialized():
    model = TinyCheckpointModel()
    _load_openpi_checkpoint(
        model, model.state_dict(), use_tactile_modules=True, use_tactile_expert=True
    )
    assert model.initializations == 0
    state = dict(model.state_dict())
    del state["tactile_expert_in_proj.weight"]
    with pytest.raises(RuntimeError):
        _load_openpi_checkpoint(
            model, state, use_tactile_modules=True, use_tactile_expert=True
        )


def test_only_verified_base_can_initialize_missing_te():
    model = TinyCheckpointModel()
    base = {
        key: val for key, val in model.state_dict().items() if key.startswith("action.")
    }
    with pytest.raises(RuntimeError, match="verified"):
        _load_openpi_checkpoint(
            model, base, use_tactile_modules=True, use_tactile_expert=True
        )
    _load_openpi_checkpoint(
        model,
        base,
        use_tactile_modules=True,
        use_tactile_expert=True,
        allow_missing_tactile_expert_from_verified_base=True,
    )
    assert model.initializations == 1


@pytest.mark.parametrize("task", ["sft", "eval"])
def test_training_factory_accepts_slim_recipe_without_allocating_large_model(
    release_config, monkeypatch, tmp_path, task
):
    cfg = release_config.actor.model
    cfg.openpi.task = task
    cfg.model_path = str(tmp_path)
    (tmp_path / "model.safetensors").touch()
    monkeypatch.setenv("SIMPLETOUCH_BASE_MODEL_PATH", str(tmp_path))
    captured = []

    def create(config):
        captured.append(config)
        return TinyCheckpointModel()

    monkeypatch.setattr(Pi0Config, "create", create)
    base = {
        key: val
        for key, val in TinyCheckpointModel().state_dict().items()
        if key.startswith("action.")
    }
    import safetensors.torch

    monkeypatch.setattr(safetensors.torch, "load_file", lambda *a, **kw: base)
    if task == "eval":
        from rlinf.models.embodiment.openpi_pytorch.utils import model_builders

        def wrap(config, model_config, model, **kwargs):
            assert model_config == config.openpi
            return type("EvalWrapper", (), {"model": model})()

        monkeypatch.setattr(model_builders, "_build_eval_model", wrap)
    result = get_model(cfg)
    assert result.model.initializations == 1
    assert captured[0].tactile_expert_tail_supervision_mode == "repeat"
    assert captured[0].tactile_expert_init_mode == "random"
