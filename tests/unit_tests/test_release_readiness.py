"""Regressions from the first fresh-user installation/evaluation/training audit."""

import dataclasses
import inspect
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config
from scripts.simpletouch.checkpoint import validate_checkpoint_completion
from scripts.simpletouch.download_checkpoint import validate_bundle
from scripts.simpletouch.export_checkpoint import export_checkpoint
from scripts.univtac.serve_policy import (
    _current_model_config,
    _validate_simpletac_contract,
)

ROOT = Path(__file__).resolve().parents[2]


def test_export_has_current_config_and_separate_inference_provenance(tmp_path):
    weights, norm, tokenizer, config = [
        tmp_path / n
        for n in ("weights.pt", "norm.json", "tokenizer.model", "config.yaml")
    ]
    for path in (weights, norm, tokenizer):
        path.write_bytes(b"fixture payload")
    model = Pi0Config(
        pi05=True,
        tactile_expert_enabled=True,
        tactile_preprocess_profile="univtac_task_rgb_train_ep0_49_r256_cc224_v1",
        tactile_preprocess_task="insert_hole",
    )
    OmegaConf.save(OmegaConf.create(dataclasses.asdict(model)), config)
    output = tmp_path / "export/global_step_20000"
    export_checkpoint(
        weights=weights,
        norm_stats=norm,
        tokenizer=tokenizer,
        config=config,
        task="insert_hole",
        output=output,
        source="immutable-source-revision",
    )
    validate_bundle(output, "insert_hole")
    assert output.stat().st_mode & 0o777 == 0o755
    resolved = OmegaConf.load(output / "config.yaml")
    _validate_simpletac_contract(resolved)
    assert _current_model_config(resolved).pi05
    assert (
        output / "actor/model_state_dict/full_weights.pt"
    ).read_bytes() == weights.read_bytes()
    marker = json.loads((output / "_SUCCESS").read_text())
    assert "training_fingerprint" not in marker
    assert (
        marker["inference_fingerprint"]["provenance"]["source"]
        == "immutable-source-revision"
    )
    with pytest.raises(ValueError, match="cannot resume"):
        validate_checkpoint_completion(output)
    with pytest.raises(FileExistsError):
        export_checkpoint(
            weights=weights,
            norm_stats=norm,
            tokenizer=tokenizer,
            config=config,
            task="insert_hole",
            output=output,
        )
    (output / "norm_stats.json").write_bytes(b"damaged payload")
    with pytest.raises(ValueError):
        validate_bundle(output, "insert_hole")


def test_default_loader_does_not_require_deleted_world_model_options(monkeypatch):
    from rlinf.data.datasets.openpi_pytorch.univtac import (
        univtac_sft_data_loader as loader,
    )

    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/sft"))
    with initialize_config_dir(
        version_base="1.1", config_dir=str(ROOT / "examples/sft/config")
    ):
        cfg = compose(
            config_name="simpletac_univtac",
            overrides=[
                "actor.model.openpi.tactile_expert_future_loss_weight=0",
                "data.num_workers=0",
            ],
        )
    assert "tactile_world_pool_grid" not in cfg.actor.model.openpi

    class ReachedDataset(Exception):
        pass

    dataset_signature = inspect.signature(loader.UniVTACHDF5Dataset)

    def dataset(**kwargs):
        dataset_signature.bind(**kwargs)
        assert kwargs["tactile_world_pool_grid"] == 14
        raise ReachedDataset()

    monkeypatch.setattr(loader, "UniVTACHDF5Dataset", dataset)
    with pytest.raises(ReachedDataset):
        loader.build_univtac_sft_dataloader(cfg, 1, 0, "/not-read")


def test_installer_only_references_shipped_requirements():
    script = (ROOT / "requirements/install.sh").read_text()
    assert "embodied/models/openpi.txt" not in script
    assert "download_assets.sh" not in script
    assert (ROOT / "requirements/policy.txt").is_file()
    assert "uv pip check" in script
    assert "override-dependencies" not in (ROOT / "pyproject.toml").read_text()
