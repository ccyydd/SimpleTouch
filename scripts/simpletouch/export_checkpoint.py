#!/usr/bin/env python3
"""Package unchanged native weights with explicit current-release inference metadata.

This is an export, not a training resume or a legacy configuration migration.
The caller supplies a current model configuration; obsolete options are rejected.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from omegaconf import OmegaConf

from scripts.simpletouch.checkpoint import (
    INFERENCE_FINGERPRINT_SCHEMA_VERSION,
    INFERENCE_REQUIRED_PATHS,
    SUCCESS_MARKER_SCHEMA_VERSION,
    canonical_json_digest,
    checkpoint_step,
    sha256_file,
    validate_checkpoint_completion,
)
from scripts.univtac.serve_policy import (
    _current_model_config,
    _validate_simpletac_contract,
)
from rlinf.data.datasets.openpi_pytorch.univtac import UNIVTAC_CAMERA_TYPES
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_preprocess_profile import (
    resolve_t3_preprocess_profile,
)


def export_checkpoint(
    *,
    weights: Path,
    config: Path,
    norm_stats: Path,
    tokenizer: Path,
    task: str,
    output: Path,
    source: str = "",
) -> Path:
    """Write a new inference-only bundle; never modify the source files."""
    output = output.expanduser().absolute()
    step = checkpoint_step(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for path in (weights, config, norm_stats, tokenizer):
        if not path.is_file():
            raise FileNotFoundError(path)
    loaded = OmegaConf.load(config)
    configured_task = OmegaConf.select(
        loaded, "data.task", default=OmegaConf.select(loaded, "task")
    )
    if configured_task is not None and configured_task != task:
        raise ValueError(f"Config task {configured_task!r} differs from {task!r}")
    model = _current_model_config(loaded)
    if not model.tactile_expert_enabled or not model.tactile_encoder_frozen:
        raise ValueError("Released checkpoints require an enabled TE and frozen T3")
    if model.tactile_preprocess_task != task:
        raise ValueError(
            "The configured T3 preprocessing task must match the export task"
        )
    profile = resolve_t3_preprocess_profile(
        model.tactile_preprocess_profile, task=task
    ).as_fingerprint()
    norm_digest, tokenizer_digest = sha256_file(norm_stats), sha256_file(tokenizer)
    values = dataclasses.asdict(model)
    # These assets are supplied by the user at deployment, not looked up at
    # paths from the training machine. Their identities stay bound by SHA-256.
    values.update(
        tactile_encoder_path="",
        defer_pretrained_weight_init=False,
        task=task,
        camera_type=UNIVTAC_CAMERA_TYPES[task],
        normalization="quantile",
        norm_stats_path="norm_stats.json",
        norm_stats_sha256=norm_digest,
        tokenizer_sha256=tokenizer_digest,
        env_action_dim=8,
        tactile_preprocess=profile,
    )
    _validate_simpletac_contract(OmegaConf.create(values))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.export-", dir=output.parent)
    )
    try:
        target = staging / "actor/model_state_dict/full_weights.pt"
        target.parent.mkdir(parents=True)
        shutil.copyfile(weights, target)
        if target.stat().st_size != weights.stat().st_size:
            raise OSError("Weight copy size changed during export")
        shutil.copyfile(norm_stats, staging / "norm_stats.json")
        OmegaConf.save(OmegaConf.create(values), staging / "config.yaml")
        fingerprint = {
            "schema_version": INFERENCE_FINGERPRINT_SCHEMA_VERSION,
            "data": {
                "task": task,
                "norm_stats_sha256": norm_digest,
                "tokenizer_sha256": tokenizer_digest,
            },
            "model": {"openpi": {"tactile_preprocess": profile}},
            "provenance": {
                "source": source,
                "exported_with": "simpletouch_univtac_release_v1",
            },
        }
        manifest = {}
        for relative in sorted(INFERENCE_REQUIRED_PATHS):
            path = staging / relative
            manifest[relative] = {"size_bytes": path.stat().st_size}
            if relative != "actor/model_state_dict/full_weights.pt":
                manifest[relative]["sha256"] = sha256_file(path)
        marker = {
            "schema_version": SUCCESS_MARKER_SCHEMA_VERSION,
            "bundle_type": "inference",
            "step": step,
            "inference_fingerprint": fingerprint,
            "inference_fingerprint_sha256": canonical_json_digest(fingerprint),
            "required_files": manifest,
        }
        (staging / "_SUCCESS").write_text(json.dumps(marker, indent=2) + "\n")
        # The complete bundle appears only after the weights and metadata exist.
        staging.chmod(0o755)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_checkpoint_completion(output, required_paths=set(INFERENCE_REQUIRED_PATHS))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "config", "norm-stats", "tokenizer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--task", required=True, choices=tuple(UNIVTAC_CAMERA_TYPES))
    parser.add_argument(
        "--source",
        default="",
        help="Original checkpoint or immutable repository revision",
    )
    print(export_checkpoint(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
