#!/usr/bin/env python3
"""Download an inference-only SimpleTouch checkpoint from Hugging Face."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.simpletouch.checkpoint import (  # noqa: E402
    INFERENCE_REQUIRED_PATHS,
    completion_metadata,
    validate_checkpoint_completion,
)

TASKS = (
    "lift_bottle",
    "pull_out_key",
    "lift_can",
    "put_bottle_in_shelf",
    "insert_hole",
    "insert_tube",
)
DEFAULT_REPO_ID = "cyang203912/SimpleTouch"
STEP_DIR = "global_step_20000"


def validate_bundle(checkpoint: Path, task: str) -> None:
    """Check the published payload and its original task provenance."""
    marker = validate_checkpoint_completion(
        checkpoint,
        required_paths=set(INFERENCE_REQUIRED_PATHS),
        verify_mode="full",
    )
    recorded_task = completion_metadata(marker).get("data", {}).get("task")
    if recorded_task != task:
        raise ValueError(
            f"Checkpoint task mismatch: expected {task!r}, marker has {recorded_task!r}"
        )


def download_checkpoint(
    task: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    output_root: Path = REPO / "checkpoints",
    revision: str | None = None,
) -> Path:
    """Return a verified checkpoint, using the local copy when complete."""
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}")
    output_root = output_root.expanduser().resolve()
    checkpoint = output_root / task / STEP_DIR
    try:
        if revision is not None:
            raise FileNotFoundError("An explicit revision requires a fresh download")
        validate_bundle(checkpoint, task)
    except (FileNotFoundError, ValueError):
        from huggingface_hub import snapshot_download

        prefix = f"{task}/{STEP_DIR}/"
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=output_root,
            # Include the Hub's standard query file with real bundle downloads.
            # Complete local bundles return above without making Hub requests.
            allow_patterns=["config.json"] + [
                prefix + name
                for name in (*sorted(INFERENCE_REQUIRED_PATHS), "_SUCCESS")
            ],
        )
        validate_bundle(checkpoint, task)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("SIMPLETOUCH_HF_REPO", DEFAULT_REPO_ID),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get("SIMPLETOUCH_CHECKPOINT_ROOT", REPO / "checkpoints")
        ),
    )
    parser.add_argument("--revision")
    args = parser.parse_args()
    print(
        download_checkpoint(
            args.task,
            repo_id=args.repo_id,
            output_root=args.output_root,
            revision=args.revision,
        )
    )


if __name__ == "__main__":
    main()
