"""Run the installed UniVTAC evaluator against a bounded, private seed shard.

The simulation/action loop stays in UniVTAC. Only output routing and the outer
seed schedule are adapted. Source anchors fail closed if that interface changes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def adapt_source(source: str) -> str:
    """Route either reviewed output layout without changing simulation code."""
    legacy = {
        "    run_id = build_run_id()": "    run_id = _managed_result_root.name",
        "    run_root = Path('eval_result') / policy_name / task_file_name / run_id": "    run_root = _managed_result_root",
        "    env_cfg.worker_name = worker_name(0)": "    env_cfg.worker_name = _managed_worker_name",
    }
    public_save = "    env_cfg.save_dir = Path('eval_result') / policy_name / task_file_name / deploy_config_file.stem / curr_time"
    public_seeds = "    seed_path = task.save_root.parent / 'seeds.json'"

    def exact_count(line: str) -> int:
        count = source.splitlines().count(line)
        # Reject changed expressions that merely start with a reviewed anchor.
        return count if source.count(line) == count else -1

    legacy_counts = [exact_count(old) for old in legacy]
    if exact_count(public_save) == 1 and not any(legacy_counts):
        # Public 0dafa102 BaseTask uses cfg.save_dir directly, without appending
        # a worker name. Its flat metadata/video layout is read by the merger.
        if exact_count(public_seeds) != 1:
            raise RuntimeError("Unsupported public UniVTAC seed-file layout")
        replacements = {
            public_save: "    env_cfg.save_dir = _managed_result_root",
            public_seeds: "    seed_path = task.save_root / 'seeds.json'",
        }
    elif exact_count(public_save) == 0 and legacy_counts == [1, 1, 1]:
        replacements = legacy
    else:
        raise RuntimeError("Unsupported or mixed UniVTAC evaluator output layout")
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def bounded_evaluator(original, seeds: list[int]):
    """Keep errors on their assigned seed; never spill into another shard."""

    def evaluate(**kwargs):
        success = 0
        if kwargs.get("expert_check"):
            raise ValueError(
                "Managed fixed-seed evaluation does not skip expert-check seeds"
            )
        for seed in seeds:
            result = original(
                **{
                    **kwargs,
                    "start_seed": seed,
                    "max_seed": seed,
                    "test_total_num": 1,
                }
            )
            success += result["succ_num"]
        return {"test_num": len(seeds), "succ_num": success}

    return evaluate


def main() -> None:
    """Load the installed driver, then replace only its seed scheduler."""
    plan = json.loads(Path(os.environ["SIMPLETAC_MANAGED_PLAN"]).read_text())
    seeds = plan["seeds"]
    if (
        not seeds
        or any(type(s) is not int or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Shard seeds must be distinct nonnegative integers")
    root = Path(plan["result_root"])
    if not root.is_absolute() or not plan["worker_name"].startswith("worker_"):
        raise ValueError("Invalid private shard output path/name")
    if any((root / name).exists() for name in ("log.log", "metadata", "metadata.json")):
        raise FileExistsError(f"Refusing to reuse worker output: {root}")
    os.environ["SIMPLETAC_WORKER_RESULT_DIR"] = str(root)
    os.environ["SIMPLETAC_WORKER_NAME"] = plan["worker_name"]
    upstream = Path.cwd() / "scripts/eval_policy.py"
    source = adapt_source(upstream.read_text())
    sys.path.insert(0, str(upstream.parent))
    namespace = {
        "__name__": "_univtac_managed_driver",
        "__file__": str(upstream),
        "_managed_result_root": root,
        "_managed_worker_name": plan["worker_name"],
    }
    exec(compile(source, str(upstream), "exec"), namespace)
    namespace["eval_policy"] = bounded_evaluator(namespace["eval_policy"], seeds)
    namespace["main"]()


if __name__ == "__main__":
    main()
