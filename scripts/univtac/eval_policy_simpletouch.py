"""Run UniVTAC evaluation while keeping results outside its checkout."""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path


def adapt_source(source: str) -> str:
    """Redirect only the output root; fail if the upstream interface changes."""
    layouts = {
        "    run_root = Path('eval_result') / policy_name / task_file_name / run_id": "    run_root = _simpletouch_eval_root / policy_name / task_file_name / run_id",
        "    env_cfg.save_dir = Path('eval_result') / policy_name / task_file_name / deploy_config_file.stem / curr_time": "    env_cfg.save_dir = _simpletouch_eval_root / policy_name / task_file_name / _simpletouch_run_id",
    }
    counts = [source.splitlines().count(old) for old in layouts]
    if sum(counts) != 1 or any(
        source.count(old) != count for old, count in zip(layouts, counts)
    ):
        raise RuntimeError("Unsupported UniVTAC evaluator output layout")
    for old, new in layouts.items():
        if old in source:
            return source.replace(old, new)
    raise AssertionError("No supported evaluator layout")


def main() -> None:
    """Execute the installed evaluator without changing its evaluation loop."""
    root = Path(os.environ["SIMPLETOUCH_EVAL_ROOT"]).expanduser()
    if not root.is_absolute() or root == Path(root.anchor):
        raise ValueError("SIMPLETOUCH_EVAL_ROOT must be an absolute, non-root path")
    upstream = Path.cwd() / "scripts/eval_policy.py"
    source = adapt_source(upstream.read_text())
    sys.path.insert(0, str(upstream.parent))
    namespace = {
        "__name__": "_univtac_simpletouch_driver",
        "__file__": str(upstream),
        "_simpletouch_eval_root": root.resolve(),
        "_simpletouch_run_id": datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        + "_"
        + uuid.uuid4().hex[:8],
    }
    exec(compile(source, str(upstream), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
