"""Execute the reference native CLI, adapting only output paths.

The native eval_policy function and seed-range CLI are unchanged. Simulator
outputs stay in durable .eval_runs directories. The visible video/worker_0
directory contains relative links to those originals, with merged metadata.
"""
from __future__ import annotations

from datetime import datetime
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid

if __package__:
    from .eval_policy_managed import adapt_source as route_legacy_output
    from .managed_eval_results import consolidate, fingerprint, write_json
else:
    from eval_policy_managed import adapt_source as route_legacy_output
    from managed_eval_results import consolidate, fingerprint, write_json


def adapt_source(source):
    source = route_legacy_output(source)
    replacements = {
        '    log_path = run_root / "log.log"':
            '    log_path = run_root / "logs" / f"{_managed_worker_name}.log"',
        '    write_json(run_root / "metadata.json", summary)':
            '    write_json(run_root / "workers" / _managed_worker_name / "native-summary.json", summary)',
        '    policy_init_cost = time.perf_counter() - init_start':
            '    policy_init_cost = time.perf_counter() - init_start\n'
            '    _native_runtime_metadata.update(horizon=policy.execution_horizon, '
            'use_tactile=str(policy.use_tactile).lower())',
    }
    for old, new in replacements.items():
        if source.splitlines().count(old) != 1 or source.count(old) != 1:
            raise RuntimeError(f"Unsupported native output assignment: {old}")
        source = source.replace(old, new)
    return source


def native_contract(namespace, upstream):
    """Describe the actual native CLI settings without changing its seed loop."""
    cli = namespace["args_cli"]
    config, config_path = namespace["get_config"](
        cli.deploy_config, default_root=upstream.parent.parent / "policy", type="yaml")
    _, task_path = namespace["get_config"](
        cli.task_config, default_root=upstream.parent.parent / "task_config", type="yaml")
    start = cli.start_seed if cli.start_seed != -1 else 1000000 * (1 + config.get("seed", 0))
    total = cli.total_num
    if cli.max_seed != -1:
        total = min(total, max(0, cli.max_seed - start + 1))
    if total < 1:
        raise ValueError("Native evaluation needs a nonempty seed range")
    setting = os.environ.get("SIMPLETAC_EXECUTION_HORIZON", str(config.get("execution_horizon", "task")))
    runtime = namespace.get("_native_runtime_metadata", {})
    horizon = runtime.get("horizon", int(setting) if setting.isdigit() else setting)
    checkpoint = os.environ.get("CKPT_PATH", "server:" + os.environ.get("SIMPLETAC_HOST", "127.0.0.1")
                                + ":" + os.environ.get("SIMPLETAC_PORT", "8000"))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    contract = {
        "checkpoint": checkpoint, "task": cli.task_name,
        "start_seed": start, "episodes": total, "horizon": horizon,
        "temporal_ensemble": str(os.environ.get("SIMPLETAC_TEMPORAL_ENSEMBLE", config.get("temporal_ensemble", False))).lower(),
        "ensemble_k": float(os.environ.get("SIMPLETAC_ENSEMBLE_K", config.get("ensemble_k", 0.01))),
        "task_config": str(cli.task_config),
        "use_tactile": runtime.get("use_tactile", os.environ.get("SIMPLETAC_USE_TACTILE", str(config.get("use_tactile", "auto")))),
        "tactile_use_all_tokens": os.environ.get("SIMPLETAC_TACTILE_USE_ALL_TOKENS", str(config.get("tactile_use_all_tokens", "auto"))),
        "cpu_threads": int(os.environ.get("ISAAC_CPU_THREADS", "16")),
        "livestream": os.environ.get("UNIVTAC_LIVESTREAM", "2"),
        "policy_noise": os.environ.get("SIMPLETAC_POLICY_NOISE", config.get("policy_noise", "legacy_stochastic")),
        "environment_profile": os.environ.get("UNIVTAC_ENVIRONMENT", "reference"),
        "task_config_sha256": sha(task_path),
        "evaluation_code_sha256": {str(upstream): sha(upstream)},
        "deploy_config_sha256": sha(config_path),
    }
    for key, name in [("checkpoint_config_sha256", "config.yaml"),
                      ("completion_marker_sha256", "_SUCCESS")]:
        path = Path(checkpoint) / name
        if path.is_file():
            contract[key] = sha(path)
    return contract


def main():
    if "--expert_check" in sys.argv:
        raise ValueError("Parallel fixed-seed evaluation does not skip expert-check seeds")
    root = os.environ.get("SIMPLETAC_NATIVE_RESULT_ROOT")
    if root is None:
        run = os.environ.get("SIMPLETOUCH_EVAL_RUN_ID") or (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_" + uuid.uuid4().hex[:8])
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", run):
            raise ValueError("Run ID must be one path component")
        root = Path(os.environ["SIMPLETOUCH_EVAL_ROOT"]) / "SimpleTac" / sys.argv[1] / run
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("Result root must be absolute")
    worker = os.environ.get("SIMPLETAC_WORKER_NAME", "worker_0")
    if root == Path(root.anchor) or not re.fullmatch(r"worker_[A-Za-z0-9_]+", worker):
        raise ValueError("Invalid output root or worker name")
    if root.is_symlink() or any((root / name).is_symlink() for name in ["video", "scene", "metadata"]):
        raise ValueError("Result directories must be ordinary directories")
    crosshost = os.environ.get("SIMPLETAC_CROSSHOST") == "true"
    final_root = root
    attempt = None
    before = None
    with contextlib.ExitStack() as resources:
        if not crosshost:
            # Keep raw videos and scene diagnostics out of the public result layout.
            if any((root / name).exists() for name in ["video", "metadata", "metadata.json", "log.log"]):
                raise FileExistsError(f"Refusing to replace existing evaluation results: {root}")
            root.mkdir(parents=True, exist_ok=True)
            lock = resources.enter_context(open(root / ".managed_eval.lock", "a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if any((root / name).exists() for name in ["video", "metadata", "metadata.json", "log.log"]):
                raise FileExistsError(f"Refusing to replace existing evaluation results: {root}")
            before = fingerprint(root)
            attempt = root / ".eval_runs" / ("native_" + uuid.uuid4().hex)
            root = attempt / "workers" / worker
        if (root / "metadata" / (worker + ".json")).exists():
            raise FileExistsError(f"Refusing to reuse completed worker output: {root}/{worker}")
        upstream = Path.cwd() / "scripts/eval_policy.py"
        sys.path.insert(0, str(upstream.parent))
        namespace = {"__name__": "_univtac_native_cli", "__file__": str(upstream),
                     "_managed_result_root": root, "_managed_worker_name": worker,
                     "_native_runtime_metadata": {}}
        exec(compile(adapt_source(upstream.read_text()), str(upstream), "exec"), namespace)
        if not crosshost:
            contract = native_contract(namespace, upstream)
            write_json(attempt / "attempt.json", {
                "contract": contract, "resume_from": contract["start_seed"],
                "before_fingerprint": before,
                "workers": [{"worker_name": worker, "result_root": str(root),
                             "seeds": list(range(contract["start_seed"], contract["start_seed"] + contract["episodes"]))}],
            })
        namespace["main"]()
        if not crosshost:
            plan = json.loads((attempt / "attempt.json").read_text())
            plan["contract"] = native_contract(namespace, upstream)
            write_json(attempt / "attempt.json", plan)
            consolidate(final_root, attempt)


if __name__ == "__main__":
    main()
