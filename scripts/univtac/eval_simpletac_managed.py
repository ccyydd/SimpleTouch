"""Multi-GPU, fixed-seed evaluation and recoverable in-place resumption.

Use the SimpleTac policy Python. Each worker owns one GPU and one private output
directory. Only the coordinator publishes canonical results after workers exit.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.univtac.managed_eval_results import (
    MANIFEST,
    consolidate,
    fingerprint,
    read_records,
    seed_shards,
    validate_prefix,
    write_json,
)
from scripts.univtac.reference_runtime import reference_root, verify_reference

LOGGER = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]
UNIVTAC = (
    Path(os.environ["UNIVTAC_DIR"]).expanduser().resolve()
    if os.environ.get("UNIVTAC_DIR")
    else None
)


def parser() -> argparse.ArgumentParser:
    """Build the explicit CLI; unset resume options inherit the saved contract."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--task")
    p.add_argument(
        "--gpus", default="0", help="Distinct physical GPU indices, e.g. 0,1,2,3"
    )
    p.add_argument(
        "--episodes", type=int, help="Total final episode count, not additional count"
    )
    p.add_argument(
        "--start-episode",
        "--start-seed",
        dest="start_seed",
        type=int,
        help="First episode ID/seed; default 1000000",
    )
    p.add_argument(
        "--result-dir",
        type=Path,
        help="New canonical result directory (must not exist)",
    )
    p.add_argument("--resume-dir", type=Path)
    p.add_argument(
        "--resume-from-episode",
        type=int,
        help="Inclusive episode ID/seed, e.g. 1000050",
    )
    p.add_argument(
        "--confirm-legacy-resume",
        action="store_true",
        help="Confirm supplied checkpoint/settings belong to an old directory without a manifest",
    )
    p.add_argument(
        "--horizon",
        type=int,
        help="Execution steps per query: IH/Insert Tube default 25, others 50; resume inherits its saved value",
    )
    p.add_argument("--task-config")
    p.add_argument("--use-tactile", choices=["auto", "true", "false"])
    p.add_argument("--cpu-threads", type=int)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/print a plan without launching or writing",
    )
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="Recover last interrupted merge without rerunning episodes",
    )
    return p


def small_hash(path: Path) -> str:
    """Hash only small identity/configuration files, never the large checkpoint."""
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError(f"Unexpectedly large identity file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_plan(args: argparse.Namespace) -> tuple[Path, dict, int, list[int]]:
    """Resolve resume semantics and reject conflicting checkpoint/settings."""
    if args.result_dir and args.resume_dir:
        raise ValueError("Use --result-dir or --resume-dir, not both")
    if args.resume_from_episode is not None and not args.resume_dir:
        raise ValueError("--resume-from-episode requires --resume-dir")
    gpu_parts = args.gpus.split(",")
    if not all(x.strip().isdigit() for x in gpu_parts):
        raise ValueError("--gpus must contain physical GPU integers")
    gpus = [int(x) for x in gpu_parts]
    if len(set(gpus)) != len(gpus):
        raise ValueError("Duplicate GPU indices")
    saved = {}
    root = args.resume_dir.resolve() if args.resume_dir else None
    if root:
        if not root.is_dir() or root == Path(root.anchor):
            raise ValueError("Resume target must be an existing evaluation directory")
        if (root / MANIFEST).exists():
            saved = json.loads((root / MANIFEST).read_text())["contract"]
        elif not args.confirm_legacy_resume or not args.checkpoint:
            raise ValueError(
                "Legacy resume requires --checkpoint and --confirm-legacy-resume; identity is not recorded in old episode metadata"
            )
        if args.resume_from_episode is None and not args.merge_only:
            raise ValueError(
                "Choose an explicit --resume-from-episode ID (e.g. 1000050)"
            )
    checkpoint = args.checkpoint or saved.get("checkpoint")
    task = args.task or saved.get("task") or (root.parent.name if root else None)
    if not checkpoint or not task or not task.replace("_", "").isalnum():
        raise ValueError(
            "A valid --checkpoint and --task are required for a new evaluation"
        )
    checkpoint = Path(checkpoint).resolve()
    if UNIVTAC is None or not (UNIVTAC / "scripts/eval_policy.py").is_file():
        raise ValueError("Set UNIVTAC_DIR to a valid UniVTAC checkout")
    environment = os.environ.get(
        "UNIVTAC_ENVIRONMENT", saved.get("environment_profile", "reference")
    )
    if environment not in {"reference", "public"}:
        raise ValueError("UNIVTAC_ENVIRONMENT must be reference or public")
    evaluation_source = reference_root() if environment == "reference" else UNIVTAC
    if environment == "reference":
        verify_reference(evaluation_source)
    values = {
        "checkpoint": str(checkpoint),
        "task": task,
        "start_seed": args.start_seed
        if args.start_seed is not None
        else saved.get("start_seed", 1000000),
        "episodes": args.episodes
        if args.episodes is not None
        else saved.get("episodes", 100),
    }
    if str(saved.get("temporal_ensemble", "false")).lower() not in {"false", "0", "no"}:
        raise ValueError("Cannot resume a temporal-ensemble evaluation")
    defaults = {
        "horizon": 25 if task in {"insert_hole", "insert_tube"} else 50,
        "temporal_ensemble": "false",
        "ensemble_k": 0.01,
        "task_config": "demo",
        "use_tactile": "auto",
        "tactile_use_all_tokens": "auto",
        "cpu_threads": 16,
    }
    for key, default in defaults.items():
        value = getattr(args, key, None)
        values[key] = saved.get(key, default) if value is None else value
    if values["horizon"] < 1 or values["cpu_threads"] < 1:
        raise ValueError("Invalid horizon/thread setting")
    values["checkpoint_config_sha256"] = small_hash(checkpoint / "config.yaml")
    values["completion_marker_sha256"] = small_hash(checkpoint / "_SUCCESS")
    if environment == "reference":
        values["environment_profile"] = environment
        values["reference_manifest_sha256"] = small_hash(evaluation_source / "manifest.json")
    config = Path(values["task_config"])
    if not config.is_absolute():
        config = (
            evaluation_source
            / "task_config"
            / (config.name if config.suffix else config.name + ".yml")
        )
    values["task_config_sha256"] = small_hash(config)
    # Keep simulation/sensor code and deployed policy plugin identical on resume.
    values["evaluation_code_sha256"] = {
        str(p): small_hash(p)
        for p in [
            evaluation_source / "scripts/eval_policy.py",
            evaluation_source / "envs/_base_task.py",
            evaluation_source / f"envs/{task}.py",
            REPO / "examples/UniVTAC/plugin/policy/SimpleTac/deploy_policy.py",
        ]
    }
    values["livestream"] = os.environ.get(
        "UNIVTAC_LIVESTREAM", saved.get("livestream", "2")
    )
    if environment == "reference" and values["livestream"] != "2":
        raise ValueError("Reference evaluation requires UNIVTAC_LIVESTREAM=2")
    values["policy_noise"] = "legacy_stochastic"
    if saved and values != saved:
        changed = [k for k in set(saved) | set(values) if saved.get(k) != values.get(k)]
        raise ValueError(f"Resume would change evaluation contract: {changed}")
    boundary = (
        args.resume_from_episode
        if args.resume_from_episode is not None
        else values["start_seed"]
    )
    seed_shards(values["start_seed"], values["episodes"], boundary, len(gpus))
    if root:
        records = read_records(root)
        if not saved and not records:
            raise ValueError("Legacy directory has no readable per-seed metadata")
        validate_prefix(records, values["start_seed"], boundary)
        expected = set(
            range(values["start_seed"], values["start_seed"] + values["episodes"])
        )
        if set(records) - expected:
            raise ValueError(
                "Old run includes additional seeds; explicitly set --start-seed/--episodes to include them"
            )
        if not saved and root.parent.name != task:
            raise ValueError("Resume directory task does not match --task")
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f")
        if args.result_dir:
            root = args.result_dir.resolve()
        else:
            eval_root = Path(
                os.environ.get("SIMPLETOUCH_EVAL_ROOT", REPO / "results/UniVTAC")
            ).expanduser()
            if not eval_root.is_absolute() or eval_root == Path(eval_root.anchor):
                raise ValueError(
                    "SIMPLETOUCH_EVAL_ROOT must be an absolute, non-root path"
                )
            root = (eval_root / "SimpleTac" / task / f"{stamp}_managed").resolve()
        if root.exists():
            raise FileExistsError(
                "New result directory already exists; use explicit resume"
            )
    return root, values, boundary, gpus


def active_process_check(root: Path, checkpoint: str) -> None:
    """Reject active local writers/duplicate checkpoint servers, never stop them."""
    if (root / MANIFEST).exists():
        state = json.loads((root / MANIFEST).read_text())
        if state.get("status") in {"running", "cleanup_required"}:
            plan = json.loads(
                (Path(state["last_attempt"]) / "attempt.json").read_text()
            )
            if plan["host"] != socket.gethostname():
                raise RuntimeError(
                    f"Recover this unfinished attempt on its originating host: {plan['host']}"
                )
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            command = (
                (proc / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
            if "serve_policy.py" in command and checkpoint in command:
                raise RuntimeError(
                    f"Checkpoint already has a policy server (PID {proc.name})"
                )
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith(str(root) + "/"):
                    # Read-only inspectors are harmless; check write access bits.
                    info = (proc / "fdinfo" / fd.name).read_text()
                    flags = next(
                        x.split()[1]
                        for x in info.splitlines()
                        if x.startswith("flags:")
                    )
                    if int(flags, 8) & 3 and int(proc.name) != os.getpid():
                        raise RuntimeError(
                            f"Result directory has an active writer (PID {proc.name})"
                        )
        except (PermissionError, FileNotFoundError, ProcessLookupError, StopIteration):
            continue


def reserve_gpus(gpus: list[int], stack: contextlib.ExitStack) -> dict[int, int]:
    """Reserve only idle GPUs; inherited locks prevent a preflight/launch race."""
    descriptors = {}
    for gpu in gpus:
        lock = stack.enter_context(open(f"/tmp/simpletac_eval_gpu_{gpu}.lock", "a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        usage = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        memory, utilization = map(int, usage.strip().split(","))
        if memory >= 100 or utilization >= 10:
            raise RuntimeError(f"GPU {gpu} is not idle: {usage.strip()}")
        for port in [28000 + gpu, 18000 + gpu]:
            with socket.socket() as sock:
                sock.bind(("0.0.0.0", port))
        descriptors[gpu] = lock.fileno()
    return descriptors


def worker_env(
    contract: dict, worker: dict, plan_file: Path, fd: int
) -> dict[str, str]:
    """Build an explicit environment, without inheriting stale per-run overrides."""
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("SIMPLETAC_"):
            env.pop(key)
    for key in [
        "CUDA_VISIBLE_DEVICES",
        "NORM_STATS",
        "EVAL_GPU_LOCK_FILE",
        "SIMPLETAC_MANAGED_PLAN",
        "SIMPLETAC_RESERVED_GPU_FD",
        "GPU_IDS",
        "RESUME_EVAL_DIR",
        "UNIVTAC_REFERENCE_READY",
        "UNIVTAC_EVAL_DRIVER",
    ]:
        env.pop(key, None)
    gpu = worker["gpu"]
    env.update(
        CKPT_PATH=contract["checkpoint"],
        TASK=contract["task"],
        TASK_CONFIG=contract["task_config"],
        GPU_ID=str(gpu),
        POLICY_GPU_ID=str(gpu),
        SIM_GPU_ID=str(gpu),
        PORT=str(28000 + gpu),
        ISAAC_HTTP_PORT=str(18000 + gpu),
        LOG_DIR=worker["log_dir"],
        EXECUTION_HORIZON=str(contract["horizon"]),
        TOTAL_NUM=str(len(worker["seeds"])),
        ISAAC_CPU_THREADS=str(contract["cpu_threads"]),
        USE_TACTILE=contract["use_tactile"],
        UNIVTAC_LIVESTREAM=contract["livestream"],
        UNIVTAC_ENVIRONMENT=contract.get("environment_profile", "public"),
        UNIVTAC_EVAL_DRIVER="simpletouch",
        SIMPLETAC_MANAGED_PLAN=str(plan_file),
        SIMPLETAC_RESERVED_GPU_FD=str(fd),
        FIX_CHECKPOINT_PERMISSIONS="false",
        ALLOW_LEGACY_CHECKPOINT_WITHOUT_SUCCESS_MARKER="false",
        ALLOW_MISSING_TASK_METADATA="false",
        VULKAN_PREFLIGHT="true",
    )
    return env


def execute(args: argparse.Namespace) -> dict:
    """Launch once, wait for children, and merge even when a worker fails."""
    if args.merge_only:
        if not args.resume_dir:
            raise ValueError("--merge-only requires --resume-dir")
        root = args.resume_dir.resolve()
        saved = json.loads((root / MANIFEST).read_text())
        attempt = Path(saved["last_attempt"])
        if attempt.parent != root / ".eval_runs":
            raise ValueError("Invalid recorded attempt path")
        with open(root / ".managed_eval.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            active_process_check(root, saved["contract"]["checkpoint"])
            return consolidate(root, attempt)
    root, contract, boundary, gpus = resolve_plan(args)
    shards = seed_shards(
        contract["start_seed"], contract["episodes"], boundary, len(gpus)
    )
    gpus = gpus[: len(shards)]
    if args.dry_run:
        return {
            "result_dir": str(root),
            "contract": contract,
            "resume_from": boundary,
            "shards": [
                {"gpu": gpu, "seeds": seeds} for gpu, seeds in zip(gpus, shards)
            ],
        }
    sys.path.insert(0, str(REPO))
    from scripts.simpletouch.checkpoint import (
        INFERENCE_REQUIRED_PATHS,
        validate_checkpoint_completion,
    )

    validate_checkpoint_completion(
        Path(contract["checkpoint"]),
        required_paths=set(INFERENCE_REQUIRED_PATHS),
        verify_mode="full",
    )
    for dependency in ["ffmpeg", "nvidia-smi"]:
        import shutil

        if not shutil.which(dependency):
            raise RuntimeError(f"Required executable missing: {dependency}")
    from scripts.univtac.eval_policy_managed import adapt_source

    evaluation_source = reference_root() if contract.get("environment_profile") == "reference" else UNIVTAC
    adapt_source((evaluation_source / "scripts/eval_policy.py").read_text())
    active_process_check(root, contract["checkpoint"])
    with contextlib.ExitStack() as stack:
        gpu_fds = reserve_gpus(gpus, stack)
        root.mkdir(parents=True, exist_ok=bool(args.resume_dir))
        lock = stack.enter_context(open(root / ".managed_eval.lock", "a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Repeat after taking the shared-directory lock.
        active_process_check(root, contract["checkpoint"])
        attempt = (
            root
            / ".eval_runs"
            / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        )
        attempt.mkdir(parents=True)
        workers = [
            {
                "gpu": gpu,
                "seeds": seeds,
                "worker_name": f"worker_{index}",
                "result_root": str(attempt / "workers" / f"worker_{index}"),
                "log_dir": str(attempt / "logs" / f"worker_{index}"),
            }
            for index, (gpu, seeds) in enumerate(zip(gpus, shards))
        ]
        plan = {
            "contract": contract,
            "resume_from": boundary,
            "workers": workers,
            "before_fingerprint": fingerprint(root),
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        write_json(attempt / "attempt.json", plan)
        write_json(
            root / MANIFEST,
            {"contract": contract, "status": "running", "last_attempt": str(attempt)},
        )
        LOGGER.info("Unified result directory: %s", root)
        children = []
        interrupted = False
        previous_handler = signal.signal(
            signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
        )
        try:
            for index, worker in enumerate(workers):
                plan_file = attempt / f"worker_{index}.json"
                write_json(plan_file, worker)
                Path(worker["log_dir"]).mkdir(parents=True)
                output = stack.enter_context(
                    open(Path(worker["log_dir"]) / "launcher.log", "x")
                )
                child = subprocess.Popen(
                    ["bash", str(REPO / "scripts/univtac/run_eval_simpletac.sh")],
                    cwd=REPO,
                    env=worker_env(contract, worker, plan_file, gpu_fds[worker["gpu"]]),
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(gpu_fds[worker["gpu"]],),
                )
                children.append(child)
                worker["launcher_pid"] = child.pid
                write_json(attempt / "attempt.json", plan)
                LOGGER.info(
                    "GPU%s: %s episodes, launcher PID%s",
                    worker["gpu"],
                    len(worker["seeds"]),
                    child.pid,
                )
            while any(child.poll() is None for child in children):
                time.sleep(1)
        except (KeyboardInterrupt, Exception):
            interrupted = True
            LOGGER.exception(
                "Managed launch interrupted/failed; stopping only its own workers"
            )
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            for child in children:
                if child.poll() is None:
                    child.terminate()  # wrapper owns and cleans up policy + simulation
            for child in children:
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    # Do not merge files while a writer might remain alive.
                    write_json(
                        root / MANIFEST,
                        {
                            "contract": contract,
                            "status": "cleanup_required",
                            "last_attempt": str(attempt),
                        },
                    )
                    raise RuntimeError(
                        f"Worker {child.pid} has not exited; merge deferred"
                    )
        plan["worker_exit_codes"] = [child.returncode for child in children]
        plan["interrupted"] = interrupted
        write_json(attempt / "attempt.json", plan)
        summary = consolidate(root, attempt)
        summary["worker_exit_codes"] = plan["worker_exit_codes"]
        return summary


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parser().parse_args()
    try:
        result = execute(args)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.dry_run and (
        result.get("status") != "complete" or any(result.get("worker_exit_codes", []))
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
