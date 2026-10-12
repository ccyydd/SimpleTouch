"""Planning and recoverable result consolidation for managed UniVTAC runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Callable

MANIFEST = ".simpletac_eval.json"
ARTIFACT = re.compile(r"^(\d+)(?:_(success|failed|error))?\.(mp4|hdf5)$")
OUTPUT_NAMES = ["video", "hdf5", "metadata", "log.log", "metadata.json", "scene", "logs", "workers"]


def archive_root(root: Path) -> Path:
    """Locate the durable attempt directory for canonical or staged outputs."""
    root = root.absolute()
    if ".eval_runs" in root.parts:
        return Path(*root.parts[:root.parts.index(".eval_runs") + 1])
    return root / ".eval_runs"


def artifact_target(path: Path, root: Path) -> Path:
    """Allow only relative artifact links to regular files in this run archive."""
    if path.is_symlink():
        if Path(os.readlink(path)).is_absolute():
            raise ValueError(f"Artifact link must be relative: {path}")
        try:
            target = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Broken artifact link: {path}") from exc
        if not target.is_relative_to(archive_root(root).resolve()):
            raise ValueError(f"Artifact link escapes .eval_runs: {path}")
        if not target.is_file() or target.stat().st_size == 0:
            raise ValueError(f"Artifact link target must be a nonempty file: {path}")
        return target
    return path


def publish_artifact(source: Path, target: Path, root: Path) -> None:
    """Atomically expose an artifact without creating another video copy."""
    source = artifact_target(source, root).resolve(strict=True)
    if (not source.is_relative_to(archive_root(root).resolve()) or not source.is_file()
            or source.stat().st_size == 0):
        raise ValueError(f"Artifact source must be a nonempty regular file in .eval_runs: {source}")
    if not target.absolute().is_relative_to(root.absolute()):
        raise ValueError(f"Artifact export must stay inside the result directory: {target}")
    for parent in [target.parent, *target.parent.parents]:
        if parent.is_symlink():
            raise ValueError(f"Artifact directory must be ordinary: {parent}")
        if parent == root.absolute():
            break
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise ValueError(f"Artifact directory must be ordinary: {target.parent}")
    # A killed process can leave its temporary link behind. Keep that scratch
    # entry outside visible artifact inventories so recovery never reads it.
    scratch = archive_root(root) / ".publication_tmp"
    if scratch.is_symlink():
        raise ValueError(f"Artifact scratch directory must be ordinary: {scratch}")
    scratch.mkdir(parents=True, exist_ok=True)
    temporary = scratch / f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        temporary.symlink_to(os.path.relpath(source, target.parent))
        os.replace(temporary, target)
    finally:
        if temporary.is_symlink():
            temporary.unlink()


def write_json(path: Path, data: object) -> None:
    """Publish small metadata atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def seed_shards(
    start_seed: int, episodes: int, resume_from: int, workers: int
) -> list[list[int]]:
    """Episode identifiers equal seeds (default 1000000), never ordinal indices."""
    if (
        start_seed < 0
        or episodes < 1
        or workers < 1
        or not start_seed <= resume_from < start_seed + episodes
    ):
        raise ValueError(
            "Invalid episode/worker range; episode IDs equal seeds, default 1000000"
        )
    seeds = list(range(resume_from, start_seed + episodes))
    count = min(workers, len(seeds))
    return [seeds[i::count] for i in range(count)]


def read_records(root: Path) -> dict[int, dict]:
    """Read reviewed flat/public or worker/canonical episode metadata.

    Public UniVTAC writes a seed map to metadata.json; the canonical merger
    uses that filename for its summary instead. Never interpret a summary as
    episode records, or silently combine two competing source layouts.
    """
    records = {}
    directory = root / "metadata"
    flat = root / "metadata.json"
    if directory.is_symlink() or flat.is_symlink():
        raise ValueError(f"Symlink not allowed in evaluation metadata: {root}")
    paths = sorted(directory.glob("worker_*.json"))
    payloads = []
    if flat.exists():
        payload = json.loads(flat.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid metadata: {flat}")
        if all(str(key).isdigit() for key in payload):
            if paths:
                raise ValueError(f"Mixed public and worker metadata layouts: {root}")
            payloads.append((flat, payload))
        elif any(str(key).isdigit() for key in payload):
            raise ValueError(f"Mixed summary and seed metadata: {flat}")
        elif (
            not paths
            and not {"policy_name", "run_root", "total_episodes"} <= payload.keys()
        ):
            raise ValueError(f"Unrecognized metadata layout: {flat}")
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Symlink not allowed in evaluation metadata: {path}")
        payloads.append((path, json.loads(path.read_text())))
    for path, payload in payloads:
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid metadata: {path}")
        for key, value in payload.items():
            if not str(key).isdigit() or not isinstance(value, dict):
                raise ValueError(f"Invalid seed record in {path}: {key}")
            seed = int(key)
            if seed in records:
                raise ValueError(f"Duplicate seed {seed} in {root}")
            if value.get("result") not in {"success", "failed", "error"}:
                raise ValueError(f"Unknown episode result for seed {seed}")
            records[seed] = dict(value)
    return records


def artifact_files(
    root: Path, kind: str, only_seeds: set[int] | None = None
) -> dict[int, Path]:
    """Inventory recognized seed artifacts, without traversing hidden attempts."""
    files = {}
    if (root / kind).is_symlink():
        raise ValueError(f"Symlink not allowed in evaluation artifacts: {root / kind}")
    for path in sorted((root / kind).rglob("*")):
        artifact_target(path, root)
        if not path.is_file():
            continue
        match = ARTIFACT.fullmatch(path.name)
        if not match or match[3] != {"video": "mp4", "hdf5": "hdf5"}[kind]:
            raise ValueError(
                f"Unrecognized artifact (refusing destructive merge): {path}"
            )
        seed = int(match[1])
        if only_seeds is not None and seed not in only_seeds:
            continue
        if seed in files:
            raise ValueError(f"Duplicate {kind} for seed {seed}: {root}")
        files[seed] = path
    return files


def fingerprint(root: Path) -> str:
    """Detect writes by a legacy/other-host evaluator before merging."""
    items = []
    for name in OUTPUT_NAMES:
        base = root / name
        if base.is_symlink():
            raise ValueError(f"Refusing symlink in result directory: {base}")
        paths = sorted(base.rglob("*")) if base.is_dir() else [base]
        for path in paths:
            if path.is_symlink():
                if name not in {"video", "hdf5"}:
                    raise ValueError(f"Refusing symlink in result directory: {path}")
                artifact_target(path, root)
            if path.is_file():
                stat = path.stat()
                items.append(
                    (str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns,
                     os.readlink(path) if path.is_symlink() else None)
                )
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def check_video(path: Path) -> tuple[bool, str]:
    """Fully decode, counting zero-frame/corrupt videos as errors, not failures."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-nostdin",
                "-threads",
                "1",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-progress",
                "pipe:1",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return False, "video_decode_timeout"
    frames = [int(x) for x in re.findall(r"^frame=(\d+)$", result.stdout, re.M)]
    ok = result.returncode == 0 and max(frames, default=0) > 0
    return ok, "" if ok else "zero_frame_or_decode_error"


def validate_prefix(records: dict[int, dict], start: int, resume_from: int) -> None:
    """Never silently retain a hole before the requested resume boundary."""
    missing = set(range(start, resume_from)) - records.keys()
    if missing:
        raise ValueError(
            f"Missing prefix episodes; resume earlier. Seeds: {sorted(missing)}"
        )


def consolidate(root: Path, attempt: Path, decode: Callable = check_video) -> dict:
    """Merge staged shards; backup old canonical outputs before any replacement.

    Restart with the same attempt after an interrupted merge: backup creation is
    journaled and all new files remain in private worker directories.
    """
    plan = json.loads((attempt / "attempt.json").read_text())
    contract = plan["contract"]
    start, total, boundary = (
        contract["start_seed"],
        contract["episodes"],
        plan["resume_from"],
    )
    expected = set(range(start, start + total))
    selected = set(range(boundary, start + total))
    journal = attempt / "merge.json"
    if not journal.exists():
        if fingerprint(root) != plan["before_fingerprint"]:
            raise RuntimeError(
                "Result directory changed while workers ran; refusing to overwrite it"
            )
        # Validate all directories before moving anything.
        before = read_records(root)
        if set(before) - expected:
            raise ValueError(
                "Existing seeds exceed requested range; increase --episodes"
            )
        validate_prefix(before, start, boundary)
        # Interrupted/duplicate tail clips will be archived, not retained.
        artifact_files(root, "video", set(before) - selected)
        artifact_files(root, "hdf5", set(before) - selected)
        assigned, observed = set(), set()
        for worker in plan["workers"]:
            seeds = set(worker["seeds"])
            if assigned & seeds:
                raise ValueError("Overlapping worker seed plans")
            assigned |= seeds
            output = Path(worker["result_root"])
            rows = read_records(output)
            if set(rows) - seeds or observed & rows.keys():
                raise ValueError("Worker emitted duplicate or unassigned seeds")
            observed |= rows.keys()
            artifact_files(output, "video")
            artifact_files(output, "hdf5")
        if assigned != selected:
            raise ValueError("Worker plan does not cover the selected episode range")
        # Moving the final directories changes relative link bases. Journal
        # resolved targets before the first rename so recovery can rebase them
        # even after a crash between moving a directory and fixing its links.
        links = {
            str(path.relative_to(root)): str(artifact_target(path, root))
            for kind in ["video", "hdf5"]
            for path in (root / kind).rglob("*") if path.is_symlink()
        }
        write_json(journal, {"phase": "backup", "links": links})
    state = json.loads(journal.read_text())
    if state["phase"] == "done":
        artifact_files(root, "video")
        artifact_files(root, "hdf5")
        return json.loads((root / "metadata.json").read_text())
    backup = attempt / "previous"
    backup.mkdir(exist_ok=True)
    if state["phase"] == "backup":
        for name in OUTPUT_NAMES:
            src, dst = root / name, backup / name
            if src.exists() and not dst.exists():
                os.replace(src, dst)
        for relative, source in state.get("links", {}).items():
            publish_artifact(Path(source), backup / relative, root)
        write_json(journal, {"phase": "publish"})
    # Rebuild from immutable originals + staged outputs, including after a crash.
    old = read_records(backup)
    records = {s: v for s, v in old.items() if s not in selected}
    origins = dict.fromkeys(records, backup)
    for worker in plan["workers"]:
        source = Path(worker["result_root"])
        for seed, meta in read_records(source).items():
            records[seed] = dict(meta)
            origins[seed] = source
    inventories = {
        p: (
            artifact_files(p, "video", {s for s in origins if origins[s] == p}),
            artifact_files(p, "hdf5", {s for s in origins if origins[s] == p}),
        )
        for p in set(origins.values())
    }
    # Persist classifications before publishing any video. A transient decode
    # timeout on recovery must not leave two differently labelled seed clips.
    prepared = attempt / "prepared_records.json"
    if prepared.exists():
        records = {int(s): m for s, m in json.loads(prepared.read_text()).items()}
        if set(records) != set(origins):
            raise RuntimeError("Worker outputs changed after merge preparation")
    else:
        for seed, meta in records.items():
            video = inventories[origins[seed]][0].get(seed)
            valid, reason = (
                decode(video) if video is not None else (False, "missing_video")
            )
            if not valid:
                meta["raw_result"] = meta.get("raw_result", meta["result"])
                meta["result"], meta["video_error"] = "error", reason
            meta["episode"] = seed
        write_json(prepared, {str(s): m for s, m in records.items()})
    for seed, meta in records.items():
        videos, hdf5 = inventories[origins[seed]]
        video = videos.get(seed)
        if video is not None and video.stat().st_size:
            target = root / "video/worker_0" / f"{seed}_{meta['result']}.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            publish_artifact(video, target, root)
        if seed in hdf5:
            target = root / "hdf5/worker_0" / f"{seed}.hdf5"
            target.parent.mkdir(parents=True, exist_ok=True)
            publish_artifact(hdf5[seed], target, root)
    counts = {
        k: sum(m["result"] == k for m in records.values())
        for k in ["success", "failed", "error"]
    }
    missing = sorted(expected - records.keys())
    valid_count = counts["success"] + counts["failed"]
    summary = {
        **counts,
        "policy_name": "SimpleTac",
        "task_name": contract["task"],
        "run_id": root.name,
        "run_root": str(root),
        "total_episodes": len(records),
        "requested_episodes": total,
        "success_rate": counts["success"] / valid_count if valid_count else None,
        "success_rate_denominator": "success_plus_failed",
        "valid_episodes": valid_count,
        "raw_success_rate": counts["success"] / total,
        "valid_success_rate": counts["success"] / valid_count if valid_count else None,
        "missing_seeds": missing,
        "status": "incomplete" if missing else "completed_with_errors" if counts["error"] else "complete",
        "contract": contract,
        "attempt": str(attempt),
        "worker_exit_codes": plan.get("worker_exit_codes", []),
        "workers": {"worker_0": {**counts, "episodes": len(records)}},
    }
    write_json(
        root / "metadata/worker_0.json", {str(s): records[s] for s in sorted(records)}
    )
    write_json(root / "metadata.json", summary)
    lines = [
        f"Task Name: {contract['task']}",
        f"Checkpoint: {contract['checkpoint']}",
        f"Eval Config: {json.dumps(contract, ensure_ascii=False)}",
    ]
    for seed, meta in sorted(records.items()):
        lines.append(
            f"[{meta['episode']:<3}] Seed {seed} {meta['result']} after {float(meta.get('cost_time', 0)):.2f} s."
        )
    label = "Partial Result" if missing or counts["error"] else "Final Result"
    lines += [
        f"{label}: {counts['success']}/{valid_count} valid episodes success; requested={total}.",
        f"Valid success: {counts['success']}/{valid_count}; error={counts['error']}; missing={missing}",
        f"Evaluation status: {summary['status']}",
    ]
    temporary = root / "log.log.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, root / "log.log")
    write_json(
        root / MANIFEST,
        {
            "contract": contract,
            "status": summary["status"],
            "last_attempt": str(attempt),
        },
    )
    write_json(journal, {"phase": "done"})
    return summary
