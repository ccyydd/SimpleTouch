"""Prepare a private UniVTAC runtime matching the September evaluation.

Only immutable assets and installed dependencies come from the supplied
checkout. Environment code, camera configuration and the native evaluator are
frozen in this repository, with hashes checked before each preparation.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil


def reference_root():
    return Path(__file__).resolve().parents[2] / "examples/UniVTAC/reference_20260913"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_reference(reference):
    reference = Path(reference)
    manifest = json.loads((reference / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(reference / name) != expected:
            raise RuntimeError(f"Reference source changed: {name}")
    return manifest


def prepare_runtime(upstream, target, kit_root=None, reference=None):
    """Copy reviewed Python/config files, without editing the supplied checkout."""
    upstream, target = Path(upstream).resolve(), Path(target).absolute()
    reference = Path(reference) if reference else reference_root()
    manifest = verify_reference(reference)
    if target == upstream or target == reference.resolve():
        raise ValueError("Runtime must be separate from its source")
    for name, expected in manifest["checked_assets"].items():
        if digest(upstream / name) != expected:
            raise RuntimeError(f"Runtime asset differs from the reference: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / (target.name + ".prepare.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = target / "reference-runtime.json"
        if target.exists() and not marker.exists():
            raise FileExistsError(f"Refusing an unrecognized runtime: {target}")
        if not target.exists():
            target.mkdir()
            # Mutable policy plugins and all output paths remain private.
            for name in ["third_party", "encoder"]:
                source = upstream / name
                if not source.is_dir():
                    raise FileNotFoundError(source)
                (target / name).symlink_to(source, target_is_directory=True)
            # Keep newly created asset files on the runtime's NAS volume too.
            shutil.copytree(upstream / "assets", target / "assets")
            (target / "policy").mkdir()
            for source in (upstream / "policy").iterdir():
                if source.name not in ["SimpleTac", "__pycache__"]:
                    (target / "policy" / source.name).symlink_to(
                        source, target_is_directory=source.is_dir())
            for name in manifest["files"]:
                dest = target / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(reference / name, dest)
            marker.write_text(json.dumps({"initializing": True}) + "\n")
        # Always verify mutable runtime copies, rather than silently overwriting
        # an unexpected change in an existing installation.
        for name, expected in manifest["files"].items():
            if name == "scripts/eval_policy.py":
                continue
            if digest(target / name) != expected:
                raise RuntimeError(f"Private runtime source changed: {name}")
        for name, expected in manifest["checked_assets"].items():
            if digest(target / name) != expected:
                raise RuntimeError(f"Private runtime asset changed: {name}")
        source = (reference / "scripts/eval_policy.py").read_text()
        evaluator = target / "scripts/eval_policy.py"
        if kit_root:
            kit_root = Path(kit_root).absolute()
            kit_root.mkdir(parents=True, exist_ok=True)
            anchor = 'args_cli.livestream = int(os.environ.get("UNIVTAC_LIVESTREAM", "2"))'
            if source.splitlines().count(anchor) != 1:
                raise RuntimeError("Unsupported reference livestream assignment")
            source = source.replace(anchor, anchor + "\nargs_cli.kit_args += "
                                    + repr(" --portable-root " + str(kit_root)))
        evaluator.write_text(source)
        record = {"schema": 1, "reference": str(reference.resolve()),
                  "reference_manifest_sha256": digest(reference / "manifest.json"),
                  "upstream_assets": str(upstream), "environment": manifest["evaluation_environment"],
                  "reference_eval_sha256": manifest["files"]["scripts/eval_policy.py"],
                  "private_eval_sha256": digest(evaluator),
                  "startup_changes": ["per-GPU Kit portable root"] if kit_root else []}
        marker.write_text(json.dumps(record, indent=2) + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--kit-root")
    args = parser.parse_args()
    os.umask(0o002)
    print(prepare_runtime(args.upstream, args.target, args.kit_root))


if __name__ == "__main__":
    main()
