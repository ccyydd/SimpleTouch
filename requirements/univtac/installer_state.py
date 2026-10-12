# Copyright (c) 2026 SimpleTouch contributors. All rights reserved.
"""Standard-library-only safety helpers for the isolated simulator installer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

OWNER = ".simpletouch-univtac-owner.json"
MANIFEST = "install-manifest.json"


def atomic_json(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_root(prefix: str) -> Path:
    root = Path(prefix)
    if not root.is_absolute() or root == Path(root.anchor):
        raise ValueError("--prefix must be an absolute, non-root directory")
    # Reject symlink components rather than accidentally owning a shared target.
    if root != root.resolve():
        raise ValueError("--prefix must be a canonical path without symlink components")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_uid != os.getuid():
        raise ValueError("Installation prefix is not owned by this user")
    marker = root / OWNER
    owner = {
        "installer": "simpletouch-univtac-v1",
        "uid": os.getuid(),
        "root": str(root),
    }
    if marker.exists():
        if marker.is_symlink() or json.loads(marker.read_text()) != owner:
            raise ValueError("Invalid installation ownership marker")
    else:
        if any(root.iterdir()):
            raise ValueError(
                "Refusing a nonempty directory without an ownership marker"
            )
        # Atomic exclusive creation: concurrent first launches cannot overwrite ownership.
        with marker.open("x") as handle:
            json.dump(owner, handle, sort_keys=True)
    return root


def check_manifest(root: Path, repo: Path, gpu: str, arch: str) -> None:
    sources = [repo / "requirements/install_univtac.sh"]
    sources.extend(sorted((repo / "requirements/univtac").glob("*")))
    sources.extend(sorted((repo / "packages/openpi-client").rglob("*.py")))
    sources.append(repo / "packages/openpi-client/pyproject.toml")
    hashes = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
        if path.is_file()
    }
    desired = {
        "schema": 1,
        "status": "Installation input manifest; runtime verification is separate",
        "prefix": str(root),
        "repo": str(repo),
        "build_gpu": gpu,
        "cuda_arch": arch,
        "isaac_eula_explicitly_accepted": True,
        "inputs_sha256": hashes,
    }
    manifest = root / MANIFEST
    if manifest.exists():
        if manifest.is_symlink() or json.loads(manifest.read_text()) != desired:
            raise ValueError(
                "Installation inputs changed; use a new prefix (no automatic migration)"
            )
    else:
        atomic_json(manifest, desired)
    for name in (
        "env",
        "sources",
        "tools",
        "cache",
        "tmp",
        "logs",
        "state",
        "runtime",
        "overlay-ports",
        "overlay-ports/tinygltf",
        "sources/UniVTAC",
        "sources/vcpkg",
        "sources/UniVTAC/third_party",
        "sources/UniVTAC/third_party/IsaacLab",
        "sources/UniVTAC/third_party/curobo",
        "sources/UniVTAC/third_party/TacEx",
        "sources/UniVTAC/third_party/TacEx/source",
        "sources/openpi-client",
        "tools/native",
    ):
        path = root / name
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(f"Refusing redirected installation path: {path}")


def copy_client(source: Path, destination: Path) -> None:
    """Copy this small code-only package; never build editable metadata in the release repo."""
    paths = [source / "pyproject.toml", *sorted((source / "src").rglob("*.py"))]
    for path in paths:
        target = destination / path.relative_to(source)
        if target.resolve() != target:
            raise ValueError(f"Refusing redirected client source: {target}")
        if target.exists():
            if target.read_bytes() != path.read_bytes():
                raise ValueError(f"Copied client was edited: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


def apply_scatter_patch(checkout: Path, specification: Path) -> None:
    patch = json.loads(specification.read_text())
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != patch["commit"]:
        raise ValueError(
            "UniVTAC revision does not match the reviewed compatibility patch"
        )
    path = checkout / patch["path"]
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("Refusing redirected TacEx setup.py")
    original = subprocess.check_output(
        ["git", "-C", str(checkout), "show", f"HEAD:{patch['path']}"], text=True
    )
    if original.count(patch["old"]) != 1:
        raise ValueError("Upstream torch_scatter declaration changed")
    expected = original.replace(patch["old"], patch["new"])
    current = path.read_text()
    if current == expected:
        return  # Resume is idempotent, but arbitrary edits are never overwritten.
    if current != original:
        raise ValueError("Refusing to patch an unexpectedly modified TacEx setup.py")
    path.write_text(expected)


def tinygltf_overlay_contents(
    portfile: str, metadata: str, patch: dict
) -> dict[str, str]:
    if json.loads(metadata).get("version") != patch["version"]:
        raise ValueError("Unexpected TinyGLTF port version")
    for old, new in (
        (patch["old_ref"], patch["new_ref"]),
        (patch["old_sha512"], patch["new_sha512"]),
    ):
        if portfile.count(old) != 1:
            raise ValueError("TinyGLTF source declaration changed")
        portfile = portfile.replace(old, new)
    return {"portfile.cmake": portfile, "vcpkg.json": metadata}


def prepare_tinygltf_overlay(
    registry: Path, destination: Path, specification: Path
) -> None:
    patch = json.loads(specification.read_text())
    revision = subprocess.check_output(
        ["git", "-C", str(registry), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != patch["registry_commit"]:
        raise ValueError("Unexpected vcpkg registry revision")

    def read_original(name: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(registry), "show", f"{patch['port_tree']}:{name}"],
            text=True,
        )

    files = tinygltf_overlay_contents(
        read_original("portfile.cmake"), read_original("vcpkg.json"), patch
    )
    if destination.resolve() != destination:
        raise ValueError("Refusing redirected overlay directory")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = destination / name
        if path.is_symlink() or (path.exists() and path.read_text() != content):
            raise ValueError(f"Refusing to overwrite changed overlay: {path}")
        if not path.exists():
            with path.open("x") as handle:
                handle.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("prefix")
    manifest = commands.add_parser("manifest")
    manifest.add_argument("prefix")
    manifest.add_argument("repo")
    manifest.add_argument("gpu")
    manifest.add_argument("arch")
    patch = commands.add_parser("patch")
    patch.add_argument("checkout")
    patch.add_argument("specification")
    client = commands.add_parser("copy-client")
    client.add_argument("source")
    client.add_argument("destination")
    overlay = commands.add_parser("tinygltf-overlay")
    overlay.add_argument("registry")
    overlay.add_argument("destination")
    overlay.add_argument("specification")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_root(args.prefix)
    elif args.command == "manifest":
        check_manifest(Path(args.prefix), Path(args.repo), args.gpu, args.arch)
    elif args.command == "patch":
        apply_scatter_patch(Path(args.checkout), Path(args.specification))
    elif args.command == "copy-client":
        copy_client(Path(args.source), Path(args.destination))
    else:
        prepare_tinygltf_overlay(
            Path(args.registry), Path(args.destination), Path(args.specification)
        )


if __name__ == "__main__":
    main()
