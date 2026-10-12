# Copyright (c) 2026 SimpleTouch contributors. All rights reserved.
"""Offline safety tests. These do not certify a simulator dependency installation."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "requirements/install_univtac.sh"
SUPPORT = REPO / "requirements/univtac"
spec = importlib.util.spec_from_file_location(
    "univtac_installer_state", SUPPORT / "installer_state.py"
)
assert spec is not None and spec.loader is not None
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


class UniVTACInstallerEntryTest(unittest.TestCase):
    def run_entry(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), *args], text=True, capture_output=True, check=False
        )

    def test_help_is_offline_and_requires_runtime_smoke(self) -> None:
        result = self.run_entry("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("evaluation smoke test", result.stdout)
        self.assertIn("--accept-isaac-eula", result.stdout)
        self.assertIn("--jobs", result.stdout)

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_private_graphics_runtime_prepares_headless_materialx_dependencies(
        self,
    ) -> None:
        helper = REPO / "scripts/univtac/setup_nvidia_vulkan.sh"
        subprocess.run(["bash", "-n", str(helper)], check=True)
        source = helper.read_text()
        self.assertIn("apt-get download libegl1 libegl-mesa0 libxt6", source)
        self.assertIn('ldd "$egl/libXt.so.6"', source)
        self.assertIn("libxt-dependencies.log", source)
        self.assertNotIn("apt-get install", source)

    def test_rejects_missing_values_and_unknown_options(self) -> None:
        for args in (("--prefix",), ("--gpu", "--accept-isaac-eula"), ("--unknown",)):
            with self.subTest(args=args):
                self.assertEqual(self.run_entry(*args).returncode, 2)

    def test_eula_required_before_any_directory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "untouched"
            result = self.run_entry(
                "--prefix", str(prefix), "--gpu", "0", "--cuda-arch", "80"
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("EULA", result.stderr)
            self.assertFalse(prefix.exists())

    def test_invalid_build_parallelism_is_rejected(self) -> None:
        for jobs in ("0", "-1", "1.5", "auto"):
            with self.subTest(jobs=jobs):
                result = self.run_entry("--jobs", jobs)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--jobs must be", result.stderr)

    def test_unsupported_gpu_architecture_stops_before_directory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "untouched"
            for arch in ("60", "100", "120"):
                with self.subTest(arch=arch):
                    result = self.run_entry(
                        "--prefix",
                        str(prefix),
                        "--gpu",
                        "0",
                        "--cuda-arch",
                        arch,
                        "--accept-isaac-eula",
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("Unsupported --cuda-arch", result.stderr)
                    self.assertIn("CUDA 12.4", result.stderr)
                    self.assertFalse(prefix.exists())

    def test_core_versions_and_no_upstream_all_in_one_installers(self) -> None:
        source = SCRIPT.read_text()
        constraints = (SUPPORT / "constraints.txt").read_text()
        self.assertIn("torch==2.5.1+cu124", constraints)
        self.assertIn("torchvision==0.20.1+cu124", constraints)
        self.assertIn("cmake==3.30.1", constraints)
        self.assertIn("flock -n 9", source)
        self.assertIn("CONDA_REGISTER_ENVS=false", source)
        self.assertIn("UV_NO_CONFIG=1", source)
        self.assertEqual(source.count("unset ISAAC_PATH EXP_PATH CARB_APP_PATH"), 2)
        self.assertIn("export LC_ALL=C LANG=C LANGUAGE=C", source)
        self.assertIn("VCPKG_KEEP_ENV_VARS='LC_ALL;LANG;LANGUAGE'", source)
        self.assertIn("pip check", source)
        self.assertNotIn("pip_check ||", source)
        self.assertNotIn("sudo ", source)
        self.assertNotIn("source ~/.bashrc", source)
        self.assertNotIn("isaaclab.sh --install", source)
        self.assertNotIn("tacex.sh -i", source)
        self.assertNotIn("scripts/reinforcement_learning", source)
        self.assertNotIn("snapshot_download", source)
        self.assertLess(
            source.index("stage 01-toolchain"), source.index("stage 03-isaac")
        )
        self.assertIn(
            "for extension in isaaclab isaaclab_assets isaaclab_tasks;", source
        )
        self.assertIn("for extension in tacex tacex_assets;", source)
        self.assertIn("-DUIPC_BUILD_PYBIND=1", source)
        self.assertIn("-DUIPC_BUILD_EXAMPLES=OFF", source)
        self.assertIn("-DUIPC_BUILD_TESTS=OFF", source)
        self.assertIn("-DUIPC_BUILD_BENCHMARKS=OFF", source)
        self.assertIn("mktemp -d /tmp/simpletouch-uipc.XXXXXX", source)
        self.assertIn('export TMPDIR="$build_tmp"', source)
        self.assertIn('cmake --build "$build" --parallel "$jobs"', source)
        self.assertIn("export UNIVTAC_PYTHON=", source)
        self.assertIn('GIT_LFS_SKIP_SMUDGE=1 git -C "$destination" checkout', source)

    def test_failed_stage_or_pip_check_never_writes_completion_stamp(self) -> None:
        # Exercise the actual shell stage function, without any installer/network commands.
        definition = (
            SCRIPT.read_text().split("stage() {", 1)[1].split("clone_at() {", 1)[0]
        )
        for command, check in (("false", "return 0"), ("true", "return 7")):
            with (
                self.subTest(command=command, check=check),
                tempfile.TemporaryDirectory() as directory,
            ):
                prefix = Path(directory)
                (prefix / "logs").mkdir()
                (prefix / "state").mkdir()
                program = (
                    "set -euo pipefail\n"
                    f"prefix={shlex.quote(directory)}\n"
                    f"pip_check() {{ {check}; }}\n"
                    f"stage() {{{definition}\n"
                    f"stage sample {command}\n"
                )
                result = subprocess.run(
                    ["bash", "-c", program], capture_output=True, text=True
                )
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse((prefix / "state/sample.done").exists())

    def test_simulator_wrapper_restores_only_explicit_graphics_runtime(self) -> None:
        template = (
            SCRIPT.read_text().split("  cat <<'SH'\n", 1)[1].split("\nSH\n", 1)[0]
        )
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory).resolve()
            (prefix / "env/bin").mkdir(parents=True)
            interpreter = prefix / "env/bin/python"
            interpreter.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$LD_LIBRARY_PATH" "$PYTHONPATH" '
                '"${ISAAC_PATH-unset}" "${EXP_PATH-unset}" "${CARB_APP_PATH-unset}"\n'
            )
            interpreter.chmod(0o700)
            graphics = prefix / "graphics.sh"
            graphics.write_text(
                'export LD_LIBRARY_PATH="/private-driver/lib:$LD_LIBRARY_PATH"\n'
            )
            program = (
                f"set -euo pipefail\nprefix={shlex.quote(str(prefix))}\n{template}"
            )
            env = dict(
                os.environ,
                LD_LIBRARY_PATH="/policy/cu126",
                PYTHONPATH="/policy-env/lib/python3.11/site-packages",
                ISAAC_PATH="/old-isaac",
                EXP_PATH="/old-experience",
                CARB_APP_PATH="/old-carb",
                VULKAN_RUNTIME_ENV=str(graphics),
            )
            result = subprocess.run(
                ["bash", "-c", program], env=env, text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("/private-driver/lib:", result.stdout)
            self.assertIn(f"{prefix}/env/lib", result.stdout)
            self.assertNotIn("/policy/cu126", result.stdout)
            self.assertNotIn("/policy-env", result.stdout)
            self.assertNotIn("/old-", result.stdout)
            self.assertEqual(result.stdout.splitlines()[-3:], ["unset"] * 3)
            self.assertIn(f"{prefix}/sources/UniVTAC", result.stdout)
            # The selected environment must precede any installed public task
            # source, while unrelated policy packages remain isolated.
            selected = prefix / 'runtime/reference/UniVTAC'
            selected.mkdir(parents=True)
            (selected / 'reference-runtime.json').write_text('{"schema": 1}')
            env['UNIVTAC_DIR'] = str(selected)
            result = subprocess.run(
                ['bash', '-c', program], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            paths = result.stdout.splitlines()[1].split(':')
            self.assertIn(str(selected), paths)
            self.assertNotIn(str(prefix / 'sources/UniVTAC'), paths)
            self.assertNotIn('/policy-env/lib/python3.11/site-packages', paths)
            env['UNIVTAC_DIR'] = str(prefix / 'unverified')
            result = subprocess.run(
                ['bash', '-c', program], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            env.pop('UNIVTAC_DIR')
            env["VULKAN_RUNTIME_ENV"] = str(prefix / "missing.sh")
            result = subprocess.run(
                ["bash", "-c", program], env=env, text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Missing VULKAN_RUNTIME_ENV", result.stderr)

    def test_resume_can_repair_later_interrupted_stage(self) -> None:
        definition = (
            SCRIPT.read_text().split("stage() {", 1)[1].split("clone_at() {", 1)[0]
        )
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            (prefix / "logs").mkdir()
            (prefix / "state").mkdir()
            (prefix / "state/first.done").touch()
            program = (
                "set -euo pipefail\n"
                f"prefix={shlex.quote(directory)}\n"
                'pip_check() { test -f "$prefix/repaired"; }\n'
                f"stage() {{{definition}\n"
                "stage first false\n"
                'stage second touch "$prefix/repaired"\n'
                "pip_check\n"
            )
            result = subprocess.run(
                ["bash", "-c", program], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((prefix / "state/second.done").exists())

    def test_simulator_launcher_does_not_append_caller_pythonpath(self) -> None:
        source = (REPO / "scripts/univtac/eval_simpletac.sh").read_text()
        self.assertNotIn("${PYTHONPATH:+:$PYTHONPATH}", source)


class UniVTACInstallerStateTest(unittest.TestCase):
    def test_tinygltf_overlay_only_changes_source_reference_and_checksum(self) -> None:
        patch = json.loads((SUPPORT / "tinygltf-source.json").read_text())
        port = f"vcpkg_from_github({patch['old_ref']}\nSHA512 {patch['old_sha512']})\n"
        metadata = json.dumps(
            {
                "name": "tinygltf",
                "version": "2.9.3",
                "dependencies": ["nlohmann-json", "stb"],
            }
        )
        files = state.tinygltf_overlay_contents(port, metadata, patch)
        self.assertEqual(files["vcpkg.json"], metadata)
        self.assertIn(patch["new_ref"], files["portfile.cmake"])
        self.assertIn(patch["new_sha512"], files["portfile.cmake"])
        with self.assertRaisesRegex(ValueError, "version"):
            state.tinygltf_overlay_contents(
                port, metadata.replace("2.9.3", "3.0.0"), patch
            )
        with self.assertRaisesRegex(ValueError, "declaration"):
            state.tinygltf_overlay_contents(
                port.replace(patch["old_ref"], "REF unknown"), metadata, patch
            )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def test_owned_root_is_resumable(self) -> None:
        prefix = self.root / "installation"
        state.prepare_root(str(prefix))
        (prefix / "a-log").write_text("preserve me")
        self.assertEqual(state.prepare_root(str(prefix)), prefix)
        self.assertEqual((prefix / "a-log").read_text(), "preserve me")

    def test_foreign_nonempty_directory_is_preserved(self) -> None:
        prefix = self.root / "foreign"
        prefix.mkdir()
        (prefix / "user-file").write_text("original")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            state.prepare_root(str(prefix))
        self.assertEqual((prefix / "user-file").read_text(), "original")
        self.assertFalse((prefix / state.OWNER).exists())

    def test_symlink_and_root_are_rejected(self) -> None:
        target = self.root / "target"
        target.mkdir()
        link = self.root / "link"
        link.symlink_to(target, target_is_directory=True)
        for path in (str(link), "/", "relative/path"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                state.prepare_root(path)

    def test_wrong_owner_marker_is_rejected(self) -> None:
        prefix = self.root / "owned"
        state.prepare_root(str(prefix))
        marker = prefix / state.OWNER
        value = json.loads(marker.read_text())
        value["uid"] = os.getuid() + 1
        marker.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "ownership"):
            state.prepare_root(str(prefix))

    def test_manifest_rejects_changed_inputs_and_configuration(self) -> None:
        prefix = self.root / "owned"
        state.prepare_root(str(prefix))
        state.check_manifest(prefix, REPO, "0", "80")
        state.check_manifest(prefix, REPO, "0", "80")
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            state.check_manifest(prefix, REPO, "1", "80")
        manifest = prefix / state.MANIFEST
        value = json.loads(manifest.read_text())
        key = next(iter(value["inputs_sha256"]))
        value["inputs_sha256"][key] = "modified"
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            state.check_manifest(prefix, REPO, "0", "80")

    def test_manifest_rejects_redirected_environment(self) -> None:
        prefix = self.root / "owned"
        state.prepare_root(str(prefix))
        (prefix / "env").symlink_to(self.root / "shared")
        with self.assertRaisesRegex(ValueError, "redirected"):
            state.check_manifest(prefix, REPO, "0", "80")

    def test_client_copy_is_isolated_and_rejects_local_edits(self) -> None:
        destination = self.root / "client"
        source = REPO / "packages/openpi-client"
        state.copy_client(source, destination)
        state.copy_client(source, destination)
        self.assertEqual(
            (source / "pyproject.toml").read_bytes(),
            (destination / "pyproject.toml").read_bytes(),
        )
        (destination / "pyproject.toml").write_text("local edit")
        with self.assertRaisesRegex(ValueError, "was edited"):
            state.copy_client(source, destination)

    def make_patch_fixture(self) -> tuple[Path, Path, str, str]:
        checkout = self.root / "checkout"
        checkout.mkdir()
        self.git(checkout, "init", "-q")
        original = 'dependency = "old-reviewed-wheel"\n'
        expected = 'dependency = "new-reviewed-wheel"\n'
        (checkout / "setup.py").write_text(original)
        self.git(checkout, "add", "setup.py")
        self.git(
            checkout,
            "-c",
            "user.name=Installer Test",
            "-c",
            "user.email=installer@example.invalid",
            "commit",
            "-qm",
            "fixture",
        )
        revision = self.git(checkout, "rev-parse", "HEAD").strip()
        patch = self.root / "patch.json"
        patch.write_text(
            json.dumps(
                {
                    "commit": revision,
                    "path": "setup.py",
                    "old": "old-reviewed-wheel",
                    "new": "new-reviewed-wheel",
                }
            )
        )
        return checkout, patch, original, expected

    @staticmethod
    def git(checkout: Path, *args: str) -> str:
        return subprocess.check_output(["git", "-C", str(checkout), *args], text=True)

    def test_exact_patch_is_idempotent(self) -> None:
        checkout, patch, _, expected = self.make_patch_fixture()
        state.apply_scatter_patch(checkout, patch)
        state.apply_scatter_patch(checkout, patch)
        self.assertEqual((checkout / "setup.py").read_text(), expected)

    def test_patch_rejects_unknown_local_edits(self) -> None:
        checkout, patch, original, _ = self.make_patch_fixture()
        (checkout / "setup.py").write_text(original + "user change\n")
        with self.assertRaisesRegex(ValueError, "unexpectedly modified"):
            state.apply_scatter_patch(checkout, patch)
        self.assertIn("user change", (checkout / "setup.py").read_text())

    def test_patch_rejects_wrong_commit(self) -> None:
        checkout, patch, original, _ = self.make_patch_fixture()
        value = json.loads(patch.read_text())
        value["commit"] = "0" * 40
        patch.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "revision"):
            state.apply_scatter_patch(checkout, patch)
        self.assertEqual((checkout / "setup.py").read_text(), original)


if __name__ == "__main__":
    unittest.main()
