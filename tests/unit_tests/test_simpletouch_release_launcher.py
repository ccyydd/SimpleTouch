"""Lightweight checks for the public UniVTAC training entry point."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts/simpletouch/train_univtac.sh"
EVAL_LAUNCHER = REPO_ROOT / "scripts/simpletouch/evaluate_univtac.sh"


class SimpleTouchReleaseLauncherTest(unittest.TestCase):
    def _run(self, *, task: str, num_gpus: str = "8") -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            SIMPLETOUCH_DRY_RUN="1",
            SIMPLETOUCH_TASK=task,
            SIMPLETOUCH_NUM_GPUS=num_gpus,
            SIMPLETOUCH_DATA_ROOT="/tmp/simpletouch-test-data",
            SIMPLETOUCH_NORM_STATS_PATH="/tmp/simpletouch-test-norm.json",
            SIMPLETOUCH_BASE_MODEL_PATH="/tmp/simpletouch-test-base",
            SIMPLETOUCH_T3_PATH="/tmp/simpletouch-test-t3",
            SIMPLETOUCH_TOKENIZER_PATH="/tmp/simpletouch-test-tokenizer",
            SIMPLETOUCH_FUTURE_CACHE_ROOT="/tmp/simpletouch-test-cache",
        )
        return subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_canonical_camera_mapping_and_batch(self) -> None:
        for task, camera in (
            ("lift_bottle", "head"),
            ("pull_out_key", "head"),
            ("lift_can", "all"),
            ("put_bottle_in_shelf", "head"),
            ("insert_hole", "head"),
            ("insert_tube", "all"),
        ):
            with self.subTest(task=task):
                result = self._run(task=task)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"data.camera_type={camera}", result.stdout)
                self.assertIn("actor.micro_batch_size=8", result.stdout)
                self.assertIn("actor.global_batch_size=64", result.stdout)

    def test_rejects_unknown_task_or_incompatible_gpu_count(self) -> None:
        self.assertNotEqual(self._run(task="unknown").returncode, 0)
        self.assertNotEqual(self._run(task="insert_hole", num_gpus="3").returncode, 0)

    def test_policy_only_install_scope_matches_readme(self) -> None:
        installer = (REPO_ROOT / "requirements/install.sh").read_text()
        readme = (REPO_ROOT / "README.md").read_text()
        self.assertIn('--model openpi --env policy-only', installer)
        self.assertIn("--model openpi --env policy-only", readme)

        unsupported = subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "requirements/install.sh"),
                "embodied",
                "--model",
                "openvla",
                "--env",
                "policy-only",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(unsupported.returncode, 2)
        self.assertIn("supports only", unsupported.stderr)

    def test_multigpu_evaluation_uses_managed_launcher(self) -> None:
        env = os.environ.copy()
        env.update(
            TASK="insert_hole",
            CKPT_PATH="/tmp/simpletouch-test-checkpoint",
            SIMPLETOUCH_EVAL_GPUS="0,1,2,3",
            SIMPLETOUCH_PYTHON="/bin/echo",
        )
        result = subprocess.run(
            ["bash", str(EVAL_LAUNCHER)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("eval_simpletac_managed.py", result.stdout)
        self.assertIn("--checkpoint /tmp/simpletouch-test-checkpoint", result.stdout)
        self.assertIn("--task insert_hole --gpus 0,1,2,3", result.stdout)


if __name__ == "__main__":
    unittest.main()
