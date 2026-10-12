"""CPU-only checks for single native evaluation export and provenance."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.univtac import eval_policy_native as native
from scripts.univtac.managed_eval_results import consolidate


class NativePublicationTest(unittest.TestCase):
    def test_single_native_publishes_reference_layout_without_copying_video(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            upstream = temp / "UniVTAC"
            (upstream / "scripts").mkdir(parents=True)
            (upstream / "scripts/eval_policy.py").write_text("# fake runtime")
            output = temp / "eval_result"
            fake = '''
from pathlib import Path
import json
def main():
    root = _managed_result_root
    (root / "metadata").mkdir(parents=True)
    (root / "metadata" / (_managed_worker_name + ".json")).write_text(json.dumps({
        "1000000": {"result": "success", "cost_step": 10, "cost_time": 1.2}}))
    video = root / "video" / _managed_worker_name / "1000000_success.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"only video original")
    (root / "scene" / _managed_worker_name).mkdir(parents=True)
    (root / "scene" / _managed_worker_name / "systems.json").write_text("{}")
'''
            contract = {"checkpoint": "/weights/global_step_20000", "task": "lift_bottle",
                        "start_seed": 1000000, "episodes": 1}
            with (mock.patch.dict(os.environ, {"SIMPLETOUCH_EVAL_ROOT": str(output),
                                               "SIMPLETOUCH_EVAL_RUN_ID": "reference"}, clear=True),
                  mock.patch.object(native.sys, "argv", ["native.py", "lift_bottle"]),
                  mock.patch.object(native.Path, "cwd", return_value=upstream),
                  mock.patch.object(native, "adapt_source", return_value=fake),
                  mock.patch.object(native, "native_contract", return_value=contract),
                  mock.patch.object(native, "consolidate", side_effect=lambda r, a:
                                    consolidate(r, a, lambda p: (True, "")))):
                native.main()
            run = output / "SimpleTac/lift_bottle/reference"
            video = run / "video/worker_0/1000000_success.mp4"
            self.assertTrue(video.is_symlink())
            self.assertFalse(os.path.isabs(os.readlink(video)))
            self.assertEqual(video.read_bytes(), b"only video original")
            self.assertTrue(video.resolve().is_relative_to(run / ".eval_runs"))
            self.assertEqual(len([p for p in run.rglob("*.mp4") if not p.is_symlink()]), 1)
            self.assertFalse((run / "scene").exists())
            self.assertTrue((run / ".managed_eval.lock").is_file())
            self.assertEqual(len(list((run / ".eval_runs").rglob("systems.json"))), 1)
            rows = json.loads((run / "metadata/worker_0.json").read_text())
            self.assertEqual(rows["1000000"], {"result": "success", "cost_step": 10,
                                             "cost_time": 1.2, "episode": 1000000})
            summary = json.loads((run / "metadata.json").read_text())
            self.assertEqual(summary["requested_episodes"], 1)
            self.assertEqual(summary["status"], "complete")
            self.assertIn("Final Result: 1/1", (run / "log.log").read_text())

    def test_native_contract_records_actual_cli_and_file_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            upstream = root / "scripts/eval_policy.py"
            upstream.parent.mkdir()
            upstream.write_text("# installed native source")
            deploy, task = root / "deploy.yml", root / "demo.yml"
            deploy.write_text("seed: 2")
            task.write_text("video_frequency: 2")
            weights = root / "weights"
            weights.mkdir()
            (weights / "config.yaml").write_text("policy configuration")
            (weights / "_SUCCESS").write_text("complete")
            cli = SimpleNamespace(deploy_config=str(deploy), task_config=str(task),
                                  task_name="insert_tube", start_seed=-1,
                                  max_seed=3000001, total_num=100)
            namespace = {"args_cli": cli, "_native_runtime_metadata": {"horizon": 25, "use_tactile": "false"}, "get_config": lambda name, **kwargs:
                         ({"seed": 2} if name == str(deploy) else {}, Path(name))}
            with mock.patch.dict(os.environ, {"CKPT_PATH": str(weights)}, clear=True):
                contract = native.native_contract(namespace, upstream)
            self.assertEqual((contract["start_seed"], contract["episodes"], contract["horizon"]),
                             (3000000, 2, 25))
            self.assertEqual(contract["checkpoint_config_sha256"],
                             hashlib.sha256((weights / "config.yaml").read_bytes()).hexdigest())
            self.assertEqual(contract["evaluation_code_sha256"][str(upstream)],
                             hashlib.sha256(upstream.read_bytes()).hexdigest())
            with mock.patch.dict(os.environ, {"SIMPLETAC_EXECUTION_HORIZON": "auto"}, clear=True):
                namespace["_native_runtime_metadata"] = {"horizon": 50, "use_tactile": "false"}
                self.assertEqual(native.native_contract(namespace, upstream)["horizon"], 50)


if __name__ == "__main__":
    unittest.main()
