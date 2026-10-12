# Copyright (c) 2026 SimpleTouch contributors. All rights reserved.
"""CPU-only regression coverage for public UniVTAC 0dafa102 output layouts.

The small fixtures mirror reviewed evaluator/BaseTask/VideoHandler assignments;
they do not certify Isaac Sim execution or video encoding.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.univtac.eval_policy_managed import adapt_source, bounded_evaluator
from scripts.univtac.managed_eval_results import (
    artifact_files,
    consolidate,
    fingerprint,
    read_records,
    seed_shards,
    write_json,
)

PUBLIC_SAVE = "    env_cfg.save_dir = Path('eval_result') / policy_name / task_file_name / deploy_config_file.stem / curr_time"
PUBLIC_SEEDS = "    seed_path = task.save_root.parent / 'seeds.json'"
PUBLIC_FIXTURE = (
    "args_cli.livestream = 2\n"
    "def main():\n"
    "    env_cfg = SimpleNamespace()\n"
    f"{PUBLIC_SAVE}\n"
    "    return env_cfg\n"
    "def eval_policy(task):\n"
    f"{PUBLIC_SEEDS}\n"
    "    return seed_path\n"
)
LEGACY_FIXTURE = (
    "def main():\n"
    "    run_id = build_run_id()\n"
    "    run_root = Path('eval_result') / policy_name / task_file_name / run_id\n"
    "    env_cfg.worker_name = worker_name(0)\n"
)


class PublicAdapterTest(unittest.TestCase):
    def test_public_routes_exact_worker_root_and_private_seed_file(self):
        root = Path("/owned/attempt/workers/worker_1")
        namespace = {
            "Path": Path,
            "SimpleNamespace": SimpleNamespace,
            "args_cli": SimpleNamespace(),
            "_managed_result_root": root,
        }
        adapted = adapt_source(PUBLIC_FIXTURE)
        exec(adapted, namespace)
        cfg = namespace["main"]()
        # Public BaseTask._setup_save uses Path(cfg.save_dir), not root/worker.
        task = SimpleNamespace(save_root=Path(cfg.save_dir))
        self.assertEqual(task.save_root, root)
        self.assertEqual(namespace["eval_policy"](task), root / "seeds.json")
        self.assertEqual(namespace["args_cli"].livestream, 2)
        restored = adapted.replace(
            "    env_cfg.save_dir = _managed_result_root", PUBLIC_SAVE
        ).replace("    seed_path = task.save_root / 'seeds.json'", PUBLIC_SEEDS)
        self.assertEqual(restored, PUBLIC_FIXTURE)

    def test_legacy_layout_remains_supported(self):
        adapted = adapt_source(LEGACY_FIXTURE)
        self.assertIn("run_id = _managed_result_root.name", adapted)
        self.assertIn("run_root = _managed_result_root", adapted)
        self.assertIn("env_cfg.worker_name = _managed_worker_name", adapted)

    def test_unknown_duplicate_and_mixed_anchors_fail_closed(self):
        for source in (
            PUBLIC_FIXTURE.replace(PUBLIC_SAVE, PUBLIC_SAVE + " / 'changed'"),
            PUBLIC_FIXTURE + PUBLIC_SAVE + "\n",
            PUBLIC_FIXTURE.replace(PUBLIC_SEEDS, "    seed_path = unknown"),
            PUBLIC_FIXTURE + LEGACY_FIXTURE,
            LEGACY_FIXTURE.replace("worker_name(0)", "worker_name(1)"),
        ):
            with self.subTest(source=source), self.assertRaises(RuntimeError):
                adapt_source(source)

    def test_error_seed_does_not_consume_another_workers_seed(self):
        calls = []

        def upstream(**kwargs):
            calls.append(kwargs)
            # Public upstream decrements test_num on errors. The managed outer
            # schedule still counts the assigned seed, without borrowing one.
            return {"succ_num": 0, "test_num": 0}

        result = bounded_evaluator(upstream, [1000000, 1000002])(
            expert_check=False, start_seed=0, max_seed=-1, test_total_num=999
        )
        self.assertEqual(result, {"succ_num": 0, "test_num": 2})
        self.assertEqual([c["start_seed"] for c in calls], [1000000, 1000002])
        self.assertTrue(all(c["start_seed"] == c["max_seed"] for c in calls))
        self.assertTrue(all(c["test_total_num"] == 1 for c in calls))


class PublicMergeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "insert_hole/run"
        self.root.mkdir(parents=True)

    def public_record(self, root, seed, result="success"):
        path = root / "metadata.json"
        rows = json.loads(path.read_text()) if path.exists() else {}
        rows[str(seed)] = {"result": result, "cost_time": 1.5, "cost_step": 12}
        write_json(path, rows)
        # Public VideoHandler.close(result) renames video/<seed>.mp4 this way.
        video = root / "video" / f"{seed}_{result}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"fake-video")
        (root / "log.log").write_text(f"Original worker detail for {seed}\n")
        return video

    def plan(self, boundary=1000000, total=4):
        attempt = self.root / ".eval_runs/attempt"
        workers = [
            {
                "worker_name": f"worker_{i}",
                "seeds": seeds,
                "result_root": str(attempt / "workers" / f"worker_{i}"),
            }
            for i, seeds in enumerate(seed_shards(1000000, total, boundary, 2))
        ]
        write_json(
            attempt / "attempt.json",
            {
                "contract": {
                    "start_seed": 1000000,
                    "episodes": total,
                    "task": "insert_hole",
                    "checkpoint": "/weights",
                },
                "resume_from": boundary,
                "before_fingerprint": fingerprint(self.root),
                "workers": workers,
            },
        )
        return attempt, workers

    def test_flat_workers_merge_video_metadata_and_summary(self):
        attempt, workers = self.plan()
        for worker in workers:
            output = Path(worker["result_root"])
            for seed in worker["seeds"]:
                self.public_record(
                    output, seed, "error" if seed == 1000002 else "success"
                )
        summary = consolidate(self.root, attempt, lambda _: (True, ""))
        self.assertEqual(summary["status"], "completed_with_errors")
        self.assertEqual((summary["success"], summary["error"]), (3, 1))
        self.assertEqual(summary["missing_seeds"], [])
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["raw_success_rate"], 0.75)
        self.assertEqual(set(read_records(self.root)), set(range(1000000, 1000004)))
        videos = artifact_files(self.root, "video")
        self.assertEqual(videos[1000002].name, "1000002_error.mp4")
        self.assertTrue(all(p.parent.name == "worker_0" for p in videos.values()))
        self.assertIn("Partial Result: 3/3 valid episodes success; requested=4.", (self.root / "log.log").read_text())
        # Detailed upstream logs remain intact under their private attempt.
        self.assertTrue(
            all((Path(w["result_root"]) / "log.log").exists() for w in workers)
        )

    def test_public_prefix_resume_is_archived_and_readable(self):
        self.public_record(self.root, 1000000)
        self.public_record(self.root, 1000001, "failed")
        attempt, workers = self.plan(boundary=1000001, total=3)
        for worker in workers:
            for seed in worker["seeds"]:
                self.public_record(Path(worker["result_root"]), seed)
        summary = consolidate(self.root, attempt, lambda _: (True, ""))
        self.assertEqual(summary["success"], 3)
        self.assertEqual(
            read_records(attempt / "previous")[1000001]["result"], "failed"
        )
        self.assertEqual(read_records(self.root)[1000000]["result"], "success")
        self.assertTrue((attempt / "previous/video/1000001_failed.mp4").exists())

    def test_summary_is_not_a_seed_map(self):
        write_json(
            self.root / "metadata.json",
            {
                "policy_name": "SimpleTac",
                "run_root": str(self.root),
                "total_episodes": 0,
            },
        )
        self.assertEqual(read_records(self.root), {})

    def test_malformed_mixed_and_symlink_metadata_fail_closed(self):
        flat = self.root / "metadata.json"
        for payload in (
            [],
            {"1000000": {"result": "unknown"}},
            {"1000000": {"result": "success"}, "total_episodes": 1},
            {"unexpected": "format"},
        ):
            with self.subTest(payload=payload):
                write_json(flat, payload)
                with self.assertRaises(ValueError):
                    read_records(self.root)
        write_json(flat, {"1000000": {"result": "success"}})
        write_json(
            self.root / "metadata/worker_0.json", {"1000001": {"result": "success"}}
        )
        with self.assertRaisesRegex(ValueError, "Mixed"):
            read_records(self.root)
        flat.unlink()
        flat.symlink_to(self.root / "metadata/worker_0.json")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            read_records(self.root)


if __name__ == "__main__":
    unittest.main()
