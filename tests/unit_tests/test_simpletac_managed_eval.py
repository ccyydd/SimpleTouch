"""CPU-only tests for episode scheduling, resume safety and canonical merging."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.univtac import eval_simpletac_managed as launcher
from scripts.univtac.eval_policy_managed import adapt_source, bounded_evaluator
from scripts.univtac.eval_policy_simpletouch import adapt_source as adapt_public_source
from scripts.univtac.managed_eval_results import (
    MANIFEST,
    artifact_files,
    consolidate,
    fingerprint,
    read_records,
    seed_shards,
    write_json,
)


class ManagedResultsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "insert_hole/run"
        self.root.mkdir(parents=True)

    def record(self, root, seed, result="success", content=b"valid", worker="worker_0"):
        file = root / "metadata" / f"{worker}.json"
        rows = json.loads(file.read_text()) if file.exists() else {}
        rows[str(seed)] = {"result": result, "cost_time": 1.2, "cost_step": 10}
        write_json(file, rows)
        video = root / "video" / worker / f"{seed}_{result}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(content)
        return video

    def plan(self, boundary=1000000, total=4, count=2, name="attempt"):
        attempt = self.root / ".eval_runs" / name
        workers = [
            {
                "worker_name": f"worker_{i}",
                "seeds": seeds,
                "result_root": str(attempt / f"workers/worker_{i}"),
            }
            for i, seeds in enumerate(seed_shards(1000000, total, boundary, count))
        ]
        contract = {
            "start_seed": 1000000,
            "episodes": total,
            "task": "insert_hole",
            "checkpoint": "/weights/20k",
        }
        write_json(
            attempt / "attempt.json",
            {
                "contract": contract,
                "resume_from": boundary,
                "workers": workers,
                "before_fingerprint": fingerprint(self.root),
            },
        )
        return attempt, workers

    @staticmethod
    def decode(path):
        return (True, "") if path.read_bytes() == b"valid" else (False, "bad_video")

    def test_shards_absolute_episode_ids(self):
        shards = seed_shards(1000000, 100, 1000050, 4)
        flat = sum(shards, [])
        self.assertEqual(set(flat), set(range(1000050, 1000100)))
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual([len(x) for x in shards], [13, 13, 12, 12])
        self.assertEqual(seed_shards(1000000, 2, 1000001, 8), [[1000001]])

    def test_invalid_ordinal_is_rejected(self):
        for boundary in [1, 50, 999999, 1000100]:
            with self.assertRaises(ValueError):
                seed_shards(1000000, 100, boundary, 4)

    def test_full_merge_one_visible_worker(self):
        attempt, workers = self.plan()
        for worker in workers:
            for seed in worker["seeds"]:
                self.record(
                    Path(worker["result_root"]),
                    seed,
                    "success" if seed % 2 == 0 else "failed",
                )
        result = consolidate(self.root, attempt, self.decode)
        self.assertEqual(
            (result["success"], result["failed"], result["error"]), (2, 2, 0)
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["raw_success_rate"], 0.5)
        self.assertEqual(len(list((self.root / "video/worker_0").glob("*.mp4"))), 4)
        self.assertEqual(
            list((self.root / "metadata").glob("*.json")),
            [self.root / "metadata/worker_0.json"],
        )
        self.assertEqual(read_records(self.root)[1000003]["episode"], 1000003)
        self.assertIn("Seed 1000003", (self.root / "log.log").read_text())
        clips = list((self.root / "video/worker_0").glob("*.mp4"))
        self.assertTrue(all(p.is_symlink() and not os.path.isabs(os.readlink(p)) for p in clips))
        self.assertEqual(len([p for p in self.root.rglob("*.mp4") if not p.is_symlink()]), 4)
        self.assertTrue(all(p.resolve().is_relative_to(attempt) for p in clips))

    def test_resume_preserves_prefix_and_archives_tail(self):
        first = self.record(self.root, 1000000)
        self.record(self.root, 1000001, "failed")
        self.record(self.root, 1000002, "failed", b"bad")
        # An abandoned partial clip alongside the selected failed video is OK.
        (self.root / "video/worker_0/1000002.mp4").write_bytes(b"partial")
        attempt, workers = self.plan(boundary=1000002)
        for worker in workers:
            for seed in worker["seeds"]:
                self.record(Path(worker["result_root"]), seed)
        result = consolidate(self.root, attempt, self.decode)
        self.assertEqual(result["success"], 3)
        self.assertEqual(first.read_bytes(), b"valid")
        self.assertTrue(
            (attempt / "previous/video/worker_0/1000002_failed.mp4").exists()
        )
        self.assertFalse((self.root / "video/worker_0/1000002_failed.mp4").exists())
        self.assertEqual(len(artifact_files(self.root, "video")), 4)

    def test_legacy_native_scene_and_process_logs_are_archived(self):
        self.record(self.root, 1000000)
        for name in ["scene", "logs", "workers"]:
            (self.root / name).mkdir()
            (self.root / name / "internal.json").write_text("{}")
        attempt, workers = self.plan(boundary=1000001)
        for worker in workers:
            for seed in worker["seeds"]:
                self.record(Path(worker["result_root"]), seed)
        consolidate(self.root, attempt, self.decode)
        for name in ["scene", "logs", "workers"]:
            self.assertFalse((self.root / name).exists())
            self.assertTrue((attempt / "previous" / name / "internal.json").is_file())

    def test_missing_worker_never_reuses_old_tail_as_new(self):
        self.record(self.root, 1000000)
        self.record(self.root, 1000001)
        attempt, workers = self.plan(boundary=1000001)
        self.record(Path(workers[0]["result_root"]), 1000001, "failed")
        result = consolidate(self.root, attempt, self.decode)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_seeds"], [1000002, 1000003])
        self.assertEqual(result["requested_episodes"], 4)

    def test_bad_and_missing_video_are_error(self):
        attempt, workers = self.plan()
        for worker in workers:
            for seed in worker["seeds"]:
                video = self.record(
                    Path(worker["result_root"]),
                    seed,
                    content=b"valid" if seed == 1000000 else b"corrupt",
                )
                if seed == 1000003:
                    video.unlink()
        result = consolidate(self.root, attempt, self.decode)
        self.assertEqual(result["error"], 3)
        self.assertEqual(result["valid_success_rate"], 1)
        self.assertEqual(result["success_rate"], 1)
        self.assertEqual(result["valid_episodes"], 1)
        self.assertEqual(result["status"], "completed_with_errors")
        self.assertEqual(result["raw_success_rate"], 0.25)
        self.assertEqual(read_records(self.root)[1000001]["raw_result"], "success")

    def test_duplicate_and_unassigned_results_rejected_before_backup(self):
        for mode in ["duplicate", "unassigned"]:
            with self.subTest(mode=mode):
                root = self.root / mode
                root.mkdir()
                saved = self.root
                self.root = root
                attempt, workers = self.plan()
                self.record(Path(workers[0]["result_root"]), 1000000)
                self.record(
                    Path(workers[1]["result_root"]),
                    1000000 if mode == "duplicate" else 1000099,
                )
                with self.assertRaises(ValueError):
                    consolidate(root, attempt, self.decode)
                self.assertFalse((attempt / "previous").exists())
                self.root = saved

    def test_concurrent_writer_rejected(self):
        attempt, _ = self.plan()
        (self.root / "log.log").write_text("Another live evaluator wrote this")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            consolidate(self.root, attempt, self.decode)

    def test_prefix_hole_rejected(self):
        self.record(self.root, 1000000)
        attempt, _ = self.plan(boundary=1000002)
        with self.assertRaisesRegex(ValueError, "prefix"):
            consolidate(self.root, attempt, self.decode)

    def test_merge_is_idempotent_after_completion(self):
        attempt, workers = self.plan()
        for w in workers:
            for seed in w["seeds"]:
                self.record(Path(w["result_root"]), seed)
        a = consolidate(self.root, attempt, self.decode)
        before = fingerprint(self.root)
        b = consolidate(
            self.root, attempt, lambda _: self.fail("must not decode twice")
        )
        self.assertEqual(a, b)
        self.assertEqual(before, fingerprint(self.root))

    def test_resume_interrupted_backup(self):
        self.record(self.root, 1000000)
        attempt, workers = self.plan(boundary=1000001)
        for w in workers:
            for seed in w["seeds"]:
                self.record(Path(w["result_root"]), seed)
        write_json(attempt / "merge.json", {"phase": "backup"})
        (attempt / "previous").mkdir()
        os.replace(self.root / "video", attempt / "previous/video")
        result = consolidate(self.root, attempt, self.decode)
        self.assertEqual(result["success"], 4)

    def test_interrupted_publish_reuses_classification(self):
        attempt, workers = self.plan()
        for w in workers:
            for seed in w["seeds"]:
                self.record(Path(w["result_root"]), seed)
        with mock.patch(
            "scripts.univtac.managed_eval_results.publish_artifact",
            side_effect=OSError("interrupted"),
        ):
            with self.assertRaises(OSError):
                consolidate(self.root, attempt, self.decode)
        result = consolidate(
            self.root,
            attempt,
            lambda _: self.fail("classification must be stable on recovery"),
        )
        self.assertEqual(result["success"], 4)
        self.assertEqual(len(artifact_files(self.root, "video")), 4)

    def test_metadata_duplicates_and_symlinks_rejected(self):
        self.record(self.root, 1000000)
        self.record(self.root, 1000000, worker="worker_1")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_records(self.root)
        link = self.root / "video/linked.mp4"
        link.symlink_to(self.root / "video/worker_0/1000000_success.mp4")
        with self.assertRaisesRegex(ValueError, "link"):
            artifact_files(self.root, "video")

    def test_relative_links_survive_resume_and_backup(self):
        first, workers = self.plan(name="first")
        for worker in workers:
            for seed in worker["seeds"]:
                self.record(Path(worker["result_root"]), seed)
        consolidate(self.root, first, self.decode)
        original = (self.root / "video/worker_0/1000000_success.mp4").resolve()
        second, workers = self.plan(boundary=1000002, name="second")
        for worker in workers:
            for seed in worker["seeds"]:
                self.record(Path(worker["result_root"]), seed, "failed")
        result = consolidate(self.root, second, self.decode)
        self.assertEqual((result["success"], result["failed"]), (2, 2))
        self.assertEqual((self.root / "video/worker_0/1000000_success.mp4").resolve(), original)
        saved = second / "previous/video/worker_0/1000000_success.mp4"
        self.assertEqual(saved.resolve(strict=True), original)
        self.assertEqual(len(artifact_files(self.root, "video")), 4)

    def test_crash_after_moving_relative_links_rebases_from_journal(self):
        first, workers = self.plan(name="first")
        for worker in workers:
            for seed in worker["seeds"]:
                self.record(Path(worker["result_root"]), seed)
        consolidate(self.root, first, self.decode)
        second, workers = self.plan(boundary=1000002, name="second")
        for worker in workers:
            for seed in worker["seeds"]:
                self.record(Path(worker["result_root"]), seed)
        original_replace = os.replace
        def interrupted(src, dst):
            original_replace(src, dst)
            if Path(src) == self.root / "video":
                raise OSError("crash after directory move")
        with mock.patch("scripts.univtac.managed_eval_results.os.replace", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "crash"):
                consolidate(self.root, second, self.decode)
        result = consolidate(self.root, second, self.decode)
        self.assertEqual(result["success"], 4)
        self.assertEqual(len(artifact_files(self.root, "video")), 4)

    def test_leaf_links_reject_external_absolute_empty_and_broken_targets(self):
        clip = self.root / "video/worker_0/1000000_success.mp4"
        clip.parent.mkdir(parents=True)
        outside = Path(self.temp.name) / "outside.mp4"
        outside.write_bytes(b"valid")
        inside = self.root / ".eval_runs/raw/1000000_success.mp4"
        inside.parent.mkdir(parents=True)
        inside.write_bytes(b"")
        for target in [str(outside), os.path.relpath(outside, clip.parent),
                       os.path.relpath(inside, clip.parent), "missing.mp4"]:
            with self.subTest(target=target):
                clip.symlink_to(target)
                with self.assertRaises(ValueError):
                    artifact_files(self.root, "video")
                with self.assertRaises(ValueError):
                    fingerprint(self.root)
                clip.unlink()

    def test_error_label_links_to_original_clip_without_renaming_it(self):
        attempt, workers = self.plan(total=1, count=1)
        original = self.record(Path(workers[0]["result_root"]), 1000000, content=b"corrupt")
        result = consolidate(self.root, attempt, self.decode)
        exported = self.root / "video/worker_0/1000000_error.mp4"
        self.assertEqual(result["error"], 1)
        self.assertTrue(original.exists())
        self.assertEqual(exported.resolve(), original)
        self.assertEqual(read_records(self.root)[1000000]["raw_result"], "success")

    def test_abandoned_atomic_export_link_is_hidden_and_does_not_block_recovery(self):
        attempt, workers = self.plan(total=1, count=1)
        original = self.record(Path(workers[0]["result_root"]), 1000000)
        with mock.patch("scripts.univtac.managed_eval_results.publish_artifact",
                        side_effect=OSError("interrupted publication")):
            with self.assertRaises(OSError):
                consolidate(self.root, attempt, self.decode)
        scratch = self.root / ".eval_runs/.publication_tmp"
        scratch.mkdir()
        # Model a SIGKILL after creating scratch symlink and before replacing
        # the public name. Such a leaf can be dangling until the atomic move.
        (scratch / "1000000_success.mp4.1234.abandoned").symlink_to("../../somewhere")
        result = consolidate(self.root, attempt, self.decode)
        self.assertEqual(result["success"], 1)
        self.assertEqual(artifact_files(self.root, "video")[1000000].resolve(), original)


class DriverTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.upstream = Path(self.temp.name) / "UniVTAC"
        (self.upstream / "scripts").mkdir(parents=True)
        (self.upstream / "scripts/eval_policy.py").write_text("# test upstream")
        upstream_patch = mock.patch.object(launcher, "UNIVTAC", self.upstream)
        upstream_patch.start()
        self.addCleanup(upstream_patch.stop)

    def test_coordinator_routes_workers_and_merges_after_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            upstream = Path(tmp) / "upstream/scripts/eval_policy.py"
            upstream.parent.mkdir(parents=True)
            upstream.touch()
            contract = {
                "checkpoint": "/weights",
                "task": "insert_hole",
                "task_config": "demo",
                "horizon": 50,
                "temporal_ensemble": "false",
                "ensemble_k": 0.01,
                "cpu_threads": 16,
                "use_tactile": "auto",
                "tactile_use_all_tokens": "auto",
                "livestream": "2",
                "start_seed": 1000000,
                "episodes": 4,
            }
            launched = []

            def launch(*args, **kwargs):
                plan = json.loads(
                    Path(kwargs["env"]["SIMPLETAC_MANAGED_PLAN"]).read_text()
                )
                launched.append(plan)
                output = Path(plan["result_root"])
                write_json(
                    output / "metadata/worker_0.json",
                    {str(seed): {"result": "success"} for seed in plan["seeds"]},
                )
                (output / "video/worker_0").mkdir(parents=True)
                for seed in plan["seeds"]:
                    (output / f"video/worker_0/{seed}_success.mp4").write_bytes(b"valid")
                return SimpleNamespace(
                    pid=100 + len(launched),
                    returncode=0,
                    poll=lambda: 0,
                    wait=lambda **kw: 0,
                )

            args = launcher.parser().parse_args([])
            with (
                mock.patch.object(
                    launcher,
                    "resolve_plan",
                    return_value=(root, contract, 1000000, [2, 3]),
                ),
                mock.patch.object(launcher, "UNIVTAC", upstream.parent.parent),
                mock.patch.object(
                    launcher, "reserve_gpus", return_value={2: 21, 3: 22}
                ),
                mock.patch.object(launcher, "active_process_check"),
                mock.patch("shutil.which", return_value="/executable"),
                mock.patch(
                    "scripts.simpletouch.checkpoint.validate_checkpoint_completion",
                    return_value=None,
                ),
                mock.patch("scripts.univtac.eval_policy_managed.adapt_source"),
                mock.patch.object(launcher.subprocess, "Popen", side_effect=launch),
                mock.patch.object(
                    launcher,
                    "consolidate",
                    side_effect=lambda r, a: consolidate(r, a, lambda p: (True, "")),
                ),
            ):
                result = launcher.execute(args)
            self.assertEqual(result["success"], 4)
            self.assertEqual(
                [p["seeds"] for p in launched], [[1000000, 1000002], [1000001, 1000003]]
            )
            self.assertEqual(len(artifact_files(root, "video")), 4)

    def test_error_cannot_advance_into_another_shard(self):
        calls = []

        def original(**kw):
            calls.append(kw)
            return {"succ_num": int(kw["start_seed"] == 1000002), "test_num": 0}

        result = bounded_evaluator(original, [1000000, 1000002])(
            expert_check=False, task="fake"
        )
        self.assertEqual(result, {"succ_num": 1, "test_num": 2})
        self.assertEqual([x["start_seed"] for x in calls], [1000000, 1000002])
        self.assertTrue(
            all(
                x["max_seed"] == x["start_seed"] and x["test_total_num"] == 1
                for x in calls
            )
        )

    def test_driver_adapter_routes_output_despite_local_os_import(self):
        source = """from pathlib import Path
def main():
    run_id = build_run_id()
    run_root = Path('eval_result') / policy_name / task_file_name / run_id
    env_cfg.worker_name = worker_name(0)
    import os
    return run_id, run_root, env_cfg.worker_name
"""
        env = {
            "_managed_result_root": Path("/private/worker"),
            "_managed_worker_name": "worker_3",
            "env_cfg": SimpleNamespace(),
        }
        exec(adapt_source(source), env)
        self.assertEqual(env["main"](), ("worker", Path("/private/worker"), "worker_3"))
        with self.assertRaises(RuntimeError):
            adapt_source(source.replace("worker_name(0)", "worker_name(1)"))

    def test_public_driver_routes_results_outside_univtac(self):
        source = """from pathlib import Path
def main():
    run_id = 'run-1'
    policy_name = 'SimpleTac'
    task_file_name = 'insert_hole'
    run_root = Path('eval_result') / policy_name / task_file_name / run_id
    return run_root
"""
        namespace = {"_simpletouch_eval_root": Path("/output/UniVTAC")}
        exec(adapt_public_source(source), namespace)
        self.assertEqual(
            namespace["main"](), Path("/output/UniVTAC/SimpleTac/insert_hole/run-1")
        )
        with self.assertRaisesRegex(RuntimeError, "output layout"):
            adapt_public_source(source.replace("Path('eval_result')", "Path('other')"))

    def test_public_upstream_driver_routes_results_without_changing_simulation(self):
        source = """from pathlib import Path
args_cli.livestream = 2
args_cli.enable_cameras = True
def main():
    policy_name = 'SimpleTac'
    task_file_name = 'insert_hole'
    env_cfg.save_dir = Path('eval_result') / policy_name / task_file_name / deploy_config_file.stem / curr_time
    return env_cfg.save_dir
"""
        adapted = adapt_public_source(source)
        namespace = {
            "args_cli": SimpleNamespace(),
            "env_cfg": SimpleNamespace(),
            "_simpletouch_eval_root": Path("/output/UniVTAC"),
            "_simpletouch_run_id": "unique-run",
        }
        exec(adapted, namespace)
        self.assertEqual(
            namespace["main"](),
            Path("/output/UniVTAC/SimpleTac/insert_hole/unique-run"),
        )
        self.assertEqual(namespace["args_cli"].livestream, 2)
        self.assertTrue(namespace["args_cli"].enable_cameras)
        self.assertEqual(adapted.splitlines()[:3], source.splitlines()[:3])
        with self.assertRaisesRegex(RuntimeError, "output layout"):
            adapt_public_source(source + source)
        with self.assertRaisesRegex(RuntimeError, "output layout"):
            adapt_public_source(
                source.replace(" / curr_time", " / curr_time / 'changed'")
            )

    def test_managed_default_result_root_is_not_univtac_checkout(self):
        with (
            mock.patch.object(launcher, "small_hash", return_value="identity"),
            mock.patch.dict(os.environ, {"SIMPLETOUCH_EVAL_ROOT": self.temp.name}),
        ):
            args = launcher.parser().parse_args(
                ["--checkpoint", "/weights", "--task", "insert_hole"]
            )
            root, _, _, _ = launcher.resolve_plan(args)
        self.assertEqual(
            root.parent, (Path(self.temp.name) / "SimpleTac/insert_hole").resolve()
        )
        self.assertFalse(root.is_relative_to(self.upstream))

    def test_worker_env_preserves_eval_contract(self):
        contract = {
            "checkpoint": "/weights",
            "task": "insert_hole",
            "task_config": "demo",
            "horizon": 50,
            "temporal_ensemble": "false",
            "ensemble_k": 0.01,
            "cpu_threads": 16,
            "use_tactile": "auto",
            "tactile_use_all_tokens": "auto",
            "livestream": "2",
        }
        worker = {"gpu": 3, "seeds": [1000000], "log_dir": "/logs"}
        with mock.patch.dict(
            os.environ,
            {
                "SIMPLETAC_EXECUTION_HORIZON": "25",
                "CUDA_VISIBLE_DEVICES": "7",
                "NORM_STATS": "/wrong",
            },
        ):
            env = launcher.worker_env(contract, worker, Path("/plan.json"), 17)
        self.assertEqual(env["EXECUTION_HORIZON"], "50")
        self.assertEqual(env["UNIVTAC_LIVESTREAM"], "2")
        self.assertEqual(env["PORT"], "28003")
        self.assertEqual(env["ISAAC_HTTP_PORT"], "18003")
        self.assertEqual(env["SIMPLETAC_RESERVED_GPU_FD"], "17")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
        self.assertNotIn("NORM_STATS", env)
        self.assertNotIn("SIMPLETAC_EXECUTION_HORIZON", env)

    def test_parser_episode_identifiers(self):
        args = launcher.parser().parse_args(
            [
                "--resume-dir",
                "/old",
                "--resume-from-episode",
                "1000050",
                "--gpus",
                "2,3",
            ]
        )
        self.assertEqual(args.resume_from_episode, 1000050)

    def test_plan_resume_inherits_contract_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ckpt = base / "weights"
            ckpt.mkdir()
            for name in ["config.yaml", "_SUCCESS"]:
                (ckpt / name).write_text("test")
            with mock.patch.object(launcher, "small_hash", return_value="identity"):
                new = launcher.parser().parse_args(
                    [
                        "--checkpoint",
                        str(ckpt),
                        "--task",
                        "insert_hole",
                        "--result-dir",
                        str(base / "run"),
                        "--episodes",
                        "2",
                    ]
                )
                root, contract, boundary, _ = launcher.resolve_plan(new)
                self.assertFalse(root.exists())
                self.assertEqual(boundary, 1000000)
                write_json(root / MANIFEST, {"contract": contract})
                write_json(
                    root / "metadata/worker_0.json", {"1000000": {"result": "success"}}
                )
                resume = launcher.parser().parse_args(
                    ["--resume-dir", str(root), "--resume-from-episode", "1000001"]
                )
                self.assertEqual(launcher.resolve_plan(resume)[1], contract)
                self.assertEqual(contract["horizon"], 25)
                resume.horizon = 50
                with self.assertRaisesRegex(ValueError, "contract"):
                    launcher.resolve_plan(resume)

    def test_task_horizon_defaults_and_overrides(self):
        for task, expected in [
            ("insert_hole", 25),
            ("insert_tube", 25),
            ("put_bottle_in_shelf", 50),
            ("pull_out_key", 50),
        ]:
            for override in (None, 25, 50):
                with self.subTest(task=task, override=override):
                    with (
                        tempfile.TemporaryDirectory() as tmp,
                        mock.patch.object(
                            launcher, "small_hash", return_value="identity"
                        ),
                    ):
                        args = launcher.parser().parse_args(
                            [
                                "--checkpoint",
                                "/weights",
                                "--task",
                                task,
                                "--result-dir",
                                str(Path(tmp) / "new"),
                            ]
                        )
                        args.horizon = override
                        self.assertEqual(
                            launcher.resolve_plan(args)[1]["horizon"],
                            expected if override is None else override,
                        )

    def test_resume_preserves_legacy_h50_despite_new_task_default(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(launcher, "small_hash", return_value="identity"),
        ):
            args = launcher.parser().parse_args(
                [
                    "--checkpoint",
                    "/weights",
                    "--task",
                    "insert_hole",
                    "--horizon",
                    "50",
                    "--result-dir",
                    str(Path(tmp) / "old"),
                ]
            )
            root, contract, _, _ = launcher.resolve_plan(args)
            write_json(root / MANIFEST, {"contract": contract})
            resume = launcher.parser().parse_args(
                ["--resume-dir", str(root), "--resume-from-episode", "1000000"]
            )
            self.assertEqual(launcher.resolve_plan(resume)[1]["horizon"], 50)

    def test_shell_default_defers_to_task_and_preserves_overrides(self):
        for name in ("eval_simpletac.sh", "run_eval_simpletac.sh"):
            source = (launcher.REPO / "scripts/univtac" / name).read_text()
            prefix = source[
                : source.index(
                    "\nvalidate_gpu_id()"
                    if name.startswith("run_")
                    else "\nvalidate_tcp_port()"
                )
            ]
            for override in (None, "25", "50", "auto"):
                with self.subTest(script=name, override=override):
                    env = {"PATH": "/usr/bin:/bin", "CKPT_PATH": "/weights"}
                    if override is not None:
                        env["EXECUTION_HORIZON"] = override
                    result = subprocess.run(
                        ["bash", "-c", prefix + '\nprintf "%s" "$EXECUTION_HORIZON"'],
                        env=env,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    self.assertEqual(result.stdout, override or "task")

    def test_legacy_resume_requires_explicit_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = launcher.parser().parse_args(
                ["--resume-dir", tmp, "--resume-from-episode", "1000050"]
            )
            with self.assertRaisesRegex(ValueError, "Legacy resume requires"):
                launcher.resolve_plan(args)


if __name__ == "__main__":
    unittest.main()
