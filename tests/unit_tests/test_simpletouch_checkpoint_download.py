"""CPU-only checks for published inference bundle download and identity."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.simpletouch import download_checkpoint as download  # noqa: E402


class CheckpointDownloadTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    @staticmethod
    def marker(task: str) -> dict:
        return {"training_fingerprint": {"data": {"task": task}}}

    def test_valid_local_bundle_is_reused(self):
        with mock.patch.object(
            download, "validate_checkpoint_completion", return_value=self.marker("insert_hole")
        ) as validate:
            path = download.download_checkpoint("insert_hole", output_root=self.root)
        self.assertEqual(path, self.root / "insert_hole/global_step_20000")
        self.assertEqual(
            validate.call_args.kwargs["required_paths"],
            set(download.INFERENCE_REQUIRED_PATHS),
        )
        self.assertEqual(validate.call_args.kwargs["verify_mode"], "full")

    def test_incomplete_bundle_downloads_only_inference_files(self):
        def validate_or_raise(*_args, **_kwargs):
            if not fetched:
                raise FileNotFoundError("not downloaded")
            return self.marker("pull_out_key")

        fetched = []

        def fake_snapshot_download(**kwargs):
            fetched.append(kwargs)

        with mock.patch.object(
            download, "validate_checkpoint_completion", side_effect=validate_or_raise
        ), mock.patch.dict(
            sys.modules,
            {"huggingface_hub": SimpleNamespace(snapshot_download=fake_snapshot_download)},
        ):
            path = download.download_checkpoint("pull_out_key", output_root=self.root)
        self.assertEqual(path, self.root / "pull_out_key/global_step_20000")
        self.assertEqual(len(fetched), 1)
        self.assertEqual(fetched[0]["repo_id"], download.DEFAULT_REPO_ID)
        self.assertEqual(
            set(fetched[0]["allow_patterns"]),
            {"config.json"} | {
                f"pull_out_key/global_step_20000/{name}"
                for name in (*download.INFERENCE_REQUIRED_PATHS, "_SUCCESS")
            },
        )

    def test_valid_local_bundle_does_not_contact_hub(self):
        snapshot = mock.Mock()
        with mock.patch.object(
            download, "validate_checkpoint_completion", return_value=self.marker("insert_hole")
        ), mock.patch.dict(
            sys.modules,
            {"huggingface_hub": SimpleNamespace(snapshot_download=snapshot)},
        ):
            download.download_checkpoint("insert_hole", output_root=self.root)
        snapshot.assert_not_called()

    def test_explicit_revision_is_forwarded_to_the_snapshot(self):
        snapshot = mock.Mock()
        with mock.patch.object(
            download, "validate_checkpoint_completion", return_value=self.marker("insert_tube")
        ), mock.patch.dict(
            sys.modules,
            {"huggingface_hub": SimpleNamespace(snapshot_download=snapshot)},
        ):
            download.download_checkpoint(
                "insert_tube", output_root=self.root, revision="fixed-release"
            )
        self.assertEqual(snapshot.call_args.kwargs["revision"], "fixed-release")
        self.assertEqual(snapshot.call_args.kwargs["allow_patterns"].count("config.json"), 1)

    def test_hub_index_matches_released_bundles(self):
        index = json.loads((download.REPO / "assets/huggingface/config.json").read_text())
        self.assertEqual(set(index["task_checkpoints"]), set(download.TASKS))
        self.assertEqual(
            set(index["checkpoint_files"]),
            set(download.INFERENCE_REQUIRED_PATHS) | {"_SUCCESS"},
        )
        for task, entry in index["task_checkpoints"].items():
            self.assertEqual(entry["path"], f"{task}/{download.STEP_DIR}")

    def test_wrong_task_is_rejected(self):
        with mock.patch.object(
            download, "validate_checkpoint_completion", return_value=self.marker("lift_can")
        ):
            with self.assertRaisesRegex(ValueError, "task mismatch"):
                download.validate_bundle(self.root / "global_step_20000", "insert_hole")

    def test_unsupported_task_is_rejected_before_download(self):
        with self.assertRaisesRegex(ValueError, "Unsupported task"):
            download.download_checkpoint("not_a_task", output_root=self.root)


if __name__ == "__main__":
    unittest.main()
