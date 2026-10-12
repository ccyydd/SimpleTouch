"""Tests for executable UniVTAC action normalization statistics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from rlinf.data.datasets.openpi_pytorch.univtac.univtac_sft_data_loader import (
    _load_stats,
)
from rlinf.data.datasets.openpi_pytorch.univtac.univtac_sft_dataset import (
    UNIVTAC_ACTION_ALIGNMENT,
    UNIVTAC_TERMINAL_HANDLING,
)
from scripts.univtac import recompute_norm_stats as norm_script
from scripts.univtac.recompute_norm_stats import _episode_samples, _statistics


@pytest.fixture(autouse=True)
def portable_mountinfo(monkeypatch, tmp_path):
    """Use a deterministic local mount for metadata tests on non-Linux hosts."""
    if not Path("/proc/self/mountinfo").exists():
        monkeypatch.setattr(
            norm_script,
            "_read_mountinfo",
            lambda: [(Path("/"), "local", "test-device", "/")],
        )
        boot = tmp_path / "test-boot-id"
        boot.write_text("01234567-89ab-cdef-0123-456789abcdef")
        monkeypatch.setattr(norm_script, "_BOOT_ID_PATH", boot)


def test_episode_samples_start_at_current_frame_and_repeat_terminal():
    joints = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)

    states, actions = _episode_samples(joints, horizon=2)
    chunks = actions.reshape(3, 2, 8)

    np.testing.assert_array_equal(states, joints)
    np.testing.assert_array_equal(
        chunks[0, :, :7],
        joints[0:2, :7] - joints[0, :7],
    )
    np.testing.assert_array_equal(chunks[0, :, 7], joints[0:2, 7])
    np.testing.assert_array_equal(
        chunks[1, :, :7],
        np.stack([np.zeros(7, dtype=np.float32), joints[2, :7] - joints[1, :7]]),
    )
    np.testing.assert_array_equal(chunks[2, :, :7], np.zeros((2, 7)))
    np.testing.assert_array_equal(chunks[2, :, 7], np.full(2, joints[2, 7]))


def test_statistics_reject_non_finite_values():
    values = np.zeros((2, 8), dtype=np.float32)
    values[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        _statistics(values)


def _write_stats(path: Path, *, alignment: str = UNIVTAC_ACTION_ALIGNMENT) -> None:
    values = {key: [0.0] * 8 for key in ("mean", "std", "q01", "q99")}
    path.write_text(
        json.dumps(
            {
                "norm_stats": {"state": values, "actions": values},
                "metadata": {
                    "task": "insert_hole",
                    "episodes": [{"episode": 0, "frames": 3}],
                    "action_horizon": 50,
                    "action_alignment": alignment,
                    "terminal_handling": UNIVTAC_TERMINAL_HANDLING,
                    "action_semantics": "arm_delta_from_q_t_gripper_absolute",
                },
            }
        ),
        encoding="utf-8",
    )


def test_norm_stats_require_exact_task_alignment_and_episode_metadata(tmp_path: Path):
    path = tmp_path / "norm_stats.json"
    _write_stats(path)

    stats = _load_stats(
        str(path),
        expected_task="insert_hole",
        expected_action_horizon=50,
        expected_episodes=[0],
    )

    assert set(stats) == {"state", "actions"}


def test_norm_stats_reject_legacy_or_stale_alignment(tmp_path: Path):
    path = tmp_path / "norm_stats.json"
    _write_stats(path, alignment="state_t_actions_t_plus_1_to_t_plus_horizon")

    with pytest.raises(ValueError, match="metadata mismatch"):
        _load_stats(
            str(path),
            expected_task="insert_hole",
            expected_action_horizon=50,
        )


def test_norm_stats_reject_missing_metadata(tmp_path: Path):
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="no alignment metadata"):
        _load_stats(
            str(path),
            expected_task="insert_hole",
            expected_action_horizon=50,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mean", [0.0] * 7, "must have shape"),
        ("std", [float("nan")] + [0.0] * 7, "non-finite"),
        ("std", [-1.0] + [0.0] * 7, "nonnegative"),
        ("q99", [-1.0] + [0.0] * 7, "q99 below q01"),
    ],
)
def test_norm_stats_reject_invalid_arrays(
    tmp_path: Path,
    field: str,
    value: list[float],
    message: str,
):
    path = tmp_path / "norm_stats.json"
    _write_stats(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["norm_stats"]["actions"][field] = value
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises((ValueError, KeyError), match=message):
        _load_stats(
            str(path),
            expected_task="insert_hole",
            expected_action_horizon=50,
        )


def _write_joint_episode(path: Path, *, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joints = np.arange(4 * 8, dtype=np.float32).reshape(4, 8) / 100.0 + offset
    with h5py.File(path, "w") as file:
        file.require_group("embodiment").create_dataset("joint", data=joints)


def _norm_args(tmp_path: Path) -> argparse.Namespace:
    data_root = tmp_path / "data"
    for episode in range(2):
        _write_joint_episode(
            data_root / "insert_hole" / "clean" / f"{episode}.hdf5",
            offset=float(episode),
        )
    return argparse.Namespace(
        task="insert_hole",
        data_root=str(data_root),
        episode_start=0,
        episode_end=1,
        action_horizon=2,
        output=str(tmp_path / "norm_stats.json"),
        validate_only=False,
        verify_mode="full",
        write_attestation=False,
        overwrite=True,
    )


def _generate_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> argparse.Namespace:
    args = _norm_args(tmp_path)
    monkeypatch.setattr(norm_script, "_parse_args", lambda: args)
    norm_script.main()
    return args


def test_validate_only_accepts_exact_recomputed_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args = _generate_stats(tmp_path, monkeypatch)
    output = Path(args.output)
    before = output.read_bytes()
    before_mtime_ns = output.stat().st_mtime_ns
    args.validate_only = True
    args.overwrite = False

    norm_script.main()

    assert output.read_bytes() == before
    assert output.stat().st_mtime_ns == before_mtime_ns


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda document: document["metadata"]["episodes"][0].__setitem__(
                "sha256", "0" * 64
            ),
            "metadata mismatch",
        ),
        (
            lambda document: document["norm_stats"]["actions"]["mean"].__setitem__(
                0,
                document["norm_stats"]["actions"]["mean"][0] + 0.5,
            ),
            "does not match the source data",
        ),
        (
            lambda document: document["norm_stats"]["state"].__setitem__(
                "q01", [0.0] * 7
            ),
            "must have shape",
        ),
        (
            lambda document: document["norm_stats"]["state"]["mean"].__setitem__(
                0, float("nan")
            ),
            "non-finite",
        ),
        (
            lambda document: document["norm_stats"]["actions"]["q01"].__setitem__(
                0,
                document["norm_stats"]["actions"]["q99"][0] + 1.0,
            ),
            "q01 greater than q99",
        ),
        (
            lambda document: document["metadata"].__setitem__("action_horizon", 3),
            "metadata mismatch",
        ),
    ],
)
def test_validate_only_rejects_tampered_stats_source_or_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
):
    args = _generate_stats(tmp_path, monkeypatch)
    output = Path(args.output)
    document = json.loads(output.read_text(encoding="utf-8"))
    mutation(document)
    output.write_text(json.dumps(document), encoding="utf-8")
    args.validate_only = True
    args.overwrite = False

    with pytest.raises(ValueError, match=message):
        norm_script.main()


def test_validate_only_and_overwrite_are_mutually_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args = _norm_args(tmp_path)
    args.validate_only = True
    args.overwrite = True
    monkeypatch.setattr(norm_script, "_parse_args", lambda: args)

    with pytest.raises(ValueError, match="mutually exclusive"):
        norm_script.main()


def _write_full_attestation(
    args: argparse.Namespace,
) -> Path:
    args.validate_only = True
    args.overwrite = False
    args.verify_mode = "full"
    args.write_attestation = True
    norm_script.main()
    return norm_script._attestation_path(Path(args.output))


def test_full_attestation_then_attested_validation_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    args = _generate_stats(tmp_path, monkeypatch)
    output = Path(args.output)
    norm_before = output.read_bytes()
    norm_mtime_before = output.stat().st_mtime_ns
    attestation_path = _write_full_attestation(args)
    attestation_before = attestation_path.read_bytes()
    attestation_mtime_before = attestation_path.stat().st_mtime_ns
    attestation = json.loads(attestation_before)
    assert set(attestation["episodes"][0]) == norm_script._ATTESTED_SOURCE_KEYS
    filesystem_identity = attestation["episodes"][0]["filesystem_identity"]
    expected_filesystem_keys = (
        norm_script._FILESYSTEM_IDENTITY_COMMON_KEYS
        if filesystem_identity["kind"] == "nfs_export"
        else norm_script._LOCAL_FILESYSTEM_IDENTITY_KEYS
    )
    assert set(filesystem_identity) == expected_filesystem_keys
    assert attestation["norm_stats_sha256"] == norm_script._sha256_file(output)
    assert attestation_path.stat().st_mode & 0o777 == 0o644

    args.verify_mode = "attested"
    args.write_attestation = False
    monkeypatch.setattr(
        norm_script,
        "_collect_expected",
        lambda **kwargs: pytest.fail("attested mode must not read source payloads"),
    )
    norm_script.main()

    assert output.read_bytes() == norm_before
    assert output.stat().st_mtime_ns == norm_mtime_before
    assert attestation_path.read_bytes() == attestation_before
    assert attestation_path.stat().st_mtime_ns == attestation_mtime_before
    captured = capsys.readouterr().out
    assert "validation_mode=full" in captured
    assert "validation_mode=attested" in captured


def test_auto_without_attestation_falls_back_to_full_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    args = _generate_stats(tmp_path, monkeypatch)
    output = Path(args.output)
    before = output.read_bytes()
    args.validate_only = True
    args.overwrite = False
    args.verify_mode = "auto"

    norm_script.main()

    assert output.read_bytes() == before
    assert not norm_script._attestation_path(output).exists()
    captured = capsys.readouterr().out
    assert "status=fallback from=attested to=full" in captured
    assert "validation_mode=full" in captured


def test_auto_fallback_can_atomically_refresh_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    args = _generate_stats(tmp_path, monkeypatch)
    output = Path(args.output)
    args.validate_only = True
    args.overwrite = False
    args.verify_mode = "auto"
    args.write_attestation = True

    norm_script.main()

    attestation_path = norm_script._attestation_path(output)
    assert attestation_path.is_file()
    captured = capsys.readouterr().out
    assert "status=fallback from=attested to=full" in captured
    assert f"attestation={attestation_path}" in captured


def test_generation_and_full_validation_do_not_read_mountinfo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args = _norm_args(tmp_path)
    monkeypatch.setattr(norm_script, "_parse_args", lambda: args)
    monkeypatch.setattr(
        norm_script,
        "_read_mountinfo",
        lambda: pytest.fail(
            "generation and non-attesting full validation must not read mountinfo"
        ),
    )

    norm_script.main()
    args.validate_only = True
    args.overwrite = False
    args.verify_mode = "full"
    norm_script.main()

    assert not norm_script._attestation_path(Path(args.output)).exists()


def test_auto_mountinfo_failure_falls_back_to_full_without_mountinfo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    args = _generate_stats(tmp_path, monkeypatch)
    _write_full_attestation(args)

    def mountinfo_unavailable():
        raise OSError("mountinfo unavailable")

    monkeypatch.setattr(
        norm_script,
        "_read_mountinfo",
        mountinfo_unavailable,
    )
    args.verify_mode = "auto"
    args.write_attestation = False

    norm_script.main()

    captured = capsys.readouterr().out
    assert "status=fallback from=attested to=full" in captured
    assert "mountinfo unavailable" in captured
    assert "validation_mode=full" in captured


def test_attested_validation_rejects_changed_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args = _generate_stats(tmp_path, monkeypatch)
    _write_full_attestation(args)
    episode = Path(args.data_root) / args.task / "clean" / "0.hdf5"
    source_stat = episode.stat()
    os.utime(
        episode,
        ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000),
    )

    args.verify_mode = "attested"
    args.write_attestation = False
    with pytest.raises(ValueError, match="file identity mismatch"):
        norm_script.main()


def test_attested_validation_binds_norm_sha_and_source_code_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args = _generate_stats(tmp_path, monkeypatch)
    attestation_path = _write_full_attestation(args)
    output = Path(args.output)

    document = json.loads(output.read_text(encoding="utf-8"))
    document["norm_stats"]["actions"]["mean"][0] += 0.25
    output.write_text(json.dumps(document), encoding="utf-8")
    args.verify_mode = "attested"
    args.write_attestation = False
    with pytest.raises(ValueError, match="SHA256 does not match"):
        norm_script.main()

    args = _norm_args(tmp_path / "source_hash")
    monkeypatch.setattr(norm_script, "_parse_args", lambda: args)
    norm_script.main()
    attestation_path = _write_full_attestation(args)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    source_name = next(iter(attestation["source_code_sha256"]))
    attestation["source_code_sha256"][source_name] = "0" * 64
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    args.verify_mode = "attested"
    args.write_attestation = False
    with pytest.raises(ValueError, match="attestation contract mismatch"):
        norm_script.main()


def test_auto_invalid_attestation_falls_back_without_rewriting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    args = _generate_stats(tmp_path, monkeypatch)
    attestation_path = _write_full_attestation(args)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    source_name = next(iter(attestation["source_code_sha256"]))
    attestation["source_code_sha256"][source_name] = "0" * 64
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    before = attestation_path.read_bytes()

    args.verify_mode = "auto"
    args.write_attestation = False
    norm_script.main()

    assert attestation_path.read_bytes() == before
    captured = capsys.readouterr().out
    assert "status=fallback from=attested to=full" in captured
    assert "validation_mode=full" in captured


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("episode_start", False),
        ("episode_end", 1.0),
        ("action_horizon", 2.0),
    ],
)
def test_attested_validation_rejects_non_integer_top_level_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
):
    args = _generate_stats(tmp_path, monkeypatch)
    attestation_path = _write_full_attestation(args)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation[field] = value
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    args.verify_mode = "attested"
    args.write_attestation = False

    with pytest.raises(ValueError, match=f"attestation {field} must be an integer"):
        norm_script.main()


def test_nfs_identity_is_stable_across_local_mount_device_numbers(
    tmp_path: Path,
):
    source = tmp_path / "dataset" / "episode.hdf5"
    source.parent.mkdir()
    source.write_bytes(b"episode")
    mounts = [
        (
            tmp_path.resolve(),
            "nfs4",
            "NAS.EXAMPLE.COM.:/univtac",
            "/",
        )
    ]

    first = norm_script._filesystem_identity(source, device=101, mounts=mounts)
    second = norm_script._filesystem_identity(source, device=909, mounts=mounts)

    assert first == second
    assert first == {
        "kind": "nfs_export",
        "fs_type": "nfs",
        "mount_source": "nas.example.com:/univtac",
        "mount_root": "/",
        "relative_path": "dataset/episode.hdf5",
    }
    assert "device" not in first


def test_local_identity_remains_bound_to_host_and_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "episode.hdf5"
    source.write_bytes(b"episode")
    boot_id = tmp_path / "boot_id"
    boot_id.write_text(
        "12345678-1234-1234-1234-123456789abc\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(norm_script, "_BOOT_ID_PATH", boot_id)
    mounts = [(tmp_path.resolve(), "ext4", "/dev/disk-a", "/")]

    first = norm_script._filesystem_identity(source, device=101, mounts=mounts)
    other_device = norm_script._filesystem_identity(source, device=102, mounts=mounts)
    other_export = norm_script._filesystem_identity(
        source,
        device=101,
        mounts=[(tmp_path.resolve(), "ext4", "/dev/disk-b", "/")],
    )

    assert first != other_device
    assert first != other_export
    assert first["host_boot_id"] == boot_id.read_text(encoding="utf-8").strip()


def test_attested_validation_accepts_nfs_device_remapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mount = (tmp_path.resolve(), "nfs", "nas.example.com:/", "/")
    monkeypatch.setattr(norm_script, "_read_mountinfo", lambda: [mount])
    args = _generate_stats(tmp_path, monkeypatch)
    _write_full_attestation(args)
    filesystem_identity = norm_script._filesystem_identity

    def remapped_identity(
        path: Path,
        *,
        device: int,
        mounts: list[tuple[Path, str, str, str]] | None = None,
    ) -> dict[str, int | str]:
        return filesystem_identity(path, device=device + 12345, mounts=mounts)

    monkeypatch.setattr(norm_script, "_filesystem_identity", remapped_identity)
    args.verify_mode = "attested"
    args.write_attestation = False

    norm_script.main()


def test_attested_validation_rejects_local_device_remapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    boot_id = tmp_path / "boot_id"
    boot_id.write_text(
        "12345678-1234-1234-1234-123456789abc\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(norm_script, "_BOOT_ID_PATH", boot_id)
    mount = (tmp_path.resolve(), "ext4", "/dev/disk-a", "/")
    monkeypatch.setattr(norm_script, "_read_mountinfo", lambda: [mount])
    args = _generate_stats(tmp_path, monkeypatch)
    _write_full_attestation(args)
    filesystem_identity = norm_script._filesystem_identity

    def remapped_identity(
        path: Path,
        *,
        device: int,
        mounts: list[tuple[Path, str, str, str]] | None = None,
    ) -> dict[str, int | str]:
        return filesystem_identity(path, device=device + 1, mounts=mounts)

    monkeypatch.setattr(norm_script, "_filesystem_identity", remapped_identity)
    args.verify_mode = "attested"
    args.write_attestation = False

    with pytest.raises(ValueError, match="filesystem identity mismatch"):
        norm_script.main()


@pytest.mark.parametrize("mutation", ("different_export", "extra_schema_key"))
def test_attested_validation_rejects_wrong_export_or_identity_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
):
    mount = (tmp_path.resolve(), "nfs", "nas.example.com:/", "/")
    monkeypatch.setattr(norm_script, "_read_mountinfo", lambda: [mount])
    args = _generate_stats(tmp_path, monkeypatch)
    attestation_path = _write_full_attestation(args)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    identity = attestation["episodes"][0]["filesystem_identity"]
    if mutation == "different_export":
        identity["mount_source"] = "other.example.com:/"
        message = "filesystem identity mismatch"
    else:
        identity["unexpected"] = "not-allowed"
        message = "filesystem_identity schema mismatch"
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    args.verify_mode = "attested"
    args.write_attestation = False

    with pytest.raises(ValueError, match=message):
        norm_script.main()


@pytest.mark.parametrize(
    ("validate_only", "verify_mode", "message"),
    [
        (False, "full", "requires --validate-only"),
        (True, "attested", "requires --verify-mode=full or auto"),
    ],
)
def test_write_attestation_requires_full_or_auto_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validate_only: bool,
    verify_mode: str,
    message: str,
):
    args = _norm_args(tmp_path)
    args.validate_only = validate_only
    args.verify_mode = verify_mode
    args.write_attestation = True
    args.overwrite = False
    monkeypatch.setattr(norm_script, "_parse_args", lambda: args)

    with pytest.raises(ValueError, match=message):
        norm_script.main()
