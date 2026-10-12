"""Integrity tests for versioned SimpleTac frozen-T3 caches."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from rlinf.models.embodiment.openpi_pytorch.pi0_model import t3_cache_manifest
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_cache_manifest import (
    T3_WORLD_CACHE_INCOMPLETE,
    T3_WORLD_CACHE_LOCK,
    T3_WORLD_CACHE_SCHEMA_VERSION,
    build_t3_world_cache_manifest,
    exclusive_t3_world_cache_lock,
    validate_t3_world_cache_files,
    validate_t3_world_cache_manifest,
    validate_t3_world_cache_sources,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_preprocess_profile import (
    T3_UNIVTAC_PREPROCESS_PROFILE,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.t3_tactile_encoder import (
    T3_LEGACY_CACHE_IMPLEMENTATION_VERSION,
    T3_LEGACY_PREPROCESS_VERSION,
)
from scripts.univtac import cache_t3_latents as cache_script
from scripts.univtac import validate_t3_cache_manifest as fast_cache_validator


def _checkpoint(path: Path) -> Path:
    (path / "encoders").mkdir(parents=True)
    (path / "encoders" / "mini.pth").write_bytes(b"mini")
    (path / "trunk.pth").write_bytes(b"trunk")
    return path


def _complete_manifest(tmp_path: Path, *, finite: bool = True):
    source = tmp_path / "data" / "insert_hole" / "clean" / "0.hdf5"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"episode")
    cache_root = tmp_path / "cache"
    cache = cache_root / "insert_hole" / "clean" / "0.npy"
    cache.parent.mkdir(parents=True)
    values = np.zeros((2, 2, 2, 1024), dtype=np.float16)
    if not finite:
        values[0, 0, 0, 0] = np.nan
    np.save(cache, values)
    manifest = build_t3_world_cache_manifest(
        task="insert_hole",
        episode_lengths={0: 2},
        pool_grid=1,
        checkpoint_dir=_checkpoint(tmp_path / "t3"),
        episode_paths={0: source},
        include_source_sha256=True,
        cache_paths={0: cache},
    )
    return manifest, source, cache_root


def _episode(path: Path, *, lengths: dict[str, int] | None = None) -> Path:
    lengths = lengths or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    streams = {
        "embodiment/joint": (lengths.get("joint", 2), 8),
        "observation/head/rgb": (lengths.get("head", 2),),
        "observation/wrist/rgb": (lengths.get("wrist", 2),),
        "tactile/left_gsmini/rgb_marker": (lengths.get("left", 2),),
        "tactile/right_gsmini/rgb_marker": (lengths.get("right", 2),),
    }
    with h5py.File(path, "w") as file:
        for key, shape in streams.items():
            parent, name = key.rsplit("/", 1)
            file.require_group(parent).create_dataset(
                name,
                data=np.zeros(shape, dtype=np.float32),
            )
    return path


def _validation_fixture(tmp_path: Path):
    data_root = tmp_path / "data"
    episode = _episode(data_root / "insert_hole" / "clean" / "0.hdf5")
    cache_root = tmp_path / "cache"
    cache = cache_root / "insert_hole" / "clean" / "0.npy"
    cache.parent.mkdir(parents=True)
    np.save(cache, np.zeros((2, 2, 2, 1024), dtype=np.float16))
    checkpoint = _checkpoint(tmp_path / "t3")
    manifest = build_t3_world_cache_manifest(
        task="insert_hole",
        episode_lengths={0: 2},
        pool_grid=1,
        checkpoint_dir=checkpoint,
        episode_paths={0: episode},
        include_source_sha256=True,
        cache_paths={0: cache},
    )
    (cache_root / "insert_hole" / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        task="insert_hole",
        data_root=str(data_root),
        output_root=str(cache_root),
        tactile_encoder=str(checkpoint),
        episode_start=0,
        episode_end=0,
        batch_size=1,
        pool_grid=1,
        device="cuda",
        validate_samples=0,
        validate_only=True,
        overwrite=False,
    )
    return args, episode, cache_root


def test_complete_cache_manifest_binds_decoder_source_and_files(tmp_path: Path):
    manifest, source, cache_root = _complete_manifest(tmp_path)

    assert manifest["schema_version"] == T3_WORLD_CACHE_SCHEMA_VERSION
    assert manifest["preprocess"]["univtac_rgb_decoder_sha256"]
    profile = manifest["preprocess"]["profile"]
    assert profile["profile_id"] == "legacy_imagenet_v1"
    assert profile["input"]["field"] == "rgb_marker"
    assert profile["input"]["channel_order"] == "RGB"
    assert profile["geometry"]["resize_shorter"] == 256
    assert profile["geometry"]["crop_size"] == [224, 224]
    assert profile["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert manifest["episodes"]["0"]["source"]["sha256"]
    assert manifest["episodes"]["0"]["cache"]["sha256"]
    validate_t3_world_cache_sources(
        manifest,
        {0: source},
        verify_sha256=True,
    )
    validate_t3_world_cache_files(
        manifest,
        cache_root,
        "insert_hole",
        verify_sha256=True,
        verify_finite=True,
    )


def test_cache_manifest_rejects_stale_rgb_decoder(tmp_path: Path):
    expected, _, _ = _complete_manifest(tmp_path)
    stale = json.loads(json.dumps(expected))
    stale["preprocess"]["univtac_rgb_decoder_sha256"] = "stale"

    with pytest.raises(ValueError, match="univtac_rgb_decoder_sha256"):
        validate_t3_world_cache_manifest(
            stale,
            expected,
            manifest_path=tmp_path / "manifest.json",
        )


def _cross_runtime_manifest(tmp_path: Path):
    producer, _, _ = _complete_manifest(tmp_path)
    producer["runtime"].update(opencv="4.13.0", torch="2.7.1+cu126", torch_cuda="12.6")
    consumer = copy.deepcopy(producer)
    consumer["runtime"].update(
        opencv="4.11.0", torch="2.6.0+rocm6.3.4", torch_cuda="None"
    )
    return producer, consumer


def test_consumer_accepts_cross_runtime_without_mutating_provenance(tmp_path: Path):
    producer, consumer = _cross_runtime_manifest(tmp_path)
    before = copy.deepcopy((producer, consumer))
    with pytest.warns(UserWarning, match="immutable cache consumer accepted"):
        validate_t3_world_cache_manifest(
            producer, consumer, manifest_path="manifest.json", runtime_policy="consumer"
        )
    assert (producer, consumer) == before
    # Cache generation and replay verification must still reject this pair.
    with pytest.raises(ValueError, match="runtime.torch"):
        validate_t3_world_cache_manifest(
            producer, consumer, manifest_path="manifest.json"
        )


@pytest.mark.parametrize(
    "field,value",
    [
        (("encoder", "checkpoint_sha256"), "wrong"),
        (("encoder", "source_sha256"), "wrong"),
        (("preprocess", "univtac_rgb_decoder_sha256"), "wrong"),
        (("preprocess", "cache_generator_sha256"), "wrong"),
        (("preprocess", "profile", "input", "field"), "wrong"),
        (("output", "pool_grid"), 6),
        (("output", "dtype"), "float32"),
        (("output", "sensor_order"), ["right_gsmini_rgb", "left_gsmini_rgb"]),
        (("episodes", "0", "frames"), 999),
        (("task",), "put_bottle_in_shelf"),
        (("status",), "incomplete"),
        (("schema_version",), 999),
    ],
)
def test_consumer_rejects_semantic_changes_across_runtime(tmp_path, field, value):
    producer, consumer = _cross_runtime_manifest(tmp_path)
    target = producer
    for part in field[:-1]:
        target = target[part]
    target[field[-1]] = value
    with pytest.raises(ValueError, match="mismatched fields"):
        validate_t3_world_cache_manifest(
            producer, consumer, manifest_path="manifest.json", runtime_policy="consumer"
        )


@pytest.mark.parametrize("runtime", [None, {}, {"torch": ""}])
def test_consumer_requires_producer_runtime_provenance(tmp_path, runtime):
    producer, consumer = _cross_runtime_manifest(tmp_path)
    producer["runtime"] = runtime
    with pytest.raises(ValueError, match="runtime provenance"):
        validate_t3_world_cache_manifest(
            producer, consumer, manifest_path="manifest.json", runtime_policy="consumer"
        )


def test_unknown_runtime_policy_rejected(tmp_path):
    producer, consumer = _cross_runtime_manifest(tmp_path)
    with pytest.raises(ValueError, match="Unknown T3 cache runtime policy"):
        validate_t3_world_cache_manifest(
            producer, consumer, manifest_path="manifest.json", runtime_policy="typo"
        )


def test_schema2_official_v3_cache_is_whitelisted_only_for_legacy_profile(
    tmp_path: Path,
):
    expected, _, _ = _complete_manifest(tmp_path)
    legacy = json.loads(json.dumps(expected))
    legacy["schema_version"] = 2
    legacy["encoder"]["implementation_version"] = T3_LEGACY_CACHE_IMPLEMENTATION_VERSION
    legacy["encoder"]["source_sha256"] = (
        "af3df5fd3917708a2f9d5f78acb4d481a614f327eebb1ca3307e49fcf22372f9"
    )
    legacy["preprocess"] = {
        "t3": T3_LEGACY_PREPROCESS_VERSION,
        "univtac_rgb_decoder_sha256": (
            "71c14b4ee43499026160ffbd4df76ba585162482782cd65dd5f9f4b1528c8c0b"
        ),
        "cache_generator_sha256": (
            "c20fd2e8e50d7639f4c5aeb89e8d39d2c59e266a264c7bb8f99e838a2d279307"
        ),
    }

    validate_t3_world_cache_manifest(
        legacy,
        expected,
        manifest_path=tmp_path / "manifest.json",
    )

    stale = json.loads(json.dumps(legacy))
    stale["preprocess"]["cache_generator_sha256"] = "stale"
    with pytest.raises(ValueError, match="compatibility whitelist"):
        validate_t3_world_cache_manifest(
            stale,
            expected,
            manifest_path=tmp_path / "manifest.json",
        )

    tasknorm_expected = build_t3_world_cache_manifest(
        task="insert_hole",
        episode_lengths={0: 2},
        pool_grid=1,
        checkpoint_dir=tmp_path / "t3",
        preprocess_profile=T3_UNIVTAC_PREPROCESS_PROFILE,
    )
    with pytest.raises(ValueError, match="mismatched fields"):
        validate_t3_world_cache_manifest(
            legacy,
            tasknorm_expected,
            manifest_path=tmp_path / "manifest.json",
        )


def test_cache_validation_rejects_changed_source(tmp_path: Path):
    manifest, source, _ = _complete_manifest(tmp_path)
    source.write_bytes(b"changed episode")

    with pytest.raises(ValueError, match="Source metadata mismatch"):
        validate_t3_world_cache_sources(
            manifest,
            {0: source},
            verify_sha256=True,
        )


def test_cache_validation_rejects_non_finite_latent(tmp_path: Path):
    manifest, _, cache_root = _complete_manifest(tmp_path, finite=False)

    with pytest.raises(ValueError, match="non-finite"):
        validate_t3_world_cache_files(
            manifest,
            cache_root,
            "insert_hole",
            verify_sha256=True,
            verify_finite=True,
        )


def test_cache_writer_lock_is_exclusive_and_crash_recoverable(tmp_path: Path):
    task_root = tmp_path / "cache" / "insert_hole"

    with exclusive_t3_world_cache_lock(task_root):
        assert (task_root / T3_WORLD_CACHE_LOCK).is_file()
        with pytest.raises(RuntimeError, match="Another T3 cache writer"):
            with exclusive_t3_world_cache_lock(task_root):
                pass

    # Releasing the first context must make the persistent lock file reusable.
    with exclusive_t3_world_cache_lock(task_root):
        pass


def test_cache_episode_preflight_rejects_stream_length_mismatch(tmp_path: Path):
    episode = _episode(
        tmp_path / "0.hdf5",
        lengths={"right": 1},
    )

    with pytest.raises(ValueError, match="inconsistent stream lengths"):
        cache_script._inspect_episode(
            episode,
            (
                "tactile/left_gsmini/rgb_marker",
                "tactile/right_gsmini/rgb_marker",
            ),
        )


def test_cache_episode_does_not_require_policy_cameras(tmp_path: Path):
    episode = _episode(tmp_path / "0.hdf5")
    with h5py.File(episode, "a") as file:
        del file["observation"]
        for side in ("left", "right"):
            file.move(f"tactile/{side}_gsmini/rgb_marker", f"tactile/{side}_gsmini/rgb")
    assert (
        cache_script._inspect_episode(
            episode, ("tactile/left_gsmini/rgb", "tactile/right_gsmini/rgb")
        )
        == 2
    )
    with h5py.File(episode, "a") as file:
        del file["tactile/right_gsmini/rgb"]
    with pytest.raises(KeyError, match="missing required streams"):
        cache_script._inspect_episode(
            episode, ("tactile/left_gsmini/rgb", "tactile/right_gsmini/rgb")
        )


def test_camera_optional_generator_preserves_exact_prior_cache(tmp_path: Path):
    expected, _, _ = _complete_manifest(tmp_path)
    previous, current = t3_cache_manifest._CAMERA_OPTIONAL_GENERATOR_PAIR
    # Exercise this explicitly audited historical migration, not the hash of
    # whatever generator revision happens to be checked out today.
    expected["preprocess"]["cache_generator_sha256"] = current
    actual = copy.deepcopy(expected)
    actual["preprocess"]["cache_generator_sha256"] = previous
    validate_t3_world_cache_manifest(actual, expected, manifest_path="test")
    for section, field in (
        ("encoder", "checkpoint_sha256"),
        ("preprocess", "profile"),
        ("preprocess", "cache_generator_sha256"),
        ("output", "pool_grid"),
    ):
        damaged = copy.deepcopy(actual)
        damaged[section][field] = "changed"
        with pytest.raises(ValueError, match="incompatible"):
            validate_t3_world_cache_manifest(damaged, expected, manifest_path="test")
    later = copy.deepcopy(expected)
    later["preprocess"]["cache_generator_sha256"] = "later-unaudited-generator"
    with pytest.raises(ValueError, match="incompatible"):
        validate_t3_world_cache_manifest(actual, later, manifest_path="test")


def test_validate_only_zero_samples_does_not_create_t3_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args, _, _ = _validation_fixture(tmp_path)
    monkeypatch.setattr(cache_script, "_parse_args", lambda: args)

    def forbidden_encoder(*_args, **_kwargs):
        raise AssertionError("validate-only --validate-samples=0 loaded T3")

    monkeypatch.setattr(cache_script, "_create_encoder", forbidden_encoder)

    cache_script.main()


def test_fast_cache_validator_checks_manifest_without_payload_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args, _, _ = _validation_fixture(tmp_path)
    fast_args = argparse.Namespace(
        task=args.task,
        data_root=args.data_root,
        output_root=args.output_root,
        tactile_encoder=args.tactile_encoder,
        episode_start=args.episode_start,
        episode_end=args.episode_end,
        pool_grid=args.pool_grid,
    )
    monkeypatch.setattr(fast_cache_validator, "_parse_args", lambda: fast_args)

    fast_cache_validator.main()


def test_fast_cache_validator_consumes_cross_platform_cache(tmp_path, monkeypatch):
    args, _, cache_root = _validation_fixture(tmp_path)
    manifest_path = cache_root / args.task / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime"].update(torch="2.7.1+cu126", torch_cuda="12.6")
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(fast_cache_validator, "_parse_args", lambda: args)
    consumer_runtime = dict(
        manifest["runtime"], torch="2.6.0+rocm6.3.4", torch_cuda="None"
    )
    monkeypatch.setattr(
        t3_cache_manifest, "_runtime_versions", lambda: consumer_runtime
    )
    with pytest.warns(UserWarning, match="immutable cache consumer accepted"):
        fast_cache_validator.main()


def test_write_failure_keeps_incomplete_marker_and_success_clears_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args, _, cache_root = _validation_fixture(tmp_path)
    args.validate_only = False
    args.overwrite = True
    monkeypatch.setattr(cache_script, "_parse_args", lambda: args)
    monkeypatch.setattr(
        cache_script,
        "_create_encoder",
        lambda *_args, **_kwargs: (object(), object()),
    )

    def fail_cache(*_args, **_kwargs):
        raise RuntimeError("synthetic cache failure")

    monkeypatch.setattr(cache_script, "_cache_episode", fail_cache)
    with pytest.raises(RuntimeError, match="synthetic cache failure"):
        cache_script.main()
    incomplete = cache_root / "insert_hole" / T3_WORLD_CACHE_INCOMPLETE
    assert incomplete.is_file()

    def write_cache(_encoder, _episode, output_path, **_kwargs):
        np.save(output_path, np.zeros((2, 2, 2, 1024), dtype=np.float16))

    monkeypatch.setattr(cache_script, "_cache_episode", write_cache)
    monkeypatch.setattr(cache_script, "_validate_episode", lambda *_a, **_k: None)
    cache_script.main()
    assert not incomplete.exists()
