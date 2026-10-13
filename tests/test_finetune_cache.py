from copy import deepcopy
from pathlib import Path

import pytest

from fakegpu._finetune_cache import (
    CACHE_VARIABLES,
    capture_cache_regime,
    validate_cache_regime,
)


def _roots(tmp_path):
    env = {}
    for variable in CACHE_VARIABLES:
        path = tmp_path / variable
        path.mkdir(parents=True)
        env[variable] = str(path)
    return env


def test_uncontrolled_does_not_scan_environment():
    value = capture_cache_regime(environ={"TRITON_CACHE_DIR": "/missing"})
    assert value["mode"] == "uncontrolled" and value["files"] == []
    validate_cache_regime(value)


def test_controlled_manifest_is_relocatable_and_sensitive_to_content(tmp_path):
    first, second = _roots(tmp_path / "a"), _roots(tmp_path / "b")
    assert capture_cache_regime(
        "empty_kernel_cache", environ=first
    ) == capture_cache_regime("empty_kernel_cache", environ=second)
    for env in (first, second):
        (Path(env["TRITON_CACHE_DIR"]) / "kernel.bin").write_bytes(b"kernel-a")
    left = capture_cache_regime("populated_kernel_cache", environ=first)
    right = capture_cache_regime("populated_kernel_cache", environ=second)
    assert left == right
    validate_cache_regime(left)
    (Path(second["TRITON_CACHE_DIR"]) / "kernel.bin").write_bytes(b"kernel-b")
    assert left != capture_cache_regime("populated_kernel_cache", environ=second)


def test_controlled_state_cannot_be_declared_without_matching_files(tmp_path):
    env = _roots(tmp_path)
    with pytest.raises(ValueError, match="nonempty"):
        capture_cache_regime("populated_kernel_cache", environ=env)
    path = Path(env["TRITON_CACHE_DIR"]) / "existing.bin"
    path.write_bytes(b"keep")
    with pytest.raises(ValueError, match="empty"):
        capture_cache_regime("empty_kernel_cache", environ=env)
    assert path.read_bytes() == b"keep"
    del env["CUDA_CACHE_PATH"]
    with pytest.raises(ValueError, match="explicit CUDA_CACHE_PATH"):
        capture_cache_regime("populated_kernel_cache", environ=env)


def test_overlapping_or_linked_cache_roots_are_rejected(tmp_path):
    env = _roots(tmp_path)
    env["CUDA_CACHE_PATH"] = env["TRITON_CACHE_DIR"]
    with pytest.raises(ValueError, match="disjoint"):
        capture_cache_regime("empty_kernel_cache", environ=env)
    env["CUDA_CACHE_PATH"] = str(tmp_path / "CUDA_CACHE_PATH")
    (Path(env["TRITON_CACHE_DIR"]) / "link").symlink_to(tmp_path / "CUDA_CACHE_PATH")
    with pytest.raises(ValueError, match="symlinks"):
        capture_cache_regime("empty_kernel_cache", environ=env)


@pytest.mark.parametrize(
    "change", ["unknown", "contents", "digest", "duplicate", "path"]
)
def test_invalid_serialized_regime_is_rejected(tmp_path, change):
    env = _roots(tmp_path)
    (Path(env["TRITON_CACHE_DIR"]) / "kernel.bin").write_bytes(b"kernel")
    value = capture_cache_regime("populated_kernel_cache", environ=env)
    if change == "unknown":
        value["mode"] = "warm-ish"
    elif change == "contents":
        value["mode"] = "empty_kernel_cache"
    elif change == "digest":
        value["files"][0]["sha256"] = "z" * 64
    elif change == "duplicate":
        value["files"].append(deepcopy(value["files"][0]))
    else:
        value["files"][0]["path"] = "../kernel.bin"
    with pytest.raises(ValueError, match="cache"):
        validate_cache_regime(value)
