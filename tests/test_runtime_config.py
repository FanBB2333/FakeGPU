from __future__ import annotations

import os
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock

import pytest

from fakegpu import _api, _runtime


@pytest.fixture
def runtime_options(tmp_path: Path) -> dict:
    return {
        "mode": "simulate",
        "oom_policy": "clamp",
        "unsupported_api": " WARN ",
        "dist_mode": "simulate",
        "cluster_config": tmp_path / "cluster.yaml",
        "coordinator_addr": "127.0.0.1:29501",
        "coordinator_transport": "tcp",
        "profile": "rtx3090ti",
        "device_count": 2,
        "devices": ["rtx3090ti", "test-512m"],
    }


@pytest.fixture
def native_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for library in _api._PRELOAD_LIBS:
        (tmp_path / library).touch()
    monkeypatch.setattr(_api, "library_dir", lambda **kwargs: tmp_path)
    return tmp_path


def test_config_is_immutable(runtime_options: dict) -> None:
    config = _api._FakeGpuRuntimeConfig(**runtime_options)
    with pytest.raises(FrozenInstanceError):
        config.device_count = 3


def test_env_maps_all_options_without_mutating_base(
    native_directory: Path, runtime_options: dict,
) -> None:
    base = {"SENTINEL": "preserved"}
    environment = _api.env(lib_dir=native_directory, base_env=base, **runtime_options)
    expected = {
        "FAKEGPU_RUNTIME": "native",
        "FAKEGPU_MODE": "simulate",
        "FAKEGPU_OOM_POLICY": "clamp",
        "FAKEGPU_UNSUPPORTED_API": "warn",
        "FAKEGPU_DIST_MODE": "simulate",
        "FAKEGPU_CLUSTER_CONFIG": str(runtime_options["cluster_config"]),
        "FAKEGPU_COORDINATOR_ADDR": "127.0.0.1:29501",
        "FAKEGPU_COORDINATOR_TRANSPORT": "tcp",
        "FAKEGPU_PROFILE": "rtx3090ti",
        "FAKEGPU_DEVICE_COUNT": "2",
        "FAKEGPU_PROFILES": "rtx3090ti,test-512m",
    }
    assert expected.items() <= environment.items()
    assert base == {"SENTINEL": "preserved"}
    assert environment["SENTINEL"] == "preserved"
    assert environment[_api._LIBRARY_PATH_VAR] == str(native_directory)


@pytest.mark.parametrize("mode", ["simulate", "hybrid", "passthrough"])
def test_preload_boundaries_survive_config_refactor(native_directory: Path, mode: str) -> None:
    environment = _api.env(lib_dir=native_directory, base_env={}, mode=mode)
    preload = environment[_api._PRELOAD_VAR]
    assert ("cudart" in preload) == (mode == "simulate")
    assert ("libcuda" in preload) == (mode != "passthrough")
    assert (_api._LIBRARY_PATH_VAR in environment) == (mode == "simulate")


@pytest.mark.parametrize("device_count", [0, -1])
def test_invalid_device_count(native_directory: Path, device_count: int) -> None:
    with pytest.raises(ValueError, match="device_count must be > 0"):
        _api.env(lib_dir=native_directory, base_env={}, device_count=device_count)


def test_run_forwards_config_and_subprocess_options(
    native_directory: Path, runtime_options: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess(["probe"], 0, "ok", "")
    runner = Mock(return_value=completed)
    monkeypatch.setattr(_api.subprocess, "run", runner)
    result = _api.run(["probe"], lib_dir=native_directory, check=False, capture_output=True, **runtime_options)
    assert result is completed
    arguments = runner.call_args.kwargs
    assert arguments["env"]["FAKEGPU_PROFILES"] == "rtx3090ti,test-512m"
    assert arguments["env"]["FAKEGPU_UNSUPPORTED_API"] == "warn"
    assert arguments["check"] is False
    assert arguments["capture_output"] is True


@pytest.mark.parametrize("update_env", [True, False])
def test_native_init_applies_config_with_optional_preload(
    native_directory: Path, runtime_options: dict, monkeypatch: pytest.MonkeyPatch, update_env: bool,
) -> None:
    monkeypatch.setattr(_api, "_initialized", False)
    monkeypatch.setattr(_api, "_loaded_dir", None)
    monkeypatch.setattr(_api, "_handles", {})
    monkeypatch.setattr(_api.ctypes, "CDLL", Mock(return_value=object()))
    monkeypatch.setattr(_api.os, "environ", {})
    result = _runtime.init(runtime="native", lib_dir=native_directory, update_env=update_env, **runtime_options)
    assert result.runtime == "native"
    assert os.environ["FAKEGPU_DEVICE_COUNT"] == "2"
    assert os.environ["FAKEGPU_CLUSTER_CONFIG"] == str(runtime_options["cluster_config"])
    assert (_api._PRELOAD_VAR in os.environ) == update_env
    assert (_api._LIBRARY_PATH_VAR in os.environ) == update_env
