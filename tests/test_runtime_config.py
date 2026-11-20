from __future__ import annotations

import os
import subprocess
import warnings
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
        "profile": "rtx3090ti-reference",
        "device_count": 2,
        "devices": ["rtx3090ti-reference", "test-512m"],
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
        "FAKEGPU_PROFILE": "rtx3090ti-reference",
        "FAKEGPU_DEVICE_COUNT": "2",
        "FAKEGPU_PROFILES": "rtx3090ti-reference,test-512m",
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


@pytest.mark.parametrize("visible_devices", ["1", "1,0", "GPU-test-uuid", ""])
def test_masked_hybrid_uses_physical_nvml_enumeration(native_directory: Path, visible_devices: str) -> None:
    environment = _api.env(
        lib_dir=native_directory,
        base_env={"CUDA_VISIBLE_DEVICES": visible_devices},
        mode="hybrid",
    )
    preload = environment[_api._PRELOAD_VAR]
    assert "libcuda" in preload
    assert "libnccl" in preload
    assert "nvidia-ml" not in preload
    assert environment["CUDA_VISIBLE_DEVICES"] == visible_devices


def test_child_preload_uses_its_own_visibility_environment(native_directory: Path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    environment = _api.env(lib_dir=native_directory, base_env={}, mode="hybrid")
    assert "nvidia-ml" in environment[_api._PRELOAD_VAR]


def test_repeated_mode_changes_remove_stale_fakegpu_preloads(native_directory: Path) -> None:
    environment = {_api._PRELOAD_VAR: "/other/profiler.so", "CUDA_VISIBLE_DEVICES": "1"}
    for mode in ("simulate", "simulate", "hybrid", "passthrough", "simulate"):
        environment = _api.env(lib_dir=native_directory, base_env=environment, mode=mode)
        entries = environment[_api._PRELOAD_VAR].split(":")
        assert entries.count("/other/profiler.so") == 1
        assert len(entries) == len(set(entries))
        assert ("cudart" in environment[_api._PRELOAD_VAR]) == (mode == "simulate")
        assert ("nvidia-ml" in environment[_api._PRELOAD_VAR]) == (mode == "simulate")
        if mode == "passthrough":
            assert entries == ["/other/profiler.so"]


def test_mode_switch_removes_unversioned_preloads(native_directory: Path) -> None:
    old_paths = [str(native_directory / _api._fallback_name(lib)) for lib in _api._PRELOAD_LIBS]
    for separator in (":", " " if not _api._is_macos() else ":"):
        environment = _api.env(
            lib_dir=native_directory,
            base_env={_api._PRELOAD_VAR: separator.join(["/other/profiler.so", *old_paths])},
            mode="passthrough",
        )
        assert environment[_api._PRELOAD_VAR] == "/other/profiler.so"


@pytest.mark.parametrize("mode", ["simulate", "hybrid"])
def test_preload_names_the_files_of_a_fallback_only_directory(tmp_path: Path, mode: str) -> None:
    # library_dir() accepts a directory holding only the fallback names. A
    # preload entry naming a missing file is skipped by ld.so ("cannot be
    # preloaded ... ignored") and makes dyld terminate the process.
    for library in _api._PRELOAD_LIBS:
        (tmp_path / _api._fallback_name(library)).touch()
    assert _api.library_dir(lib_dir=tmp_path) == tmp_path.resolve()

    environment = _api.env(lib_dir=tmp_path, base_env={}, mode=mode)

    entries = environment[_api._PRELOAD_VAR].split(":")
    assert entries
    assert all(Path(entry).is_file() for entry in entries)


@pytest.fixture
def macos_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_api, "_is_macos", lambda: True)
    monkeypatch.setattr(_api, "_macos_injection_warning_emitted", False)


# Apple's System Integrity Protection Guide lists /System, /bin, /sbin and
# /usr (except /usr/local) as system-only; SIP purges DYLD_* variables when it
# launches an executable from there.
@pytest.mark.parametrize(
    ("command", "protected"),
    [
        ("/bin/sh", True),
        ("/bin/bash", True),
        ("/sbin/ping", True),
        ("/usr/sbin/sysctl", True),
        ("/usr/bin/env", True),
        ("/System/Library/Frameworks/Python.framework/python3", True),
        ("/usr/local/bin/python3", False),
        ("/opt/homebrew/bin/python3", False),
    ],
)
def test_macos_warning_covers_sip_protected_locations(
    macos_warning: None, monkeypatch: pytest.MonkeyPatch, command: str, protected: bool,
) -> None:
    monkeypatch.setattr(_api, "_resolve_executable_path", lambda executable: os.fspath(executable))
    monkeypatch.setattr(_api, "_script_interpreter", lambda path: "")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _api._warn_if_macos_injection_may_be_blocked(command)
    assert bool(caught) == protected


@pytest.mark.parametrize(
    ("interpreter", "protected"),
    [("/usr/bin/env python3", True), ("/bin/sh", True), ("{tmp}/python3", False)],
)
def test_macos_warning_follows_the_script_interpreter(
    macos_warning: None, tmp_path: Path, interpreter: str, protected: bool,
) -> None:
    script = tmp_path / "train.sh"
    script.write_text(f"#!{interpreter.format(tmp=tmp_path)}\necho ok\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _api._warn_if_macos_injection_may_be_blocked(script)
    assert bool(caught) == protected
    if protected:
        assert "interpreter" in str(caught[0].message)


def test_launcher_exit_status_follows_env_for_commands_it_cannot_run(
    native_directory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    # POSIX env(1): 127 when the utility is not found, 126 when it is found
    # but cannot be invoked.
    from fakegpu.__main__ import main

    monkeypatch.setattr(_api, "_macos_injection_warning_emitted", True)
    assert main(["fakegpu-no-such-command-xyz"]) == 127
    assert "fakegpu-no-such-command-xyz" in capsys.readouterr().err

    script = tmp_path / "not-executable.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o644)
    assert main([str(script)]) == 126


def test_native_init_honors_physical_visibility(native_directory: Path, monkeypatch) -> None:
    monkeypatch.setattr(_api, "_initialized", False)
    monkeypatch.setattr(_api, "_loaded_dir", None)
    monkeypatch.setattr(_api, "_handles", {})
    monkeypatch.setattr(_api.ctypes, "CDLL", Mock(return_value=object()))
    monkeypatch.setattr(_api.os, "environ", {"CUDA_VISIBLE_DEVICES": "1"})
    result = _api.init(lib_dir=native_directory, mode="hybrid")
    assert all("nvidia-ml" not in name for name in result.handles)
    assert "nvidia-ml" not in os.environ[_api._PRELOAD_VAR]


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
    assert arguments["env"]["FAKEGPU_PROFILES"] == "rtx3090ti-reference,test-512m"
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


# pynvml (nvidia-ml-py) loads CDLL("libnvidia-ml.so.1") on every non-Windows
# platform; nvitop loads CDLL("libcudart.so") on Linux. The build directory
# links those names and the wheel ships copies (a wheel cannot hold
# symlinks). A copy loads as a second image and must share FakeGPU's state.
_ALIAS_PROBE = """
import ctypes, sys
import pynvml
pynvml.nvmlInit()
count = pynvml.nvmlDeviceGetCount()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
runtime = ctypes.CDLL(sys.argv[1])
pointer = ctypes.c_void_p()
assert runtime.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(256 << 20)) == 0
print(count, pynvml.nvmlDeviceGetMemoryInfo(handle).used >> 20)
"""


@pytest.mark.parametrize("layout", ["build", "wheel"])
def test_unmodified_pynvml_and_nvitop_names_load_fakegpu_libraries(
    tmp_path: Path, layout: str,
) -> None:
    import shutil
    import sys

    pytest.importorskip("pynvml")
    try:
        build = _api.library_dir()
    except FileNotFoundError:
        pytest.skip("native libraries are not built")
    macos = sys.platform == "darwin"
    alias = "libnvidia-ml.so.1" if macos else "libcudart.so"
    runtime = "libcudart.dylib" if macos else alias
    assert (build / alias).exists(), "rebuild: the build lacks " + alias
    lib_dir = build
    if layout == "wheel":
        lib_dir = tmp_path / "_native"
        lib_dir.mkdir()
        for name in (*_api._PRELOAD_LIBS, alias):
            shutil.copyfile((build / name).resolve(), lib_dir / name)
    env = {key: value for key, value in os.environ.items() if not key.startswith("FAKEGPU_")}
    env.update(
        FAKEGPU_DEVICE_COUNT="2",
        FAKEGPU_TERMINAL_REPORT="0",
        FAKEGPU_REPORT_PATH=str(tmp_path / "report.json"),
    )
    completed = subprocess.run(
        [sys.executable, "-m", "fakegpu", "--lib-dir", str(lib_dir),
         sys.executable, "-c", _ALIAS_PROBE, runtime],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["2", "256"]
