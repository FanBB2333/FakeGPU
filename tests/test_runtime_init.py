"""Smoke tests for the side-effect-free fakegpu import and runtime router."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.setdefault("XONSH_HISTORY_BACKEND", "dummy")
    pythonpath = str(ROOT)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        env=env,
        cwd=str(ROOT),
    )


def _assert_ok(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise AssertionError(
            f"{label} failed with rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def test_import_is_side_effect_free() -> None:
    code = f"""
from pathlib import Path
import sys
import tempfile

root = Path({str(ROOT)!r})
shadow = Path(tempfile.mkdtemp(prefix="fakegpu-shadow-"))
(shadow / "torch.py").write_text("raise RuntimeError('torch import should not happen during import fakegpu')\\n")
sys.path.insert(0, str(shadow))
sys.path.insert(0, str(root))

import fakegpu

assert callable(fakegpu.init)
assert callable(fakegpu.patch_torch)
print("side-effect-free import passed")
"""
    _assert_ok(_run(code), "side-effect-free import")


def test_runtime_router_dispatch() -> None:
    code = f"""
from pathlib import Path
import sys
from unittest.mock import patch

root = Path({str(ROOT)!r})
sys.path.insert(0, str(root))

import fakegpu

# "auto" always resolves to fakecuda: the vendored upstream backend ships with
# the package, so no probing is needed.
with patch("fakegpu._runtime._init_fakecuda_runtime", return_value={{"route": "fakecuda"}}) as fake_init, \\
     patch("fakegpu._runtime._init_native_runtime", return_value={{"route": "native"}}) as native_init:
    auto_result = fakegpu.init(runtime="auto")
    assert auto_result["route"] == "fakecuda"
    fake_init.assert_called_once()
    native_init.assert_not_called()

with patch("fakegpu._runtime._init_native_runtime", return_value={{"route": "native"}}) as native_init:
    native_result = fakegpu.init(runtime="native")
    assert native_result["route"] == "native"
    native_init.assert_called_once()

with patch("fakegpu._runtime._init_fakecuda_runtime", return_value={{"route": "fakecuda"}}) as fake_init:
    fake_result = fakegpu.init(runtime="fakecuda")
    assert fake_result["route"] == "fakecuda"
    fake_init.assert_called_once()

try:
    fakegpu.init(runtime="invalid")
except ValueError:
    pass
else:
    raise AssertionError("invalid runtime should raise ValueError")

print("runtime router dispatch passed")
"""
    _assert_ok(_run(code), "runtime router dispatch")


def test_torch_accelerator_routes_through_fakecuda() -> None:
    code = f"""
from pathlib import Path
import sys

root = Path({str(ROOT)!r})
sys.path.insert(0, str(root))

from fakegpu.torch_patch import patch

patch(num_devices=2)

import torch

if hasattr(torch, "accelerator"):
    assert torch.accelerator.current_accelerator().type == "cuda"
    assert torch.accelerator.current_accelerator(check_available=True).type == "cuda"
    assert torch.accelerator.is_available() is True
    assert torch.accelerator.device_count() == 2
    assert torch.accelerator.current_device_index() == 0

    stream = torch.accelerator.current_stream()
    assert stream.is_capturing() is False

    torch.accelerator.set_device_index(1)
    assert torch.accelerator.current_device_index() == 1
    torch.accelerator.set_device_index(0)
    torch.accelerator.synchronize()

model = torch.nn.Linear(2, 1).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = model(torch.ones(1, 2, device="cuda")).sum()
loss.backward()
optimizer.step()

print("torch accelerator compatibility passed")
"""
    _assert_ok(_run(code), "torch accelerator compatibility")


def test_native_mode_preload_boundaries() -> None:
    from fakegpu._api import _preload_libs_for_mode

    simulate = _preload_libs_for_mode("simulate")
    hybrid = _preload_libs_for_mode("hybrid")
    passthrough = _preload_libs_for_mode("passthrough")

    assert any("cudart" in lib for lib in simulate)
    assert not any("cudart" in lib for lib in hybrid)
    assert not any("cudart" in lib for lib in passthrough)
    assert any("libcuda" in lib for lib in hybrid)
    assert any("nvidia-ml" in lib for lib in hybrid)
    assert passthrough == ()


def test_unsupported_api_policy_environment() -> None:
    from fakegpu._api import _apply_config_env, _FakeGpuRuntimeConfig

    env: dict[str, str] = {}
    _apply_config_env(env, _FakeGpuRuntimeConfig(unsupported_api=" WARN "))
    assert env["FAKEGPU_RUNTIME"] == "native"
    assert env["FAKEGPU_UNSUPPORTED_API"] == "warn"

    try:
        _apply_config_env({}, _FakeGpuRuntimeConfig(unsupported_api="ignore"))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid unsupported API policy should raise ValueError")


def test_fakecuda_runtime_options_are_exported() -> None:
    code = f"""
from pathlib import Path
import os
import sys

root = Path({str(ROOT)!r})
sys.path.insert(0, str(root))

import fakegpu

fakegpu.init(
    runtime="fakecuda",
    mode="hybrid",
    oom_policy="managed",
    unsupported_api="error",
    dist_mode="simulate",
)
assert os.environ["FAKEGPU_MODE"] == "hybrid"
assert os.environ["FAKEGPU_OOM_POLICY"] == "managed"
assert os.environ["FAKEGPU_UNSUPPORTED_API"] == "error"
assert os.environ["FAKEGPU_DIST_MODE"] == "simulate"
print("fakecuda runtime options passed")
"""
    _assert_ok(_run(code), "fakecuda runtime options")


def test_inplace_hybrid_env_removes_stale_library_path(monkeypatch) -> None:
    from fakegpu._api import (
        _LIBRARY_PATH_VAR,
        _FakeGpuRuntimeConfig,
        _apply_env_inplace,
    )

    monkeypatch.setenv(_LIBRARY_PATH_VAR, str(ROOT / "build"))
    _apply_env_inplace(
        ROOT / "build",
        _FakeGpuRuntimeConfig(mode="hybrid"),
    )

    assert _LIBRARY_PATH_VAR not in os.environ


def main() -> None:
    test_import_is_side_effect_free()
    test_runtime_router_dispatch()
    test_torch_accelerator_routes_through_fakecuda()
    test_native_mode_preload_boundaries()
    test_unsupported_api_policy_environment()
    print("runtime init smoke passed")


if __name__ == "__main__":
    main()
