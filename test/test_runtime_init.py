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
assert callable(fakegpu.init_privateuse1)
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

with patch("fakegpu._runtime._detect_custom_torch_fakegpu_available", return_value=True), \\
     patch("fakegpu._runtime._init_fakecuda_runtime", return_value={{"route": "fakecuda"}}) as fake_init, \\
     patch("fakegpu._runtime._init_native_runtime", return_value={{"route": "native"}}) as native_init:
    auto_result = fakegpu.init(runtime="auto")
    assert auto_result["route"] == "fakecuda"
    fake_init.assert_called_once()
    native_init.assert_not_called()

with patch("fakegpu._runtime._detect_custom_torch_fakegpu_available", return_value=False), \\
     patch("fakegpu._runtime._init_fakecuda_runtime", return_value={{"route": "fakecuda"}}) as fake_init, \\
     patch("fakegpu._runtime._init_native_runtime", return_value={{"route": "native"}}) as native_init:
    auto_result = fakegpu.init(runtime="auto")
    assert auto_result["route"] == "native"
    native_init.assert_called_once()
    fake_init.assert_not_called()

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


def test_editable_custom_torch_detection() -> None:
    code = f"""
from pathlib import Path
import sys
from unittest.mock import patch

root = Path({str(ROOT)!r})
sys.path.insert(0, str(root))

from fakegpu import _runtime

with patch.dict(sys.modules, {{}}, clear=False):
    sys.modules.pop("torch.fakegpu", None)
    with patch("fakegpu._runtime.importlib.util.find_spec", return_value=object()), \\
         patch.object(sys, "path", [""]):
        assert _runtime._detect_custom_torch_fakegpu_available() is True

print("editable custom torch detection passed")
"""
    _assert_ok(_run(code), "editable custom torch detection")


def main() -> None:
    test_import_is_side_effect_free()
    test_runtime_router_dispatch()
    test_editable_custom_torch_detection()
    print("runtime init smoke passed")


if __name__ == "__main__":
    main()
