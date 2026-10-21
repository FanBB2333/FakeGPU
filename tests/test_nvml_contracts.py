"""Native NVML answers checked against nvml.h and the nvidia-ml-py (pynvml) bindings.

Expected values come from nvml.h (CUDA 12.9 nvidia-nvml-dev) and pynvml's ctypes
definitions, not from FakeGPU itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = Path(os.environ.get("FAKEGPU_BUILD_DIR", ROOT / "build"))
LIBRARY = BUILD_DIR / ("libnvidia-ml.dylib" if sys.platform == "darwin" else "libnvidia-ml.so.1")

pytestmark = pytest.mark.skipif(not LIBRARY.exists(), reason="native libraries are not built")


def _nvml(tmp_path: Path, body: str, **environment: str):
    """Run ``body`` with pynvml bound to the native library; return its ``result``."""
    pytest.importorskip("pynvml")
    code = textwrap.dedent(
        """
        import ctypes, json
        import pynvml as N
        lib = ctypes.CDLL({library!r})
        N.nvmlLib = lib
        N._nvmlGetFunctionPointer_cache.clear()
        N.nvmlInit()
        result = None
        """
    ).format(library=str(LIBRARY)) + textwrap.dedent(body) + "\nN.nvmlShutdown()\nprint(json.dumps(result))\n"
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("FAKEGPU_", "CUDA_VISIBLE_DEVICES"))
    }
    env.update(
        PYTHONPATH=str(ROOT), FAKEGPU_HOME=str(tmp_path), FAKEGPU_MACHINE="off",
        FAKEGPU_TERMINAL_REPORT="0", FAKEGPU_REPORT_PATH=str(tmp_path / "report.json"),
        **environment,
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_architecture_uses_nvml_device_arch_ids(tmp_path: Path) -> None:
    profiles = ["gtx980", "p100", "v100", "t4", "a100", "l4", "h100", "b200"]
    result = _nvml(
        tmp_path,
        "result = [N.nvmlDeviceGetArchitecture(N.nvmlDeviceGetHandleByIndex(i)) for i in range(8)]",
        FAKEGPU_PROFILES=",".join(profiles),
    )
    import pynvml

    assert result == [
        pynvml.NVML_DEVICE_ARCH_MAXWELL, pynvml.NVML_DEVICE_ARCH_PASCAL, pynvml.NVML_DEVICE_ARCH_VOLTA,
        pynvml.NVML_DEVICE_ARCH_TURING, pynvml.NVML_DEVICE_ARCH_AMPERE, pynvml.NVML_DEVICE_ARCH_ADA,
        pynvml.NVML_DEVICE_ARCH_HOPPER, pynvml.NVML_DEVICE_ARCH_BLACKWELL,
    ]


def test_encoder_stats_write_unsigned_int_outputs(tmp_path: Path) -> None:
    # nvml.h: nvmlDeviceGetEncoderStats(device, unsigned int*, unsigned int*, unsigned int*).
    result = _nvml(
        tmp_path,
        """
        values = (ctypes.c_uint * 4)(11, 22, 33, 0xDEADBEEF)
        rc = lib.nvmlDeviceGetEncoderStats(N.nvmlDeviceGetHandleByIndex(0),
                                           *(ctypes.byref(values, 4 * i) for i in range(3)))
        result = [rc, list(values)]
        """,
        FAKEGPU_PROFILE="a100", FAKEGPU_DEVICE_COUNT="1",
    )
    assert result == [0, [0, 0, 0, 0xDEADBEEF]]
