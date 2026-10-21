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


def test_pci_bus_ids_use_nvml_formats_and_lookup_accepts_documented_forms(tmp_path: Path) -> None:
    # nvml.h: busId "%08X:%02X:%02X.0", busIdLegacy "%04X:%02X:%02X.0"; the lookup
    # accepts %x:%x:%x.%x, %x:%x:%x and %x:%x.%x.
    result = _nvml(
        tmp_path,
        """
        pci = N.nvmlDeviceGetPciInfo(N.nvmlDeviceGetHandleByIndex(9))
        found = []
        for key in (pci.busId, pci.busIdLegacy, "0:a:0", "a:0.0", "00000000:0a:00.0"):
            found.append(N.nvmlDeviceGetIndex(N.nvmlDeviceGetHandleByPciBusId(key)))
        try:
            N.nvmlDeviceGetHandleByPciBusId("0000:7F:00.0")
            missing = None
        except N.NVMLError as exc:
            missing = exc.value
        result = [pci.busId, pci.busIdLegacy, pci.domain, pci.bus, pci.device, found, missing]
        """,
        FAKEGPU_PROFILE="a100", FAKEGPU_DEVICE_COUNT="10",
    )
    assert result == ["00000000:0A:00.0", "0000:0A:00.0", 0, 0x0A, 0, [9] * 5, 6]


def test_power_limits_stay_within_constraints_and_unknown_power_is_not_zero_watts(tmp_path: Path) -> None:
    # nvml.h: limits are in mW and the enforced/default limit is one of the
    # "possible values of power management limits" bounded by the constraints.
    # The a100 and l4 catalogs know only the power cap (400 W, 72 W, matching
    # fakegpu/data/a100_monitor_reference.json for the a100); l2 knows neither.
    result = _nvml(
        tmp_path,
        """
        def query(fn, *args):
            try:
                return fn(*args)
            except N.NVMLError as exc:
                return ["error", exc.value]
        result = []
        for i in range(3):
            h = N.nvmlDeviceGetHandleByIndex(i)
            fields = N.nvmlDeviceGetFieldValues(h, [
                N.NVML_FI_DEV_POWER_MIN_LIMIT, N.NVML_FI_DEV_POWER_MAX_LIMIT, N.NVML_FI_DEV_POWER_CURRENT_LIMIT])
            result.append([
                query(N.nvmlDeviceGetPowerManagementLimit, h), query(N.nvmlDeviceGetPowerManagementDefaultLimit, h),
                query(N.nvmlDeviceGetEnforcedPowerLimit, h), query(N.nvmlDeviceGetPowerManagementLimitConstraints, h),
                [[f.nvmlReturn, f.value.uiVal] for f in fields],
            ])
        """,
        FAKEGPU_PROFILES="a100,l4,l2",
    )
    not_supported = ["error", 3]
    assert result == [
        [400000, 400000, 400000, [100000, 400000], [[0, 100000], [0, 400000], [0, 400000]]],
        [72000, 72000, 72000, [72000, 72000], [[0, 72000], [0, 72000], [0, 72000]]],
        [not_supported, not_supported, not_supported, not_supported, [[3, 0], [3, 0], [3, 0]]],
    ]


def test_clock_queries_answer_the_requested_clock_domain(tmp_path: Path) -> None:
    # nvml.h: applications/boost/P-state clocks take an nvmlClockType_t, and the
    # supported-clock lists return INSUFFICIENT_SIZE for a too-small count, which
    # pynvml's first (count 0) call relies on; an unsupported memory clock is NOT_FOUND.
    result = _nvml(
        tmp_path,
        """
        h = N.nvmlDeviceGetHandleByIndex(0)
        mem, graphics = N.NVML_CLOCK_MEM, N.NVML_CLOCK_GRAPHICS
        memory_clock = N.nvmlDeviceGetMaxClockInfo(h, mem)
        try:
            N.nvmlDeviceGetSupportedGraphicsClocks(h, memory_clock + 1)
            unsupported = None
        except N.NVMLError as exc:
            unsupported = exc.value
        result = [
            memory_clock, N.nvmlDeviceGetMaxClockInfo(h, graphics),
            N.nvmlDeviceGetApplicationsClock(h, mem), N.nvmlDeviceGetDefaultApplicationsClock(h, mem),
            N.nvmlDeviceGetMaxCustomerBoostClock(h, mem), N.nvmlDeviceGetMinMaxClockOfPState(h, mem, 0)[1],
            N.nvmlDeviceGetApplicationsClock(h, graphics),
            N.nvmlDeviceGetSupportedMemoryClocks(h), N.nvmlDeviceGetSupportedGraphicsClocks(h, memory_clock),
            unsupported,
        ]
        """,
        FAKEGPU_PROFILE="a100", FAKEGPU_DEVICE_COUNT="1",
    )
    memory, graphics = result[0], result[1]
    assert memory != graphics
    assert result[2:] == [memory, memory, memory, memory, graphics, [memory], [graphics], 6]


def test_driver_model_is_not_supported_off_windows(tmp_path: Path) -> None:
    # nvml.h nvmlDeviceGetDriverModel_v2: NVML_ERROR_NOT_SUPPORTED if the platform
    # is not windows. Reporting 0 would read as NVML_DRIVER_WDDM.
    result = _nvml(
        tmp_path,
        """
        h = N.nvmlDeviceGetHandleByIndex(0)
        result = []
        for fn in (N.nvmlDeviceGetDriverModel, N.nvmlDeviceGetDriverModel_v1):
            try:
                result.append(fn(h))
            except N.NVMLError as exc:
                result.append(exc.value)
        """,
        FAKEGPU_PROFILE="a100", FAKEGPU_DEVICE_COUNT="1",
    )
    assert result == [3, 3]


def test_bridge_power_fields_follow_power_queries(tmp_path: Path) -> None:
    # docs/en/fakegpu-smi.md: without reference telemetry the bridge reports
    # power draw as unavailable; the field-value path must agree.
    pytest.importorskip("pynvml")
    from fakegpu.smi import SmiStatePublisher

    SmiStatePublisher(
        tmp_path / "state" / "state.json",
        lambda: {"devices": [{"index": 0, "name": "Fake A100", "profile_id": "a100",
                              "total_memory": 80 * 2**30, "reported_memory": 2**20}]},
    ).publish_once(running=True)
    code = textwrap.dedent(
        f"""
        import json
        from fakegpu.nvml import NvmlStateBridge
        with NvmlStateBridge(state_dir={str(tmp_path / "state")!r}, lib_dir={str(BUILD_DIR)!r}):
            import pynvml as N
            N.nvmlInit()
            h = N.nvmlDeviceGetHandleByIndex(0)
            try:
                usage = N.nvmlDeviceGetPowerUsage(h)
            except N.NVMLError as exc:
                usage = ["error", exc.value]
            fields = N.nvmlDeviceGetFieldValues(h, [N.NVML_FI_DEV_POWER_INSTANT, N.NVML_FI_DEV_POWER_CURRENT_LIMIT])
            print(json.dumps([usage, N.nvmlDeviceGetPowerManagementLimit(h),
                              [[f.nvmlReturn, f.value.uiVal] for f in fields]]))
            N.nvmlShutdown()
        """
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT), FAKEGPU_HOME=str(tmp_path), FAKEGPU_MACHINE="off")
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == [["error", 3], 400000, [[3, 0], [0, 400000]]]
