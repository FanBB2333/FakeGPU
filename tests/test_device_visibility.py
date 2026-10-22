"""CUDA_VISIBLE_DEVICES and device identity across CUDA, NVML, nvidia-smi and torch.

Expected behavior comes from the CUDA Programming Guide ("Environment
Variables": CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER), cuda.h, nvml.h, and one
real run of the CUDA 12.4 runtime with driver 580 and torch 2.9.1 on an
Ampere GPU (sm_80). That run gave:

* mask "0,7" or "0abc": one device; "", "-1" or "7,0": cuInit returns
  CUDA_ERROR_NO_DEVICE (100), cuDeviceGetCount CUDA_ERROR_NOT_INITIALIZED (3),
  cudaGetDeviceCount cudaErrorNoDevice (100) and leaves the count unwritten;
  "0,0": cuInit and cudaGetDeviceCount return 101 (invalid device);
* NVML still counts every GPU under any mask;
* cuDeviceGetUuid's 16 bytes are the hex digits of nvmlDeviceGetUUID's string;
* cuDeviceGetPCIBusId and cudaDeviceGetPCIBusId print NVML's busIdLegacy
  ("%04X:%02X:%02X.0", uppercase); a 13-byte buffer fits, an 8-byte one gets
  the truncated string and error 1; lookups accept upper or lower case, an 8-
  digit domain, "bus:device.function" and "domain:bus:device", and an unknown
  bus is error 101;
* the PCI_DEVICE_ID attribute is the slot number (0), not the product id;
* torch's ``uuid`` prints 8-4-4-4-12 without "GPU-" and ``bytes`` lists the
  16 octets; ``pci_bus_id``/``pci_device_id``/``pci_domain_id`` are integers.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fakegpu._run_bootstrap import visible_gpus
from fakegpu._smi_environment import _synthetic_mig_uuid
from fakegpu._smi_state import _synthetic_pci_address, _synthetic_pci_bus_id, _synthetic_uuid

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = Path(os.environ.get("FAKEGPU_BUILD_DIR", ROOT / "build"))
DARWIN = sys.platform == "darwin"
LIBS = {
    "nvml": BUILD_DIR / ("libnvidia-ml.dylib" if DARWIN else "libnvidia-ml.so.1"),
    "cuda": BUILD_DIR / ("libcuda.dylib" if DARWIN else "libcuda.so.1"),
    "cudart": BUILD_DIR / ("libcudart.dylib" if DARWIN else "libcudart.so.12"),
    "nccl": BUILD_DIR / ("libnccl.dylib" if DARWIN else "libnccl.so"),
}
native = pytest.mark.skipif(not LIBS["cudart"].exists(), reason="native libraries are not built")
GIB = 1024**3

# Reads every identity surface of the native libraries in one process.
PROBE = textwrap.dedent(
    """
    import ctypes, json, os, sys
    libs = json.loads(sys.argv[1])
    actions = json.loads(sys.argv[2])
    nvml = ctypes.CDLL(libs["nvml"], mode=ctypes.RTLD_GLOBAL)
    cuda = ctypes.CDLL(libs["cuda"], mode=ctypes.RTLD_GLOBAL)
    rt = ctypes.CDLL(libs["cudart"], mode=ctypes.RTLD_GLOBAL)

    class Pci(ctypes.Structure):
        _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                    ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                    ("busId", ctypes.c_char * 32)]

    class Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def nvml_devices():
        count = ctypes.c_uint()
        nvml.nvmlDeviceGetCount_v2(ctypes.byref(count))
        devices = []
        for index in range(count.value):
            handle = ctypes.c_void_p()
            nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle))
            uuid = ctypes.create_string_buffer(96)
            nvml.nvmlDeviceGetUUID(handle, uuid, 96)
            pci = Pci()
            nvml.nvmlDeviceGetPciInfo_v3(handle, ctypes.byref(pci))
            memory = Memory()
            nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
            devices.append({"uuid": uuid.value.decode(), "busId": pci.busId.decode(),
                            "busIdLegacy": pci.busIdLegacy.decode(), "used": memory.used})
        return devices

    nvml.nvmlInit_v2()
    out = {"nvml_before": nvml_devices()}
    out["cuInit"] = cuda.cuInit(0)
    count = ctypes.c_int(-7)
    out["cuDeviceGetCount"] = [cuda.cuDeviceGetCount(ctypes.byref(count)), count.value]
    devices = []
    for ordinal in range(max(count.value, 0)):
        name = ctypes.create_string_buffer(256)
        cuda.cuDeviceGetName(name, 256, ordinal)
        total = ctypes.c_size_t()
        cuda.cuDeviceTotalMem_v2(ctypes.byref(total), ordinal)
        uuid = (ctypes.c_ubyte * 16)()
        bus_id = ctypes.create_string_buffer(64)
        attributes = []
        for attribute in (33, 34, 50):  # PCI_BUS_ID, PCI_DEVICE_ID, PCI_DOMAIN_ID
            value = ctypes.c_int(-1)
            cuda.cuDeviceGetAttribute(ctypes.byref(value), attribute, ordinal)
            attributes.append(value.value)
        runtime_bus_id = ctypes.create_string_buffer(64)
        devices.append({
            "name": name.value.decode(),
            "total": total.value,
            "uuid_rc": cuda.cuDeviceGetUuid(uuid, ordinal),
            "uuid": bytes(uuid).hex(),
            "bus_id_rc": cuda.cuDeviceGetPCIBusId(bus_id, 64, ordinal),
            "bus_id": bus_id.value.decode(),
            "runtime_bus_id_rc": rt.cudaDeviceGetPCIBusId(runtime_bus_id, 64, ordinal),
            "runtime_bus_id": runtime_bus_id.value.decode(),
            "pci": attributes,
        })
    out["devices"] = devices
    runtime_count = ctypes.c_int(-7)
    out["cudaGetDeviceCount"] = [rt.cudaGetDeviceCount(ctypes.byref(runtime_count)), runtime_count.value]
    current = ctypes.c_int(-7)
    out["cudaGetDevice"] = [rt.cudaGetDevice(ctypes.byref(current)), current.value]
    out["cudaSetDevice0"] = rt.cudaSetDevice(0)
    for action in actions:
        kind = action[0]
        if kind == "malloc":  # ["malloc", ordinal, bytes]
            rt.cudaSetDevice(action[1])
            pointer = ctypes.c_void_p()
            rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            out.setdefault("malloc", []).append(rt.cudaMalloc(ctypes.byref(pointer), action[2]))
            out.setdefault("pointer_device", []).append(None)
            attributes = (ctypes.c_int * 8)()
            device = ctypes.c_int(-1)
            cuda.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
            cuda.cuPointerGetAttribute(ctypes.byref(device), 9, pointer.value or 0)  # DEVICE_ORDINAL
            out["pointer_device"][-1] = device.value
        elif kind == "lookup":  # ["lookup", text]
            device = ctypes.c_int(-7)
            rc = cuda.cuDeviceGetByPCIBusId(ctypes.byref(device), action[1].encode())
            runtime_device = ctypes.c_int(-7)
            runtime_rc = rt.cudaDeviceGetByPCIBusId(ctypes.byref(runtime_device), action[1].encode())
            out.setdefault("lookup", []).append([rc, device.value, runtime_rc, runtime_device.value])
        elif kind == "short_bus_id":  # ["short_bus_id", ordinal, length]
            text = ctypes.create_string_buffer(64)
            rc = cuda.cuDeviceGetPCIBusId(text, action[2], action[1])
            runtime_text = ctypes.create_string_buffer(64)
            runtime_rc = rt.cudaDeviceGetPCIBusId(runtime_text, action[2], action[1])
            out.setdefault("short_bus_id", []).append([rc, text.value.decode(), runtime_rc, runtime_text.value.decode()])
        elif kind == "peer":  # ["peer", a, b]
            value = ctypes.c_int(-1)
            out.setdefault("peer", []).append([cuda.cuDeviceCanAccessPeer(ctypes.byref(value), action[1], action[2]), value.value])
    out["nvml_after"] = nvml_devices()
    print(json.dumps(out))
    """
)


def _probe(tmp_path: Path, mask: str | None, actions: list | None = None, **environment: str) -> dict:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("FAKEGPU_", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"))}
    env.update(
        FAKEGPU_MODE="simulate", FAKEGPU_HOME=str(tmp_path), FAKEGPU_MACHINE="off",
        FAKEGPU_TERMINAL_REPORT="0", FAKEGPU_REPORT_PATH=os.devnull, **environment,
    )
    if mask is not None:
        env["CUDA_VISIBLE_DEVICES"] = mask
    completed = subprocess.run(
        [sys.executable, "-c", PROBE, json.dumps({k: str(v) for k, v in LIBS.items()}), json.dumps(actions or [])],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _uuid_hex(uuid: str) -> str:
    return uuid[4:].replace("-", "")


@native
@pytest.mark.parametrize(
    ("mask", "names"),
    [
        (None, ["T4", "A100"]),
        ("1", ["A100"]),
        ("1,0", ["A100", "T4"]),
        ("0,7", ["T4"]),  # the list stops at the first invalid entry
        ("0abc", ["T4"]),  # an index reads like strtoul
        ("+1", ["A100"]),
    ],
)
def test_cuda_ordinals_follow_cuda_visible_devices(tmp_path: Path, mask: str | None, names: list[str]) -> None:
    result = _probe(tmp_path, mask, FAKEGPU_PROFILES="t4,a100")
    assert result["cuInit"] == 0
    assert result["cuDeviceGetCount"] == [0, len(names)]
    assert result["cudaGetDeviceCount"] == [0, len(names)]
    assert [name.split()[-1].split("-")[0] for name in (d["name"] for d in result["devices"])] == names
    # NVML keeps the whole inventory in PCI order, whatever the mask.
    assert len(result["nvml_before"]) == len(result["nvml_after"]) == 2


@native
@pytest.mark.parametrize(
    ("mask", "init", "runtime"),
    [("", 100, 100), ("-1", 100, 100), ("7,0", 100, 100), ("GPU-", 100, 100), ("0,0", 101, 101), ("1,GPU-x,1", 1, 0)],
)
def test_masks_without_a_usable_device_fail_cuda_initialization(
    tmp_path: Path, mask: str, init: int, runtime: int
) -> None:
    result = _probe(tmp_path, mask, FAKEGPU_PROFILES="t4,a100")
    if init == 1:  # "1,GPU-x,1": the list ends at the unknown UUID before the repeat
        assert result["cuDeviceGetCount"] == [0, 1]
        return
    assert result["cuInit"] == init
    assert result["cuDeviceGetCount"] == [3, -7]  # CUDA_ERROR_NOT_INITIALIZED, count untouched
    assert result["cudaGetDeviceCount"] == [runtime, -7]
    assert result["cudaGetDevice"][0] == runtime
    assert result["cudaSetDevice0"] == runtime
    assert len(result["nvml_after"]) == 2


@native
def test_uuid_masks_select_by_nvml_uuid_or_unique_prefix(tmp_path: Path) -> None:
    nvml = _probe(tmp_path, None, FAKEGPU_PROFILES="t4,a100")["nvml_before"]
    full = nvml[1]["uuid"]
    for mask in (full, full[:12], f"{full},0"):
        result = _probe(tmp_path, mask, FAKEGPU_PROFILES="t4,a100")
        ordinals = [d["uuid"] for d in result["devices"]]
        expected = [_uuid_hex(full)] + ([_uuid_hex(nvml[0]["uuid"])] if mask.endswith(",0") else [])
        assert ordinals == expected, mask
    # The same GPU by UUID and by index is a repeat.
    assert _probe(tmp_path, f"1,{full}", FAKEGPU_PROFILES="t4,a100")["cuInit"] == 101


@native
def test_allocations_land_on_the_physical_gpu_nvml_reports(tmp_path: Path) -> None:
    result = _probe(tmp_path, "1", [["malloc", 0, GIB]], FAKEGPU_PROFILES="t4,a100")
    assert result["malloc"] == [0]
    assert result["pointer_device"] == [0]  # CUDA ordinal
    assert [device["used"] for device in result["nvml_after"]] == [0, GIB]


@native
def test_peer_access_maps_ordinals_to_the_physical_topology(tmp_path: Path) -> None:
    # Physical GPUs 0-1 and 2-3 share NVLink; PCIe peer access is off.
    groups = {"FAKEGPU_NVLINK_GROUPS": "0,1;2,3", "FAKEGPU_SLOTS": ";".join(["p2p=0"] * 4)}
    result = _probe(tmp_path, "2,3,0", [["peer", 0, 1], ["peer", 0, 2]], FAKEGPU_PROFILES="a100:4", **groups)
    unmasked = _probe(tmp_path, None, [["peer", 2, 3], ["peer", 2, 0]], FAKEGPU_PROFILES="a100:4", **groups)
    assert result["peer"] == unmasked["peer"] == [[0, 1], [0, 0]]


@native
def test_cuda_and_nvml_share_uuid_and_pci_bus_id(tmp_path: Path) -> None:
    profiles = ["t4"] * 10 + ["a100"] * 2
    result = _probe(
        tmp_path, None, [["short_bus_id", 10, 8], ["short_bus_id", 10, 13]], FAKEGPU_PROFILES="t4:10,a100:2"
    )
    host = socket.gethostname()
    for ordinal, (device, nvml) in enumerate(zip(result["devices"], result["nvml_before"])):
        # One identity function in native and Python code.
        assert nvml["uuid"] == _synthetic_uuid(host, ordinal, profiles[ordinal])
        assert nvml["busId"] == _synthetic_pci_bus_id(ordinal)
        assert device["uuid_rc"] == 0 and device["uuid"] == _uuid_hex(nvml["uuid"])
        assert device["bus_id_rc"] == device["runtime_bus_id_rc"] == 0
        assert device["bus_id"] == device["runtime_bus_id"] == nvml["busIdLegacy"]
        domain, bus, slot = _synthetic_pci_address(ordinal)
        assert device["pci"] == [bus, slot, domain]
    assert result["devices"][10]["bus_id"] == "0000:0B:00.0"  # uppercase, like the real driver
    assert result["nvml_before"][10]["busId"] == "00000000:0B:00.0"
    # 13 bytes hold the id; 8 bytes get the truncated string and error 1.
    assert result["short_bus_id"] == [[1, "0000:0B", 1, "0000:0B"], [0, "0000:0B:00.0", 0, "0000:0B:00.0"]]


@native
def test_pci_bus_id_lookup_accepts_the_driver_spellings(tmp_path: Path) -> None:
    texts = ["0000:02:00.0", "0000:02:00.0".lower(), "00000000:02:00.0", "02:00.0", "0000:02:00", "0000:7F:1F.7"]
    result = _probe(tmp_path, "1,0", [["lookup", text] for text in texts], FAKEGPU_PROFILES="t4,a100")
    # Physical 1 (bus 02) is CUDA ordinal 0 under "1,0"; an unknown bus is
    # cudaErrorInvalidDevice / CUDA_ERROR_INVALID_DEVICE with the output untouched.
    assert result["lookup"] == [[0, 0, 0, 0]] * 5 + [[101, -7, 101, -7]]
    hidden = _probe(tmp_path, "1", [["lookup", "0000:01:00.0"]], FAKEGPU_PROFILES="t4,a100")
    assert hidden["lookup"] == [[101, -7, 101, -7]]


@native
def test_mig_masks_expose_one_instance(tmp_path: Path) -> None:
    layout = {"FAKEGPU_PROFILES": "a100:2", "FAKEGPU_MIG_LAYOUT": "1:1g.10gb:10240:2"}
    nvml = _probe(tmp_path, None, **layout)["nvml_before"]
    parent = nvml[1]["uuid"]
    instance = _synthetic_mig_uuid(parent, 1, 1)
    assert instance.startswith("MIG-") and len(instance) == 40
    for mask in (instance, instance[:13], f"MIG-{parent}/1/1", f"{instance},0"):
        result = _probe(tmp_path, mask, [["malloc", 0, GIB]], **layout)
        assert len(result["devices"]) == 1, mask
        assert result["devices"][0]["total"] == 10240 * 1024**2
        assert result["devices"][0]["uuid"] == _uuid_hex(instance)
        assert result["devices"][0]["bus_id"] == nvml[1]["busIdLegacy"]
        assert [device["used"] for device in result["nvml_after"]] == [0, GIB]
    # Only single MIG instance enumeration is supported: a later MIG id ends the list.
    assert len(_probe(tmp_path, f"0,{instance}", **layout)["devices"]) == 1


@native
def test_hidden_devices_stay_in_the_native_smi_state(tmp_path: Path) -> None:
    state = tmp_path / "smi.json"
    _probe(tmp_path, "1", [["malloc", 0, GIB]], FAKEGPU_PROFILES="t4,a100", FAKEGPU_SMI_STATE_PATH=str(state))
    devices = json.loads(state.read_text())["devices"]
    assert [device["index"] for device in devices] == [0, 1]
    assert [device["tracked_memory"] for device in devices] == [0, GIB]
    host = socket.gethostname()
    assert [device["uuid"] for device in devices] == [_synthetic_uuid(host, 0, "t4"), _synthetic_uuid(host, 1, "a100")]


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        ("0,0", ([], [])),  # a repeat makes CUDA initialization fail
        ("0abc", (["a100"], [0])),
        ("+2", (["h100"], [2])),
        ("MIG-0", ([], [])),
    ],
)
def test_visible_gpus_reads_entries_like_cuda(mask: str, expected: tuple) -> None:
    assert visible_gpus("a100,a100,h100", mask, host="node") == expected


def test_visible_gpus_accepts_gpu_uuids_and_unique_prefixes() -> None:
    uuids = [_synthetic_uuid("node", index, profile) for index, profile in enumerate(["a100", "a100", "h100"])]
    assert visible_gpus("a100,a100,h100", uuids[2], host="node") == (["h100"], [2])
    assert visible_gpus("a100,a100,h100", f"{uuids[2][:14]},{uuids[0]}", host="node") == (["h100", "a100"], [2, 0])
    assert visible_gpus("a100,a100,h100", "GPU-", host="node") == ([], [])  # ambiguous
    assert visible_gpus("a100,a100,h100", f"{uuids[1]},1", host="node") == ([], [])  # the same GPU twice


FAKECUDA = textwrap.dedent(
    """
    import json, os, socket, warnings
    import fakegpu
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fakegpu.init(runtime="fakecuda", devices="t4,a100,l4")
    import torch
    out = {"backend": result.backend, "count": torch.cuda.device_count(),
           "physical": os.environ.get("FAKEGPU_PHYSICAL_INDICES"),
           "warnings": [str(w.message) for w in caught if "CUDA_VISIBLE_DEVICES" in str(w.message)]}
    if out["count"]:
        props = [torch.cuda.get_device_properties(i) for i in range(out["count"])]
        out["names"] = [p.name for p in props]
        out["uuid"] = [str(p.uuid) for p in props]
        out["uuid_bytes"] = [p.uuid.bytes for p in props]
        out["pci"] = [[p.pci_bus_id, p.pci_device_id, p.pci_domain_id] for p in props]
        out["repr"] = repr(props[0])
    out["host"] = socket.gethostname()
    print(json.dumps(out))
    """
)


def _fakecuda(tmp_path: Path, mask: str | None, **environment: str) -> dict:
    pytest.importorskip("torch")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("FAKEGPU_", "CUDA_VISIBLE_DEVICES"))}
    env.update(PYTHONPATH=str(ROOT), FAKEGPU_HOME=str(tmp_path), FAKEGPU_MACHINE="off",
               FAKEGPU_TERMINAL_REPORT="0", **environment)
    if mask is not None:
        env["CUDA_VISIBLE_DEVICES"] = mask
    completed = subprocess.run([sys.executable, "-c", FAKECUDA], cwd=tmp_path, env=env, text=True,
                               capture_output=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_fakecuda_init_applies_cuda_visible_devices(tmp_path: Path) -> None:
    result = _fakecuda(tmp_path, "2,0")
    assert result["count"] == 2
    assert result["physical"] == "2,0"
    assert "L4" in result["names"][0] and "T4" in result["names"][1]
    host = result["host"]
    # Each GPU keeps the identity nvidia-smi and NVML give it on this host.
    assert result["uuid"] == [_synthetic_uuid(host, 2, "l4")[4:], _synthetic_uuid(host, 0, "t4")[4:]]
    assert result["uuid_bytes"][0] == list(bytes.fromhex(_uuid_hex(_synthetic_uuid(host, 2, "l4"))))
    assert result["pci"] == [[3, 0, 0], [1, 0, 0]]
    assert "uuid=" in result["repr"] and "pci_bus_id=3" in result["repr"]


def test_fakecuda_init_hides_every_gpu_for_an_empty_mask(tmp_path: Path) -> None:
    result = _fakecuda(tmp_path, "")
    assert result["backend"] == "none"
    assert result["count"] == 0  # torch.cuda stays unpatched and CUDA shows no device
    assert result["warnings"]


def test_fakecuda_init_keeps_a_layout_fakegpu_run_already_filtered(tmp_path: Path) -> None:
    # fakegpu run passes the visible GPUs and FAKEGPU_PHYSICAL_INDICES; the
    # mask must not be applied a second time.
    result = _fakecuda(tmp_path, "2", FAKEGPU_PHYSICAL_INDICES="2")
    assert result["count"] == 3
    assert result["pci"][0] == [3, 0, 0]  # cuda:0 is physical GPU 2
