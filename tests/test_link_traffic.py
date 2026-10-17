from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fakegpu._smi_state import build_inventory
from fakegpu.link_topology import GpuSlot, LinkTopology
from fakegpu.link_traffic import (
    LinkTraffic,
    Route,
    copies_summary,
    empty_record,
    format_rate,
    format_seconds,
    merge_records,
    normalize_record,
    route,
)
from fakegpu.platforms import resolve_platform
from fakegpu.smi import SmiStatePublisher, main

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = Path(os.environ.get("FAKEGPU_BUILD_DIR", ROOT / "build"))
GEN4_X16 = 31.508  # per direction, as fakegpu.link_topology rounds it
GEN5_X16 = 63.015


def _topology(profiles: list[str], platform: str = "auto", **options) -> LinkTopology:
    return resolve_platform(profiles, platform, **options).topology


def test_routes_follow_nvlink_pcie_peer_access_and_host_links() -> None:
    board = _topology(["a100"] * 8)
    assert route(board, 0, 1) == Route("nvlink", "nvlink", pytest.approx(12 * 25.0))
    assert route(board, None, 3) == Route(None, "pcie", pytest.approx(GEN4_X16))
    assert route(board, 3, None) == Route("pcie", None, pytest.approx(GEN4_X16))
    assert route(board, 2, 2) == Route(None, None, None)

    server = _topology(["a100-pcie-80gb"] * 8)
    assert server.pcie_relation(0, 1) == "PIX"
    assert route(server, 0, 1) == Route("pcie", "pcie", pytest.approx(GEN4_X16))
    # Across sockets there is no peer access: the copy is staged through host
    # memory over both CPU links.
    assert not server.can_access_peer(0, 7)
    assert route(server, 0, 7) == Route("pcie", "pcie", pytest.approx(GEN4_X16))

    # NVIDIA disables PCIe peer access on the RTX 4090; x8/x8 halves the link.
    desktop = _topology(["rtx4090"] * 2)
    assert route(desktop, 0, 1) == Route("pcie", "pcie", 15.754)

    station = _topology(["gb300-dgx-station", "rtx-pro-6000-blackwell"])
    assert route(station, None, 0) == Route(None, "c2c", 450.0)
    assert route(station, 0, 1) == Route("c2c", "pcie", pytest.approx(GEN5_X16))

    spark = _topology(["gb10"])
    unified = route(spark, None, 0)
    assert (unified.source_link, unified.target_link) == (None, None)
    assert unified.bandwidth_gbps and unified.bandwidth_gbps > 0


def test_copies_on_one_link_run_back_to_back() -> None:
    clock = [0.0]
    topology = LinkTopology((GpuSlot(host_gbps=10.0), GpuSlot(host_gbps=10.0)))
    traffic = LinkTraffic(lambda: topology, clock=lambda: clock[0])

    assert traffic.record(None, 0, 10**9) == pytest.approx(0.1)
    # Queued behind the first copy on the same link and direction.
    assert traffic.record(None, 0, 10**9) == pytest.approx(0.1)
    clock[0] = 0.15
    record = traffic.snapshot()[0]
    assert record["copies"]["host_to_device"] == {
        "calls": 2,
        "bytes": 2 * 10**9,
        "modeled_seconds": pytest.approx(0.2),
    }
    assert record["modeled_seconds"] == pytest.approx(0.2)
    pcie = record["links"]["pcie"]
    assert pcie["rx_bytes"] == pytest.approx(1.5e9, abs=1)
    assert pcie["rx_bytes_per_second"] == pytest.approx(1.5e9, abs=1)
    assert pcie["tx_bytes"] == pcie["tx_bytes_per_second"] == 0

    clock[0] = 1.1
    pcie = traffic.snapshot()[0]["links"]["pcie"]
    assert pcie["rx_bytes"] == 2 * 10**9
    # Only the last 0.1 s of the schedule falls in the one-second window.
    assert pcie["rx_bytes_per_second"] == pytest.approx(1e9, abs=1)

    clock[0] = 3.0
    assert traffic.record(0, None, 5 * 10**8) == pytest.approx(0.05)
    clock[0] = 3.025
    first, second = traffic.snapshot()[0], traffic.snapshot()[1]
    assert first["links"]["pcie"]["rx_bytes_per_second"] == 0
    assert first["links"]["pcie"]["tx_bytes"] == pytest.approx(2.5e8, abs=1)
    assert second["copies"]["host_to_device"]["calls"] == 0

    # Copies within one GPU, empty copies and unknown GPUs are not link traffic.
    for source, target, size in ((0, 0, 100), (None, 0, 0), (None, 5, 100)):
        assert traffic.record(source, target, size) == 0.0
    traffic.record(0, 1, 10**9)
    snapshot = traffic.snapshot()
    assert snapshot[0]["copies"]["peer_send"]["bytes"] == 10**9
    assert snapshot[1]["copies"]["peer_receive"]["bytes"] == 10**9
    assert snapshot[1]["copies"]["peer_receive"]["modeled_seconds"] == pytest.approx(1 / GEN4_X16)


def test_records_normalize_merge_and_format() -> None:
    record = empty_record("c2c", 450.0)
    record["copies"]["host_to_device"] = {"calls": 1, "bytes": 3 * 2**30, "modeled_seconds": 0.007}
    record["links"]["c2c"]["rx_bytes_per_second"] = 2**30
    normalized = normalize_record(json.loads(json.dumps(record)))
    assert normalized == {**record, "modeled_seconds": pytest.approx(0.007)}
    merged = merge_records([normalized, normalize_record(record)])
    assert merged["copies"]["host_to_device"]["bytes"] == 6 * 2**30
    assert merged["links"]["c2c"]["rx_bytes_per_second"] == 2 * 2**30
    assert normalize_record("invalid") is None
    garbage = normalize_record({"host_link": "warp", "copies": {"host_to_device": {"bytes": -4}}})
    assert garbage["host_link"] == "pcie"
    assert garbage["copies"]["host_to_device"]["bytes"] == 0

    assert format_seconds(33.3e-6) == "33.3 us"
    assert format_seconds(0.0682) == "68.2 ms"
    assert format_seconds(2.5) == "2.50 s"
    assert format_rate(0) == "0 B/s"
    assert format_rate(2 * 2**30) == "2.0 GiB/s"
    assert copies_summary(record, lambda value: f"{value // 2**30} GiB") == "H2D 3 GiB / 7.0 ms"
    assert copies_summary(empty_record(), str) == "none"


def _transfer_snapshot(bytes_per_second: int) -> dict:
    record = empty_record("pcie", GEN4_X16)
    record["copies"]["host_to_device"] = {"calls": 2, "bytes": 2**30, "modeled_seconds": 0.034}
    record["links"]["pcie"].update(rx_bytes=2**30, rx_bytes_per_second=bytes_per_second)
    return {
        "devices": [
            {
                "index": 0,
                "name": "Fake NVIDIA A100",
                "profile_id": "a100",
                "total_memory": 80 * 2**30,
                "transfer": record,
            }
        ]
    }


def test_smi_sums_publishers_and_shows_modeled_copies(tmp_path: Path, capsys) -> None:
    running = SmiStatePublisher(tmp_path / "running.json", lambda: _transfer_snapshot(2**30))
    exited = SmiStatePublisher(tmp_path / "exited.json", lambda: _transfer_snapshot(2**29))
    states = [running.publish_once(running=True), exited.publish_once(running=False)]
    device = build_inventory(states)["devices"][0]
    assert device["transfer"]["copies"]["host_to_device"]["bytes"] == 2 * 2**30
    assert device["transfer"]["modeled_seconds"] == pytest.approx(0.068)
    # An exited publisher keeps its counts but moves no data now.
    assert device["transfer"]["links"]["pcie"]["rx_bytes_per_second"] == 2**30

    assert main(["--state", str(tmp_path / "running.json"), "-q"]) == 0
    output = capsys.readouterr().out
    assert "Copies (modeled): H2D 1.0 GiB / 34.0 ms" in output
    assert "Link throughput (modeled): PCIe TX 0 B/s RX 1.0 GiB/s (last 1 s)" in output

    assert main([
        "--state", str(tmp_path / "running.json"),
        "--query-gpu=index,pcie.throughput.rx,pcie.throughput.tx,copy.h2d,copy.modeled_time",
        "--format=csv,noheader",
    ]) == 0
    assert capsys.readouterr().out.strip() == "0,1048576 KB/s,0 KB/s,1024.0 MiB,0.034 s"


def test_fakecuda_records_copies_over_the_platform_links() -> None:
    code = textwrap.dedent(
        """
        import json, os
        os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"
        import fakegpu
        fakegpu.init(runtime="fakecuda", devices="a100:2", platform="auto")
        import torch
        from fakegpu import torch_patch

        host = torch.ones(256, 1024, dtype=torch.float32)  # 1 MiB
        first = host.cuda(0)
        host.to("cuda:1")
        first.to("cuda:1")
        first.cpu()
        first.copy_(host)
        host.copy_(first)
        first.sum().item()
        torch.nn.Linear(16, 16).cuda(1)
        torch.tensor([1.0, 2.0], device="cuda:0")
        torch.nn.Linear(16, 16).to("cuda:0").to("cpu")
        print(json.dumps(torch_patch._transfer_snapshot()))
        """
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for name in ("FAKEGPU_PLATFORM", "FAKEGPU_SLOTS", "FAKEGPU_NVLINK_FABRIC", "FAKEGPU_NVLINK_GROUPS"):
        env.pop(name, None)
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    snapshot = {int(key): value for key, value in json.loads(completed.stdout.strip().splitlines()[-1]).items()}
    mib = 2**20
    linear = (16 * 16 + 16) * 4
    first, second = snapshot[0]["copies"], snapshot[1]["copies"]
    assert (first["host_to_device"]["calls"], first["host_to_device"]["bytes"]) == (5, 2 * mib + 8 + linear)
    assert (first["device_to_host"]["calls"], first["device_to_host"]["bytes"]) == (5, 2 * mib + 4 + linear)
    assert first["peer_send"]["bytes"] == second["peer_receive"]["bytes"] == mib
    assert (second["host_to_device"]["calls"], second["host_to_device"]["bytes"]) == (3, mib + linear)
    # A100s on the 4-GPU HGX board: Gen4 x16 to the host, NV4 between GPUs.
    assert first["host_to_device"]["modeled_seconds"] == pytest.approx(
        first["host_to_device"]["bytes"] / (GEN4_X16 * 1e9)
    )
    assert first["peer_send"]["modeled_seconds"] == pytest.approx(mib / 100e9)
    assert snapshot[0]["links"]["nvlink"]["tx_bytes"] == mib


def _library(name: str) -> Path:
    suffix = {"darwin": ".dylib"}.get(sys.platform, ".so.1")
    return BUILD_DIR / f"{name}{suffix}"


@pytest.mark.skipif(not _library("libnvidia-ml").exists(), reason="native libraries are not built")
def test_nvml_bridge_reports_modeled_links(tmp_path: Path) -> None:
    pytest.importorskip("pynvml")
    record = empty_record("pcie", GEN4_X16)
    record["links"]["pcie"].update(tx_bytes_per_second=3 * 2**20, rx_bytes_per_second=2**30)
    record["links"]["nvlink"].update(tx_bytes=12 * 2**20, rx_bytes=6 * 2**20)
    state = SmiStatePublisher(
        tmp_path / "state.json",
        lambda: {
            "devices": [
                {"index": 0, "name": "Fake A100", "profile_id": "a100", "total_memory": 80 * 2**30, "transfer": record}
            ]
        },
    ).publish_once(running=True)
    state["devices"][0]["topology"]["pcie"].update(generation=3, width=8, max_generation=3, max_width=8)
    state["devices"][0]["topology"]["nvlink"]["active_links"] = 12
    (tmp_path / "state.json").write_text(json.dumps(state))
    code = textwrap.dedent(
        f"""
        import ctypes, json
        from fakegpu.nvml import NvmlStateBridge
        with NvmlStateBridge(state_dir={str(tmp_path)!r}, lib_dir={str(BUILD_DIR)!r}):
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            fields = pynvml.nvmlDeviceGetFieldValues(handle, [
                pynvml.NVML_FI_DEV_NVLINK_LINK_COUNT,
                (pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX, 0),
                (pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, 11),
            ])
            print(json.dumps([
                pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle),
                pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle),
                pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES),
                pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES),
                fields[0].value.uiVal,
                fields[1].value.ullVal,
                fields[2].value.ullVal,
            ]))
            pynvml.nvmlShutdown()
        """
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == [3, 8, 3 * 1024, 2**20, 12, 1024, 512]
