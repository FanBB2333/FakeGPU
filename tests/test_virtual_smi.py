from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import fakegpu.smi as smi_module
from fakegpu import library_dir
from fakegpu.profile_catalog import load_profiles
from fakegpu.smi import (
    SCHEMA_VERSION,
    SmiStatePublisher,
    main,
    render_health_events,
    render_mig_view,
    render_nvlink_status,
    render_table,
    render_topology_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def _snapshot() -> dict:
    return {
        "tracking_confidence": "C2_torch_tensor_lifetime",
        "stage": "forward",
        "devices": [
            {
                "index": 0,
                "name": "Fake NVIDIA Test GPU",
                "profile_id": "test",
                "total_memory": 8 * 2**30,
                "current_memory": 3 * 2**30,
                "peak_memory": 4 * 2**30,
            }
        ],
    }


def test_publisher_and_virtual_smi_include_process_memory(
    tmp_path: Path, capsys
) -> None:
    path = tmp_path / "state.json"
    publisher = SmiStatePublisher(
        path,
        _snapshot,
        runtime_overhead_bytes=256 * 2**20,
    )
    state = publisher.publish_once(running=True)
    assert state["schema_version"] == SCHEMA_VERSION
    assert state["fakegpu"]["version"]
    assert state["fakegpu"]["profile_catalog"]["profile_count"] == len(load_profiles())
    assert (
        state["fakegpu"]["native_capabilities"]["explicit_api_count"]
        == 109
    )
    assert state["software"]["cuda_version"]
    assert state["devices"][0]["reported_memory"] == 3328 * 2**20
    assert state["devices"][0]["reported_peak_memory"] == 4352 * 2**20
    assert state["devices"][0]["free_memory"] == 4864 * 2**20
    assert state["devices"][0]["uuid"].startswith("GPU-")
    assert state["devices"][0]["pci_bus_id"] == "00000000:01:00.0"
    assert state["publisher"]["health"]["attempted_writes"] == 1
    assert state["publisher"]["health"]["successful_writes"] == 1
    assert state["publisher"]["health"]["failed_writes"] == 0
    assert state["publisher"]["limits"]["detail_entries"] == 64
    assert state["publisher"]["limits"]["max_state_bytes"] == 2**20
    assert path.stat().st_size <= 2**20
    assert not list(tmp_path.glob(".state.json.*.tmp"))
    assert state["stage"] == "forward"
    assert main(["--state", str(path)]) == 0
    output = capsys.readouterr().out
    assert "FakeGPU-SMI" in output
    assert "3328 MiB / 8192 MiB" in output
    assert "4352 MiB" in output
    assert "3072 MiB" in output
    assert "4096 MiB" in output
    assert "forward" in output
    assert "C2_torch_tensor_lifetime" in output
    assert "test" in output


def test_virtual_smi_lists_details_and_queries_gpu_fields(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "state.json"
    publisher = SmiStatePublisher(
        path,
        lambda: {
            "runtime_backend": "upstream",
            "allocator_model": "cuda_caching_allocator.v1",
            "dispatch_tracking": {
                "enabled": True,
                "operator_calls": 7,
            },
            "devices": [
                {
                    "index": 0,
                    "name": "FakeGPU Test Profile 512MB",
                    "profile_id": "test-512m",
                    "total_memory": 512 * 2**20,
                    "current_memory": 64 * 2**20,
                    "peak_memory": 96 * 2**20,
                    "current_reserved_memory": 80 * 2**20,
                    "peak_reserved_memory": 112 * 2**20,
                    "inactive_split_bytes": 16 * 2**20,
                    "segment_count": 3,
                    "allocation_count": 9,
                    "free_count": 4,
                    "current_bytes_by_category": {
                        "activation": 64 * 2**20,
                    },
                    "peak_by_stage": {"forward": 96 * 2**20},
                    "largest_allocations": [
                        {
                            "bytes": 32 * 2**20,
                            "category": "activation",
                            "source": "torch_dispatch",
                        }
                    ],
                    "allocator_model": "cuda_caching_allocator.v1",
                }
            ],
        },
    )
    state = publisher.publish_once(running=True)

    assert main(["--state", str(path), "-L"]) == 0
    listed = capsys.readouterr().out
    # nvidia-smi -L: "GPU %u: %s (UUID: %s)", nothing else on the line.
    assert re.fullmatch(
        r"GPU 0: FakeGPU Test Profile 512MB \(UUID: GPU-[0-9a-f-]{36}\)\n",
        listed,
    )
    # torch.utils.collect_env anonymizes the list by removing the UUID.
    assert re.sub(r" \(UUID: .+?\)", "", listed) == (
        "GPU 0: FakeGPU Test Profile 512MB\n"
    )

    assert main(["--state", str(path), "-q"]) == 0
    detail = capsys.readouterr().out
    assert "FakeGPU-SMI detailed report" in detail
    assert "Profile: test-512m" in detail
    assert "PCI Bus ID: 00000000:01:00.0" in detail
    assert "Architecture: ampere, compute capability 8.0" in detail
    assert "108 SMs" in detail
    assert "native capabilities 5 groups / 109 APIs / 96" in detail
    assert "activation=64.0 MiB" in detail
    assert "memory enabled, dispatch enabled" in detail
    assert "7 calls" in detail
    assert "0.25s interval" in detail
    assert "Publisher health: 1 / 1 writes, 0 failures" in detail
    assert "64 detail entries, 1.0 MiB state size" in detail

    assert (
        main(
            [
                "--state",
                str(path),
                "--query-gpu",
                (
                    "index,name,profile.id,compute_cap,memory.total,"
                    "memory.used,fakegpu.version"
                ),
                "--format",
                "csv,noheader,nounits",
                "-i",
                "test-512m",
            ]
        )
        == 0
    )
    row = capsys.readouterr().out.strip()
    assert row.startswith(
        "0, FakeGPU Test Profile 512MB, test-512m, 8.0, 512, 80, "
    )

    assert (
        main(
            [
                "--state",
                str(path),
                "--query-gpu",
                "timestamp,memory.total",
                "--format",
                "csv,nounits",
            ]
        )
        == 0
    )
    query_lines = capsys.readouterr().out.splitlines()
    assert query_lines[0] == "timestamp, memory.total [MiB]"
    assert query_lines[1].split(", ")[0]

    assert (
        main(
            [
                "--state",
                str(path),
                "--query-runtime",
                (
                    "pid,fakegpu.version,runtime.backend,runtime.mode,"
                    "policy.oom,tracking.dispatch,catalog.profiles,"
                    "catalog.native_apis,dispatch.calls,"
                    "publisher.successful_writes,publisher.failed_writes,"
                    "publisher.detail_limit,software.cuda,status"
                ),
                "--format",
                "json",
            ]
        )
        == 0
    )
    runtime_query = json.loads(capsys.readouterr().out)
    assert runtime_query["query"] == "runtime"
    assert runtime_query["records"] == [
        {
            "pid": os.getpid(),
            "fakegpu.version": state["fakegpu"]["version"],
            "runtime.backend": "upstream",
            "runtime.mode": "simulate",
            "policy.oom": "default",
            "tracking.dispatch": True,
            "catalog.profiles": len(load_profiles()),
            "catalog.native_apis": 109,
            "dispatch.calls": 7,
            "publisher.successful_writes": 1,
            "publisher.failed_writes": 0,
            "publisher.detail_limit": 64,
            "software.cuda": state["software"]["cuda_version"],
            "status": "running",
        }
    ]


def test_virtual_smi_models_topology_nvlink_views_and_queries(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = tmp_path / "topology.json"
    monkeypatch.setenv("FAKEGPU_NVLINK_GROUPS", "0,1;2,3")
    monkeypatch.setenv("FAKEGPU_NVLINK_BANDWIDTH_GBPS", "800")
    state = SmiStatePublisher(
        path,
        lambda: {
            "devices": [
                {
                    "index": index,
                    "name": f"Fake GPU {index}",
                    "profile_id": "a100",
                    "total_memory": 80 * 2**30,
                }
                for index in range(4)
            ]
        },
    ).publish_once(running=True)

    topology = state["topology"]
    assert topology["schema_version"] == (
        "fakegpu.device_topology.v2"
    )
    assert topology["source"] == "modeled_environment"
    assert topology["valid"] is True
    assert topology["link_count"] == 2
    assert {
        (link["source_index"], link["target_index"])
        for link in topology["links"]
    } == {(0, 1), (2, 3)}
    assert state["devices"][0]["topology"]["nvlink"][
        "active_links"
    ] == 1
    assert state["devices"][0]["topology"]["nvlink"]["peers"][0][
        "pci_bus_id"
    ] == "00000000:02:00.0"

    assert main(["topo", "-m", "--state", str(path)]) == 0
    matrix = capsys.readouterr().out
    lines = matrix.splitlines()
    # nvidia-smi 580.178.04 topo -m: tab-separated cells, a header row
    # underlined with ANSI escapes, " X " on the diagonal and an empty cell
    # after NUMA Affinity.
    assert lines[0] == (
        "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity"
        "\tGPU NUMA ID\x1b[0m"
    )
    assert lines[1] == "GPU0\t X \tNV1\tPIX\tPIX\tN/A\t0\t\tN/A"
    assert lines[3] == "GPU2\tPIX\tPIX\t X \tNV1\tN/A\t0\t\tN/A"
    assert lines[5:8] == ["", "Legend:", ""]
    # torch._inductor.comm_analysis._has_nvlink reads the part before
    # "Legend"; FakeGPU's notes follow the legend.
    before_legend, after_legend = matrix.split("Legend")
    assert "NV" in before_legend and "FakeGPU" not in before_legend
    assert "FakeGPU modeled topology matrix" in after_legend
    assert "modeled, not measured" in after_legend

    assert main(["nvlink", "-s", "--state", str(path)]) == 0
    status = capsys.readouterr().out
    assert "Bandwidth values are configuration inputs" in status
    assert "Link 0: Active -> GPU 1" in status
    assert "800 GB/s" in status

    assert (
        main(
            [
                "--state",
                str(path),
                "--query-gpu",
                (
                    "index,topology.source,nvlink.active_links,"
                    "nvlink.peer_count,nvlink.bandwidth"
                ),
                "--format",
                "json",
                "-i",
                "0",
            ]
        )
        == 0
    )
    query = json.loads(capsys.readouterr().out)
    assert query["records"] == [
        {
            "index": 0,
            "topology.source": "modeled_environment",
            "nvlink.active_links": 1,
            "nvlink.peer_count": 1,
            "nvlink.bandwidth": 800.0,
        }
    ]

    inventory = smi_module.build_inventory([state])
    assert "GPU0" in render_topology_matrix(inventory)
    assert "Link 0: Active" in render_nvlink_status(inventory)


def test_virtual_smi_rejects_invalid_modeled_nvlink_config_safely(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FAKEGPU_NVLINK_GROUPS", "0,9")
    state = SmiStatePublisher(
        tmp_path / "invalid-topology.json",
        lambda: {
            "devices": [
                {"index": 0, "total_memory": 2**30},
                {"index": 1, "total_memory": 2**30},
            ]
        },
    ).publish_once(running=True)

    assert state["topology"]["configured"] is True
    assert state["topology"]["valid"] is False
    assert state["topology"]["source"] == (
        "modeled_environment_invalid"
    )
    assert "invalid device index" in state["topology"]["error"]
    assert state["topology"]["links"] == []
    assert all(
        device["topology"]["nvlink"]["active_links"] == 0
        for device in state["devices"]
    )


def test_virtual_smi_rejects_non_ascii_nvlink_device_indices(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FAKEGPU_NVLINK_GROUPS", "0,²")
    state = SmiStatePublisher(
        tmp_path / "invalid-non-ascii-topology.json",
        lambda: {
            "devices": [
                {"index": 0, "total_memory": 2**30},
                {"index": 1, "total_memory": 2**30},
            ]
        },
    ).publish_once(running=True)

    assert state["topology"]["configured"] is True
    assert state["topology"]["valid"] is False
    assert "invalid device index" in state["topology"]["error"]
    assert state["topology"]["links"] == []


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_virtual_smi_positive_float_rejects_non_finite_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        smi_module._positive_float(value)


def test_virtual_smi_models_mig_instances_views_and_queries(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = tmp_path / "mig-layout.json"
    monkeypatch.setenv(
        "FAKEGPU_MIG_LAYOUT",
        "0:1g.10gb:10240:2;1:2g.20gb:20480",
    )
    state = SmiStatePublisher(
        path,
        lambda: {
            "devices": [
                {
                    "index": index,
                    "name": f"Fake GPU {index}",
                    "profile_id": "a100",
                    "total_memory": 80 * 2**30,
                }
                for index in range(2)
            ]
        },
    ).publish_once(running=True)

    layout = state["mig"]
    assert layout["schema_version"] == "fakegpu.mig_layout.v1"
    assert layout["source"] == "modeled_environment"
    assert layout["valid"] is True
    assert layout["enabled_device_count"] == 2
    assert layout["instance_count"] == 3
    assert layout["allocated_memory_bytes"] == 40 * 2**30

    gpu0_mig = state["devices"][0]["mig"]
    assert gpu0_mig["mode"] == "enabled"
    assert gpu0_mig["instance_count"] == 2
    assert gpu0_mig["allocated_memory_bytes"] == 20 * 2**30
    assert gpu0_mig["unallocated_memory_bytes"] == 60 * 2**30
    assert gpu0_mig["instances"][0]["profile"] == "1g.10gb"
    assert gpu0_mig["instances"][0]["uuid"].startswith("MIG-")
    assert gpu0_mig["instances"][0]["memory_used_bytes"] is None
    assert gpu0_mig["instances"][1]["gpu_instance_id"] == 1

    assert main(["mig", "-lgi", "--state", str(path)]) == 0
    gpu_instances = capsys.readouterr().out
    assert "FakeGPU modeled MIG instances" in gpu_instances
    assert "GPU instance 0: profile 1g.10gb" in gpu_instances
    assert "Compute instance" not in gpu_instances
    assert "per-instance runtime memory usage is unobserved" in (
        gpu_instances
    )

    assert main(["mig", "-lci", "--state", str(path)]) == 0
    compute_instances = capsys.readouterr().out
    assert "Compute instance 0 (GI 0)" in compute_instances
    assert "memory tracking unobserved" in compute_instances

    assert main(["--state", str(path), "-L"]) == 0
    listed = capsys.readouterr().out.splitlines()
    # nvidia-smi -L prints MIG devices as "  %-15s Device %2u: (%s: %s)":
    # "  MIG 3g.40gb     Device  0: (UUID: MIG-...)" in a real H100 capture.
    assert re.fullmatch(r"GPU 0: Fake GPU 0 \(UUID: GPU-[0-9a-f-]{36}\)", listed[0])
    assert listed[1] == (
        f"  MIG 1g.10gb     Device  0: (UUID: {gpu0_mig['instances'][0]['uuid']})"
    )
    assert listed[2].startswith("  MIG 1g.10gb     Device  1: (UUID: MIG-")
    assert listed[3].startswith("GPU 1: Fake GPU 1 (UUID: GPU-")
    assert listed[4].startswith("  MIG 2g.20gb     Device  0: (UUID: MIG-")
    assert len(listed) == 5

    assert (
        main(
            [
                "--state",
                str(path),
                "--query-gpu",
                (
                    "index,mig.mode,mig.instance_count,"
                    "mig.allocated_memory,mig.unallocated_memory"
                ),
                "--format",
                "json",
                "-i",
                gpu0_mig["instances"][0]["uuid"],
            ]
        )
        == 0
    )
    query = json.loads(capsys.readouterr().out)
    assert query["records"] == [
        {
            "index": 0,
            "mig.mode": "enabled",
            "mig.instance_count": 2,
            "mig.allocated_memory": 20480,
            "mig.unallocated_memory": 61440,
        }
    ]

    inventory = smi_module.build_inventory([state])
    assert inventory["mig_instance_count"] == 3
    assert "GPU instance 0" in render_mig_view(inventory)


def test_virtual_smi_rejects_invalid_mig_layout_without_instances(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "FAKEGPU_MIG_LAYOUT",
        "0:4g.40gb:40960:3",
    )
    state = SmiStatePublisher(
        tmp_path / "invalid-mig-layout.json",
        lambda: {
            "devices": [
                {
                    "index": 0,
                    "name": "Fake GPU 0",
                    "profile_id": "a100",
                    "total_memory": 80 * 2**30,
                }
            ]
        },
    ).publish_once(running=True)

    assert state["mig"]["configured"] is True
    assert state["mig"]["valid"] is False
    assert state["mig"]["source"] == (
        "modeled_environment_invalid"
    )
    assert state["mig"]["instance_count"] == 0
    assert "8-slice per-device limit" in state["mig"]["error"]
    device_mig = state["devices"][0]["mig"]
    assert device_mig["mode"] == "configuration_error"
    assert device_mig["instances"] == []
    assert device_mig["allocated_memory_bytes"] == 0


def test_virtual_smi_models_fault_health_events_and_queries(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = tmp_path / "fault-events.json"
    monkeypatch.setenv(
        "FAKEGPU_FAULT_EVENTS",
        (
            "0:XID_79:critical;"
            "1:nvlink_crc:error:3;"
            "1:NVLINK_CRC:error:2"
        ),
    )
    state = SmiStatePublisher(
        path,
        lambda: {
            "devices": [
                {
                    "index": index,
                    "name": f"Fake GPU {index}",
                    "profile_id": "a100",
                    "total_memory": 80 * 2**30,
                }
                for index in range(2)
            ]
        },
    ).publish_once(running=True)

    faults = state["faults"]
    assert faults["schema_version"] == "fakegpu.fault_model.v1"
    assert faults["source"] == "modeled_environment"
    assert faults["valid"] is True
    assert faults["status"] == "failed"
    assert faults["max_severity"] == "critical"
    assert faults["event_count"] == 6
    assert faults["event_types_total"] == 2
    assert state["devices"][0]["health"]["status"] == "failed"
    assert state["devices"][0]["health"]["hardware_health"] == (
        "unobserved"
    )
    assert state["devices"][1]["health"]["status"] == "degraded"
    assert state["devices"][1]["health"]["events"][0]["count"] == 5

    assert main(["events", "--state", str(path)]) == 0
    rendered = capsys.readouterr().out
    assert "Hardware health is unobserved" in rendered
    assert "Reliability status: failed" in rendered
    assert "| critical | modeled_fault |" in rendered
    assert "XID_79" in rendered
    assert "NVLINK_CRC" in rendered

    assert (
        main(
            [
                "--state",
                str(path),
                "--query-gpu",
                (
                    "index,health.status,health.hardware,"
                    "health.max_severity,health.event_count,"
                    "health.event_types"
                ),
                "--format",
                "json",
            ]
        )
        == 0
    )
    query = json.loads(capsys.readouterr().out)
    assert query["records"][0] == {
        "index": 0,
        "health.status": "failed",
        "health.hardware": "unobserved",
        "health.max_severity": "critical",
        "health.event_count": 1,
        "health.event_types": 1,
    }
    assert query["records"][1]["health.status"] == "degraded"
    assert query["records"][1]["health.event_count"] == 5

    inventory = smi_module.build_inventory([state])
    assert "XID_79" in render_health_events(inventory)


def test_virtual_smi_rejects_invalid_fault_config_without_events(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = tmp_path / "invalid-faults.json"
    monkeypatch.setenv(
        "FAKEGPU_FAULT_EVENTS",
        "0:XID_79:panic",
    )
    state = SmiStatePublisher(
        path,
        lambda: {
            "devices": [
                {"index": 0, "total_memory": 2**30},
                {"index": 1, "total_memory": 2**30},
            ]
        },
    ).publish_once(running=True)

    assert state["faults"]["configured"] is True
    assert state["faults"]["valid"] is False
    assert state["faults"]["events"] == []
    assert state["faults"]["status"] == "configuration_error"
    assert all(
        device["health"]["status"] == "configuration_error"
        and device["health"]["events"] == []
        for device in state["devices"]
    )

    assert main(["events", "--state", str(path)]) == 0
    rendered = capsys.readouterr().out
    assert "FAULT_CONFIG" in rendered
    assert "severity must be info, warning, error, or critical" in (
        rendered
    )
    assert "XID_79" not in rendered


@pytest.mark.parametrize("health", [[1], 5, "x"])
def test_virtual_smi_events_ignore_non_object_publisher_health(
    tmp_path: Path,
    capsys,
    health: object,
) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "timestamp_ns": time.time_ns(),
                "pid": 42,
                "running": True,
                "publisher": {"health": health},
                "devices": [],
            }
        ),
        encoding="utf-8",
    )

    assert main(["events", "--state", str(path)]) == 0
    rendered = capsys.readouterr().out
    assert "No modeled fault or runtime reliability events found." in rendered


def test_virtual_smi_treats_non_finite_device_numbers_as_missing(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "timestamp_ns": time.time_ns(),
                "pid": 42,
                "running": True,
                "devices": [{"index": 0, "total_memory": float("inf")}],
            }
        ),
        encoding="utf-8",
    )

    assert main(["--json", "--state", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["inventory"]["devices"][0]["memory"]["total_bytes"] == 0


def _native_smi_state(
    tmp_path: Path,
    environment: dict[str, str],
    *,
    hold_seconds: float = 0.0,
) -> dict:
    try:
        lib_dir = library_dir()
    except FileNotFoundError:
        pytest.skip("native libraries are not built")
    suffix = ".dylib" if sys.platform == "darwin" else ".so.1"
    path = tmp_path / "native-state.json"
    code = (
        "import ctypes, sys, time\n"
        "nvml = ctypes.CDLL(sys.argv[1])\n"
        "assert nvml.nvmlInit_v2() == 0\n"
        "time.sleep(float(sys.argv[2]))\n"
        "nvml.nvmlShutdown()\n"
    )
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("FAKEGPU_")
    }
    env.update(
        FAKEGPU_DEVICE_COUNT="2",
        FAKEGPU_SMI_STATE_PATH=str(path),
        FAKEGPU_REPORT_PATH=str(tmp_path / "native-report.json"),
        FAKEGPU_TERMINAL_REPORT="0",
        **environment,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(lib_dir / f"libnvidia-ml{suffix}"),
            str(hold_seconds),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=60,
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _python_smi_state(
    tmp_path: Path,
    monkeypatch,
    environment: dict[str, str],
) -> dict:
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    return SmiStatePublisher(
        tmp_path / "python-state.json",
        lambda: {
            "devices": [
                {
                    "index": index,
                    "profile_id": "a100",
                    "total_memory": 80 * 2**30,
                }
                for index in range(2)
            ]
        },
    ).publish_once(running=True)


@pytest.mark.parametrize(
    ("section", "environment"),
    [
        (
            "faults",
            {"FAKEGPU_FAULT_EVENTS": "0:XID_79:critical:-18446744073709551615"},
        ),
        ("faults", {"FAKEGPU_FAULT_EVENTS": "-0:XID_79:critical"}),
        ("faults", {"FAKEGPU_FAULT_EVENTS": "0::critical"}),
        ("faults", {"FAKEGPU_FAULT_EVENTS": "0:XID::5"}),
        ("faults", {"FAKEGPU_FAULT_EVENTS": "0:XID: :2"}),
        ("mig", {"FAKEGPU_MIG_LAYOUT": "-0:1g.10gb:10240"}),
        ("topology", {"FAKEGPU_NVLINK_GROUPS": "-0,1"}),
        ("topology", {"FAKEGPU_NVLINK_FABRIC": "mesh:0-5:1:50"}),
        ("topology", {"FAKEGPU_NVLINK_FABRIC": "switch:0,1:2:25;mesh:1,3:2:12.5"}),
    ],
)
def test_native_smi_environment_errors_match_python(
    tmp_path: Path,
    monkeypatch,
    section: str,
    environment: dict[str, str],
) -> None:
    native = _native_smi_state(tmp_path, environment)[section]
    python = _python_smi_state(tmp_path, monkeypatch, environment)[section]

    assert python["valid"] is False
    assert native["valid"] is False
    assert native["error"] == python["error"]


def test_native_smi_publisher_waits_out_a_huge_interval(tmp_path: Path) -> None:
    state = _native_smi_state(
        tmp_path,
        {"FAKEGPU_SMI_INTERVAL_MS": "1e300"},
        hold_seconds=0.5,
    )

    assert state["publisher"]["health"]["attempted_writes"] == 2


def test_native_smi_limits_clamp_negative_values_like_python(
    tmp_path: Path,
    monkeypatch,
) -> None:
    environment = {
        "FAKEGPU_SMI_DETAIL_LIMIT": "-1",
        "FAKEGPU_SMI_MAX_STATE_BYTES": "-5",
    }
    native = _native_smi_state(tmp_path, environment)["publisher"]["limits"]
    python = _python_smi_state(tmp_path, monkeypatch, environment)["publisher"]["limits"]

    assert native == python == {"detail_entries": 0, "max_state_bytes": 65536}


def test_virtual_smi_process_query_filters_and_marks_stale_state(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "state.json"
    state = SmiStatePublisher(
        path,
        lambda: {
            "devices": [
                {
                    "index": 0,
                    "name": "Fake A100",
                    "profile_id": "a100",
                    "total_memory": 80 * 2**30,
                    "current_memory": 2**30,
                },
                {
                    "index": 1,
                    "name": "Fake H100",
                    "profile_id": "h100",
                    "total_memory": 80 * 2**30,
                    "current_memory": 2 * 2**30,
                },
            ]
        },
    ).publish_once(running=True)
    state["timestamp_ns"] = time.time_ns() - 10 * 10**9
    path.write_text(json.dumps(state), encoding="utf-8")

    assert (
        main(
            [
                "--state",
                str(path),
                "--query-compute-apps",
                (
                    "gpu_index,gpu_uuid,pid,process_name,"
                    "used_gpu_memory,status,state.age"
                ),
                "--format",
                "json",
                "--stale-after-seconds",
                "1",
                "-i",
                "1",
            ]
        )
        == 0
    )
    query = json.loads(capsys.readouterr().out)
    assert query["schema_version"] == "fakegpu.smi_query.v1"
    assert len(query["records"]) == 1
    record = query["records"][0]
    assert record["gpu_index"] == 1
    assert record["used_gpu_memory"] == 2048
    assert record["status"] == "stale"
    assert record["state.age"] >= 10

    assert main(["--state", str(path), "--json", "-i", "0"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == "fakegpu.smi_report.v1"
    assert report["inventory"]["device_count"] == 1
    assert report["inventory"]["devices"][0]["index"] == 0


def test_virtual_smi_reads_legacy_state_schema(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "fakegpu.smi_state.v1",
                "timestamp_ns": time.time_ns(),
                "hostname": "legacy-host",
                "pid": 42,
                "process_name": "python legacy.py",
                "runtime": "fakecuda",
                "running": True,
                "tracking_confidence": "C2_torch_tensor_lifetime",
                "stage": "forward",
                "devices": [
                    {
                        "index": 0,
                        "name": "Fake A100",
                        "profile_id": "a100",
                        "total_memory": 80 * 2**30,
                        "tracked_memory": 2**30,
                        "reported_memory": 2**30,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main(["--state", str(path), "-q"]) == 0
    output = capsys.readouterr().out
    assert "schema fakegpu.smi_state.v1" in output
    assert "Architecture: ampere, compute capability 8.0" in output
    assert "python legacy.py" in output


def test_virtual_smi_reports_native_runtime_activity(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "native.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "timestamp_ns": time.time_ns(),
                "hostname": "native-host",
                "pid": 77,
                "process_name": "native-workload",
                "runtime": "native",
                "running": True,
                "tracking_confidence": (
                    "C2_native_allocation_lifetime"
                ),
                "stage": "native",
                "allocator_model": (
                    "direct_native_allocations.v1"
                ),
                "fakegpu": {
                    "version": "1.5.5",
                    "runtime": "native",
                    "backend": "native_interception",
                    "mode": "simulate",
                    "memory_tracking_enabled": True,
                },
                "devices": [
                    {
                        "index": 0,
                        "name": "Fake NVIDIA A100-SXM4-80GB",
                        "profile_id": "a100",
                        "profile": {
                            "id": "a100",
                            "architecture": "ampere",
                            "compute_capability": "8.0",
                        },
                        "uuid": (
                            "GPU-00000000-abcd-ef01-2345-"
                            "6789abcdef00"
                        ),
                        "pci_bus_id": "00000000:01:00.0",
                        "total_memory": 80 * 2**30,
                        "tracked_memory": 64 * 2**20,
                        "peak_tracked_memory": 96 * 2**20,
                        "reserved_memory": 64 * 2**20,
                        "peak_reserved_memory": 96 * 2**20,
                        "reported_memory": 64 * 2**20,
                        "reported_peak_memory": 96 * 2**20,
                        "allocator_model": (
                            "direct_native_allocations.v1"
                        ),
                        "allocation_count": 4,
                        "free_count": 2,
                        "native_activity": {
                            "io_calls": 3,
                            "io_bytes": 6 * 2**20,
                            "kernel_launches": 2,
                            "gemm_calls": 1,
                            "gemm_flops": 4096,
                            "compatibility_events": 1,
                            "unsupported_api_calls": 2,
                            "kernels": {"native_kernel": 2},
                            "unsupported_apis": [
                                {
                                    "operation": "cudaLaunchKernel",
                                    "behavior": "not_executed",
                                    "policy": "warn",
                                    "count": 2,
                                }
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main(["--state", str(path), "-q"]) == 0
    detail = capsys.readouterr().out
    assert "runtime native, backend native_interception" in detail
    assert "Profile: a100 [reference]" in detail
    assert "Memory profile: dedicated" in detail
    assert "2 kernel launches, 1 GEMM calls / 4096 FLOP" in detail
    assert "3 IO calls / 6.0 MiB" in detail
    assert "native_kernel=2" in detail
    assert "cudaLaunchKernel: 2 calls [not_executed, warn]" in detail
    assert "native capabilities 5 groups / 109 APIs / 96" in detail

    assert (
        main(
            [
                "--state",
                str(path),
                "--query-gpu",
                (
                    "runtime,runtime.backend,native.io_calls,"
                    "native.io_bytes,native.kernel_launches,"
                    "native.gemm_calls,native.gemm_flops,"
                    "native.unsupported_api_calls"
                ),
                "--format",
                "json",
            ]
        )
        == 0
    )
    query = json.loads(capsys.readouterr().out)
    assert query["records"] == [
        {
            "runtime": "native",
            "runtime.backend": "native_interception",
            "native.io_calls": 3,
            "native.io_bytes": 6,
            "native.kernel_launches": 2,
            "native.gemm_calls": 1,
            "native.gemm_flops": 4096,
            "native.unsupported_api_calls": 2,
        }
    ]


def test_render_table_marks_exited_process() -> None:
    state = {
        "hostname": "host",
        "pid": 42,
        "process_name": "python model.py",
        "running": False,
        "devices": [
            {
                "index": 0,
                "name": "Fake GPU",
                "total_memory": 1024,
                "tracked_memory": 512,
                "reported_memory": 768,
            }
        ],
    }
    rendered = render_table([state])
    assert "(exited)" in rendered
    assert "python model.py" in rendered


def test_virtual_smi_rejects_unknown_schema(tmp_path: Path, capsys) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"schema_version": "unknown"}), encoding="utf-8")
    assert main(["--state", str(path)]) == 1
    assert "unsupported schema" in capsys.readouterr().err


def test_virtual_smi_rejects_non_object_state(tmp_path: Path, capsys) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps([]), encoding="utf-8")
    assert main(["--state", str(path)]) == 1
    assert "state root must be an object" in capsys.readouterr().err


def test_render_table_distinguishes_hosts_profiles_and_stages() -> None:
    states = []
    for host, pid, profile, stage in (
        ("host-a", 10, "a100", "forward"),
        ("host-b", 20, "h100", "backward"),
        ("host-a", 30, "h100", "decode"),
    ):
        states.append(
            {
                "hostname": host,
                "pid": pid,
                "process_name": "python train.py",
                "running": True,
                "tracking_confidence": "C2_torch_tensor_lifetime",
                "stage": stage,
                "devices": [
                    {
                        "index": 0,
                        "name": f"Fake {profile.upper()}",
                        "profile_id": profile,
                        "total_memory": 8 * 2**30,
                        "tracked_memory": 2**30,
                        "peak_tracked_memory": 2 * 2**30,
                        "reported_memory": 2**30,
                        "reported_peak_memory": 2 * 2**30,
                    }
                ],
            }
        )

    rendered = render_table(states)
    assert "| host-a | 0 | a100 |" in rendered
    assert "| host-a | 0 | h100 |" in rendered
    assert "| host-b | 0 | h100 |" in rendered
    assert "forward" in rendered
    assert "backward" in rendered
    assert "decode" in rendered
    assert "2048 MiB" in rendered


def test_virtual_smi_loop_emits_bounded_ndjson(tmp_path: Path, capsys) -> None:
    path = tmp_path / "state.json"
    SmiStatePublisher(path, _snapshot).publish_once(running=True)

    assert (
        main(
            [
                "--state-dir",
                str(tmp_path),
                "--json",
                "--loop",
                "0.001",
                "--count",
                "2",
            ]
        )
        == 0
    )
    samples = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(samples) == 2
    assert all(sample["errors"] == [] for sample in samples)
    assert all(sample["states"][0]["stage"] == "forward" for sample in samples)


def test_virtual_smi_loop_rediscovers_state_directory(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = tmp_path / "late-state.json"

    def publish_during_wait(_seconds: float) -> None:
        SmiStatePublisher(path, _snapshot).publish_once(running=True)

    monkeypatch.setattr(smi_module.time, "sleep", publish_during_wait)
    assert main(["--state-dir", str(tmp_path), "--loop", "1", "--count", "2"]) == 0
    output = capsys.readouterr().out
    assert output.count("No published FakeCUDA processes found.") == 1
    assert "forward" in output


def test_publisher_uses_environment_stage_and_counts_overhead_without_total(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "state.json"
    monkeypatch.setenv("FAKEGPU_PREFLIGHT_STAGE", "optimizer_step")
    publisher = SmiStatePublisher(
        path,
        lambda: {
            "devices": [
                {
                    "index": 0,
                    "current_memory": 2**20,
                    "peak_memory": 2 * 2**20,
                    "total_memory": 0,
                }
            ]
        },
        runtime_overhead_bytes=3 * 2**20,
    )

    state = publisher.publish_once(running=True)
    assert state["stage"] == "optimizer_step"
    assert state["devices"][0]["reported_memory"] == 4 * 2**20
    assert state["devices"][0]["reported_peak_memory"] == 5 * 2**20


def test_publisher_limits_details_and_preserves_last_valid_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bounded-state.json"
    snapshot = {
        "devices": [
            {
                "index": 0,
                "name": "Fake GPU",
                "total_memory": 2**30,
                "largest_allocations": [
                    {"bytes": index + 1, "source": f"allocation-{index}"}
                    for index in range(5)
                ],
            }
        ]
    }
    publisher = SmiStatePublisher(
        path,
        lambda: snapshot,
        detail_limit=2,
        max_state_bytes=64 * 1024,
    )

    state = publisher.publish_once(running=True)
    device = state["devices"][0]
    assert len(device["largest_allocations"]) == 2
    assert device["largest_allocations_total"] == 5
    assert device["largest_allocations_retained"] == 2
    previous = path.read_bytes()

    snapshot["devices"][0]["name"] = "x" * (70 * 1024)
    with pytest.raises(ValueError, match="serialized state"):
        publisher.publish_once(running=True)
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob(".bounded-state.json.*.tmp"))

    snapshot["devices"][0]["name"] = "Fake GPU"
    recovered = publisher.publish_once(running=True)
    health = recovered["publisher"]["health"]
    assert health["attempted_writes"] == 3
    assert health["successful_writes"] == 2
    assert health["failed_writes"] == 1
    assert health["last_error"] == "ValueError"


def test_publisher_background_failure_warns_once(tmp_path: Path) -> None:
    publisher = SmiStatePublisher(tmp_path / "state.json", _snapshot)

    class StopAfterTwoFailures:
        calls = 0

        def wait(self, _timeout: float) -> bool:
            self.calls += 1
            return self.calls > 2

    def fail_publish(*, running: bool) -> dict:
        assert running is True
        raise OSError("disk full")

    publisher._stop = StopAfterTwoFailures()  # type: ignore[assignment]
    publisher.publish_once = fail_publish  # type: ignore[method-assign]
    with pytest.warns(RuntimeWarning, match="background state publish") as caught:
        publisher._run()

    assert len(caught) == 1


def test_publisher_caps_a_huge_interval_like_the_native_publisher(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    publisher = SmiStatePublisher(path, _snapshot, interval_seconds=1e297)
    publisher.start()
    publisher.stop()

    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["running"] is False
    assert state["publisher"]["interval_seconds"] == 1e9


@pytest.mark.parametrize("ending", ["exit", "sigterm"])
def test_replaced_publisher_does_not_overwrite_final_state(
    tmp_path: Path,
    ending: str,
) -> None:
    path = tmp_path / "state.json"
    code = "\n".join(
        [
            "import os, signal, sys, time",
            "from fakegpu.smi import SmiStatePublisher",
            "def snapshot(name):",
            "    return lambda: {'devices': [{'index': 0, 'name': name}]}",
            "old = SmiStatePublisher(sys.argv[1], snapshot('old'), interval_seconds=60)",
            "old.start()",
            "old.stop()",
            "new = SmiStatePublisher(sys.argv[1], snapshot('new'), interval_seconds=60)",
            "new.start()",
            "new.publish_once(running=True)",
            "if sys.argv[2] == 'sigterm':",
            "    os.kill(os.getpid(), signal.SIGTERM)",
            "    time.sleep(10)",
        ]
    )

    completed = subprocess.run(
        [sys.executable, "-c", code, str(path), ending],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    expected_code = -signal.SIGTERM if ending == "sigterm" else 0
    assert completed.returncode == expected_code, completed.stderr
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["running"] is False
    assert state["devices"][0]["name"] == "new"
    assert state["publisher"]["health"]["attempted_writes"] >= 2
    if ending == "sigterm":
        assert state["exit_signal"] == "SIGTERM"
    else:
        assert "exit_signal" not in state


def test_fakecuda_runtime_publishes_profile_stage_and_peak(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime-state.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(ROOT)
        if not env.get("PYTHONPATH")
        else str(ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    env["FAKEGPU_SMI_STATE_PATH"] = str(state_path)
    env["FAKEGPU_TERMINAL_REPORT"] = "0"
    code = "\n".join(
        [
            "import json",
            "import fakegpu",
            "import torch",
            "fakegpu.init(runtime='fakecuda', profile='test-512m', device_count=1)",
            "with fakegpu.stage('forward'):",
            "    tensor = torch.empty((1024, 1024), device='cuda', dtype=torch.float32)",
            "    from fakegpu import torch_patch",
            "    state = torch_patch._smi_publisher.publish_once(running=True)",
            "    print(json.dumps(state, sort_keys=True))",
        ]
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    state = json.loads(completed.stdout.strip().splitlines()[-1])
    assert state["schema_version"] == SCHEMA_VERSION
    assert state["stage"] == "forward"
    assert state["tracking_confidence"] == "C3_torch_dispatch_lifetime"
    assert state["fakegpu"]["backend"] == "upstream"
    assert state["fakegpu"]["profile_catalog"]["profile_count"] == len(load_profiles())
    assert (
        state["fakegpu"]["native_capabilities"]["group_count"]
        == 5
    )
    assert state["devices"][0]["profile_id"] == "test-512m"
    assert state["devices"][0]["profile"]["architecture"] == "ampere"
    assert state["devices"][0]["profile"]["sm_count"] == 108
    assert (
        state["devices"][0]["allocator_model"]
        == "cuda_caching_allocator.v1"
    )
    assert state["devices"][0]["allocation_count"] >= 1
    assert state["devices"][0]["largest_allocations"]
    assert state["devices"][0]["tracked_memory"] >= 4 * 2**20
    assert state["devices"][0]["peak_tracked_memory"] >= 4 * 2**20
    assert state["devices"][0]["reported_peak_memory"] >= 4 * 2**20


def test_fakecuda_first_published_state_names_the_backend(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime-state.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(ROOT)
        if not env.get("PYTHONPATH")
        else str(ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    env["FAKEGPU_SMI_STATE_PATH"] = str(state_path)
    env["FAKEGPU_SMI_INTERVAL_MS"] = "600000"
    env["FAKEGPU_TERMINAL_REPORT"] = "0"
    env["FAKEGPU_MACHINE"] = "off"
    code = "\n".join(
        [
            "import fakegpu",
            "fakegpu.init(runtime='fakecuda', profile='test-512m', device_count=1)",
            f"print(open({str(state_path)!r}).read())",
        ]
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    state = json.loads(completed.stdout)
    assert state["fakegpu"]["backend"] == "upstream"


def test_virtual_smi_count_requires_loop(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert "FakeGPU-SMI" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc_info:
        main(["--state", "state.json", "--count", "2"])
    assert exc_info.value.code == 2
    assert "--count requires --loop" in capsys.readouterr().err

    assert main(["--help-query-gpu"]) == 0
    query_help = capsys.readouterr().out
    assert "memory.total [MiB]" in query_help
    assert "fakegpu.version" in query_help

    assert main(["--help-query-runtime"]) == 0
    runtime_query_help = capsys.readouterr().out
    assert "publisher.failed_writes" in runtime_query_help
    assert "catalog.native_apis" in runtime_query_help

    with pytest.raises(SystemExit) as exc_info:
        main(["--query-gpu", "unknown.field"])
    assert exc_info.value.code == 2
    assert "unsupported query field" in capsys.readouterr().err


def _csv_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.split(",")]


def test_virtual_smi_accepts_every_nvidia_smi_query_field_and_alias(
    tmp_path: Path,
    capsys,
) -> None:
    # Field names, aliases and headers from nvidia-smi 590.48.01
    # --help-query-gpu / --help-query-compute-apps. Fields FakeGPU does not
    # model print nvidia-smi's [N/A] instead of failing the whole query.
    path = tmp_path / "state.json"
    SmiStatePublisher(path, _snapshot).publish_once(running=True)

    # Lightning's get_nvidia_gpu_stats query.
    assert main([
        "--state", str(path),
        "--query-gpu=utilization.gpu,memory.used,memory.free,"
        "utilization.memory,fan.speed,temperature.gpu,temperature.memory",
        "--format=csv,nounits,noheader",
        "--id=0",
    ]) == 0
    assert _csv_cells(capsys.readouterr().out.strip()) == [
        "[N/A]", "3072", "5120", "[N/A]", "[N/A]", "[N/A]", "[N/A]",
    ]

    # Aliases are headed by the first name nvidia-smi lists.
    assert main([
        "--state", str(path),
        "--query-gpu=gpu_name,gpu_bus_id,gpu_uuid,clocks.sm,clocks.mem,"
        "gpu.power.max_limit,vbios_version,power.draw",
        "--format=csv",
    ]) == 0
    header, row = capsys.readouterr().out.splitlines()
    assert _csv_cells(header) == [
        "name", "pci.bus_id", "uuid", "clocks.current.sm [MHz]",
        "clocks.current.memory [MHz]", "power.max_limit [W]",
        "vbios_version", "power.draw [W]",
    ]
    cells = _csv_cells(row)
    assert cells[:2] == ["Fake NVIDIA Test GPU", "00000000:01:00.0"]
    assert cells[-2:] == ["[N/A]", "[N/A]"]

    assert main([
        "--state", str(path),
        "--query-compute-apps=pid,name,used_memory,gpu_bus_id,gpu_serial",
        "--format=csv",
    ]) == 0
    header, row = capsys.readouterr().out.splitlines()
    assert _csv_cells(header) == [
        "pid", "process_name", "used_gpu_memory [MiB]", "gpu_bus_id",
        "gpu_serial",
    ]
    assert _csv_cells(row)[2:] == ["3072 MiB", "00000000:01:00.0", "[N/A]"]

    with pytest.raises(SystemExit) as exc_info:
        main(["--state", str(path), "--query-gpu", "utilization.gpuu"])
    assert exc_info.value.code == 2


def test_virtual_smi_query_timestamp_is_the_query_time_in_milliseconds(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    import datetime

    import fakegpu._smi_query as smi_query

    path = tmp_path / "state.json"
    SmiStatePublisher(path, _snapshot).publish_once(running=True)
    query_ns = time.time_ns() + 123_456_789
    monkeypatch.setattr(smi_query.time, "time_ns", lambda: query_ns)

    assert main([
        "--state", str(path),
        "--query-gpu=timestamp,index",
        "--format=csv,noheader",
    ]) == 0
    stamp = _csv_cells(capsys.readouterr().out.strip())[0]
    # nvidia-smi --help-query-gpu: "YYYY/MM/DD HH:MM:SS.msec", the time of
    # the query; parsed as PerfKitBenchmarker's nvidia_power collector does.
    parsed = datetime.datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f")
    assert parsed.timestamp() == pytest.approx(query_ns // 10**6 / 1e3)


def test_virtual_smi_looping_query_appends_rows_under_one_header(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = tmp_path / "state.json"
    SmiStatePublisher(path, _snapshot).publish_once(running=True)
    sleeps: list[float] = []
    monkeypatch.setattr(smi_module.time, "sleep", sleeps.append)

    # nvidia-smi's -lms is -l in milliseconds; a looping CSV query prints the
    # header once and then only rows, so one csv.DictReader reads the whole
    # stream (PerfKitBenchmarker's nvidia_power collector relies on this).
    assert main([
        "--state", str(path),
        "--query-gpu=index,memory.used",
        "--format=csv",
        "-lms", "250",
        "--count", "3",
    ]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [_csv_cells(line) for line in lines] == [
        ["index", "memory.used [MiB]"],
        ["0", "3072 MiB"],
        ["0", "3072 MiB"],
        ["0", "3072 MiB"],
    ]
    assert sleeps == [0.25, 0.25]

    # -l without a value loops every 5 seconds (nvidia-smi man page).
    sleeps.clear()
    assert main([
        "--state", str(path),
        "--query-gpu=index",
        "--format=csv,noheader",
        "-l",
        "--count", "2",
    ]) == 0
    assert capsys.readouterr().out.splitlines() == ["0", "0"]
    assert sleeps == [5.0]


def test_virtual_smi_reports_the_reference_power_ceiling_and_unknown_specs(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "state.json"
    SmiStatePublisher(
        path,
        lambda: {
            "devices": [
                {
                    "index": index,
                    "name": profile_id,
                    "profile_id": profile_id,
                    "total_memory": 64 * 2**30,
                }
                for index, profile_id in enumerate(("a100", "h100-sxm5-64gb"))
            ]
        },
    ).publish_once(running=True)

    assert main([
        "--state", str(path),
        "--query-gpu=index,power.default_limit,power.max_limit,clocks.sm,sm_count",
        "--format=csv,noheader,nounits",
    ]) == 0
    rows = [_csv_cells(line) for line in capsys.readouterr().out.splitlines()]
    # power.default_limit is nvidia-smi's default power ceiling, equal to the
    # maximum limit on an SXM part (an H100 SXM reports 700.00 W for both).
    # The a100 profile's typical workload power is unknown (0); it is not the
    # ceiling. Specifications the catalog marks unknown print [N/A], not 0.
    assert rows == [
        ["0", "400.00", "400.00", "1410", "108"],
        ["1", "[N/A]", "[N/A]", "[N/A]", "[N/A]"],
    ]

    # nvidia-smi 580.178.04 prints watts with two decimals: "700.00 W" with
    # units, "700.00" with nounits.
    assert main([
        "--state", str(path), "-i", "0",
        "--query-gpu=power.default_limit,power.min_limit,power.max_limit",
        "--format=csv,noheader",
    ]) == 0
    assert capsys.readouterr().out == "400.00 W, [N/A], 400.00 W\n"
    assert main([
        "--state", str(path), "-i", "0", "--query-gpu=power.max_limit",
        "--format=json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["records"] == [
        {"power.max_limit": 400.0}
    ]


def test_virtual_smi_selects_pci_bus_ids_in_every_nvml_form(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "state.json"
    SmiStatePublisher(
        path,
        lambda: {
            "devices": [
                {
                    "index": index,
                    "name": "Fake NVIDIA Test GPU",
                    "profile_id": "test",
                    "total_memory": 8 * 2**30,
                }
                for index in range(3)
            ]
        },
    ).publish_once(running=True)

    # nvidia-smi -i takes a PCI bus ID as domain:bus:device.function in hex;
    # nvmlDeviceGetHandleByPciBusId also accepts domain:bus:device and
    # bus:device.function (nvml.h).
    for selector in (
        "00000000:02:00.0", "0000:02:00.0", "02:00.0", "0:2:0", "0:2:0.0",
    ):
        assert main([
            "--state", str(path), "-i", selector,
            "--query-gpu=index,pci.bus_id", "--format=csv,noheader",
        ]) == 0
        assert _csv_cells(capsys.readouterr().out.strip()) == [
            "1", "00000000:02:00.0",
        ], selector


def test_virtual_smi_reports_a_modeled_driver_version_like_nvidia_smi(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    import re

    from packaging.version import Version

    monkeypatch.delenv("FAKEGPU_DRIVER_VERSION", raising=False)
    path = tmp_path / "state.json"
    SmiStatePublisher(path, _snapshot).publish_once(running=True)

    # nvidia-smi's banner reads "Driver Version: 580.178.04     CUDA Version:
    # 13.0"; torch and vLLM collect_env parse it with these expressions.
    assert main(["--state", str(path)]) == 0
    table = capsys.readouterr().out
    driver = re.search(r"Driver Version: (.*?) ", table)
    assert driver is not None and driver.group(1) == "570.195.03"
    assert "Driver Version: 570.195.03 (modeled)" in table
    assert re.search(r"CUDA Version:\s*(\d+\.\d+)", table)

    # --query-gpu=driver_version is a parseable version, as on a real GPU.
    assert main([
        "--state", str(path), "--query-gpu=driver_version",
        "--format=csv,noheader",
    ]) == 0
    assert Version(capsys.readouterr().out.strip()) == Version("570.195.3")

    assert main(["--state", str(path), "-q"]) == 0
    assert "driver 570.195.03 (modeled)" in capsys.readouterr().out

    # Older states and the native publisher wrote a "simulated" placeholder.
    legacy = json.loads(path.read_text(encoding="utf-8"))
    legacy["software"]["driver_version"] = "simulated"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert main([
        "--state", str(path), "--query-gpu=driver_version",
        "--format=csv,noheader",
    ]) == 0
    assert capsys.readouterr().out.strip() == "570.195.03"

    monkeypatch.setenv("FAKEGPU_DRIVER_VERSION", "550.54.15")
    SmiStatePublisher(path, _snapshot).publish_once(running=True)
    assert main([
        "--state", str(path), "--query-gpu=driver_version",
        "--format=csv,noheader",
    ]) == 0
    assert capsys.readouterr().out.strip() == "550.54.15"


def test_virtual_smi_writes_state_warnings_to_stderr(
    tmp_path: Path,
    capsys,
) -> None:
    SmiStatePublisher(tmp_path / "good.json", _snapshot).publish_once(running=True)
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")

    # nvidia-smi prints only its report on stdout. Line counters such as
    # MONAI's nnunetv2 runner and DeepSpeed's test helper count
    # `nvidia-smi -L` lines to find the number of GPUs.
    for view in (["-L"], [], ["-q"], ["topo", "-m"], ["--query-gpu=index", "--format=csv,noheader"]):
        assert main([*view, "--state-dir", str(tmp_path)]) == 0
        captured = capsys.readouterr()
        assert "warning" not in captured.out, view
        assert "bad.json" in captured.err, view
    assert main(["-L", "--state-dir", str(tmp_path)]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_virtual_smi_list_names_hosts_only_when_several_are_shown(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    for host in ("node-a", "node-b"):
        monkeypatch.setattr(smi_module.socket, "gethostname", lambda host=host: host)
        SmiStatePublisher(tmp_path / f"{host}.json", _snapshot).publish_once(running=True)

    assert main(["-L", "--state", str(tmp_path / "node-a.json")]) == 0
    assert capsys.readouterr().out.startswith("GPU 0: Fake NVIDIA Test GPU (UUID: GPU-")

    assert main(["-L", "--state-dir", str(tmp_path)]) == 0
    listed = capsys.readouterr().out.splitlines()
    assert [line.split(" (UUID: ")[0] for line in listed] == [
        "GPU 0 @ node-a: Fake NVIDIA Test GPU",
        "GPU 0 @ node-b: Fake NVIDIA Test GPU",
    ]


def test_virtual_smi_exits_6_when_a_selector_matches_no_gpu(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "state.json"
    SmiStatePublisher(path, _snapshot).publish_once(running=True)

    # nvidia-smi 580.178.04: -i 9, -i GPU-nonexistent and an unknown bus ID
    # print "No devices were found" on stdout and exit 6, in every view.
    for view in (
        [], ["-q"], ["--query-gpu=index", "--format=csv"],
        ["--query-compute-apps=pid", "--format=csv"],
    ):
        for selector in ("9", "GPU-nonexistent", "00000000:99:00.0"):
            assert main([*view, "--state", str(path), "-i", selector]) == 6
            assert capsys.readouterr().out == "No devices were found\n"

    # A list that matches some GPUs prints those (-i 0,9 printed GPU 0).
    assert main([
        "--state", str(path), "-i", "0,9", "--query-gpu=index", "--format=csv",
    ]) == 0
    assert capsys.readouterr().out.splitlines() == ["index", "0"]


def test_virtual_smi_accepts_format_modifiers_in_any_order(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "state.json"
    SmiStatePublisher(path, _snapshot).publish_once(running=True)

    # nvidia-smi 580.178.04 printed the same row for each of these, and
    # CSV with a header when --format was left out.
    for value in ("csv,noheader", "noheader,csv", "noheader", "csv,noheader,csv"):
        assert main([
            "--state", str(path), "--query-gpu=index,memory.total",
            f"--format={value}",
        ]) == 0
        assert capsys.readouterr().out == "0, 8192 MiB\n", value
    assert main(["--state", str(path), "--query-gpu=index", "--format=nounits,noheader,csv"]) == 0
    assert capsys.readouterr().out == "0\n"
    assert main(["--state", str(path), "--query-gpu=index"]) == 0
    assert capsys.readouterr().out == "index\n0\n"

    # "Format modifier is not recognized." exits 2.
    with pytest.raises(SystemExit) as exc_info:
        main(["--state", str(path), "--query-gpu=index", "--format=csv,bogus"])
    assert exc_info.value.code == 2
    assert "Format modifier is not recognized" in capsys.readouterr().err


def test_virtual_smi_memory_reserved_is_the_driver_reserve(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "state.json"
    SmiStatePublisher(
        path,
        lambda: {
            "devices": [
                {
                    "index": 0,
                    "name": "Fake NVIDIA Test GPU",
                    "profile_id": "test",
                    "total_memory": 8 * 2**30,
                    "current_memory": 64 * 2**20,
                    "current_reserved_memory": 80 * 2**20,
                }
            ]
        },
    ).publish_once(running=True)

    # nvidia-smi --help-query-gpu: memory.reserved is "Total memory reserved
    # by the NVIDIA driver and firmware"; on a real GPU total = reserved +
    # used + free (479 + 0 + 81081 MiB in a public H100
    # capture). FakeGPU does not model it; the allocator's reserved
    # bytes moved to memory.allocator_reserved.
    assert main([
        "--state", str(path),
        "--query-gpu=memory.reserved,memory.allocator_reserved,memory.used",
        "--format=csv",
    ]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "memory.reserved [MiB], memory.allocator_reserved [MiB], memory.used [MiB]",
        "[N/A], 80 MiB, 80 MiB",
    ]

    assert main(["--help-query-gpu"]) == 0
    help_lines = capsys.readouterr().out.splitlines()
    reserved = next(
        line for line in help_lines if line.strip().startswith("memory.reserved ")
    )
    assert "not modeled" in reserved
    assert "memory.allocator_reserved" in reserved


def test_virtual_smi_topology_notes_cannot_fake_an_nvlink(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FAKEGPU_NVLINK_GROUPS", raising=False)
    monkeypatch.delenv("FAKEGPU_PLATFORM", raising=False)
    for host in ("NVDGX-01", "NVDGX-02"):
        monkeypatch.setattr(smi_module.socket, "gethostname", lambda host=host: host)
        SmiStatePublisher(tmp_path / f"{host}.json", _snapshot).publish_once(running=True)

    # torch._inductor.comm_analysis._has_nvlink runs `nvidia-smi topo -m`
    # and checks `"NV" in out.split("Legend")[0]`. A host name that contains
    # "NV" must not read as an NVLink.
    assert main(["topo", "-m", "--state-dir", str(tmp_path)]) == 0
    matrix = capsys.readouterr().out
    assert "NV" not in matrix.split("Legend")[0]
    lines = matrix.splitlines()
    assert lines[1] == "GPU0#1\t X \tSYS\tN/A\t0\t\tN/A"
    assert lines[2] == "GPU0#2\tSYS\t X \tN/A\t0\t\tN/A"
    assert "Host: NVDGX-01 (GPU0#1)" in matrix
    assert "Host: NVDGX-02 (GPU0#2)" in matrix


def test_virtual_smi_queries_retired_pages_and_remapped_rows(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "state.json"
    SmiStatePublisher(
        path,
        lambda: {
            "devices": [
                {
                    "index": index,
                    "name": "Fake NVIDIA Test GPU",
                    "profile_id": "test",
                    "total_memory": 8 * 2**30,
                }
                for index in range(2)
            ]
        },
    ).publish_once(running=True)
    uuids = [
        record["uuid"]
        for record in json.loads(path.read_text(encoding="utf-8"))["devices"]
    ]

    # nvidia-smi 580.178.04 on an Ampere GPU lists one row per
    # retirement cause and GPU, with an [N/A] address.
    assert main([
        "--state", str(path),
        "--query-retired-pages=gpu_uuid,retired_pages.address,retired_pages.cause",
        "--format=csv",
    ]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "gpu_uuid, retired_pages.address, retired_pages.cause",
        f"{uuids[0]}, [N/A], Single Bit ECC",
        f"{uuids[0]}, [N/A], Double Bit ECC",
        f"{uuids[1]}, [N/A], Single Bit ECC",
        f"{uuids[1]}, [N/A], Double Bit ECC",
    ]

    # One row per GPU; FakeGPU does not model row remapping, so the counts
    # are [N/A], as --query-gpu=remapped_rows.* prints them.
    assert main([
        "--state", str(path), "-i", "1",
        "--query-remapped-rows=gpu_bus_id,gpu_uuid,remapped_rows.correctable,"
        "remapped_rows.uncorrectable,remapped_rows.pending,remapped_rows.failure",
        "--format=csv",
    ]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "gpu_bus_id, gpu_uuid, remapped_rows.correctable, "
        "remapped_rows.uncorrectable, remapped_rows.pending, remapped_rows.failure",
        f"00000000:02:00.0, {uuids[1]}, [N/A], [N/A], [N/A], [N/A]",
    ]
    assert main([
        "--state", str(path), "--query-remapped-rows=remapped_rows.sbe",
        "--format=csv",
    ]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "remapped_rows.correctable"

    assert main([
        "--state", str(path), "-i", "9", "--query-remapped-rows=gpu_bus_id",
        "--format=csv",
    ]) == 6
    assert capsys.readouterr().out == "No devices were found\n"

    assert main(["--help-query-retired-pages"]) == 0
    assert "retired_pages.cause" in capsys.readouterr().out

    # Accounting is not modeled: nvidia-smi 580.178.04 printed only this
    # header for a GPU with accounting disabled, aliases headed by the
    # first name.
    assert main([
        "--state", str(path),
        "--query-accounted-apps=timestamp,gpu_name,gpu_bus_id,gpu_serial,"
        "gpu_uuid,vgpu_instance,pid,gpu_utilization,gpu_util,mem_utilization,"
        "mem_util,max_memory_usage,time",
        "--format=csv",
    ]) == 0
    assert capsys.readouterr().out == (
        "timestamp, gpu_name, gpu_bus_id, gpu_serial, gpu_uuid, vgpu_instance, "
        "pid, gpu_utilization [%], gpu_utilization [%], mem_utilization [%], "
        "mem_utilization [%], max_memory_usage [MiB], time [ms]\n"
    )
