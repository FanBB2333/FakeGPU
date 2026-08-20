from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import fakegpu.smi as smi_module
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
    assert state["fakegpu"]["profile_catalog"]["profile_count"] == 82
    assert (
        state["fakegpu"]["native_capabilities"]["explicit_api_count"]
        == 26
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
    assert "Profile: test-512m" in listed
    assert "CC: 8.0" in listed
    assert "PCI: 00000000:01:00.0" in listed

    assert main(["--state", str(path), "-q"]) == 0
    detail = capsys.readouterr().out
    assert "FakeGPU-SMI detailed report" in detail
    assert "Architecture: ampere, compute capability 8.0" in detail
    assert "108 SMs" in detail
    assert "native capabilities 5 groups / 26 APIs / 24" in detail
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
        "0,FakeGPU Test Profile 512MB,test-512m,8.0,512,80,"
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
    assert query_lines[0] == "timestamp,memory.total [MiB]"
    assert query_lines[1].split(",")[0]

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
            "catalog.profiles": 82,
            "catalog.native_apis": 26,
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
        "fakegpu.device_topology.v1"
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
    assert "FakeGPU modeled topology matrix" in matrix
    assert "GPU0  X" in matrix
    assert "NV1" in matrix
    assert "modeled, not measured" in matrix

    assert main(["nvlink", "-s", "--state", str(path)]) == 0
    status = capsys.readouterr().out
    assert "Bandwidth values are configuration inputs" in status
    assert "Link 0: Active -> GPU 1" in status
    assert "800 Gbps" in status

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
    listed = capsys.readouterr().out
    assert "MIG 1g.10gb Device 0" in listed
    assert "GI: 0, CI: 0" in listed

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
    assert "native capabilities 5 groups / 26 APIs / 24" in detail

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
    assert "unsupported schema" in capsys.readouterr().out


def test_virtual_smi_rejects_non_object_state(tmp_path: Path, capsys) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps([]), encoding="utf-8")
    assert main(["--state", str(path)]) == 1
    assert "state root must be an object" in capsys.readouterr().out


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
    assert state["fakegpu"]["profile_catalog"]["profile_count"] == 82
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
