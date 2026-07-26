from __future__ import annotations

import json
from pathlib import Path

import pytest

from fakegpu.calibration import (
    BUNDLE_SCHEMA_VERSION,
    VERIFICATION_SCHEMA_VERSION,
    build_workload_calibration_bundle,
    compare_memory_reports,
    verify_calibration_reports,
)
from fakegpu.kernel_analysis import analyze_ptx, estimate_occupancy
from fakegpu.operator_profiles import (
    evaluate_operator_profile,
    load_operator_profiles,
    match_operator_profile,
)
from fakegpu.repository_analyzer import analyze_repository
from fakegpu.topology import Topology, simulate_collective
from fakegpu.trace_replay import replay_trace
from fakegpu.training_plan import (
    estimate_training_plan,
    normalize_training_config,
)


ROOT = Path(__file__).resolve().parents[1]
MEMORY_EVIDENCE_PATH = (
    ROOT / "tests" / "data" / "memory_validation_evidence.json"
)


def test_analysis_apis_are_exported_from_package() -> None:
    import fakegpu

    for name in (
        "compare_memory_reports",
        "estimate_kv_cache_memory",
        "estimate_training_plan",
        "analyze_kernel_file",
        "replay_trace",
        "simulate_collective",
    ):
        assert callable(getattr(fakegpu, name))


def _topology_config() -> dict:
    return {
        "ecmp": True,
        "switches": [
            {"id": "leaf0"},
            {"id": "leaf1"},
            {"id": "spine0"},
        ],
        "nodes": [
            {
                "id": "node0",
                "rack": "rack0",
                "nics": [
                    {
                        "id": "node0.nic0",
                        "switch": "leaf0",
                        "bandwidth_gbps": 100,
                    }
                ],
            },
            {
                "id": "node1",
                "rack": "rack0",
                "nics": [
                    {
                        "id": "node1.nic0",
                        "switch": "leaf0",
                        "bandwidth_gbps": 100,
                    }
                ],
            },
            {
                "id": "node2",
                "rack": "rack1",
                "nics": [
                    {
                        "id": "node2.nic0",
                        "switch": "leaf1",
                        "bandwidth_gbps": 100,
                    }
                ],
            },
            {
                "id": "node3",
                "rack": "rack1",
                "nics": [
                    {
                        "id": "node3.nic0",
                        "switch": "leaf1",
                        "bandwidth_gbps": 100,
                    }
                ],
            },
        ],
        "links": [
            {
                "src": "leaf0",
                "dst": "spine0",
                "bandwidth_gbps": 200,
                "latency_us": 0.5,
                "scope": "leaf_spine",
            },
            {
                "src": "leaf1",
                "dst": "spine0",
                "bandwidth_gbps": 200,
                "latency_us": 0.5,
                "scope": "leaf_spine",
            },
        ],
        "ranks": [
            {"rank": 0, "node": "node0"},
            {"rank": 1, "node": "node1"},
            {"rank": 2, "node": "node2"},
            {"rank": 3, "node": "node3"},
        ],
    }


def test_calibration_comparison_and_bundle() -> None:
    prediction = {
        "schema_version": "static_memory_estimate.v1",
        "memory_timeline": {
            "phases": [
                {
                    "phase": "forward",
                    "peak_bytes": 900,
                    "interval_bytes": {
                        "lower": 800,
                        "expected": 900,
                        "upper": 1_100,
                    },
                },
                {"phase": "backward", "peak_bytes": 1_800},
            ]
        },
        "unmodeled_components": ["cuda_context_and_loaded_modules"],
    }
    observation = {
        "schema_version": "measurement.v1",
        "memory_timeline": {
            "phases": [
                {"phase": "forward", "peak_bytes": 1_000},
                {"phase": "backward", "peak_bytes": 2_000},
            ]
        },
    }
    report = compare_memory_reports(
        prediction,
        observation,
        workload="tiny-sft",
    )

    assert report["summary"]["phase_count"] == 2
    assert report["summary"]["recommended_memory_safety_margin_bytes"] == 200
    assert report["summary"]["recommended_memory_safety_factor"] > 1
    assert report["comparisons"][1]["absolute_percentage_error"] == 0.1
    forward = next(
        item for item in report["comparisons"] if item["phase"] == "forward"
    )
    assert forward[
        "observation_within_prediction_interval"
    ] is True

    bundle = build_workload_calibration_bundle(
        [report],
        labels=["tiny-sft"],
    )
    assert bundle["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert bundle["entries"][0]["id"] == "tiny-sft"

    verification = verify_calibration_reports(
        [report],
        labels=["tiny-sft"],
        max_underestimate_percent=10,
        max_absolute_percentage_error_percent=10,
        min_interval_coverage_percent=100,
        capacity_bytes=2_100,
        min_comparisons=2,
    )
    assert verification["schema_version"] == VERIFICATION_SCHEMA_VERSION
    assert verification["status"] == "passed"
    assert verification["metrics"][
        "maximum_underestimate_percent"
    ] == pytest.approx(10)
    assert verification["metrics"]["false_safe_count"] == 0


def test_calibration_verification_rejects_oom_risk_and_signature_drift() -> None:
    report = compare_memory_reports(
        {
            "schema_version": "prediction.v1",
            "inputs": {"batch_size": 1, "dtype": "bfloat16"},
            "memory_timeline": {
                "phases": [{"phase": "peak", "peak_bytes": 900}]
            },
        },
        {
            "schema_version": "observation.v1",
            "inputs": {"batch_size": 2, "dtype": "bfloat16"},
            "memory_timeline": {
                "phases": [{"phase": "peak", "peak_bytes": 1_000}]
            },
        },
    )

    verification = verify_calibration_reports(
        [report],
        max_underestimate_percent=5,
        max_absolute_percentage_error_percent=5,
        min_interval_coverage_percent=90,
        capacity_bytes=950,
    )

    assert verification["status"] == "failed"
    assert verification["metrics"]["false_safe_count"] == 1
    assert verification["metrics"]["dimension_mismatch_count"] == 1
    assert {failure["gate"] for failure in verification["failures"]} == {
        "matching_workload_dimensions",
        "maximum_absolute_percentage_error_percent",
        "maximum_false_safe_count",
        "maximum_underestimate_percent",
        "minimum_prediction_interval_coverage_percent",
    }


def test_published_memory_evidence_matches_calibration_math() -> None:
    evidence = json.loads(MEMORY_EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert (
        evidence["schema_version"]
        == "fakegpu.memory_validation_evidence.v1"
    )
    revision = evidence["source_revision"]
    assert len(revision) == 40
    int(revision, 16)

    groups = {
        group["id"]: group for group in evidence["evidence_groups"]
    }
    assert set(groups) == {
        "controlled_aten",
        "qwen3_8b_inference",
        "qwen_sft",
        "qwen_qlora",
    }
    for group in groups.values():
        assert group["source_path"].startswith("docs/")
        assert group["source_anchor"]

    controlled = groups["controlled_aten"]
    assert controlled["workload_count"] == 13
    assert controlled["observation_count"] == 26
    assert (
        controlled[
            "published_maximum_absolute_percentage_error_percent"
        ]
        == 0.08
    )
    assert controlled["published_maximum_underestimate_percent"] == 0.08

    inference = groups["qwen3_8b_inference"]
    prediction = {
        "schema_version": "readme_evidence_prediction.v1",
        "memory_timeline": {
            "phases": [
                {
                    "phase": item["id"],
                    "peak_bytes": item["predicted_bytes"],
                }
                for item in inference["measurements"]
            ]
        },
    }
    observation = {
        "schema_version": "readme_evidence_observation.v1",
        "memory_timeline": {
            "phases": [
                {
                    "phase": item["id"],
                    "peak_bytes": item["observed_bytes"],
                }
                for item in inference["measurements"]
            ]
        },
    }
    comparison = compare_memory_reports(
        prediction,
        observation,
        workload=inference["id"],
    )
    comparisons = {
        item["phase"]: item for item in comparison["comparisons"]
    }
    for measurement in inference["measurements"]:
        actual_percent = (
            comparisons[measurement["id"]][
                "absolute_percentage_error"
            ]
            * 100
        )
        assert actual_percent == pytest.approx(
            measurement[
                "published_absolute_percentage_error_percent"
            ],
            abs=0.0000005,
        )
    assert comparison["summary"]["underprediction_phase_count"] == 2
    assert comparison["summary"]["recommended_memory_safety_margin_bytes"] > 0
    assert comparison["summary"]["recommended_memory_safety_factor"] > 1

    sft_errors = [
        item["published_absolute_percentage_error_percent"]
        for item in groups["qwen_sft"]["measurements"]
    ]
    assert len(sft_errors) == 10
    assert (min(sft_errors), max(sft_errors)) == (0.102, 1.921)

    qlora_errors = [
        item["published_absolute_percentage_error_percent"]
        for item in groups["qwen_qlora"]["measurements"]
    ]
    assert len(qlora_errors) == 10
    assert (min(qlora_errors), max(qlora_errors)) == (0.628, 1.732)


def test_training_plan_normalizes_deepspeed_zero3_offload() -> None:
    normalized = normalize_training_config(
        {
            "world_size": 4,
            "gradient_accumulation_steps": 8,
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu"},
                "overlap_comm": True,
                "reduce_bucket_size": 200,
                "allgather_bucket_size": 300,
            },
            "activation_checkpointing": {"partition_activations": True},
        }
    )
    report = estimate_training_plan(
        normalized,
        parameter_bytes=4_000,
        activation_bytes=2_000,
        optimizer="adamw",
        parameter_element_bytes=2,
    )

    assert normalized["source_framework"] == "deepspeed"
    assert report["sharding"]["parameters_sharded"] is True
    assert report["rank_local_storage"]["parameter_bytes"] == 1_000
    assert report["rank_local_storage"]["optimizer_state_bytes"] == 0
    assert report["cpu_offload"]["optimizer_and_master_parameter_bytes"] == 4_000
    assert report["communication"]["gradient_accumulation_steps"] == 8
    assert len(report["memory_timeline"]["phases"]) == 4


def test_training_plan_handles_accelerate_fsdp_and_auto_deepspeed_values() -> None:
    fsdp = normalize_training_config(
        {
            "distributed_type": "FSDP",
            "num_processes": 2,
            "mixed_precision": "bf16",
            "fsdp_config": {
                "fsdp_sharding_strategy": "SHARD_GRAD_OP",
                "fsdp_activation_checkpointing": "false",
            },
        }
    )
    fsdp_report = estimate_training_plan(
        fsdp,
        parameter_bytes=2_000,
        activation_bytes=500,
    )
    assert fsdp_report["sharding"]["parameters_sharded"] is False
    assert fsdp_report["sharding"]["gradients_sharded"] is True
    assert fsdp_report["rank_local_storage"]["gradient_bytes"] == 1_000
    assert fsdp["activation_checkpointing"] is False

    deepspeed = normalize_training_config(
        {
            "distributed_type": "DEEPSPEED",
            "num_processes": 4,
            "mixed_precision": "bf16",
            "deepspeed_config": {
                "zero_stage": 3,
                "offload_optimizer_device": "cpu",
                "gradient_accumulation_steps": "auto",
                "reduce_bucket_size": "auto",
            },
        }
    )
    assert deepspeed["zero_stage"] == 3
    assert deepspeed["optimizer_offload_device"] == "cpu"
    assert deepspeed["reduce_bucket_bytes"] == 0


def test_hierarchical_topology_routes_across_racks_and_reports_links() -> None:
    topology = Topology.from_mapping(_topology_config())
    report = simulate_collective(
        topology,
        collective="all_reduce",
        payload_bytes_per_rank=1_000_000,
        algorithm="hierarchical",
    )

    assert report["topology"]["rack_count"] == 2
    assert report["summary"]["round_count"] >= 3
    assert report["summary"]["estimated_time_us"] > 0
    assert len(report["rank_summary"]) == 4
    assert any(
        "leaf_spine" in transfer["path_scopes"]
        for transfer in report["transfers"]
    )
    assert report["summary"]["critical_links"]


def test_trace_replay_reports_overlap_pairs_memory_and_recovery() -> None:
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "name": "fused_attention",
                "cat": "compute",
                "pid": 0,
                "ts": 0,
                "dur": 100,
                "args": {
                    "fusion_id": "f0",
                    "shape": [2, 8],
                    "dtype": "bf16",
                    "allocated_bytes": 1_000,
                    "reserved_bytes": 1_200,
                },
            },
            {
                "ph": "X",
                "name": "softmax",
                "cat": "compute",
                "pid": 0,
                "ts": 10,
                "dur": 20,
                "args": {"fusion_id": "f0"},
            },
            {
                "ph": "X",
                "name": "ncclAllReduce",
                "cat": "communication",
                "pid": 0,
                "ts": 50,
                "dur": 100,
                "args": {
                    "bytes": 1_000,
                    "src_rank": 0,
                    "dst_rank": 2,
                    "links": ["leaf0->spine0"],
                },
            },
            {
                "ph": "X",
                "name": "worker_failure",
                "cat": "wait",
                "pid": 0,
                "ts": 160,
                "dur": 10,
                "args": {"generation": 0},
            },
            {
                "ph": "X",
                "name": "elastic_restart",
                "cat": "wait",
                "pid": 0,
                "ts": 180,
                "dur": 10,
                "args": {"generation": 1},
            },
            {
                "ph": "X",
                "name": "matmul",
                "cat": "compute",
                "pid": 2,
                "ts": 0,
                "dur": 120,
                "args": {},
            },
        ]
    }
    profiles = [
        {
            "id": "fused-attention",
            "match": {"operator": "fused_attention", "dtype": "bf16"},
            "model": {"workspace_bytes": 256, "flops_per_element": 4},
            "fused_operators": ["softmax"],
        }
    ]
    report = replay_trace(trace, operator_profiles=profiles)

    rank0 = next(item for item in report["rank_summary"] if item["rank"] == 0)
    assert rank0["compute_communication_overlap_us"] == 50
    assert rank0["communication_wait_us"] == 50
    assert report["rank_pairs"][0]["total_bytes"] == 1_000
    assert report["memory_timeline"]["peak_reserved_bytes"] == 1_200
    assert report["operator_profile_summary"]["matched_event_count"] == 1
    assert report["resilience"]["elastic_restart_observed"] is True
    assert len(report["events"]) == 5


def test_trace_replay_accepts_native_cluster_timeline_shape() -> None:
    report = replay_trace(
        {
            "schema_version": "cluster_report.v1",
            "operation_timeline": {
                "retained_entries": 1,
                "dropped_entries": 0,
                "entries": [
                    {
                        "index": 1,
                        "kind": "collective",
                        "operation": "all_reduce",
                        "data_type": "float32",
                        "ranks": [0, 1],
                        "logical_payload_bytes": 4_096,
                        "rendezvous_wait_us": 5,
                        "execution_time_us": 20,
                        "coordinator_duration_us": 25,
                        "modeled_time_us": 18,
                    }
                ],
            },
        }
    )
    assert report["summary"]["rank_count"] == 2
    assert report["summary"]["event_count"] == 4
    assert report["summary"]["communication_bytes"] == 8_192
    assert all(item["explicit_wait_us"] == 5 for item in report["rank_summary"])


def test_operator_profile_catalog_matches_exact_shape(tmp_path: Path) -> None:
    path = tmp_path / "operators.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "fakegpu.operator_profiles.v1",
                "profiles": [
                    {
                        "id": "custom-op",
                        "priority": 10,
                        "match": {
                            "operator_regex": "custom::.*",
                            "dtype": "fp16",
                            "shape": [2, 4],
                        },
                        "model": {
                            "workspace_bytes": 64,
                            "flops_per_element": 3,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    profiles = load_operator_profiles([path])
    profile = match_operator_profile(
        profiles,
        operator="custom::foo",
        dtype="fp16",
        shape=[2, 4],
    )
    assert profile is not None
    evaluated = evaluate_operator_profile(profile, numel=8)
    assert evaluated["flops"] == 24
    assert evaluated["workspace_bytes"] == 64


def test_ptx_static_analysis_and_profile_occupancy() -> None:
    ptx = """
.version 8.0
.target sm_89
.address_size 64
.visible .entry kernel() {
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .shared .align 4 .b8 scratch[1024];
    ld.global.f32 %f1, [%rd1];
    fma.rn.f32 %f2, %f1, %f1, %f1;
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {%r0}, {%r1}, {%r2}, {%r3};
    st.global.f32 [%rd2], %f2;
    ret;
}
"""
    analysis = analyze_ptx(ptx)
    occupancy = estimate_occupancy(
        analysis,
        profile_id="rtx4090",
        threads_per_block=128,
    )

    assert analysis["entry_points"] == ["kernel"]
    assert analysis["declared_register_count"] == 10
    assert analysis["static_shared_memory_bytes"] == 1_024
    assert analysis["instruction_classes"]["tensor_core"] == 1
    assert analysis["recognized_flops_per_static_issue"] >= 4_096
    assert 0 < occupancy["occupancy_upper_bound"] <= 1


def test_repository_analyzer_detects_aliases_decorators_and_build_files(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "project"
    repository.mkdir()
    (repository / "train.py").write_text(
        """
import triton as tr
from torch.utils.cpp_extension import load_inline as build_cuda

@tr.jit
def kernel(x):
    return x

PTX = ".version 8.0\\n.target sm_89\\n.entry generated() {}"
build_cuda(name="x", cpp_sources="", cuda_sources="__global__ void x() {}")

if __name__ == "__main__":
    kernel
""".strip(),
        encoding="utf-8",
    )
    (repository / "CMakeLists.txt").write_text(
        "project(sample LANGUAGES CXX CUDA)\nfind_package(CUDAToolkit)\n",
        encoding="utf-8",
    )
    (repository / "kernel.ptx").write_text(
        ".version 8.0\n.target sm_89\n.entry kernel() { ret; }\n",
        encoding="utf-8",
    )

    report = analyze_repository(repository)
    markers = report["python"]["call_markers"]
    codes = {item["code"] for item in report["findings"]}

    assert markers["triton.jit"] >= 1
    assert markers["torch.utils.cpp_extension.load_inline"] == 1
    assert markers["embedded_cuda_or_ptx_source"] >= 1
    assert report["build_system"]["markers"]["cmake_cuda_language"] == 1
    assert report["kernel_static_analysis"][0]["status"] == "analyzed"
    assert {
        "generated_kernel_path",
        "runtime_cuda_extension_build",
    } <= codes
