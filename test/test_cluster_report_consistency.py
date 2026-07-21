from __future__ import annotations

from copy import deepcopy

import pytest

from verification.check_cluster_report import _validate_cross_field_consistency


def _counter(calls: int = 0, bytes_value: int = 0) -> dict[str, int]:
    return {"calls": calls, "bytes": bytes_value}


def _direction(bytes_value: int) -> dict[str, int | float]:
    active = int(bytes_value > 0)
    return {
        "transfers": active,
        "total_bytes": bytes_value,
        "peak_bytes_per_operation": bytes_value,
        "model_bandwidth_gbps": 10.0,
        "avg_latency_us": 100.0,
        "estimated_time_us_total": 100.01 if active else 0.0,
        "contention_penalty_us_total": 0.0,
        "average_estimated_throughput_gbps": 0.001 if active else 0.0,
        "peak_estimated_throughput_gbps": 0.001 if active else 0.0,
    }


def _report() -> dict:
    return {
        "collectives": {
            "all_reduce": _counter(1, 16),
            "reduce": _counter(),
            "broadcast": _counter(),
            "all_gather": _counter(),
            "reduce_scatter": _counter(),
            "all_to_all": _counter(),
            "barrier": _counter(),
        },
        "point_to_point": {"operations": 0, "bytes": 0},
        "links": [
            {
                "src": "node-a",
                "dst": "node-b",
                "samples": 1,
                "bytes": 8,
                "peak_bytes_per_operation": 8,
                "bandwidth_gbps": 10.0,
                "avg_latency_us": 100.0,
                "estimated_time_us_total": 100.01,
                "contention_penalty_us_total": 0.0,
                "average_estimated_throughput_gbps": 0.001,
                "peak_estimated_throughput_gbps": 0.001,
            },
            {
                "src": "node-b",
                "dst": "node-a",
                "samples": 1,
                "bytes": 8,
                "peak_bytes_per_operation": 8,
                "bandwidth_gbps": 10.0,
                "avg_latency_us": 100.0,
                "estimated_time_us_total": 100.01,
                "contention_penalty_us_total": 0.0,
                "average_estimated_throughput_gbps": 0.001,
                "peak_estimated_throughput_gbps": 0.001,
            },
        ],
        "node_pairs": [
            {
                "node_a": "node-a",
                "node_b": "node-b",
                "a_to_b": _direction(8),
                "b_to_a": _direction(8),
            }
        ],
        "operation_timeline": {
            "retained_entries": 1,
            "dropped_entries": 0,
            "entries": [
                {
                    "kind": "collective",
                    "operation": "allreduce",
                    "logical_payload_bytes": 16,
                }
            ],
        },
    }


def test_cross_field_consistency_accepts_matching_aggregates() -> None:
    _validate_cross_field_consistency(_report())


def test_cross_field_consistency_rejects_timeline_byte_mismatch() -> None:
    report = _report()
    report["operation_timeline"]["entries"][0]["logical_payload_bytes"] = 8
    with pytest.raises(SystemExit):
        _validate_cross_field_consistency(report)


def test_cross_field_consistency_rejects_direction_link_mismatch() -> None:
    report = _report()
    report["node_pairs"][0]["a_to_b"]["total_bytes"] = 4
    with pytest.raises(SystemExit):
        _validate_cross_field_consistency(report)


def test_cross_field_consistency_supports_bounded_timeline() -> None:
    report = deepcopy(_report())
    report["collectives"]["all_reduce"] = _counter(3, 48)
    report["operation_timeline"]["dropped_entries"] = 2
    _validate_cross_field_consistency(report)
