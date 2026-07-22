from __future__ import annotations

from copy import deepcopy

import pytest

from verification.check_cluster_report import (
    _validate_cross_field_consistency,
    _validate_resilience,
)


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


def _resilience() -> dict:
    return {
        "failure_count": 1,
        "recovery_count": 1,
        "failure_events": [
            {
                "index": 1,
                "comm_id": 1,
                "seqno": 1,
                "local_rank": 2,
                "global_rank": 2,
                "source": "injected",
                "operation": "all_reduce",
                "error_code": "injected_rank_failure",
                "error_detail": "injected rank failure",
                "observed_ranks": [0, 2],
                "attempted_payload_bytes": 8,
            }
        ],
        "recovery_events": [
            {
                "index": 2,
                "parent_comm_id": 1,
                "new_comm_id": 2,
                "seqno": 1,
                "abort_parent": True,
                "excluded_ranks": [2],
                "surviving_ranks": [0, 1, 3],
                "recovery_time_us": 12.5,
            }
        ],
    }


def test_resilience_accepts_failure_and_recovery() -> None:
    _validate_resilience(
        _resilience(),
        expect_failure=True,
        expect_recovery=True,
    )


def test_resilience_accepts_rank_inferred_from_collective_timeout() -> None:
    resilience = _resilience()
    failure = resilience["failure_events"][0]
    failure.update(
        {
            "source": "collective_timeout",
            "operation": "allreduce",
            "error_code": "timeout_waiting_for_collective",
            "error_detail": "rank 2 did not submit before timeout",
            "observed_ranks": [0, 1, 3],
            "attempted_payload_bytes": 12,
        }
    )
    _validate_resilience(
        resilience,
        expect_failure=True,
        expect_recovery=True,
    )


def test_resilience_rejects_timed_out_rank_as_an_observer() -> None:
    resilience = _resilience()
    failure = resilience["failure_events"][0]
    failure["source"] = "collective_timeout"
    failure["observed_ranks"] = [0, 1, 2, 3]
    with pytest.raises(SystemExit):
        _validate_resilience(resilience)


def test_resilience_rejects_count_mismatch() -> None:
    resilience = _resilience()
    resilience["failure_count"] = 2
    with pytest.raises(SystemExit):
        _validate_resilience(resilience)


def test_resilience_rejects_overlapping_rank_sets() -> None:
    resilience = _resilience()
    resilience["recovery_events"][0]["surviving_ranks"] = [0, 1, 2, 3]
    with pytest.raises(SystemExit):
        _validate_resilience(resilience)
