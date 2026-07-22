from __future__ import annotations

import pytest

from fakegpu.fsdp_memory import (
    build_full_shard_plan,
    estimate_full_shard_sft_memory,
)


def _static_report() -> dict:
    return {
        "static_estimate": {
            "parameter_bytes": 1_000,
            "trainable_parameter_bytes": 1_000,
            "frozen_parameter_bytes": 0,
            "first_step_graph_phase_peak_bytes": 2_600,
            "optimizer_phase_peak_bytes": 4_500,
            "optimizer_state_bytes": 2_000,
            "optimizer_temporary_bytes": 500,
            "optimizer_temporary": {
                "current_parameter_temporary_count": 2,
                "retained_previous_temporary_count": 0,
            },
        }
    }


def test_full_shard_plan_accounts_for_per_unit_padding() -> None:
    plan = build_full_shard_plan(
        [
            {"name": "root", "numel": 5, "element_size": 2},
            {"name": "layer", "numel": 8, "element_size": 2},
        ],
        world_size=2,
    )

    assert plan["unsharded_parameter_bytes"] == 26
    assert plan["local_shard_parameter_bytes"] == 14
    assert plan["padding_bytes"] == 2
    assert plan["largest_unsharded_unit_bytes"] == 16
    assert plan["largest_local_shard_bytes"] == 8


def test_full_shard_projection_replaces_shardable_memory() -> None:
    plan = build_full_shard_plan(
        [
            {"name": "root", "numel": 250, "element_size": 2},
            {"name": "layer", "numel": 250, "element_size": 2},
        ],
        world_size=2,
    )

    estimate = estimate_full_shard_sft_memory(_static_report(), plan)

    assert estimate["local_parameter_bytes"] == 500
    assert estimate["local_optimizer_state_bytes"] == 1_000
    assert estimate["all_gather_workspace_bytes"] == 500
    assert estimate["graph_nonsharded_bytes"] == 600
    assert estimate["first_step_graph_peak_bytes"] == 2_100
    assert estimate["local_optimizer_temporary_bytes"] == 500
    assert estimate["optimizer_nonsharded_bytes"] == 0
    assert estimate["optimizer_peak_bytes"] == 2_500
    assert estimate["first_step_peak_bytes"] == 2_500


def test_full_shard_projection_rejects_partially_frozen_training() -> None:
    report = _static_report()
    report["static_estimate"]["trainable_parameter_bytes"] = 100
    report["static_estimate"]["frozen_parameter_bytes"] = 900
    plan = build_full_shard_plan(
        [{"name": "root", "numel": 500, "element_size": 2}],
        world_size=2,
    )

    with pytest.raises(ValueError, match="full-parameter training"):
        estimate_full_shard_sft_memory(report, plan)
