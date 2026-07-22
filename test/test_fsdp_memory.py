from __future__ import annotations

import pytest

from fakegpu.fsdp_memory import (
    build_full_shard_plan,
    build_fully_shard_plan,
    estimate_full_shard_sft_memory,
    estimate_fully_shard_sft_memory,
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
    assert estimate["reduce_scatter_workspace_bytes"] == 250
    assert estimate["graph_nonsharded_bytes"] == 600
    assert estimate["first_step_graph_peak_bytes"] == 2_350
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


def test_fully_shard_plan_tracks_per_parameter_padding_and_trainability() -> None:
    plan = build_fully_shard_plan(
        [
            {
                "name": "root",
                "is_root": True,
                "parameters": [
                    {
                        "name": "embedding",
                        "parameter_index": 0,
                        "shape": [5, 2],
                        "element_size": 2,
                        "trainable": False,
                    }
                ],
            },
            {
                "name": "layer",
                "parameters": [
                    {
                        "name": "adapter",
                        "parameter_index": 1,
                        "shape": [3, 2],
                        "element_size": 4,
                        "trainable": True,
                    }
                ],
            },
        ],
        world_size=2,
    )

    assert plan["unsharded_parameter_bytes"] == 44
    assert plan["unsharded_trainable_parameter_bytes"] == 24
    assert plan["padding_bytes"] == 12
    assert plan["rank_shards"][0]["parameter_storage_bytes"] == 28
    assert plan["rank_shards"][1]["parameter_storage_bytes"] == 28
    assert plan["rank_shards"][0]["logical_trainable_parameter_bytes"] == 16
    assert plan["rank_shards"][1]["logical_trainable_parameter_bytes"] == 8
    assert plan["rank_shards"][0]["gradient_storage_bytes"] == 16
    assert plan["root_unsharded_parameter_bytes"] == 24
    assert plan["root_local_parameter_bytes"] == [12, 12]
    assert plan["largest_nested_local_parameter_bytes"] == [16, 16]
    assert plan["forward_collective_workspace_bytes"] == [128, 128]
    assert plan["backward_prefetch_parameter_bytes"] == 56
    assert plan["backward_collective_extra_bytes"] == [24, 24]
    assert plan["optimizer_runtime_workspace_bytes"] == [32, 32]
    assert plan["largest_trainable_unsharded_gradient_bytes"] == 32
    assert plan["largest_trainable_local_gradient_bytes"] == [16, 16]
    assert plan["rank_shards"][0]["adamw_optimizer_temporary_bytes"] == 32
    assert plan["rank_shards"][1]["adamw_optimizer_temporary_bytes"] == 16


def test_fully_shard_projection_uses_forward_and_gradient_events() -> None:
    plan = build_fully_shard_plan(
        [
            {
                "name": "root",
                "is_root": True,
                "parameters": [
                    {
                        "name": "root.weight",
                        "parameter_index": 0,
                        "shape": [4, 2],
                        "element_size": 2,
                        "trainable": False,
                    }
                ],
            },
            {
                "name": "layer",
                "parameters": [
                    {
                        "name": "layer.weight",
                        "parameter_index": 1,
                        "shape": [4, 2],
                        "element_size": 2,
                        "trainable": False,
                    },
                    {
                        "name": "layer.lora_a",
                        "parameter_index": 2,
                        "shape": [4, 1],
                        "element_size": 4,
                        "trainable": True,
                    },
                    {
                        "name": "layer.lora_b",
                        "parameter_index": 3,
                        "shape": [2, 2],
                        "element_size": 4,
                        "trainable": True,
                    },
                ],
            },
        ],
        world_size=2,
    )
    report = {
        "static_estimate": {
            "parameter_bytes": 64,
            "trainable_parameter_bytes": 32,
            "frozen_parameter_bytes": 32,
            "first_step_graph_phase_peak_bytes": 164,
            "post_graph_live_bytes": 116,
            "optimizer_phase_peak_bytes": 228,
            "optimizer_state_bytes": 64,
            "optimizer_temporary_bytes": 48,
            "workspace_peak_contribution_bytes": 0,
            "optimizer_temporary": {
                "optimizer": "adamw",
                "current_parameter_temporary_count": 2,
                "retained_previous_temporary_count": 1,
            },
            "graph": {
                "peak_bytes_by_category": {
                    "frozen_parameter": 32,
                    "trainable_parameter": 32,
                    "temporary": 100,
                },
                "final_bytes_by_category": {
                    "frozen_parameter": 32,
                    "trainable_parameter": 32,
                    "gradient": 32,
                    "output": 20,
                },
                "gradient_production_phase": {
                    "peak_live_bytes": 146,
                    "peak_bytes_by_category": {
                        "frozen_parameter": 32,
                        "trainable_parameter": 32,
                        "gradient": 32,
                        "temporary": 50,
                    },
                },
            },
        }
    }

    estimate = estimate_fully_shard_sft_memory(report, plan, rank=0)

    assert estimate["local_parameter_bytes"] == 32
    assert estimate["local_gradient_bytes"] == 16
    assert estimate["root_parameter_workspace_bytes"] == 16
    assert estimate["forward_collective_workspace_bytes"] == 152
    assert estimate["backward_parameter_workspace_bytes"] == 64
    assert estimate["backward_collective_extra_bytes"] == 16
    assert estimate["optimizer_runtime_workspace_bytes"] == 48
    assert estimate["reduce_scatter_workspace_bytes"] == 16
    assert estimate["captured_graph_peak_bytes"] == 148
    assert estimate["forward_collective_floor_bytes"] == 184
    assert estimate["forward_graph_peak_bytes"] == 184
    assert estimate["gradient_phase_peak_bytes"] == 194
    assert estimate["backward_activation_floor_bytes"] == 192
    assert estimate["backward_graph_peak_bytes"] == 194
    assert estimate["first_step_graph_peak_bytes"] == 194
    assert estimate["local_optimizer_state_bytes"] == 32
    assert estimate["local_optimizer_temporary_bytes"] == 24
    assert estimate["optimizer_peak_bytes"] == 172
    assert estimate["first_step_peak_bytes"] == 194
