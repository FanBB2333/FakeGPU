from __future__ import annotations

from math import ceil
from typing import Any, Iterable, Mapping


def build_full_shard_plan(
    units: Iterable[Mapping[str, Any]],
    *,
    world_size: int,
) -> dict[str, Any]:
    """Describe the padded parameter shards produced by FSDP FULL_SHARD.

    Each unit represents one FSDP flat-parameter group. Parameters in a unit
    must share an element size, matching FSDP's flat-parameter requirement.
    """

    if world_size <= 1:
        raise ValueError("world_size must be greater than one")

    planned_units: list[dict[str, Any]] = []
    for index, raw_unit in enumerate(units):
        name = str(raw_unit.get("name", f"unit_{index}"))
        numel = int(raw_unit["numel"])
        element_size = int(raw_unit["element_size"])
        if numel < 0:
            raise ValueError(f"unit {name!r} has a negative numel")
        if element_size <= 0:
            raise ValueError(f"unit {name!r} has an invalid element size")
        if numel == 0:
            continue

        local_shard_numel = ceil(numel / world_size)
        padded_numel = local_shard_numel * world_size
        planned_units.append(
            {
                "name": name,
                "numel": numel,
                "element_size": element_size,
                "unsharded_bytes": numel * element_size,
                "local_shard_numel": local_shard_numel,
                "local_shard_bytes": local_shard_numel * element_size,
                "padding_numel": padded_numel - numel,
                "padded_unsharded_bytes": padded_numel * element_size,
            }
        )

    if not planned_units:
        raise ValueError("at least one non-empty FSDP unit is required")

    return {
        "world_size": world_size,
        "unit_count": len(planned_units),
        "units": planned_units,
        "unsharded_parameter_bytes": sum(
            int(unit["unsharded_bytes"]) for unit in planned_units
        ),
        "local_shard_parameter_bytes": sum(
            int(unit["local_shard_bytes"]) for unit in planned_units
        ),
        "padding_bytes": sum(
            int(unit["padding_numel"]) * int(unit["element_size"])
            for unit in planned_units
        ),
        "largest_unsharded_unit_bytes": max(
            int(unit["padded_unsharded_bytes"]) for unit in planned_units
        ),
        "largest_local_shard_bytes": max(
            int(unit["local_shard_bytes"]) for unit in planned_units
        ),
    }


def estimate_full_shard_sft_memory(
    static_report: Mapping[str, Any],
    sharding_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a single-GPU SFT graph onto FSDP FULL_SHARD storage.

    The input report is produced by ``qwen_sft_memory_worker.py --mode
    static``. The projection replaces full parameters, gradients, AdamW
    state, and eager optimizer temporaries with their local shards. It adds
    one padded full FSDP unit for the active all-gather. The corresponding
    experiment disables forward and backward prefetch, so only one unit is
    expected to be materialized at a time.
    """

    estimate = static_report.get("static_estimate")
    if not isinstance(estimate, Mapping):
        raise ValueError("static report does not contain static_estimate")

    world_size = int(sharding_plan.get("world_size", 0))
    if world_size <= 1:
        raise ValueError("sharding plan world_size must be greater than one")

    parameter_bytes = int(estimate["parameter_bytes"])
    trainable_parameter_bytes = int(estimate["trainable_parameter_bytes"])
    frozen_parameter_bytes = int(estimate.get("frozen_parameter_bytes", 0))
    if frozen_parameter_bytes != 0 or trainable_parameter_bytes != parameter_bytes:
        raise ValueError(
            "the current FULL_SHARD projection requires full-parameter training"
        )

    planned_parameter_bytes = int(sharding_plan["unsharded_parameter_bytes"])
    if planned_parameter_bytes != parameter_bytes:
        raise ValueError(
            "sharding plan parameter bytes do not match the static report: "
            f"{planned_parameter_bytes} != {parameter_bytes}"
        )

    local_parameter_bytes = int(sharding_plan["local_shard_parameter_bytes"])
    local_gradient_bytes = local_parameter_bytes
    all_gather_workspace_bytes = int(
        sharding_plan["largest_unsharded_unit_bytes"]
    )

    optimizer_state_bytes = int(estimate["optimizer_state_bytes"])
    local_optimizer_state_bytes = ceil(
        optimizer_state_bytes * local_parameter_bytes / parameter_bytes
    )

    optimizer_temporary = estimate.get("optimizer_temporary")
    if not isinstance(optimizer_temporary, Mapping):
        raise ValueError("static estimate does not contain optimizer_temporary")
    temporary_count = int(
        optimizer_temporary.get("current_parameter_temporary_count", 0)
    ) + int(optimizer_temporary.get("retained_previous_temporary_count", 0))
    if temporary_count <= 0:
        raise ValueError("optimizer temporary count must be positive")
    unsharded_optimizer_temporary_bytes = int(
        estimate["optimizer_temporary_bytes"]
    )
    local_optimizer_temporary_bytes = (
        temporary_count * int(sharding_plan["largest_local_shard_bytes"])
    )

    first_graph_peak = int(estimate["first_step_graph_phase_peak_bytes"])
    graph_nonsharded_bytes = (
        first_graph_peak - parameter_bytes - trainable_parameter_bytes
    )
    if graph_nonsharded_bytes < 0:
        raise ValueError("static graph peak is smaller than parameter storage")
    projected_graph_peak = (
        graph_nonsharded_bytes
        + local_parameter_bytes
        + local_gradient_bytes
        + all_gather_workspace_bytes
    )

    optimizer_peak = int(estimate["optimizer_phase_peak_bytes"])
    optimizer_nonsharded_bytes = (
        optimizer_peak
        - parameter_bytes
        - trainable_parameter_bytes
        - optimizer_state_bytes
        - unsharded_optimizer_temporary_bytes
    )
    if optimizer_nonsharded_bytes < 0:
        raise ValueError("static optimizer peak has inconsistent components")
    projected_optimizer_peak = (
        optimizer_nonsharded_bytes
        + local_parameter_bytes
        + local_gradient_bytes
        + local_optimizer_state_bytes
        + local_optimizer_temporary_bytes
    )
    projected_steady_graph_peak = (
        projected_graph_peak + local_optimizer_state_bytes
    )
    projected_first_step_peak = max(
        projected_graph_peak,
        projected_optimizer_peak,
    )

    return {
        "method": "static_aten_liveness_with_full_shard_projection",
        "world_size": world_size,
        "parameter_bytes": parameter_bytes,
        "local_parameter_bytes": local_parameter_bytes,
        "local_gradient_bytes": local_gradient_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
        "local_optimizer_state_bytes": local_optimizer_state_bytes,
        "unsharded_optimizer_temporary_bytes": (
            unsharded_optimizer_temporary_bytes
        ),
        "local_optimizer_temporary_bytes": local_optimizer_temporary_bytes,
        "graph_nonsharded_bytes": graph_nonsharded_bytes,
        "optimizer_nonsharded_bytes": optimizer_nonsharded_bytes,
        "all_gather_workspace_bytes": all_gather_workspace_bytes,
        "first_step_graph_peak_bytes": projected_graph_peak,
        "steady_state_graph_peak_bytes": projected_steady_graph_peak,
        "optimizer_peak_bytes": projected_optimizer_peak,
        "first_step_peak_bytes": projected_first_step_peak,
        "assumptions": [
            "FULL_SHARD is applied to each transformer layer and the root unit.",
            "Forward and backward parameter prefetch are disabled.",
            "Parameters, gradients, and AdamW state are evenly sharded after per-unit padding.",
            "Buffers, inputs, activations, loss tensors, and profiled operator workspaces remain replicated.",
            "One padded full FSDP unit is materialized during forward or backward compute.",
        ],
    }
