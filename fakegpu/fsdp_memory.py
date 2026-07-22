from __future__ import annotations

from math import ceil, prod
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


def build_fully_shard_plan(
    units: Iterable[Mapping[str, Any]],
    *,
    world_size: int,
) -> dict[str, Any]:
    """Describe FSDP2's per-parameter dim-0 sharding and padding.

    Unlike FSDP1 flat parameters, FSDP2 preserves each original parameter as
    a DTensor. Each parameter is chunked independently along dimension zero,
    and every rank keeps storage sized like rank zero's (largest) chunk. This
    plan also separates frozen parameters from trainable parameters so mixed
    BF16/FP32 LoRA models can be projected without treating the base model as
    gradient or optimizer storage.
    """

    if world_size <= 1:
        raise ValueError("world_size must be greater than one")

    planned_units: list[dict[str, Any]] = []
    parameter_names: set[str] = set()
    parameter_indices: set[int] = set()
    fallback_parameter_index = 0
    for unit_index, raw_unit in enumerate(units):
        unit_name = str(raw_unit.get("name", f"unit_{unit_index}"))
        raw_parameters = raw_unit.get("parameters")
        if not isinstance(raw_parameters, Iterable) or isinstance(
            raw_parameters, (str, bytes, Mapping)
        ):
            raise ValueError(f"unit {unit_name!r} must contain parameters")

        planned_parameters: list[dict[str, Any]] = []
        for raw_parameter in raw_parameters:
            if not isinstance(raw_parameter, Mapping):
                raise ValueError(
                    f"unit {unit_name!r} contains an invalid parameter"
                )
            parameter_name = str(
                raw_parameter.get(
                    "name",
                    f"{unit_name}.parameter_{len(planned_parameters)}",
                )
            )
            if parameter_name in parameter_names:
                raise ValueError(f"duplicate FSDP2 parameter {parameter_name!r}")
            parameter_names.add(parameter_name)

            shape_value = raw_parameter.get("shape")
            if not isinstance(shape_value, Iterable) or isinstance(
                shape_value, (str, bytes, Mapping)
            ):
                raise ValueError(f"parameter {parameter_name!r} has no shape")
            shape = tuple(int(dimension) for dimension in shape_value)
            if not shape:
                raise ValueError(
                    f"parameter {parameter_name!r} is scalar; FSDP2 requires "
                    "a dimension-zero shard"
                )
            if any(dimension <= 0 for dimension in shape):
                raise ValueError(
                    f"parameter {parameter_name!r} has a non-positive shape"
                )
            numel = prod(shape)
            reported_numel = int(raw_parameter.get("numel", numel))
            if reported_numel != numel:
                raise ValueError(
                    f"parameter {parameter_name!r} numel does not match shape: "
                    f"{reported_numel} != {numel}"
                )

            element_size = int(raw_parameter["element_size"])
            gradient_element_size = int(
                raw_parameter.get("gradient_element_size", element_size)
            )
            if element_size <= 0 or gradient_element_size <= 0:
                raise ValueError(
                    f"parameter {parameter_name!r} has an invalid element size"
                )
            parameter_index = int(
                raw_parameter.get("parameter_index", fallback_parameter_index)
            )
            fallback_parameter_index += 1
            if parameter_index in parameter_indices:
                raise ValueError(
                    f"duplicate FSDP2 parameter index {parameter_index}"
                )
            parameter_indices.add(parameter_index)

            trainable = bool(raw_parameter.get("trainable", True))
            rows_per_rank = ceil(shape[0] / world_size)
            trailing_numel = prod(shape[1:])
            padded_numel = rows_per_rank * trailing_numel * world_size
            rank_shards = []
            for rank in range(world_size):
                local_rows = max(
                    0,
                    min(rows_per_rank, shape[0] - rank * rows_per_rank),
                )
                local_numel = local_rows * trailing_numel
                rank_shards.append(
                    {
                        "rank": rank,
                        "shape": [local_rows, *shape[1:]],
                        "logical_parameter_bytes": local_numel * element_size,
                        "parameter_storage_bytes": (
                            rows_per_rank * trailing_numel * element_size
                        ),
                        "logical_gradient_bytes": (
                            local_numel * gradient_element_size
                            if trainable
                            else 0
                        ),
                        "gradient_storage_bytes": (
                            rows_per_rank
                            * trailing_numel
                            * gradient_element_size
                            if trainable
                            else 0
                        ),
                    }
                )

            planned_parameters.append(
                {
                    "name": parameter_name,
                    "parameter_index": parameter_index,
                    "shape": list(shape),
                    "numel": numel,
                    "dtype": str(raw_parameter.get("dtype", "unknown")),
                    "element_size": element_size,
                    "gradient_element_size": gradient_element_size,
                    "trainable": trainable,
                    "unsharded_bytes": numel * element_size,
                    "padded_unsharded_bytes": padded_numel * element_size,
                    "padding_bytes": (padded_numel - numel) * element_size,
                    "padded_unsharded_gradient_bytes": (
                        padded_numel * gradient_element_size
                        if trainable
                        else 0
                    ),
                    "rank_shards": rank_shards,
                }
            )

        if not planned_parameters:
            continue
        rank_shards = []
        for rank in range(world_size):
            parameter_rank_shards = [
                parameter["rank_shards"][rank]
                for parameter in planned_parameters
            ]
            rank_shards.append(
                {
                    "rank": rank,
                    "logical_parameter_bytes": sum(
                        int(shard["logical_parameter_bytes"])
                        for shard in parameter_rank_shards
                    ),
                    "parameter_storage_bytes": sum(
                        int(shard["parameter_storage_bytes"])
                        for shard in parameter_rank_shards
                    ),
                    "logical_trainable_parameter_bytes": sum(
                        int(shard["logical_parameter_bytes"])
                        for parameter, shard in zip(
                            planned_parameters, parameter_rank_shards
                        )
                        if bool(parameter["trainable"])
                    ),
                    "trainable_parameter_storage_bytes": sum(
                        int(shard["parameter_storage_bytes"])
                        for parameter, shard in zip(
                            planned_parameters, parameter_rank_shards
                        )
                        if bool(parameter["trainable"])
                    ),
                    "logical_gradient_bytes": sum(
                        int(shard["logical_gradient_bytes"])
                        for shard in parameter_rank_shards
                    ),
                    "gradient_storage_bytes": sum(
                        int(shard["gradient_storage_bytes"])
                        for shard in parameter_rank_shards
                    ),
                }
            )

        planned_units.append(
            {
                "name": unit_name,
                "is_root": bool(raw_unit.get("is_root", False)),
                "parameter_count": len(planned_parameters),
                "parameters": planned_parameters,
                "unsharded_parameter_bytes": sum(
                    int(parameter["unsharded_bytes"])
                    for parameter in planned_parameters
                ),
                "padded_unsharded_parameter_bytes": sum(
                    int(parameter["padded_unsharded_bytes"])
                    for parameter in planned_parameters
                ),
                "padding_bytes": sum(
                    int(parameter["padding_bytes"])
                    for parameter in planned_parameters
                ),
                "unsharded_trainable_parameter_bytes": sum(
                    int(parameter["unsharded_bytes"])
                    for parameter in planned_parameters
                    if bool(parameter["trainable"])
                ),
                "padded_unsharded_trainable_gradient_bytes": sum(
                    int(parameter["padded_unsharded_gradient_bytes"])
                    for parameter in planned_parameters
                ),
                "rank_shards": rank_shards,
            }
        )

    if not planned_units:
        raise ValueError("at least one non-empty FSDP2 unit is required")
    root_units = [unit for unit in planned_units if bool(unit["is_root"])]
    if len(root_units) > 1:
        raise ValueError("at most one FSDP2 unit may be marked as root")
    root_unit = root_units[0] if root_units else max(
        planned_units,
        key=lambda unit: int(unit["padded_unsharded_parameter_bytes"]),
    )
    nested_units = [unit for unit in planned_units if unit is not root_unit]
    largest_nested_unit = max(
        nested_units or [root_unit],
        key=lambda unit: int(unit["padded_unsharded_parameter_bytes"]),
    )
    largest_trainable_unit = max(
        planned_units,
        key=lambda unit: int(
            unit["padded_unsharded_trainable_gradient_bytes"]
        ),
    )

    flat_parameters = sorted(
        (
            parameter
            for unit in planned_units
            for parameter in unit["parameters"]
        ),
        key=lambda parameter: int(parameter["parameter_index"]),
    )
    aggregate_rank_shards = []
    for rank in range(world_size):
        unit_rank_shards = [unit["rank_shards"][rank] for unit in planned_units]
        trainable_parameters = [
            parameter for parameter in flat_parameters if bool(parameter["trainable"])
        ]
        temporary_bytes = 0
        previous_bytes = 0
        for parameter in trainable_parameters:
            current_bytes = int(
                parameter["rank_shards"][rank]["logical_parameter_bytes"]
            )
            temporary_bytes = max(
                temporary_bytes,
                2 * current_bytes + previous_bytes,
            )
            previous_bytes = current_bytes
        aggregate_rank_shards.append(
            {
                "rank": rank,
                "logical_parameter_bytes": sum(
                    int(shard["logical_parameter_bytes"])
                    for shard in unit_rank_shards
                ),
                "parameter_storage_bytes": sum(
                    int(shard["parameter_storage_bytes"])
                    for shard in unit_rank_shards
                ),
                "logical_trainable_parameter_bytes": sum(
                    int(shard["logical_trainable_parameter_bytes"])
                    for shard in unit_rank_shards
                ),
                "trainable_parameter_storage_bytes": sum(
                    int(shard["trainable_parameter_storage_bytes"])
                    for shard in unit_rank_shards
                ),
                "logical_gradient_bytes": sum(
                    int(shard["logical_gradient_bytes"])
                    for shard in unit_rank_shards
                ),
                "gradient_storage_bytes": sum(
                    int(shard["gradient_storage_bytes"])
                    for shard in unit_rank_shards
                ),
                "adamw_optimizer_temporary_bytes": temporary_bytes,
            }
        )

    root_workspace_bytes = int(root_unit["padded_unsharded_parameter_bytes"])
    root_local_workspace_bytes = [
        int(shard["parameter_storage_bytes"])
        for shard in root_unit["rank_shards"]
    ]
    largest_nested_workspace_bytes = int(
        largest_nested_unit["padded_unsharded_parameter_bytes"]
    )
    largest_nested_local_workspace_bytes = [
        int(shard["parameter_storage_bytes"])
        for shard in largest_nested_unit["rank_shards"]
    ]
    activation_nested_workspace_bytes = (
        largest_nested_workspace_bytes if nested_units else 0
    )
    activation_nested_local_workspace_bytes = (
        largest_nested_local_workspace_bytes
        if nested_units
        else [0 for _ in range(world_size)]
    )
    return {
        "method": "fsdp2_per_parameter_dim0_sharding",
        "world_size": world_size,
        "unit_count": len(planned_units),
        "parameter_count": len(flat_parameters),
        "units": planned_units,
        "rank_shards": aggregate_rank_shards,
        "unsharded_parameter_bytes": sum(
            int(unit["unsharded_parameter_bytes"])
            for unit in planned_units
        ),
        "unsharded_trainable_parameter_bytes": sum(
            int(unit["unsharded_trainable_parameter_bytes"])
            for unit in planned_units
        ),
        "padding_bytes": sum(
            int(unit["padding_bytes"]) for unit in planned_units
        ),
        "root_unit_name": str(root_unit["name"]),
        "root_unsharded_parameter_bytes": root_workspace_bytes,
        "root_local_parameter_bytes": root_local_workspace_bytes,
        "largest_nested_unit_name": str(largest_nested_unit["name"]),
        "largest_nested_unsharded_parameter_bytes": (
            largest_nested_workspace_bytes
        ),
        "largest_nested_local_parameter_bytes": (
            largest_nested_local_workspace_bytes
        ),
        "forward_collective_workspace_bytes": [
            2 * root_workspace_bytes
            + 2 * largest_nested_workspace_bytes
            + local_nested_bytes
            for local_nested_bytes in largest_nested_local_workspace_bytes
        ],
        "forward_activation_workspace_bytes": [
            root_workspace_bytes
            + activation_nested_workspace_bytes
            + local_nested_bytes
            for local_nested_bytes in activation_nested_local_workspace_bytes
        ],
        "backward_prefetch_parameter_bytes": (
            root_workspace_bytes + largest_nested_workspace_bytes
        ),
        "backward_collective_extra_bytes": [
            root_workspace_bytes for _ in range(world_size)
        ],
        "backward_activation_workspace_bytes": [
            # The active, prefetched, and retained nested generations may each
            # keep packed output and parameter copy-out storage. The active
            # generation also retains its rank-local collective input.
            root_workspace_bytes
            + 6 * activation_nested_workspace_bytes
            + local_nested_bytes
            for local_nested_bytes in activation_nested_local_workspace_bytes
        ],
        "optimizer_runtime_workspace_bytes": [
            2 * local_nested_bytes
            for local_nested_bytes in largest_nested_local_workspace_bytes
        ],
        "largest_trainable_unit_name": str(largest_trainable_unit["name"]),
        "largest_trainable_unsharded_gradient_bytes": int(
            largest_trainable_unit[
                "padded_unsharded_trainable_gradient_bytes"
            ]
        ),
        "largest_trainable_local_gradient_bytes": [
            int(shard["gradient_storage_bytes"])
            for shard in largest_trainable_unit["rank_shards"]
        ],
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
    reduce_scatter_workspace_bytes = (
        int(sharding_plan["largest_unsharded_unit_bytes"])
        - int(sharding_plan["largest_local_shard_bytes"])
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
        + reduce_scatter_workspace_bytes
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
        "reduce_scatter_workspace_bytes": reduce_scatter_workspace_bytes,
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
            "Backward retains the active unit's full gradient until reduce-scatter produces its local shard.",
        ],
    }


def estimate_fully_shard_sft_memory(
    static_report: Mapping[str, Any],
    sharding_plan: Mapping[str, Any],
    *,
    rank: int,
) -> dict[str, Any]:
    """Project a mixed frozen/trainable SFT graph onto FSDP2 storage.

    The projection uses the captured graph peak for the forward/loss event and
    the gradient-production sub-phase for backward. It substitutes exact
    per-parameter local storage, then adds the root parameter materialization,
    default backward-prefetch materialization, and the largest trainable
    reduce-scatter buffer.
    """

    estimate = static_report.get("static_estimate")
    if not isinstance(estimate, Mapping):
        raise ValueError("static report does not contain static_estimate")
    if sharding_plan.get("method") != "fsdp2_per_parameter_dim0_sharding":
        raise ValueError("a per-parameter FSDP2 sharding plan is required")

    world_size = int(sharding_plan.get("world_size", 0))
    if world_size <= 1:
        raise ValueError("sharding plan world_size must be greater than one")
    rank_shards = sharding_plan.get("rank_shards")
    if not isinstance(rank_shards, list) or not 0 <= rank < len(rank_shards):
        raise ValueError(f"rank {rank} is outside the FSDP2 sharding plan")
    rank_shard = rank_shards[rank]
    if not isinstance(rank_shard, Mapping):
        raise ValueError(f"rank {rank} has an invalid FSDP2 shard")

    parameter_bytes = int(estimate["parameter_bytes"])
    trainable_parameter_bytes = int(estimate["trainable_parameter_bytes"])
    planned_parameter_bytes = int(sharding_plan["unsharded_parameter_bytes"])
    planned_trainable_bytes = int(
        sharding_plan["unsharded_trainable_parameter_bytes"]
    )
    if planned_parameter_bytes != parameter_bytes:
        raise ValueError(
            "sharding plan parameter bytes do not match the static report: "
            f"{planned_parameter_bytes} != {parameter_bytes}"
        )
    if planned_trainable_bytes != trainable_parameter_bytes:
        raise ValueError(
            "sharding plan trainable bytes do not match the static report: "
            f"{planned_trainable_bytes} != {trainable_parameter_bytes}"
        )
    if trainable_parameter_bytes <= 0:
        raise ValueError("FSDP2 SFT projection requires trainable parameters")

    local_parameter_bytes = int(rank_shard["parameter_storage_bytes"])
    local_trainable_parameter_bytes = int(
        rank_shard["logical_trainable_parameter_bytes"]
    )
    local_gradient_bytes = int(rank_shard["gradient_storage_bytes"])
    root_workspace_bytes = int(
        sharding_plan["root_unsharded_parameter_bytes"]
    )
    backward_parameter_workspace_bytes = int(
        sharding_plan["backward_prefetch_parameter_bytes"]
    )
    forward_collective_workspace_bytes = int(
        sharding_plan["forward_collective_workspace_bytes"][rank]
    )
    forward_activation_workspace_bytes = int(
        sharding_plan["forward_activation_workspace_bytes"][rank]
    )
    backward_collective_extra_bytes = int(
        sharding_plan["backward_collective_extra_bytes"][rank]
    )
    backward_activation_workspace_bytes = int(
        sharding_plan["backward_activation_workspace_bytes"][rank]
    )
    root_local_parameter_bytes = int(
        sharding_plan["root_local_parameter_bytes"][rank]
    )
    optimizer_runtime_workspace_bytes = int(
        sharding_plan["optimizer_runtime_workspace_bytes"][rank]
    )
    largest_unsharded_gradient_bytes = int(
        sharding_plan["largest_trainable_unsharded_gradient_bytes"]
    )
    largest_local_gradient_bytes = int(
        sharding_plan["largest_trainable_local_gradient_bytes"][rank]
    )
    reduce_scatter_workspace_bytes = max(
        0,
        largest_unsharded_gradient_bytes - largest_local_gradient_bytes,
    )

    graph = estimate.get("graph")
    if not isinstance(graph, Mapping):
        raise ValueError("static estimate does not contain graph details")
    peak_categories = graph.get("peak_bytes_by_category")
    final_categories = graph.get("final_bytes_by_category")
    if not isinstance(peak_categories, Mapping) or not isinstance(
        final_categories, Mapping
    ):
        raise ValueError("static graph does not contain category liveness")

    parameter_categories = (
        "parameter",
        "trainable_parameter",
        "frozen_parameter",
    )

    def category_bytes(categories: Mapping[str, Any], names: Iterable[str]) -> int:
        return sum(int(categories.get(name, 0)) for name in names)

    def projected_gradient_bytes(unsharded_bytes: int) -> int:
        if unsharded_bytes <= 0:
            return 0
        return ceil(
            unsharded_bytes
            * local_gradient_bytes
            / trainable_parameter_bytes
        )

    first_graph_peak = int(estimate["first_step_graph_phase_peak_bytes"])
    peak_parameter_bytes = category_bytes(peak_categories, parameter_categories)
    peak_gradient_bytes = int(peak_categories.get("gradient", 0))
    peak_nonsharded_bytes = (
        first_graph_peak - peak_parameter_bytes - peak_gradient_bytes
    )
    if peak_nonsharded_bytes < 0:
        raise ValueError("static graph peak has inconsistent categories")
    peak_is_backward = peak_gradient_bytes > 0
    projected_captured_peak = (
        peak_nonsharded_bytes
        + local_parameter_bytes
        + projected_gradient_bytes(peak_gradient_bytes)
        + (
            backward_parameter_workspace_bytes
            if peak_is_backward
            else root_workspace_bytes
        )
        + (reduce_scatter_workspace_bytes if peak_is_backward else 0)
    )
    projected_forward_collective_floor = (
        local_parameter_bytes
        + forward_collective_workspace_bytes
        + int(estimate.get("buffer_bytes", 0))
        + int(estimate.get("input_bytes", 0))
        + int(estimate.get("workspace_peak_contribution_bytes", 0))
    )
    forward_phase = graph.get("forward_phase")
    if isinstance(forward_phase, Mapping):
        forward_phase_categories = forward_phase.get("peak_bytes_by_category")
        if not isinstance(forward_phase_categories, Mapping):
            raise ValueError("forward phase has no categories")
        forward_phase_peak = int(forward_phase["peak_live_bytes"]) + int(
            estimate.get("workspace_peak_contribution_bytes", 0)
        )
        forward_phase_parameter_bytes = category_bytes(
            forward_phase_categories,
            parameter_categories,
        )
        forward_phase_gradient_bytes = int(
            forward_phase_categories.get("gradient", 0)
        )
        forward_phase_nonsharded_bytes = (
            forward_phase_peak
            - forward_phase_parameter_bytes
            - forward_phase_gradient_bytes
        )
        if forward_phase_nonsharded_bytes < 0:
            raise ValueError("static forward peak has inconsistent categories")
        projected_forward_captured_peak = (
            forward_phase_nonsharded_bytes
            + local_parameter_bytes
            + projected_gradient_bytes(forward_phase_gradient_bytes)
            + forward_activation_workspace_bytes
        )
        forward_phase_method = str(
            forward_phase.get("method")
            or "captured_forward_liveness"
        )
    else:
        forward_phase_nonsharded_bytes = peak_nonsharded_bytes
        projected_forward_captured_peak = projected_captured_peak
        forward_phase_method = "full_graph_peak_fallback"
    projected_forward_peak = max(
        projected_forward_captured_peak,
        projected_forward_collective_floor,
    )

    gradient_phase = graph.get("gradient_production_phase")
    if isinstance(gradient_phase, Mapping):
        gradient_phase_categories = gradient_phase.get(
            "peak_bytes_by_category"
        )
        if not isinstance(gradient_phase_categories, Mapping):
            raise ValueError("gradient-production phase has no categories")
        gradient_phase_peak = int(gradient_phase["peak_live_bytes"]) + int(
            estimate.get("workspace_peak_contribution_bytes", 0)
        )
        gradient_phase_method = "captured_gradient_production_liveness"
    else:
        gradient_phase_categories = final_categories
        gradient_phase_peak = int(estimate["post_graph_live_bytes"]) + int(
            estimate.get("workspace_peak_contribution_bytes", 0)
        )
        gradient_phase_method = "post_graph_liveness_fallback"
    gradient_phase_parameter_bytes = category_bytes(
        gradient_phase_categories,
        parameter_categories,
    )
    gradient_phase_gradient_bytes = int(
        gradient_phase_categories.get("gradient", 0)
    )
    gradient_phase_nonsharded_bytes = (
        gradient_phase_peak
        - gradient_phase_parameter_bytes
        - gradient_phase_gradient_bytes
    )
    if gradient_phase_nonsharded_bytes < 0:
        raise ValueError(
            "static gradient-production peak has inconsistent categories"
        )
    projected_gradient_phase_peak = (
        gradient_phase_nonsharded_bytes
        + local_parameter_bytes
        + projected_gradient_bytes(gradient_phase_gradient_bytes)
        + backward_parameter_workspace_bytes
        + backward_collective_extra_bytes
        + reduce_scatter_workspace_bytes
    )
    projected_backward_retained_forward_floor = (
        projected_forward_peak + root_local_parameter_bytes
    )
    projected_backward_captured_peak = 0
    if isinstance(forward_phase, Mapping):
        backward_start = forward_phase.get("backward_start_node")
        graph_peak_node = graph.get("peak_node")
        if isinstance(backward_start, Mapping) and isinstance(
            graph_peak_node,
            Mapping,
        ):
            if int(graph_peak_node.get("index", -1)) >= int(
                backward_start.get("index", 0)
            ):
                projected_backward_captured_peak = (
                    peak_nonsharded_bytes
                    + local_parameter_bytes
                    + projected_gradient_bytes(peak_gradient_bytes)
                    + backward_activation_workspace_bytes
                )
    projected_backward_activation_floor = max(
        projected_backward_retained_forward_floor,
        projected_backward_captured_peak,
    )
    projected_backward_peak = max(
        projected_gradient_phase_peak,
        projected_backward_activation_floor,
    )
    projected_graph_peak = max(
        projected_forward_peak,
        projected_backward_peak,
    )

    optimizer_state_bytes = int(estimate["optimizer_state_bytes"])
    local_optimizer_state_bytes = ceil(
        optimizer_state_bytes
        * local_trainable_parameter_bytes
        / trainable_parameter_bytes
    )
    optimizer_temporary = estimate.get("optimizer_temporary")
    if not isinstance(optimizer_temporary, Mapping):
        raise ValueError("static estimate does not contain optimizer_temporary")
    if str(optimizer_temporary.get("optimizer")) not in {"adam", "adamw"}:
        raise ValueError("FSDP2 projection currently requires Adam or AdamW")
    unsharded_optimizer_temporary_bytes = int(
        estimate["optimizer_temporary_bytes"]
    )
    local_optimizer_temporary_bytes = int(
        rank_shard["adamw_optimizer_temporary_bytes"]
    )

    final_parameter_bytes = category_bytes(final_categories, parameter_categories)
    final_gradient_bytes = int(final_categories.get("gradient", 0))
    optimizer_peak = int(estimate["optimizer_phase_peak_bytes"])
    optimizer_nonsharded_bytes = (
        optimizer_peak
        - final_parameter_bytes
        - final_gradient_bytes
        - optimizer_state_bytes
        - unsharded_optimizer_temporary_bytes
    )
    if optimizer_nonsharded_bytes < 0:
        raise ValueError("static optimizer peak has inconsistent components")
    projected_optimizer_peak = (
        optimizer_nonsharded_bytes
        + local_parameter_bytes
        + projected_gradient_bytes(final_gradient_bytes)
        + local_optimizer_state_bytes
        + local_optimizer_temporary_bytes
        + optimizer_runtime_workspace_bytes
    )
    projected_steady_graph_peak = (
        projected_graph_peak + local_optimizer_state_bytes
    )
    projected_first_step_peak = max(
        projected_graph_peak,
        projected_optimizer_peak,
    )

    return {
        "method": "static_aten_liveness_with_fsdp2_projection",
        "forward_phase_method": forward_phase_method,
        "gradient_phase_method": gradient_phase_method,
        "world_size": world_size,
        "rank": rank,
        "parameter_bytes": parameter_bytes,
        "trainable_parameter_bytes": trainable_parameter_bytes,
        "local_parameter_bytes": local_parameter_bytes,
        "local_trainable_parameter_bytes": local_trainable_parameter_bytes,
        "local_gradient_bytes": local_gradient_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
        "local_optimizer_state_bytes": local_optimizer_state_bytes,
        "unsharded_optimizer_temporary_bytes": (
            unsharded_optimizer_temporary_bytes
        ),
        "local_optimizer_temporary_bytes": local_optimizer_temporary_bytes,
        "peak_nonsharded_bytes": peak_nonsharded_bytes,
        "forward_phase_nonsharded_bytes": forward_phase_nonsharded_bytes,
        "gradient_phase_nonsharded_bytes": gradient_phase_nonsharded_bytes,
        "optimizer_nonsharded_bytes": optimizer_nonsharded_bytes,
        "root_parameter_workspace_bytes": root_workspace_bytes,
        "forward_collective_workspace_bytes": (
            forward_collective_workspace_bytes
        ),
        "forward_activation_workspace_bytes": (
            forward_activation_workspace_bytes
        ),
        "backward_parameter_workspace_bytes": (
            backward_parameter_workspace_bytes
        ),
        "backward_collective_extra_bytes": backward_collective_extra_bytes,
        "backward_activation_workspace_bytes": (
            backward_activation_workspace_bytes
        ),
        "optimizer_runtime_workspace_bytes": optimizer_runtime_workspace_bytes,
        "reduce_scatter_workspace_bytes": reduce_scatter_workspace_bytes,
        "captured_graph_peak_bytes": projected_captured_peak,
        "forward_captured_peak_bytes": projected_forward_captured_peak,
        "forward_collective_floor_bytes": projected_forward_collective_floor,
        "forward_graph_peak_bytes": projected_forward_peak,
        "gradient_phase_peak_bytes": projected_gradient_phase_peak,
        "backward_captured_peak_bytes": projected_backward_captured_peak,
        "backward_retained_forward_floor_bytes": (
            projected_backward_retained_forward_floor
        ),
        "backward_activation_floor_bytes": projected_backward_activation_floor,
        "backward_graph_peak_bytes": projected_backward_peak,
        "first_step_graph_peak_bytes": projected_graph_peak,
        "steady_state_graph_peak_bytes": projected_steady_graph_peak,
        "optimizer_peak_bytes": projected_optimizer_peak,
        "first_step_peak_bytes": projected_first_step_peak,
        "assumptions": [
            "FSDP2 shards every original parameter independently along dimension zero.",
            "Frozen and trainable parameter dtypes retain their original element sizes.",
            "The root unit stays unsharded through the forward/loss peak.",
            "FSDP2 all-gather copy-in, packed output, and per-parameter copy-out buffers may overlap at the communication-dominated floor.",
            "The captured forward/loss phase ends at the first explicit backward ATen operator when that boundary is available.",
            "Default backward prefetch may materialize the root and largest nested unit together.",
            "The backward activation event includes the root plus current, prefetched, and retained nested-unit collective buffers.",
            "The backward floor retains the forward peak while staging the root rank-local shard.",
            "The largest active trainable gradient is restored before reduce-scatter emits its local shard.",
            "Two largest nested-unit rank-local buffers account for persistent eager FSDP2 runtime staging during optimizer execution.",
            "Buffers, inputs, activations, outputs, loss tensors, and profiled operator workspaces remain replicated.",
            "AdamW tensor states follow each rank's logical trainable DTensor shard.",
        ],
    }
