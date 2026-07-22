from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "static_memory_estimate.v1"
SUPPORTED_OPTIMIZERS = {"none", "sgd", "sgd_momentum", "adam", "adamw"}
_EFFICIENT_ATTENTION_BATCH_WORKSPACE_BYTES = 65536
_EFFICIENT_ATTENTION_SEQUENCE_ROW_BYTES = 16
_EFFICIENT_ATTENTION_BATCH_METADATA_BYTES = 64


def estimate_module_memory(
    module: Any,
    example_args: Sequence[Any],
    *,
    example_kwargs: Mapping[str, Any] | None = None,
    mode: str = "training",
    loss_fn: Callable[[Any], Any] | None = None,
    optimizer: str = "adamw",
    retain_input_storages: bool = True,
    retain_forward_outputs: bool = True,
    target_device: str = "auto",
) -> dict[str, Any]:
    """Estimate a module's tensor-storage peak from a captured ATen graph.

    The estimator creates target-device fake tensors and traces them with
    ``make_fx``. It does not allocate CUDA memory or execute real kernels.
    Training mode captures a functional forward/backward graph with
    ``torch.func.grad_and_value`` and adds persistent optimizer state and eager
    single-tensor optimizer update temporaries explicitly.

    Matched backend workspaces are included through explicit static profiles.
    Unmatched workspaces, CUDA context memory, caching-allocator fragmentation,
    and fused/foreach optimizer temporaries are reported as unmodeled because
    they require device/software-specific calibration.
    """

    import torch
    from torch.func import functional_call, grad_and_value
    trace_device = _resolve_trace_device(
        torch,
        parameters_source=module.parameters(),
        example_args=example_args,
        example_kwargs=example_kwargs,
        target_device=target_device,
    )

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"forward", "training"}:
        raise ValueError("mode must be 'forward' or 'training'")

    normalized_optimizer = str(optimizer).strip().lower()
    if normalized_optimizer not in SUPPORTED_OPTIMIZERS:
        choices = ", ".join(sorted(SUPPORTED_OPTIMIZERS))
        raise ValueError(f"unsupported optimizer {optimizer!r}; expected one of: {choices}")
    if normalized_mode == "forward" and normalized_optimizer != "none":
        normalized_optimizer = "none"

    args = tuple(example_args)
    kwargs = dict(example_kwargs or {})
    parameters = dict(module.named_parameters())
    trainable_parameters = {
        name: parameter
        for name, parameter in parameters.items()
        if bool(getattr(parameter, "requires_grad", False))
    }
    frozen_parameters = {
        name: parameter
        for name, parameter in parameters.items()
        if not bool(getattr(parameter, "requires_grad", False))
    }
    buffers = dict(module.named_buffers())

    if normalized_mode == "training" and not trainable_parameters:
        raise ValueError("training estimation requires at least one trainable parameter")

    placeholder_categories: list[str]
    gradient_output_start: int | None
    gradient_output_end: int | None

    if normalized_mode == "training":
        selected_loss_fn = loss_fn or _default_loss

        def forward_and_loss(
            trainable: dict[str, Any],
            frozen: dict[str, Any],
            module_buffers: dict[str, Any],
            call_args: tuple[Any, ...],
            call_kwargs: dict[str, Any],
        ) -> tuple[Any, Any]:
            merged_parameters = dict(frozen)
            merged_parameters.update(trainable)
            output = functional_call(
                module,
                (merged_parameters, module_buffers),
                call_args,
                call_kwargs,
            )
            loss = selected_loss_fn(output)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar torch.Tensor")
            if not loss.is_floating_point():
                raise ValueError("loss_fn must return a real floating-point tensor")
            return loss, output

        def training_step(
            trainable: dict[str, Any],
            frozen: dict[str, Any],
            module_buffers: dict[str, Any],
            call_args: tuple[Any, ...],
            call_kwargs: dict[str, Any],
        ) -> tuple[Any, dict[str, Any], Any]:
            gradients, (loss, output) = grad_and_value(
                forward_and_loss,
                has_aux=True,
            )(
                trainable,
                frozen,
                module_buffers,
                call_args,
                call_kwargs,
            )
            if retain_forward_outputs:
                return loss, gradients, output
            return loss, gradients, ()

        with _fakegpu_saved_tensor_tracking_suspended():
            graph_module = _trace_with_fake_device(
                training_step,
                (
                    trainable_parameters,
                    frozen_parameters,
                    buffers,
                    args,
                    kwargs,
                ),
                device=trace_device,
            )
        placeholder_categories = [
            *["trainable_parameter" for _ in _iter_tensor_leaves(trainable_parameters)],
            *["frozen_parameter" for _ in _iter_tensor_leaves(frozen_parameters)],
            *["buffer" for _ in _iter_tensor_leaves(buffers)],
            *["input" for _ in _iter_tensor_leaves(args)],
            *["input" for _ in _iter_tensor_leaves(kwargs)],
        ]
        gradient_output_start = 1
        gradient_output_end = 1 + len(list(_iter_tensor_leaves(trainable_parameters)))
    else:

        def forward_step(
            module_parameters: dict[str, Any],
            module_buffers: dict[str, Any],
            call_args: tuple[Any, ...],
            call_kwargs: dict[str, Any],
        ) -> Any:
            return functional_call(
                module,
                (module_parameters, module_buffers),
                call_args,
                call_kwargs,
            )

        with _fakegpu_saved_tensor_tracking_suspended():
            graph_module = _trace_with_fake_device(
                forward_step,
                (
                    parameters,
                    buffers,
                    args,
                    kwargs,
                ),
                device=trace_device,
            )
        placeholder_categories = [
            *["parameter" for _ in _iter_tensor_leaves(parameters)],
            *["buffer" for _ in _iter_tensor_leaves(buffers)],
            *["input" for _ in _iter_tensor_leaves(args)],
            *["input" for _ in _iter_tensor_leaves(kwargs)],
        ]
        gradient_output_start = None
        gradient_output_end = None

    graph = analyze_graph_memory(
        graph_module,
        placeholder_categories=placeholder_categories,
        retain_placeholder_categories=(
            {"parameter", "trainable_parameter", "frozen_parameter", "buffer", "input"}
            if retain_input_storages
            else {"parameter", "trainable_parameter", "frozen_parameter", "buffer"}
        ),
        gradient_output_start=gradient_output_start,
        gradient_output_end=gradient_output_end,
    )

    parameter_bytes = _unique_tensor_storage_bytes(parameters.values())
    trainable_parameter_bytes = _unique_tensor_storage_bytes(trainable_parameters.values())
    frozen_parameter_bytes = _unique_tensor_storage_bytes(frozen_parameters.values())
    buffer_bytes = _unique_tensor_storage_bytes(buffers.values())
    input_bytes = _unique_tensor_storage_bytes(
        [*_iter_tensor_leaves(args), *_iter_tensor_leaves(kwargs)]
    )
    optimizer_state = _optimizer_state_estimate(
        trainable_parameters.values(),
        optimizer=normalized_optimizer,
    )
    optimizer_temporary = _optimizer_temporary_estimate(
        trainable_parameters.values(),
        optimizer=normalized_optimizer,
    )
    workspace_estimate = _estimate_backend_workspace(
        graph_module,
        graph=graph,
        target_device=trace_device,
    )
    graph_phase_peak = int(graph["peak_live_bytes"]) + int(
        optimizer_state["total_bytes"]
    ) + int(workspace_estimate["effective_peak_contribution_bytes"])
    first_step_graph_phase_peak = int(graph["peak_live_bytes"]) + int(
        workspace_estimate["effective_peak_contribution_bytes"]
    )
    optimizer_phase_peak: int | None = None
    if normalized_mode == "training" and normalized_optimizer != "none":
        optimizer_phase_peak = (
            int(graph["final_live_bytes"])
            + int(optimizer_state["total_bytes"])
            + int(optimizer_temporary["total_bytes"])
        )
    estimated_peak = max(
        graph_phase_peak,
        optimizer_phase_peak if optimizer_phase_peak is not None else 0,
    )
    first_step_estimated_peak = max(
        first_step_graph_phase_peak,
        optimizer_phase_peak if optimizer_phase_peak is not None else 0,
    )
    if optimizer_phase_peak is None or graph_phase_peak > optimizer_phase_peak:
        peak_phase = "graph"
    elif optimizer_phase_peak > graph_phase_peak:
        peak_phase = "optimizer"
    else:
        peak_phase = "graph_and_optimizer"

    unmodeled_components = [
        "backend_operator_workspaces_without_profiles",
        "cuda_context_and_loaded_modules",
        "caching_allocator_fragmentation",
    ]
    if any(
        str(profile.get("confidence", "")).startswith("extrapolated_")
        for profile in workspace_estimate["profiles"]
    ):
        unmodeled_components.append("workspace_profile_shape_extrapolation")
    if normalized_mode == "training" and normalized_optimizer != "none":
        unmodeled_components.append("optimizer_fused_or_foreach_extra_temporaries")

    return {
        "schema_version": SCHEMA_VERSION,
        "method": (
            "aot_autograd_aten_storage_liveness"
            if normalized_mode == "training"
            else "fx_aten_storage_liveness"
        ),
        "tracking_confidence": (
            "S2_aot_training_liveness"
            if normalized_mode == "training"
            else "S1_fx_forward_liveness"
        ),
        "mode": normalized_mode,
        "trace_device": str(trace_device),
        "retain_forward_outputs": bool(
            normalized_mode == "training" and retain_forward_outputs
        ),
        "torch_version": str(torch.__version__),
        "parameter_bytes": parameter_bytes,
        "trainable_parameter_bytes": trainable_parameter_bytes,
        "frozen_parameter_bytes": frozen_parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "input_bytes": input_bytes,
        "graph_peak_live_bytes": int(graph["peak_live_bytes"]),
        "post_graph_live_bytes": int(graph["final_live_bytes"]),
        "optimizer_state": optimizer_state,
        "optimizer_state_bytes": int(optimizer_state["total_bytes"]),
        "optimizer_temporary": optimizer_temporary,
        "optimizer_temporary_bytes": int(optimizer_temporary["total_bytes"]),
        "first_step_graph_phase_peak_bytes": first_step_graph_phase_peak,
        "graph_phase_peak_bytes": graph_phase_peak,
        "optimizer_phase_peak_bytes": optimizer_phase_peak,
        "first_step_estimated_peak_bytes": first_step_estimated_peak,
        "steady_state_estimated_peak_bytes": estimated_peak,
        "peak_phase": peak_phase,
        "workspace_estimate": workspace_estimate,
        "workspace_estimate_bytes": int(workspace_estimate["total_bytes"]),
        "workspace_peak_contribution_bytes": int(
            workspace_estimate["effective_peak_contribution_bytes"]
        ),
        "estimated_peak_bytes": estimated_peak,
        "graph": graph,
        "unmodeled_components": unmodeled_components,
        "notes": [
            "The ATen graph is captured with fake tensors and does not execute CUDA kernels.",
            "Storage aliases are deduplicated and tensors are released after their final graph use.",
            "Storage byte arithmetic is device-independent, while target-device dispatch controls the ATen graph and fingerprint.",
            "Known target-backend auxiliary storages are modeled from operator and shape profiles.",
            "The first-step graph peak excludes optimizer state that does not exist until the first optimizer update.",
            "The steady-state graph peak includes persistent optimizer state from an earlier update.",
        ],
    }


def analyze_graph_memory(
    graph_module: Any,
    *,
    placeholder_categories: Sequence[str] | None = None,
    retain_placeholder_categories: set[str] | None = None,
    gradient_output_start: int | None = None,
    gradient_output_end: int | None = None,
) -> dict[str, Any]:
    """Analyze unique storage lifetimes in an FX/ATen graph.

    Node ``meta['val']`` entries must contain concrete-shape tensors, as
    produced by ``make_fx``. Views and other aliases share a storage key and
    therefore do not create duplicate allocations.
    """

    try:
        from torch.fx import Node
    except Exception as exc:  # pragma: no cover - torch import error surface
        raise RuntimeError(f"torch.fx is required for graph memory analysis: {exc}") from exc

    nodes = list(graph_module.graph.nodes)
    node_index = {node: index for index, node in enumerate(nodes)}
    node_storages: dict[Any, set[int]] = {}
    storage_records: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []
    categories = list(placeholder_categories or [])
    placeholder_index = 0

    for node in nodes:
        category = "temporary"
        if node.op == "placeholder":
            if placeholder_index < len(categories):
                category = str(categories[placeholder_index])
            else:
                category = "input"
            placeholder_index += 1
        elif node.op == "get_attr":
            category = "constant"

        keys: set[int] = set()
        for tensor in _iter_tensor_leaves(node.meta.get("val")):
            try:
                key, nbytes = _tensor_storage_identity(tensor)
            except Exception as exc:
                warnings.append(f"{node.name}: unable to inspect tensor storage: {exc}")
                continue
            if nbytes <= 0:
                continue
            keys.add(key)
            record = storage_records.get(key)
            if record is None:
                storage_records[key] = {
                    "storage_key": key,
                    "bytes": nbytes,
                    "category": category,
                    "producer": node.name,
                    "producer_op": node.op,
                    "producer_target": _target_name(node.target),
                    "producer_index": node_index[node],
                    "last_use_index": node_index[node],
                    "dtype": str(getattr(tensor, "dtype", "unknown")),
                    "shape": _shape_list(getattr(tensor, "shape", ())),
                    "aliases": [node.name],
                }
            else:
                record["bytes"] = max(int(record["bytes"]), nbytes)
                if node.name not in record["aliases"]:
                    record["aliases"].append(node.name)
        node_storages[node] = keys

    for node in nodes:
        consumer_index = node_index[node]
        for referenced_node in _iter_node_references((node.args, node.kwargs), Node):
            for key in node_storages.get(referenced_node, set()):
                storage_records[key]["last_use_index"] = max(
                    int(storage_records[key]["last_use_index"]),
                    consumer_index,
                )

    output_node = next((node for node in reversed(nodes) if node.op == "output"), None)
    output_references = (
        list(_iter_node_references(output_node.args[0], Node))
        if output_node is not None and output_node.args
        else []
    )
    output_keys: set[int] = set()
    for output_position, referenced_node in enumerate(output_references):
        keys = node_storages.get(referenced_node, set())
        output_keys.update(keys)
        output_category = (
            "gradient"
            if gradient_output_start is not None
            and output_position >= gradient_output_start
            and (gradient_output_end is None or output_position < gradient_output_end)
            else "output"
        )
        for key in keys:
            current = str(storage_records[key].get("category") or "")
            if current not in {
                "parameter",
                "trainable_parameter",
                "frozen_parameter",
                "buffer",
                "input",
            }:
                storage_records[key]["category"] = output_category

    retained_categories = set(retain_placeholder_categories or set())
    retained_keys = {
        key
        for key, record in storage_records.items()
        if str(record.get("category")) in retained_categories
    }
    protected_keys = output_keys | retained_keys
    final_index = max(0, len(nodes) - 1)
    for key in protected_keys:
        storage_records[key]["last_use_index"] = final_index

    records_by_producer: dict[int, list[int]] = {}
    records_by_last_use: dict[int, list[int]] = {}
    for key, record in storage_records.items():
        records_by_producer.setdefault(int(record["producer_index"]), []).append(key)
        records_by_last_use.setdefault(int(record["last_use_index"]), []).append(key)

    gradient_producer_indices = [
        int(record["producer_index"])
        for record in storage_records.values()
        if str(record.get("category") or "") == "gradient"
    ]
    gradient_phase_start_index = (
        min(gradient_producer_indices) if gradient_producer_indices else None
    )

    live_keys: set[int] = set()
    live_bytes = 0
    peak_bytes = 0
    peak_index = 0
    peak_live_keys: set[int] = set()
    gradient_phase_peak_bytes = 0
    gradient_phase_peak_index: int | None = None
    gradient_phase_peak_live_keys: set[int] = set()
    live_bytes_by_node: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for key in records_by_producer.get(index, []):
            if key not in live_keys:
                live_keys.add(key)
                live_bytes += int(storage_records[key]["bytes"])
        live_bytes_by_node[node.name] = live_bytes
        if live_bytes > peak_bytes:
            peak_bytes = live_bytes
            peak_index = index
            peak_live_keys = set(live_keys)
        if (
            gradient_phase_start_index is not None
            and index >= gradient_phase_start_index
            and live_bytes > gradient_phase_peak_bytes
        ):
            gradient_phase_peak_bytes = live_bytes
            gradient_phase_peak_index = index
            gradient_phase_peak_live_keys = set(live_keys)
        for key in records_by_last_use.get(index, []):
            if key in protected_keys or key not in live_keys:
                continue
            live_keys.remove(key)
            live_bytes -= int(storage_records[key]["bytes"])

    final_live_keys = set(live_keys)
    final_live_bytes = sum(int(storage_records[key]["bytes"]) for key in final_live_keys)

    peak_node = nodes[peak_index] if nodes else None
    category_bytes: dict[str, int] = {}
    for key in peak_live_keys:
        record = storage_records[key]
        category = str(record.get("category") or "unknown")
        category_bytes[category] = category_bytes.get(category, 0) + int(record["bytes"])

    final_category_bytes: dict[str, int] = {}
    for key in final_live_keys:
        record = storage_records[key]
        category = str(record.get("category") or "unknown")
        final_category_bytes[category] = final_category_bytes.get(category, 0) + int(
            record["bytes"]
        )

    gradient_phase: dict[str, Any] | None = None
    if gradient_phase_peak_index is not None:
        gradient_phase_category_bytes: dict[str, int] = {}
        for key in gradient_phase_peak_live_keys:
            record = storage_records[key]
            category = str(record.get("category") or "unknown")
            gradient_phase_category_bytes[category] = (
                gradient_phase_category_bytes.get(category, 0)
                + int(record["bytes"])
            )
        gradient_phase_peak_node = nodes[gradient_phase_peak_index]
        gradient_phase_start_node = nodes[gradient_phase_start_index]
        gradient_phase = {
            "start_node": {
                "index": gradient_phase_start_index,
                "name": gradient_phase_start_node.name,
                "op": gradient_phase_start_node.op,
                "target": _target_name(gradient_phase_start_node.target),
            },
            "peak_live_bytes": gradient_phase_peak_bytes,
            "peak_node": {
                "index": gradient_phase_peak_index,
                "name": gradient_phase_peak_node.name,
                "op": gradient_phase_peak_node.op,
                "target": _target_name(gradient_phase_peak_node.target),
            },
            "peak_bytes_by_category": dict(
                sorted(gradient_phase_category_bytes.items())
            ),
        }

    top_peak_storages = sorted(
        (
            {
                key: value
                for key, value in storage_records[storage_key].items()
                if key not in {"producer_index", "last_use_index"}
            }
            for storage_key in peak_live_keys
        ),
        key=lambda item: int(item.get("bytes", 0)),
        reverse=True,
    )[:20]

    fingerprint_payload = [
        {
            "op": node.op,
            "target": _target_name(node.target),
            "storages": [
                {
                    "bytes": int(storage_records[key]["bytes"]),
                    "dtype": storage_records[key]["dtype"],
                    "shape": storage_records[key]["shape"],
                }
                for key in sorted(node_storages.get(node, set()))
            ],
        }
        for node in nodes
        if node.op != "output"
    ]
    canonical = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    operator_histogram: dict[str, int] = {}
    for node in nodes:
        if node.op not in {"call_function", "call_method", "call_module"}:
            continue
        target = _target_name(node.target)
        operator_histogram[target] = operator_histogram.get(target, 0) + 1

    return {
        "node_count": len(nodes),
        "operator_count": sum(
            1 for node in nodes if node.op in {"call_function", "call_method", "call_module"}
        ),
        "operator_histogram": dict(sorted(operator_histogram.items())),
        "unique_storage_count": len(storage_records),
        "alias_node_count": sum(
            max(0, len(record.get("aliases", [])) - 1)
            for record in storage_records.values()
        ),
        "total_unique_storage_bytes": sum(
            int(record["bytes"]) for record in storage_records.values()
        ),
        "peak_live_bytes": peak_bytes,
        "live_bytes_by_node": live_bytes_by_node,
        "final_live_bytes": final_live_bytes,
        "final_live_storage_count": len(final_live_keys),
        "peak_node": {
            "index": peak_index,
            "name": getattr(peak_node, "name", None),
            "op": getattr(peak_node, "op", None),
            "target": _target_name(getattr(peak_node, "target", "")),
        },
        "peak_bytes_by_category": dict(sorted(category_bytes.items())),
        "final_bytes_by_category": dict(sorted(final_category_bytes.items())),
        "gradient_production_phase": gradient_phase,
        "top_peak_storages": top_peak_storages,
        "retained_output_storage_count": len(output_keys),
        "retained_placeholder_storage_count": len(retained_keys),
        "graph_fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "warnings": warnings,
    }


def _optimizer_state_estimate(
    parameters: Iterable[Any],
    *,
    optimizer: str,
) -> dict[str, Any]:
    parameter_tensors = _unique_tensors_by_identity(parameters)
    parameter_bytes = sum(_tensor_logical_bytes(tensor) for tensor in parameter_tensors)
    parameter_storage_count = len(_unique_tensor_storages(parameter_tensors))

    if optimizer in {"adam", "adamw"}:
        tensor_state_bytes = 2 * parameter_bytes
        scalar_state_bytes = 0
        state_tensors_per_parameter = 2
        description = (
            "two same-dtype moment tensors; non-capturable eager step scalars "
            "remain on CPU"
        )
    elif optimizer == "sgd_momentum":
        tensor_state_bytes = parameter_bytes
        scalar_state_bytes = 0
        state_tensors_per_parameter = 1
        description = "one same-dtype momentum tensor per parameter"
    else:
        tensor_state_bytes = 0
        scalar_state_bytes = 0
        state_tensors_per_parameter = 0
        description = "no persistent optimizer tensor state"

    return {
        "optimizer": optimizer,
        "parameter_tensor_count": len(parameter_tensors),
        "parameter_storage_count": parameter_storage_count,
        "parameter_logical_bytes": parameter_bytes,
        "state_tensors_per_parameter": state_tensors_per_parameter,
        "tensor_state_bytes": tensor_state_bytes,
        "scalar_state_bytes": scalar_state_bytes,
        "total_bytes": tensor_state_bytes + scalar_state_bytes,
        "description": description,
    }


def _optimizer_temporary_estimate(
    parameters: Iterable[Any],
    *,
    optimizer: str,
) -> dict[str, Any]:
    parameter_tensors = _unique_tensors_by_identity(parameters)
    tensor_bytes = [_tensor_logical_bytes(tensor) for tensor in parameter_tensors]
    largest_parameter_bytes = max(tensor_bytes, default=0)
    if optimizer in {"adam", "adamw"}:
        peak_current_parameter_bytes = 0
        peak_previous_parameter_bytes = 0
        total_bytes = 0
        previous_bytes = 0
        for current_bytes in tensor_bytes:
            candidate = 2 * current_bytes + previous_bytes
            if candidate > total_bytes:
                total_bytes = candidate
                peak_current_parameter_bytes = current_bytes
                peak_previous_parameter_bytes = previous_bytes
            previous_bytes = current_bytes
        description = (
            "two intermediates for the current parameter plus the previous "
            "parameter's retained denominator in eager single-tensor Adam/AdamW"
        )
        assumption = "single_tensor_eager"
    else:
        peak_current_parameter_bytes = 0
        peak_previous_parameter_bytes = 0
        total_bytes = 0
        description = "no explicit optimizer update temporary"
        assumption = "in_place_or_no_optimizer"
    return {
        "optimizer": optimizer,
        "assumption": assumption,
        "parameter_tensor_count": len(parameter_tensors),
        "largest_parameter_tensor_bytes": largest_parameter_bytes,
        "peak_current_parameter_tensor_bytes": peak_current_parameter_bytes,
        "peak_previous_parameter_tensor_bytes": peak_previous_parameter_bytes,
        "current_parameter_temporary_count": 2 if total_bytes else 0,
        "retained_previous_temporary_count": (
            1 if peak_previous_parameter_bytes else 0
        ),
        "total_bytes": total_bytes,
        "description": description,
    }


def _estimate_backend_workspace(
    graph_module: Any,
    *,
    graph: Mapping[str, Any],
    target_device: Any,
) -> dict[str, Any]:
    profiles: list[dict[str, Any]] = []
    graph_modeled_attention_operators: dict[str, int] = {}
    unprofiled_attention_operators: dict[str, int] = {}

    for node in graph_module.graph.nodes:
        if node.op not in {"call_function", "call_method", "call_module"}:
            continue
        target = _target_name(node.target)
        profile = _backend_workspace_profile(
            node,
            target,
            target_device=target_device,
        )
        if profile is not None:
            profiles.append(profile)
            continue
        if _is_graph_modeled_attention_operator(target):
            graph_modeled_attention_operators[target] = (
                graph_modeled_attention_operators.get(target, 0) + 1
            )
            continue
        if "attention" in target.lower():
            unprofiled_attention_operators[target] = (
                unprofiled_attention_operators.get(target, 0) + 1
            )

    summary = _workspace_peak_summary(graph, profiles)
    return {
        "method": "target_aten_operator_shape_profiles",
        "lifetime_model": "node_liveness.v1",
        "target_device": str(target_device),
        **summary,
        "profiles": profiles,
        "profiled_operator_count": len(profiles),
        "extrapolated_profile_count": sum(
            str(profile.get("confidence", "")).startswith("extrapolated_")
            for profile in profiles
        ),
        "graph_modeled_attention_operators": dict(
            sorted(graph_modeled_attention_operators.items())
        ),
        "unprofiled_attention_operators": dict(
            sorted(unprofiled_attention_operators.items())
        ),
        "notes": [
            "Graph-phase persistent profiles are added across the graph phase.",
            "Operator-local workspaces are combined only with storage live at their execution node.",
            "Flash Attention auxiliary storage uses query shape, dtype, and 64-token sequence tiles.",
            "FP32 Efficient Attention backward workspace uses batch, sequence, and query storage shape.",
            "Unprofiled operators remain listed instead of receiving a generic multiplier.",
        ],
    }


def _backend_workspace_profile(
    node: Any,
    target: str,
    *,
    target_device: Any,
) -> dict[str, Any] | None:
    if "attention" not in target.lower():
        return None
    device_type = str(
        getattr(target_device, "type", str(target_device).split(":", 1)[0])
    )
    if device_type != "cuda":
        return None
    query = _attention_query(node)
    if query is None:
        return None

    if "_scaled_dot_product_flash_attention_backward" in target:
        workspace_bytes, sequence_length, sequence_tiles = (
            _flash_attention_workspace_bytes(query, sequence_dimension=2)
        )
        if workspace_bytes <= 0:
            return None
        return {
            "operator": target,
            "node": node.name,
            "profile": "cuda_flash_attention_backward_auxiliary.v1",
            "kind": "backend_saved_or_workspace_storage",
            "lifetime": "graph_phase_persistent",
            "query_shape": _shape_list(query.shape),
            "query_dtype": str(query.dtype),
            "sequence_length": sequence_length,
            "sequence_tile_size": 64,
            "sequence_tile_count": sequence_tiles,
            "bytes": workspace_bytes,
            "confidence": _flash_attention_profile_confidence(
                query,
                sequence_dimension=2,
                scaled_operator=True,
            ),
            "validated_envelope": {
                "compute_capabilities": ["8.6", "12.0"],
                "batch_size": [2, 4],
                "sequence_length": [64, 128],
                "attention_heads": [4, 4],
                "head_dimension": [32, 32],
                "dtype": "torch.bfloat16",
            },
        }

    if "_flash_attention_backward" in target:
        workspace_bytes, sequence_length, sequence_tiles = (
            _flash_attention_workspace_bytes(query, sequence_dimension=1)
        )
        if workspace_bytes <= 0:
            return None
        return {
            "operator": target,
            "node": node.name,
            "profile": "cuda_flash_attention_backward_auxiliary.v1",
            "kind": "backend_saved_or_workspace_storage",
            "lifetime": "graph_phase_persistent",
            "query_shape": _shape_list(query.shape),
            "query_dtype": str(query.dtype),
            "sequence_length": sequence_length,
            "sequence_tile_size": 64,
            "sequence_tile_count": sequence_tiles,
            "bytes": workspace_bytes,
            "confidence": _flash_attention_profile_confidence(
                query,
                sequence_dimension=1,
                scaled_operator=False,
            ),
            "validated_envelope": {
                "source_profile": "scaled_dot_product_flash_attention_backward",
                "compute_capabilities": ["8.6", "12.0"],
            },
        }

    if (
        "_scaled_dot_product_efficient_attention_backward" in target
        and str(getattr(query, "dtype", "")) == "torch.float32"
    ):
        (
            workspace_bytes,
            batch_size,
            sequence_length,
            fixed_bytes,
            query_scratch_bytes,
            row_metadata_bytes,
        ) = _efficient_attention_backward_workspace_bytes(query)
        if workspace_bytes <= 0:
            return None
        query_shape = tuple(query.shape)
        validated_shape = (
            1 <= batch_size <= 4
            and 16 <= sequence_length <= 64
            and int(query_shape[1]) == 4
            and 16 <= int(query_shape[3]) <= 32
        )
        return {
            "operator": target,
            "node": node.name,
            "profile": "cuda_efficient_attention_backward_workspace.v1",
            "kind": "operator_workspace_storage",
            "lifetime": "operator_local",
            "query_shape": _shape_list(query.shape),
            "query_dtype": str(query.dtype),
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "fixed_bytes": fixed_bytes,
            "query_scratch_bytes": query_scratch_bytes,
            "row_metadata_bytes": row_metadata_bytes,
            "bytes": workspace_bytes,
            "confidence": (
                "validated_two_gpu_multi_shape"
                if validated_shape
                else "extrapolated_from_two_gpu_multi_shape"
            ),
            "validated_envelope": {
                "compute_capabilities": ["8.6", "12.0"],
                "batch_size": [1, 4],
                "sequence_length": [16, 64],
                "attention_heads": [4, 4],
                "head_dimension": [16, 32],
                "dtype": "torch.float32",
            },
        }

    return None


def _workspace_peak_summary(
    graph: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    graph_peak_bytes = int(graph.get("peak_live_bytes", 0) or 0)
    live_bytes_by_node = {
        str(name): int(value)
        for name, value in dict(graph.get("live_bytes_by_node") or {}).items()
    }
    persistent_bytes = sum(
        int(profile.get("bytes", 0) or 0)
        for profile in profiles
        if profile.get("lifetime") == "graph_phase_persistent"
    )
    local_bytes_by_node: dict[str, int] = {}
    for profile in profiles:
        if profile.get("lifetime") != "operator_local":
            continue
        node = str(profile.get("node") or "")
        local_bytes_by_node[node] = local_bytes_by_node.get(node, 0) + int(
            profile.get("bytes", 0) or 0
        )

    peak_node = str((graph.get("peak_node") or {}).get("name") or "")
    peak_graph_live_bytes = graph_peak_bytes
    peak_operator_workspace_bytes = 0
    combined_peak_bytes = graph_peak_bytes + persistent_bytes
    for node, operator_workspace_bytes in local_bytes_by_node.items():
        graph_live_bytes = int(live_bytes_by_node.get(node, 0))
        candidate = graph_live_bytes + persistent_bytes + operator_workspace_bytes
        if candidate > combined_peak_bytes:
            combined_peak_bytes = candidate
            peak_node = node
            peak_graph_live_bytes = graph_live_bytes
            peak_operator_workspace_bytes = operator_workspace_bytes

    profiled_bytes_sum = sum(
        int(profile.get("bytes", 0) or 0) for profile in profiles
    )
    effective_peak_contribution = max(0, combined_peak_bytes - graph_peak_bytes)
    return {
        "total_bytes": profiled_bytes_sum,
        "effective_peak_contribution_bytes": effective_peak_contribution,
        "profiled_bytes_sum": profiled_bytes_sum,
        "graph_phase_persistent_bytes": persistent_bytes,
        "operator_local_peak_bytes": max(local_bytes_by_node.values(), default=0),
        "peak_candidate": {
            "node": peak_node,
            "graph_live_bytes": peak_graph_live_bytes,
            "graph_phase_persistent_bytes": persistent_bytes,
            "operator_workspace_bytes": peak_operator_workspace_bytes,
            "combined_live_bytes": combined_peak_bytes,
        },
    }


def _attention_query(node: Any) -> Any | None:
    if len(node.args) <= 1 or not hasattr(node.args[1], "meta"):
        return None
    return _first_tensor_leaf(node.args[1].meta.get("val"))


def _is_graph_modeled_attention_operator(target: str) -> bool:
    return (
        "_scaled_dot_product_efficient_attention(" in target
        or "_scaled_dot_product_flash_attention(" in target
        or "_flash_attention_forward(" in target
    )


def _flash_attention_workspace_bytes(
    query: Any,
    *,
    sequence_dimension: int,
) -> tuple[int, int, int]:
    shape = tuple(query.shape)
    if not shape:
        return 0, 0, 0
    sequence_length = int(shape[sequence_dimension])
    sequence_tiles = max(1, (sequence_length + 63) // 64)
    query_bytes = int(query.numel()) * int(query.element_size())
    return 2 * sequence_tiles * query_bytes, sequence_length, sequence_tiles


def _flash_attention_profile_confidence(
    query: Any,
    *,
    sequence_dimension: int,
    scaled_operator: bool,
) -> str:
    shape = tuple(query.shape)
    validated_shape = (
        scaled_operator
        and len(shape) == 4
        and str(getattr(query, "dtype", "")) == "torch.bfloat16"
        and int(shape[0]) in {2, 4}
        and int(shape[1]) == 4
        and int(shape[sequence_dimension]) in {64, 128}
        and int(shape[3]) == 32
    )
    return (
        "validated_two_gpu_multi_shape"
        if validated_shape
        else "extrapolated_from_two_gpu_multi_shape"
    )


def _efficient_attention_backward_workspace_bytes(
    query: Any,
) -> tuple[int, int, int, int, int, int]:
    shape = tuple(query.shape)
    if len(shape) != 4:
        raise ValueError("Efficient Attention workspace profile requires a rank-4 query")
    batch_size = int(shape[0])
    sequence_length = int(shape[2])
    fixed_bytes = batch_size * _EFFICIENT_ATTENTION_BATCH_WORKSPACE_BYTES
    query_scratch_bytes = (
        int(query.numel()) * int(query.element_size()) if batch_size > 1 else 0
    )
    row_metadata_bytes = batch_size * (
        _EFFICIENT_ATTENTION_SEQUENCE_ROW_BYTES * sequence_length
        + _EFFICIENT_ATTENTION_BATCH_METADATA_BYTES
    )
    total_bytes = fixed_bytes + query_scratch_bytes + row_metadata_bytes
    return (
        total_bytes,
        batch_size,
        sequence_length,
        fixed_bytes,
        query_scratch_bytes,
        row_metadata_bytes,
    )


def _first_tensor_leaf(value: Any) -> Any | None:
    return next(_iter_tensor_leaves(value), None)


def _default_loss(output: Any) -> Any:
    import torch

    direct_loss = getattr(output, "loss", None)
    if isinstance(direct_loss, torch.Tensor):
        return direct_loss
    if isinstance(output, Mapping):
        mapped_loss = output.get("loss")
        if isinstance(mapped_loss, torch.Tensor):
            return mapped_loss

    tensor = next(_iter_tensor_leaves(output), None)
    if tensor is None:
        raise ValueError("unable to derive a loss because the module returned no tensor")
    if tensor.numel() == 1 and tensor.is_floating_point():
        return tensor.reshape(())
    if tensor.is_complex():
        return tensor.abs().square().mean()
    if not tensor.is_floating_point():
        raise ValueError("default loss requires a floating-point or complex output tensor")
    return tensor.square().mean()


def _fakegpu_saved_tensor_tracking_suspended():
    torch_patch = sys.modules.get("fakegpu.torch_patch")
    suspend = getattr(torch_patch, "_suspend_autograd_saved_tensor_tracking", None)
    if callable(suspend):
        return suspend()
    return contextlib.nullcontext()


def _resolve_trace_device(
    torch: Any,
    *,
    parameters_source: Iterable[Any],
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None,
    target_device: str,
) -> Any:
    normalized = str(target_device).strip().lower()
    if normalized == "auto":
        if _torch_has_cuda_build(torch):
            return torch.device("cuda", 0)
        first_tensor = next(
            _iter_tensor_leaves(
                (
                    tuple(parameters_source),
                    tuple(example_args),
                    dict(example_kwargs or {}),
                )
            ),
            None,
        )
        return torch.device(
            getattr(first_tensor, "device", torch.device("cpu"))
        )

    device = torch.device(target_device)
    if device.type == "cuda" and not _torch_has_cuda_build(torch):
        raise RuntimeError(
            "target_device='cuda' requires a PyTorch build linked with CUDA; "
            "use target_device='auto' or run the trace in a CUDA-enabled PyTorch environment"
        )
    return device


def _torch_has_cuda_build(torch: Any) -> bool:
    compiled_flag = getattr(getattr(torch, "_C", None), "_has_cuda", None)
    if compiled_flag is not None:
        return bool(compiled_flag)
    return bool(getattr(torch.version, "cuda", None))


def _trace_with_fake_device(
    function: Callable[..., Any],
    inputs: tuple[Any, ...],
    *,
    device: Any,
) -> Any:
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    tensor_memo: dict[int, Any] = {}
    byte_storage_bases: dict[int, Any] = {}
    typed_storage_bases: dict[tuple[int, Any], Any] = {}

    def convert(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            memoized = tensor_memo.get(id(value))
            if memoized is not None:
                return memoized
            if value.layout != torch.strided:
                raise NotImplementedError(
                    "static memory tracing currently supports strided tensor inputs only"
                )

            storage = value.untyped_storage()
            storage_key = int(getattr(storage, "_cdata", id(storage)))
            byte_base = byte_storage_bases.get(storage_key)
            if byte_base is None:
                storage_bytes = int(storage.nbytes())
                byte_base = torch.ops.aten.empty.memory_format(
                    [storage_bytes],
                    dtype=torch.uint8,
                    layout=torch.strided,
                    device=device,
                    pin_memory=False,
                    memory_format=torch.contiguous_format,
                )
                byte_storage_bases[storage_key] = byte_base

            typed_key = (storage_key, value.dtype)
            typed_base = typed_storage_bases.get(typed_key)
            if typed_base is None:
                element_size = max(1, int(value.element_size()))
                usable_bytes = int(storage.nbytes()) // element_size * element_size
                typed_base = byte_base[:usable_bytes].view(value.dtype)
                typed_storage_bases[typed_key] = typed_base

            converted = typed_base.as_strided(
                tuple(value.shape),
                tuple(value.stride()),
                int(value.storage_offset()),
            ).detach()
            if bool(value.requires_grad):
                converted.requires_grad_(True)
            tensor_memo[id(value)] = converted
            return converted
        if isinstance(value, Mapping):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    with mode:
        fake_inputs = tuple(convert(item) for item in inputs)
        return make_fx(function, tracing_mode="real")(*fake_inputs)


def _iter_tensor_leaves(value: Any):
    try:
        import torch
    except Exception:
        return

    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensor_leaves(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensor_leaves(item)


def _iter_node_references(value: Any, node_type: type[Any]):
    if isinstance(value, node_type):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_node_references(item, node_type)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_node_references(item, node_type)


def _tensor_storage_identity(tensor: Any) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    storage_key = int(getattr(storage, "_cdata", id(storage)))
    return storage_key, int(storage.nbytes())


def _unique_tensor_storages(tensors: Iterable[Any]) -> list[tuple[int, int, Any]]:
    unique: dict[int, tuple[int, int, Any]] = {}
    for tensor in tensors:
        try:
            key, nbytes = _tensor_storage_identity(tensor)
        except Exception:
            continue
        if nbytes <= 0:
            continue
        current = unique.get(key)
        if current is None or nbytes > current[1]:
            unique[key] = (key, nbytes, tensor)
    return list(unique.values())


def _unique_tensors_by_identity(tensors: Iterable[Any]) -> list[Any]:
    unique: dict[int, Any] = {}
    for tensor in tensors:
        unique.setdefault(id(tensor), tensor)
    return list(unique.values())


def _tensor_logical_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _unique_tensor_storage_bytes(tensors: Iterable[Any]) -> int:
    return sum(nbytes for _key, nbytes, _tensor in _unique_tensor_storages(tensors))


def _target_name(target: Any) -> str:
    schema = getattr(target, "_schema", None)
    if schema is not None:
        return str(schema)
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return str(target)


def _shape_list(shape: Any) -> list[int | str]:
    values: list[int | str] = []
    for dimension in shape:
        try:
            values.append(int(dimension))
        except (TypeError, ValueError):
            values.append(str(dimension))
    return values


__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_OPTIMIZERS",
    "analyze_graph_memory",
    "estimate_module_memory",
]
