from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._cli import (
    add_json_path_argument,
    command_prog,
    usage_error,
)
from .llm_estimator import (
    KV_CACHE_STRATEGIES,
    _forward_transient_bytes,
    estimate_decoder_inference,
    estimate_kv_cache_memory,
)
from .profile_catalog import get_profile
from .structured_io import emit_json, load_mapping


SCHEMA_VERSION = "fakegpu.llm_serving_plan.v1"
REQUEST_SET_SCHEMA_VERSION = "fakegpu.llm_serving_request_set_plan.v1"
REQUEST_MANIFEST_SCHEMA_VERSION = "fakegpu.serving_requests.v1"
WORKLOAD_SIGNATURE_SCHEMA_VERSION = "fakegpu.serving_workload.v1"
_PREFIX_CACHE_STRATEGIES = frozenset({"dynamic", "paged"})
SERVING_RUNTIMES = frozenset({"generic", "vllm"})
DEFAULT_MEMORY_UTILIZATION = 0.9
VLLM_DEFAULT_GPU_MEMORY_UTILIZATION = 0.92
DEFAULT_SPECULATIVE_TOKENS = 5


class ServingPlanError(ValueError):
    pass


def estimate_serving_kv_pool(
    *,
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    active_sequences: int,
    prompt_tokens: int,
    generated_tokens: int = 1,
    element_bytes: int = 2,
    strategy: str = "paged",
    shared_prefix_tokens: int = 0,
    quantized_bits: int = 4,
    quantized_residual_tokens: int = 128,
    block_tokens: int = 16,
    max_cache_tokens: int | None = None,
    window_tokens: int | None = None,
    elements_per_token_per_layer: int | None = None,
    cache_layout: str = "separate_key_value",
) -> dict[str, Any]:
    """Estimate a homogeneous online-serving KV pool.

    Prefix caching is modeled as one shared cache segment plus one private
    segment per active sequence. Dynamic allocation is token-exact. Paged
    allocation rounds the shared and private segments independently.
    """

    _positive_integer(active_sequences, "active_sequences")
    _positive_integer(prompt_tokens, "prompt_tokens")
    _positive_integer(generated_tokens, "generated_tokens")
    _nonnegative_integer(shared_prefix_tokens, "shared_prefix_tokens")
    if shared_prefix_tokens > prompt_tokens:
        raise ServingPlanError(
            "shared_prefix_tokens must not exceed prompt_tokens"
        )
    if strategy not in KV_CACHE_STRATEGIES:
        choices = ", ".join(sorted(KV_CACHE_STRATEGIES))
        raise ServingPlanError(
            f"unsupported KV-cache strategy {strategy!r}; "
            f"expected one of: {choices}"
        )
    if shared_prefix_tokens and strategy not in _PREFIX_CACHE_STRATEGIES:
        choices = ", ".join(sorted(_PREFIX_CACHE_STRATEGIES))
        raise ServingPlanError(
            "shared_prefix_tokens requires a prefix-shareable KV-cache "
            f"strategy: {choices}"
        )

    baseline = estimate_kv_cache_memory(
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        batch_size=active_sequences,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        element_bytes=element_bytes,
        strategy=strategy,
        quantized_bits=quantized_bits,
        quantized_residual_tokens=quantized_residual_tokens,
        block_tokens=block_tokens,
        max_cache_tokens=max_cache_tokens,
        window_tokens=window_tokens,
        elements_per_token_per_layer=elements_per_token_per_layer,
        cache_layout=cache_layout,
    )
    bytes_per_token_per_sequence = int(
        baseline["bytes_per_token_per_sequence_at_compute_dtype"]
    )

    phases: dict[str, dict[str, Any]] = {}
    phase_tokens = {
        "prefill": prompt_tokens,
        "decode": prompt_tokens + max(0, generated_tokens - 1),
    }
    baseline_phase_names = {
        "prefill": "prefill",
        "decode": "generation",
    }
    for phase_name, logical_tokens in phase_tokens.items():
        baseline_phase = dict(
            baseline[baseline_phase_names[phase_name]]
        )
        effective_tokens = int(
            baseline_phase["effective_tokens_per_sequence"]
        )
        retained_prefix_tokens = _retained_prefix_tokens(
            shared_prefix_tokens=shared_prefix_tokens,
            logical_tokens=logical_tokens,
            effective_tokens=effective_tokens,
        )
        private_tokens = effective_tokens - retained_prefix_tokens
        without_prefix_bytes = int(baseline_phase["allocated_bytes"])

        if not shared_prefix_tokens:
            shared_allocation = _empty_cache_segment()
            private_allocation = {
                "sequence_count": active_sequences,
                "logical_tokens_per_sequence": effective_tokens,
                "allocated_tokens_per_sequence": int(
                    baseline_phase["allocated_tokens_per_sequence"]
                ),
                "allocated_bytes": without_prefix_bytes,
            }
            allocated_bytes = without_prefix_bytes
            storage_logical_bytes = int(
                baseline_phase["storage_logical_bytes"]
            )
            reservation_overhead_bytes = int(
                baseline_phase["reservation_overhead_bytes"]
            )
        else:
            shared_allocation = _cache_segment(
                tokens=retained_prefix_tokens,
                sequence_count=1,
                bytes_per_token_per_sequence=(
                    bytes_per_token_per_sequence
                ),
                strategy=strategy,
                block_tokens=block_tokens,
            )
            private_allocation = _cache_segment(
                tokens=private_tokens,
                sequence_count=active_sequences,
                bytes_per_token_per_sequence=(
                    bytes_per_token_per_sequence
                ),
                strategy=strategy,
                block_tokens=block_tokens,
            )
            allocated_bytes = int(
                shared_allocation["allocated_bytes"]
            ) + int(private_allocation["allocated_bytes"])
            storage_logical_bytes = (
                bytes_per_token_per_sequence
                * (
                    retained_prefix_tokens
                    + active_sequences * private_tokens
                )
            )
            reservation_overhead_bytes = (
                allocated_bytes - storage_logical_bytes
            )

        phases[phase_name] = {
            "logical_tokens_per_sequence": logical_tokens,
            "effective_tokens_per_sequence": effective_tokens,
            "retained_shared_prefix_tokens": retained_prefix_tokens,
            "private_tokens_per_sequence": private_tokens,
            "storage_logical_bytes": storage_logical_bytes,
            "allocated_bytes": allocated_bytes,
            "without_prefix_cache_bytes": without_prefix_bytes,
            "prefix_cache_savings_bytes": (
                without_prefix_bytes - allocated_bytes
            ),
            "reservation_overhead_bytes": reservation_overhead_bytes,
            "allocation_utilization_percent": (
                storage_logical_bytes / allocated_bytes * 100
                if allocated_bytes
                else None
            ),
            "shared_segment": shared_allocation,
            "private_segments": private_allocation,
        }

    return {
        "strategy": strategy,
        "active_sequences": active_sequences,
        "cache_layout": baseline["cache_layout"],
        "elements_per_token_per_layer": baseline[
            "elements_per_token_per_layer"
        ],
        "bytes_per_token_per_sequence_at_compute_dtype": (
            bytes_per_token_per_sequence
        ),
        "storage_bits_per_element": baseline[
            "storage_bits_per_element"
        ],
        "compute_element_bytes": element_bytes,
        "block_tokens": block_tokens if strategy == "paged" else None,
        "max_cache_tokens": baseline["max_cache_tokens"],
        "window_tokens": window_tokens,
        "shared_prefix_tokens": shared_prefix_tokens,
        "prefix_cache_enabled": shared_prefix_tokens > 0,
        "prefill": phases["prefill"],
        "decode": phases["decode"],
        "formula": (
            "shared_segment_bytes + active_sequences * "
            "private_segment_bytes"
            if shared_prefix_tokens
            else baseline["formula"]
        ),
        "modeled_overheads": list(baseline["modeled_overheads"]),
    }


def load_serving_requests(path: str | Path) -> dict[str, Any]:
    """Load and validate a JSON, TOML, or YAML serving-request manifest."""

    payload = load_mapping(path)
    schema_version = payload.get("schema_version")
    if schema_version != REQUEST_MANIFEST_SCHEMA_VERSION:
        raise ServingPlanError(
            "unsupported serving request schema "
            f"{schema_version!r}; expected "
            f"{REQUEST_MANIFEST_SCHEMA_VERSION!r}"
        )
    requests = _normalize_serving_requests(payload.get("requests"))
    return {
        "schema_version": REQUEST_MANIFEST_SCHEMA_VERSION,
        "source": str(Path(path).expanduser().resolve()),
        "requests": requests,
    }


def estimate_serving_request_kv_pool(
    requests: Sequence[Mapping[str, Any]],
    *,
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    element_bytes: int = 2,
    strategy: str = "paged",
    quantized_bits: int = 4,
    quantized_residual_tokens: int = 128,
    block_tokens: int = 16,
    max_cache_tokens: int | None = None,
    window_tokens: int | None = None,
    elements_per_token_per_layer: int | None = None,
    cache_layout: str = "separate_key_value",
) -> dict[str, Any]:
    """Estimate KV storage for an ordered heterogeneous request set."""

    normalized = _normalize_serving_requests(requests)
    if strategy not in KV_CACHE_STRATEGIES:
        choices = ", ".join(sorted(KV_CACHE_STRATEGIES))
        raise ServingPlanError(
            f"unsupported KV-cache strategy {strategy!r}; "
            f"expected one of: {choices}"
        )
    if (
        any(request["shared_prefix_tokens"] for request in normalized)
        and strategy not in _PREFIX_CACHE_STRATEGIES
    ):
        choices = ", ".join(sorted(_PREFIX_CACHE_STRATEGIES))
        raise ServingPlanError(
            "shared prefix groups require a prefix-shareable KV-cache "
            f"strategy: {choices}"
        )

    baselines: dict[str, dict[str, Any]] = {}
    for request in normalized:
        baselines[request["id"]] = estimate_kv_cache_memory(
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            batch_size=1,
            prompt_tokens=request["prompt_tokens"],
            generated_tokens=request["generated_tokens"],
            element_bytes=element_bytes,
            strategy=strategy,
            quantized_bits=quantized_bits,
            quantized_residual_tokens=quantized_residual_tokens,
            block_tokens=block_tokens,
            max_cache_tokens=max_cache_tokens,
            window_tokens=window_tokens,
            elements_per_token_per_layer=(
                elements_per_token_per_layer
            ),
            cache_layout=cache_layout,
        )
    first_baseline = baselines[normalized[0]["id"]]
    bytes_per_token = int(
        first_baseline[
            "bytes_per_token_per_sequence_at_compute_dtype"
        ]
    )
    static_capacities = (
        {
            request["id"]: int(
                baselines[request["id"]]["max_cache_tokens"]
            )
            for request in normalized
        }
        if strategy == "static"
        else None
    )
    unique_static_capacities = (
        set(static_capacities.values())
        if static_capacities is not None
        else set()
    )

    phases: dict[str, dict[str, Any]] = {}
    for phase_name, baseline_phase_name in (
        ("prefill", "prefill"),
        ("decode", "generation"),
    ):
        request_details: list[dict[str, Any]] = []
        prefix_members: dict[str, list[dict[str, Any]]] = {}
        without_prefix_bytes = 0
        without_prefix_storage_logical_bytes = 0
        base_dtype_logical_bytes = 0
        quantization_savings_bytes = 0
        storage_logical_bytes = 0
        private_allocated_bytes = 0

        for request in normalized:
            baseline_phase = dict(
                baselines[request["id"]][baseline_phase_name]
            )
            logical_tokens = (
                request["prompt_tokens"]
                if phase_name == "prefill"
                else request["prompt_tokens"]
                + max(0, request["generated_tokens"] - 1)
            )
            effective_tokens = int(
                baseline_phase["effective_tokens_per_sequence"]
            )
            retained_prefix_tokens = _retained_prefix_tokens(
                shared_prefix_tokens=request[
                    "shared_prefix_tokens"
                ],
                logical_tokens=logical_tokens,
                effective_tokens=effective_tokens,
            )
            private_tokens = effective_tokens - retained_prefix_tokens
            without_bytes = int(baseline_phase["allocated_bytes"])
            without_prefix_bytes += without_bytes
            without_prefix_storage_logical_bytes += int(
                baseline_phase["storage_logical_bytes"]
            )
            base_dtype_logical_bytes += int(
                baseline_phase["base_dtype_logical_bytes"]
            )
            quantization_savings_bytes += int(
                baseline_phase["quantization_savings_bytes"]
            )

            if request["prefix_group"] is None:
                private_segment = {
                    "sequence_count": 1,
                    "logical_tokens_per_sequence": effective_tokens,
                    "allocated_tokens_per_sequence": int(
                        baseline_phase[
                            "allocated_tokens_per_sequence"
                        ]
                    ),
                    "allocated_bytes": without_bytes,
                }
                request_storage_bytes = int(
                    baseline_phase["storage_logical_bytes"]
                )
            else:
                private_segment = _cache_segment(
                    tokens=private_tokens,
                    sequence_count=1,
                    bytes_per_token_per_sequence=bytes_per_token,
                    strategy=strategy,
                    block_tokens=block_tokens,
                )
                request_storage_bytes = (
                    private_tokens * bytes_per_token
                )
                prefix_members.setdefault(
                    request["prefix_group"],
                    [],
                ).append(
                    {
                        "id": request["id"],
                        "retained_prefix_tokens": (
                            retained_prefix_tokens
                        ),
                    }
                )

            storage_logical_bytes += request_storage_bytes
            private_allocated_bytes += int(
                private_segment["allocated_bytes"]
            )
            request_details.append(
                {
                    "id": request["id"],
                    "prompt_tokens": request["prompt_tokens"],
                    "generated_tokens": request["generated_tokens"],
                    "logical_tokens": logical_tokens,
                    "effective_tokens": effective_tokens,
                    "prefix_group": request["prefix_group"],
                    "retained_shared_prefix_tokens": (
                        retained_prefix_tokens
                    ),
                    "private_tokens": private_tokens,
                    "private_segment": private_segment,
                    "without_prefix_cache_bytes": without_bytes,
                }
            )

        prefix_groups: list[dict[str, Any]] = []
        shared_allocated_bytes = 0
        for group_id in sorted(prefix_members):
            members = prefix_members[group_id]
            shared_tokens = max(
                int(member["retained_prefix_tokens"])
                for member in members
            )
            segment = _cache_segment(
                tokens=shared_tokens,
                sequence_count=1,
                bytes_per_token_per_sequence=bytes_per_token,
                strategy=strategy,
                block_tokens=block_tokens,
            )
            shared_allocated_bytes += int(segment["allocated_bytes"])
            storage_logical_bytes += shared_tokens * bytes_per_token
            prefix_groups.append(
                {
                    "id": group_id,
                    "member_ids": [
                        str(member["id"]) for member in members
                    ],
                    "retained_shared_prefix_tokens": shared_tokens,
                    "shared_segment": segment,
                }
            )

        allocated_bytes = (
            private_allocated_bytes + shared_allocated_bytes
        )
        phases[phase_name] = {
            "request_count": len(normalized),
            "base_dtype_logical_bytes": base_dtype_logical_bytes,
            "storage_logical_bytes": storage_logical_bytes,
            "allocated_bytes": allocated_bytes,
            "without_prefix_cache_bytes": without_prefix_bytes,
            "without_prefix_storage_logical_bytes": (
                without_prefix_storage_logical_bytes
            ),
            "prefix_cache_savings_bytes": (
                without_prefix_bytes - allocated_bytes
            ),
            "logical_prefix_sharing_savings_bytes": (
                without_prefix_storage_logical_bytes
                - storage_logical_bytes
            ),
            "quantization_savings_bytes": quantization_savings_bytes,
            "reservation_overhead_bytes": (
                allocated_bytes - storage_logical_bytes
            ),
            "allocation_utilization_percent": (
                storage_logical_bytes / allocated_bytes * 100
                if allocated_bytes
                else None
            ),
            "shared_allocated_bytes": shared_allocated_bytes,
            "private_allocated_bytes": private_allocated_bytes,
            "prefix_groups": prefix_groups,
            "requests": request_details,
        }

    return {
        "strategy": strategy,
        "request_count": len(normalized),
        "request_ids": [request["id"] for request in normalized],
        "cache_layout": first_baseline["cache_layout"],
        "elements_per_token_per_layer": first_baseline[
            "elements_per_token_per_layer"
        ],
        "bytes_per_token_per_sequence_at_compute_dtype": (
            bytes_per_token
        ),
        "storage_bits_per_element": first_baseline[
            "storage_bits_per_element"
        ],
        "compute_element_bytes": element_bytes,
        "quantized_residual_tokens": (
            quantized_residual_tokens
            if strategy == "quantized"
            else None
        ),
        "block_tokens": block_tokens if strategy == "paged" else None,
        "max_cache_tokens": (
            next(iter(unique_static_capacities))
            if len(unique_static_capacities) == 1
            else None
        ),
        "max_cache_tokens_by_request": static_capacities,
        "window_tokens": window_tokens,
        "prefix_cache_enabled": bool(
            any(request["shared_prefix_tokens"] for request in normalized)
        ),
        "prefill": phases["prefill"],
        "decode": phases["decode"],
        "formula": (
            "sum(shared_prefix_group_segments) + "
            "sum(request_private_segments)"
        ),
        "modeled_overheads": list(
            first_baseline["modeled_overheads"]
        ),
    }


def estimate_serving_plan(
    model_dir: str | Path,
    *,
    active_sequences: int,
    max_batch_size: int,
    prompt_tokens: int,
    generated_tokens: int = 1,
    dtype: str = "auto",
    attention_implementation: str = "sdpa",
    prefill_chunk_tokens: int | None = None,
    shared_prefix_tokens: int = 0,
    kv_cache_strategy: str = "paged",
    kv_cache_bits: int = 4,
    kv_cache_residual_tokens: int = 128,
    kv_cache_block_tokens: int = 16,
    kv_cache_max_tokens: int | None = None,
    kv_cache_window_tokens: int | None = None,
    runtime_overhead_bytes: int = 0,
    scheduler_overhead_bytes_per_sequence: int = 0,
    adapter_dirs: Sequence[str | Path] | None = None,
    draft_model_dir: str | Path | None = None,
    draft_dtype: str = "auto",
    speculative_tokens: int = DEFAULT_SPECULATIVE_TOKENS,
    speculative_acceptance_rate: float | None = None,
    target_profile: str | None = None,
    device_capacity_bytes: int | None = None,
    memory_utilization: float | None = None,
    runtime: str = "generic",
    vllm_kv_cache_memory_bytes: int | None = None,
    vllm_non_kv_cache_memory_bytes: int | None = None,
) -> dict[str, Any]:
    """Plan memory admission for a homogeneous continuous-batching pool."""

    _positive_integer(active_sequences, "active_sequences")
    _positive_integer(max_batch_size, "max_batch_size")
    _positive_integer(prompt_tokens, "prompt_tokens")
    _positive_integer(generated_tokens, "generated_tokens")
    _nonnegative_integer(shared_prefix_tokens, "shared_prefix_tokens")
    _nonnegative_integer(runtime_overhead_bytes, "runtime_overhead_bytes")
    _nonnegative_integer(
        scheduler_overhead_bytes_per_sequence,
        "scheduler_overhead_bytes_per_sequence",
    )
    if prefill_chunk_tokens is not None:
        _positive_integer(
            prefill_chunk_tokens,
            "prefill_chunk_tokens",
        )
    if shared_prefix_tokens > prompt_tokens:
        raise ServingPlanError(
            "shared_prefix_tokens must not exceed prompt_tokens"
        )
    if attention_implementation not in {"eager", "sdpa"}:
        raise ServingPlanError(
            "attention_implementation must be 'eager' or 'sdpa'"
        )
    runtime, memory_utilization = _serving_runtime_configuration(
        runtime=runtime,
        memory_utilization=memory_utilization,
        kv_cache_strategy=kv_cache_strategy,
        draft_model_dir=draft_model_dir,
        vllm_kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
        vllm_non_kv_cache_memory_bytes=(
            vllm_non_kv_cache_memory_bytes
        ),
    )
    if device_capacity_bytes is not None:
        _positive_integer(
            device_capacity_bytes,
            "device_capacity_bytes",
        )
    _validate_speculative_inputs(
        speculative_tokens=speculative_tokens,
        acceptance_rate=speculative_acceptance_rate,
    )

    target = None
    profile_capacity_bytes = None
    if target_profile:
        profile = get_profile(target_profile)
        profile_capacity_bytes = profile.memory_bytes
        target = {
            "id": profile.id,
            "name": profile.name,
            "architecture": profile.architecture,
            "compute_capability": profile.compute_capability_text,
            "memory_bytes": profile.memory_bytes,
            "memory_kind": profile.memory_kind,
            "profile_status": profile.profile_status,
        }
    selected_capacity_bytes = (
        device_capacity_bytes
        if device_capacity_bytes is not None
        else profile_capacity_bytes
    )
    capacity_source = (
        "explicit_device_capacity"
        if device_capacity_bytes is not None
        else "gpu_profile"
        if profile_capacity_bytes is not None
        else "unavailable"
    )

    base = estimate_decoder_inference(
        model_dir,
        batch_size=1,
        prompt_tokens=prompt_tokens,
        generated_tokens=1,
        dtype=dtype,
        use_cache=False,
        attention_implementation=attention_implementation,
        runtime_overhead_bytes=0,
        adapter_dirs=adapter_dirs,
    )
    dimensions = dict(base["model"])
    element_bytes = int(base["inputs"]["element_bytes"])
    parameter_bytes = int(base["memory"]["parameter_bytes"])
    speculative = _load_speculative_draft(
        draft_model_dir=draft_model_dir,
        draft_dtype=draft_dtype,
        prompt_tokens=prompt_tokens,
        attention_implementation=attention_implementation,
        target_dimensions=dimensions,
        target_parameter_bytes=parameter_bytes,
        speculative_tokens=speculative_tokens,
        acceptance_rate=speculative_acceptance_rate,
    )

    batch_memory_cache: dict[int, dict[str, Any]] = {}

    def batch_memory(sequence_count: int) -> dict[str, Any]:
        cached = batch_memory_cache.get(int(sequence_count))
        if cached is not None:
            return cached
        result = _serving_batch_memory(
            dimensions=dimensions,
            parameter_bytes=parameter_bytes,
            active_sequences=sequence_count,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            element_bytes=element_bytes,
            attention_implementation=attention_implementation,
            prefill_chunk_tokens=prefill_chunk_tokens,
            shared_prefix_tokens=shared_prefix_tokens,
            kv_cache_strategy=kv_cache_strategy,
            kv_cache_bits=kv_cache_bits,
            kv_cache_residual_tokens=kv_cache_residual_tokens,
            kv_cache_block_tokens=kv_cache_block_tokens,
            kv_cache_max_tokens=kv_cache_max_tokens,
            kv_cache_window_tokens=kv_cache_window_tokens,
            runtime_overhead_bytes=runtime_overhead_bytes,
            scheduler_overhead_bytes_per_sequence=(
                scheduler_overhead_bytes_per_sequence
            ),
            speculative=speculative,
        )
        batch_memory_cache[int(sequence_count)] = result
        return result

    requested = batch_memory(active_sequences)
    runtime_report = None
    if runtime == "vllm":
        runtime_report = _build_vllm_runtime_report(
            requested_memory=requested,
            profiled_memory=batch_memory(max_batch_size),
            selected_capacity_bytes=selected_capacity_bytes,
            gpu_memory_utilization=memory_utilization,
            kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
            non_kv_cache_memory_bytes=(
                vllm_non_kv_cache_memory_bytes
            ),
            max_model_len_tokens=prompt_tokens + generated_tokens,
            profiled_sequence_count=max_batch_size,
            profile_scope="configured_max_batch_size",
        )
        requested_for_report = _apply_vllm_kv_reservation(
            requested,
            runtime_report,
        )
        usable_capacity_bytes = runtime_report["memory"][
            "memory_limit_bytes"
        ]
        kv_cache_capacity_bytes = runtime_report["kv_cache"][
            "allocatable_bytes"
        ]
        if kv_cache_capacity_bytes is None:
            memory_limited_sequences = None
        elif runtime_report["memory"]["initialization_fits"] is False:
            memory_limited_sequences = 0
        else:
            memory_limited_sequences = _maximum_fitting_vllm_kv_count(
                max_count=max_batch_size,
                kv_cache_capacity_bytes=int(kv_cache_capacity_bytes),
                memory_for_count=batch_memory,
            )
        fits_memory = runtime_report["requested_fits_memory"]
    else:
        requested_for_report = requested
        usable_capacity_bytes = (
            math.floor(
                int(selected_capacity_bytes) * memory_utilization
            )
            if selected_capacity_bytes is not None
            else None
        )
        memory_limited_sequences = None
        if usable_capacity_bytes is not None:
            memory_limited_sequences = _maximum_fitting_sequences(
                max_batch_size=max_batch_size,
                usable_capacity_bytes=usable_capacity_bytes,
                batch_memory=batch_memory,
            )
        fits_memory = (
            int(requested["peak_bytes"]) <= usable_capacity_bytes
            if usable_capacity_bytes is not None
            else None
        )
    admissible_sequences = (
        min(max_batch_size, memory_limited_sequences)
        if memory_limited_sequences is not None
        else None
    )
    fits_configured_limit = active_sequences <= max_batch_size
    requested_fits = (
        fits_configured_limit and fits_memory
        if fits_memory is not None
        else False
        if not fits_configured_limit
        else None
    )
    available_slots = (
        max(0, admissible_sequences - active_sequences)
        if admissible_sequences is not None
        else None
    )
    if not fits_configured_limit:
        limiting_factor = "configured_max_batch_size"
    elif runtime == "vllm" and runtime_report is not None:
        if runtime_report["memory"]["initialization_fits"] is False:
            limiting_factor = "vllm_model_executor_budget"
        elif fits_memory is False:
            limiting_factor = "vllm_kv_cache_capacity"
        elif fits_memory is None:
            limiting_factor = "device_capacity_unavailable"
        elif admissible_sequences == max_batch_size:
            limiting_factor = "configured_max_batch_size"
        else:
            limiting_factor = "vllm_kv_cache_capacity"
    elif fits_memory is False:
        limiting_factor = "usable_device_memory"
    elif fits_memory is None:
        limiting_factor = "device_capacity_unavailable"
    elif admissible_sequences == max_batch_size:
        limiting_factor = "configured_max_batch_size"
    else:
        limiting_factor = "usable_device_memory"

    input_report = {
        "active_sequences": active_sequences,
        "max_batch_size": max_batch_size,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "dtype": base["inputs"]["dtype"],
        "element_bytes": element_bytes,
        "attention_implementation": attention_implementation,
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "shared_prefix_tokens": shared_prefix_tokens,
        "kv_cache_strategy": kv_cache_strategy,
        "kv_cache_bits": (
            kv_cache_bits if kv_cache_strategy == "quantized" else None
        ),
        "kv_cache_residual_tokens": (
            kv_cache_residual_tokens
            if kv_cache_strategy == "quantized"
            else None
        ),
        "kv_cache_block_tokens": (
            kv_cache_block_tokens if kv_cache_strategy == "paged" else None
        ),
        "kv_cache_max_tokens": kv_cache_max_tokens,
        "kv_cache_window_tokens": kv_cache_window_tokens,
        "runtime": runtime,
        "vllm": (
            {
                "gpu_memory_utilization": memory_utilization,
                "kv_cache_memory_bytes": (
                    vllm_kv_cache_memory_bytes
                ),
                "non_kv_cache_memory_bytes": (
                    vllm_non_kv_cache_memory_bytes
                ),
            }
            if runtime == "vllm"
            else None
        ),
        "runtime_overhead_bytes": runtime_overhead_bytes,
        "scheduler_overhead_bytes_per_sequence": (
            scheduler_overhead_bytes_per_sequence
        ),
        "speculative_decoding": _speculative_input_report(
            speculative=speculative,
            speculative_tokens=speculative_tokens,
            acceptance_rate=speculative_acceptance_rate,
        ),
    }
    workload_signature = _serving_workload_signature(
        report_schema_version=SCHEMA_VERSION,
        model=dimensions,
        checkpoint=base["checkpoint"],
        weight_storage=base["weight_storage"],
        inputs=input_report,
        speculative=speculative,
    )
    return _build_serving_report(
        schema_version=SCHEMA_VERSION,
        workload_signature=workload_signature,
        method_base=(
            "safetensors_headers_plus_decoder_shape_and_serving_pool_model"
        ),
        timeline_source_base="decoder_shape_and_serving_pool_model",
        accuracy_reason=(
            "No matching online-serving GPU observation was supplied."
        ),
        model=dimensions,
        checkpoint=base["checkpoint"],
        weight_storage=base["weight_storage"],
        inputs=input_report,
        runtime=runtime,
        runtime_report=runtime_report,
        target_profile=target,
        capacity_source=capacity_source,
        selected_capacity_bytes=selected_capacity_bytes,
        memory_utilization=memory_utilization,
        usable_capacity_bytes=usable_capacity_bytes,
        scheduler={
            "policy": "continuous_batching",
            "requested_active_sequences": active_sequences,
            "configured_max_batch_size": max_batch_size,
            "memory_limited_active_sequences": (
                memory_limited_sequences
            ),
            "admissible_active_sequences": admissible_sequences,
            "available_slots": available_slots,
            "fits_configured_limit": fits_configured_limit,
            "fits_memory": fits_memory,
            "requested_fits": requested_fits,
            "limiting_factor": limiting_factor,
        },
        requested=requested,
        requested_for_report=requested_for_report,
        speculative=speculative,
        mode_unmodeled_components=[
            "request_length_distribution_and_batch_turnover",
            "scheduler_queueing_and_preemption",
        ],
        notes=[
            (
                "The active pool is homogeneous: every sequence uses the "
                "same prompt and generation lengths."
            ),
            (
                "Chunked prefill bounds query-side transient tensors while "
                "the final chunk attends to the modeled effective context."
            ),
            (
                "Prefix caching shares retained prefix KV blocks; paged "
                "shared and private segments are rounded independently."
            ),
            (
                "Admission uses the configured memory-utilization fraction "
                "and does not predict latency, throughput, or queue time."
            ),
        ],
        vllm_note=(
            "vLLM mode profiles non-KV memory at the configured maximum "
            "batch size, reserves the remaining executor budget as whole "
            "paged-cache blocks, and admits by logical KV demand."
        ),
        speculative_note=(
            "Speculative decoding keeps target and draft weights and KV "
            "caches resident; a supplied acceptance assumption affects "
            "expected target calls but not the conservative memory peak."
        ),
    )


def estimate_serving_request_set(
    model_dir: str | Path,
    requests: Sequence[Mapping[str, Any]],
    *,
    max_batch_size: int,
    dtype: str = "auto",
    attention_implementation: str = "sdpa",
    prefill_chunk_tokens: int | None = None,
    prefill_concurrency: int = 1,
    kv_cache_strategy: str = "paged",
    kv_cache_bits: int = 4,
    kv_cache_residual_tokens: int = 128,
    kv_cache_block_tokens: int = 16,
    kv_cache_max_tokens: int | None = None,
    kv_cache_window_tokens: int | None = None,
    runtime_overhead_bytes: int = 0,
    scheduler_overhead_bytes_per_sequence: int = 0,
    adapter_dirs: Sequence[str | Path] | None = None,
    draft_model_dir: str | Path | None = None,
    draft_dtype: str = "auto",
    speculative_tokens: int = DEFAULT_SPECULATIVE_TOKENS,
    speculative_acceptance_rate: float | None = None,
    target_profile: str | None = None,
    device_capacity_bytes: int | None = None,
    memory_utilization: float | None = None,
    runtime: str = "generic",
    vllm_kv_cache_memory_bytes: int | None = None,
    vllm_non_kv_cache_memory_bytes: int | None = None,
) -> dict[str, Any]:
    """Plan ordered admission for a heterogeneous serving request set."""

    normalized = _normalize_serving_requests(requests)
    _positive_integer(max_batch_size, "max_batch_size")
    _positive_integer(prefill_concurrency, "prefill_concurrency")
    _nonnegative_integer(runtime_overhead_bytes, "runtime_overhead_bytes")
    _nonnegative_integer(
        scheduler_overhead_bytes_per_sequence,
        "scheduler_overhead_bytes_per_sequence",
    )
    if prefill_chunk_tokens is not None:
        _positive_integer(
            prefill_chunk_tokens,
            "prefill_chunk_tokens",
        )
    if attention_implementation not in {"eager", "sdpa"}:
        raise ServingPlanError(
            "attention_implementation must be 'eager' or 'sdpa'"
        )
    runtime, memory_utilization = _serving_runtime_configuration(
        runtime=runtime,
        memory_utilization=memory_utilization,
        kv_cache_strategy=kv_cache_strategy,
        draft_model_dir=draft_model_dir,
        vllm_kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
        vllm_non_kv_cache_memory_bytes=(
            vllm_non_kv_cache_memory_bytes
        ),
    )
    if device_capacity_bytes is not None:
        _positive_integer(
            device_capacity_bytes,
            "device_capacity_bytes",
        )
    _validate_speculative_inputs(
        speculative_tokens=speculative_tokens,
        acceptance_rate=speculative_acceptance_rate,
    )

    target = None
    profile_capacity_bytes = None
    if target_profile:
        profile = get_profile(target_profile)
        profile_capacity_bytes = profile.memory_bytes
        target = {
            "id": profile.id,
            "name": profile.name,
            "architecture": profile.architecture,
            "compute_capability": profile.compute_capability_text,
            "memory_bytes": profile.memory_bytes,
            "memory_kind": profile.memory_kind,
            "profile_status": profile.profile_status,
        }
    selected_capacity_bytes = (
        device_capacity_bytes
        if device_capacity_bytes is not None
        else profile_capacity_bytes
    )
    capacity_source = (
        "explicit_device_capacity"
        if device_capacity_bytes is not None
        else "gpu_profile"
        if profile_capacity_bytes is not None
        else "unavailable"
    )

    base = estimate_decoder_inference(
        model_dir,
        batch_size=1,
        prompt_tokens=max(
            request["prompt_tokens"] for request in normalized
        ),
        generated_tokens=1,
        dtype=dtype,
        use_cache=False,
        attention_implementation=attention_implementation,
        runtime_overhead_bytes=0,
        adapter_dirs=adapter_dirs,
    )
    dimensions = dict(base["model"])
    element_bytes = int(base["inputs"]["element_bytes"])
    parameter_bytes = int(base["memory"]["parameter_bytes"])
    speculative = _load_speculative_draft(
        draft_model_dir=draft_model_dir,
        draft_dtype=draft_dtype,
        prompt_tokens=max(
            request["prompt_tokens"] for request in normalized
        ),
        attention_implementation=attention_implementation,
        target_dimensions=dimensions,
        target_parameter_bytes=parameter_bytes,
        speculative_tokens=speculative_tokens,
        acceptance_rate=speculative_acceptance_rate,
    )

    memory_by_request_count: dict[int, dict[str, Any]] = {}

    def request_memory(request_count: int) -> dict[str, Any]:
        cached = memory_by_request_count.get(request_count)
        if cached is not None:
            return cached
        result = _serving_request_set_memory(
            dimensions=dimensions,
            parameter_bytes=parameter_bytes,
            requests=normalized[:request_count],
            element_bytes=element_bytes,
            attention_implementation=attention_implementation,
            prefill_chunk_tokens=prefill_chunk_tokens,
            prefill_concurrency=prefill_concurrency,
            kv_cache_strategy=kv_cache_strategy,
            kv_cache_bits=kv_cache_bits,
            kv_cache_residual_tokens=kv_cache_residual_tokens,
            kv_cache_block_tokens=kv_cache_block_tokens,
            kv_cache_max_tokens=kv_cache_max_tokens,
            kv_cache_window_tokens=kv_cache_window_tokens,
            runtime_overhead_bytes=runtime_overhead_bytes,
            scheduler_overhead_bytes_per_sequence=(
                scheduler_overhead_bytes_per_sequence
            ),
            speculative=speculative,
        )
        memory_by_request_count[request_count] = result
        return result

    requested_count = len(normalized)
    requested = request_memory(requested_count)
    candidate_count = min(max_batch_size, requested_count)
    runtime_report = None
    if runtime == "vllm":
        candidate_requests = normalized[:candidate_count]
        runtime_report = _build_vllm_runtime_report(
            requested_memory=requested,
            profiled_memory=request_memory(candidate_count),
            selected_capacity_bytes=selected_capacity_bytes,
            gpu_memory_utilization=memory_utilization,
            kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
            non_kv_cache_memory_bytes=(
                vllm_non_kv_cache_memory_bytes
            ),
            max_model_len_tokens=max(
                int(request["prompt_tokens"])
                + int(request["generated_tokens"])
                for request in candidate_requests
            ),
            profiled_sequence_count=candidate_count,
            profile_scope="configured_request_manifest_prefix",
        )
        requested_for_report = _apply_vllm_kv_reservation(
            requested,
            runtime_report,
        )
        usable_capacity_bytes = runtime_report["memory"][
            "memory_limit_bytes"
        ]
        kv_cache_capacity_bytes = runtime_report["kv_cache"][
            "allocatable_bytes"
        ]
        if kv_cache_capacity_bytes is None:
            memory_limited_count = None
        elif runtime_report["memory"]["initialization_fits"] is False:
            memory_limited_count = 0
        else:
            memory_limited_count = _maximum_fitting_vllm_kv_count(
                max_count=candidate_count,
                kv_cache_capacity_bytes=int(kv_cache_capacity_bytes),
                memory_for_count=request_memory,
            )
        fits_memory = runtime_report["requested_fits_memory"]
    else:
        requested_for_report = requested
        usable_capacity_bytes = (
            math.floor(
                int(selected_capacity_bytes) * memory_utilization
            )
            if selected_capacity_bytes is not None
            else None
        )
        memory_limited_count = None
        if usable_capacity_bytes is not None:
            memory_limited_count = _maximum_fitting_request_prefix(
                max_request_count=candidate_count,
                usable_capacity_bytes=usable_capacity_bytes,
                request_memory=request_memory,
            )
        fits_memory = (
            int(requested["peak_bytes"]) <= usable_capacity_bytes
            if usable_capacity_bytes is not None
            else None
        )
    admissible_count = memory_limited_count
    fits_configured_limit = requested_count <= max_batch_size
    requested_fits = (
        fits_configured_limit and fits_memory
        if fits_memory is not None
        else False
        if not fits_configured_limit
        else None
    )
    if not fits_configured_limit:
        limiting_factor = "configured_max_batch_size"
    elif runtime == "vllm" and runtime_report is not None:
        if runtime_report["memory"]["initialization_fits"] is False:
            limiting_factor = "vllm_model_executor_budget"
        elif fits_memory is False:
            limiting_factor = "vllm_kv_cache_capacity"
        elif fits_memory is None:
            limiting_factor = "device_capacity_unavailable"
        elif admissible_count == candidate_count:
            limiting_factor = "request_manifest_exhausted"
        else:
            limiting_factor = "vllm_kv_cache_capacity"
    elif fits_memory is False:
        limiting_factor = "usable_device_memory"
    elif fits_memory is None:
        limiting_factor = "device_capacity_unavailable"
    elif admissible_count == candidate_count:
        limiting_factor = "request_manifest_exhausted"
    else:
        limiting_factor = "usable_device_memory"

    admitted_request_ids = (
        [
            request["id"]
            for request in normalized[:admissible_count]
        ]
        if admissible_count is not None
        else None
    )
    rejected_request_ids = (
        [
            request["id"]
            for request in normalized[admissible_count:]
        ]
        if admissible_count is not None
        else None
    )

    input_report = {
        "mode": "heterogeneous_request_set",
        "requests": normalized,
        "active_sequences": requested_count,
        "max_batch_size": max_batch_size,
        "dtype": base["inputs"]["dtype"],
        "element_bytes": element_bytes,
        "attention_implementation": attention_implementation,
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "prefill_concurrency": prefill_concurrency,
        "kv_cache_strategy": kv_cache_strategy,
        "kv_cache_bits": (
            kv_cache_bits if kv_cache_strategy == "quantized" else None
        ),
        "kv_cache_residual_tokens": (
            kv_cache_residual_tokens
            if kv_cache_strategy == "quantized"
            else None
        ),
        "kv_cache_block_tokens": (
            kv_cache_block_tokens if kv_cache_strategy == "paged" else None
        ),
        "kv_cache_max_tokens": kv_cache_max_tokens,
        "kv_cache_window_tokens": kv_cache_window_tokens,
        "runtime": runtime,
        "vllm": (
            {
                "gpu_memory_utilization": memory_utilization,
                "kv_cache_memory_bytes": (
                    vllm_kv_cache_memory_bytes
                ),
                "non_kv_cache_memory_bytes": (
                    vllm_non_kv_cache_memory_bytes
                ),
            }
            if runtime == "vllm"
            else None
        ),
        "runtime_overhead_bytes": runtime_overhead_bytes,
        "scheduler_overhead_bytes_per_sequence": (
            scheduler_overhead_bytes_per_sequence
        ),
        "speculative_decoding": _speculative_input_report(
            speculative=speculative,
            speculative_tokens=speculative_tokens,
            acceptance_rate=speculative_acceptance_rate,
        ),
    }
    workload_signature = _serving_workload_signature(
        report_schema_version=REQUEST_SET_SCHEMA_VERSION,
        model=dimensions,
        checkpoint=base["checkpoint"],
        weight_storage=base["weight_storage"],
        inputs=input_report,
        speculative=speculative,
    )
    return _build_serving_report(
        schema_version=REQUEST_SET_SCHEMA_VERSION,
        workload_signature=workload_signature,
        method_base=(
            "safetensors_headers_plus_decoder_shape_and_heterogeneous_"
            "serving_request_model"
        ),
        timeline_source_base=(
            "decoder_shape_and_heterogeneous_serving_request_model"
        ),
        accuracy_reason=(
            "No matching heterogeneous online-serving GPU observation "
            "was supplied."
        ),
        model=dimensions,
        checkpoint=base["checkpoint"],
        weight_storage=base["weight_storage"],
        inputs=input_report,
        runtime=runtime,
        runtime_report=runtime_report,
        target_profile=target,
        capacity_source=capacity_source,
        selected_capacity_bytes=selected_capacity_bytes,
        memory_utilization=memory_utilization,
        usable_capacity_bytes=usable_capacity_bytes,
        scheduler={
            "policy": "continuous_batching",
            "admission_order": "request_manifest_order",
            "admission_scope": "request_manifest_only",
            "requested_active_sequences": requested_count,
            "configured_max_batch_size": max_batch_size,
            "configured_candidate_request_ids": [
                request["id"]
                for request in normalized[:candidate_count]
            ],
            "memory_limited_active_sequences": (
                memory_limited_count
            ),
            "admissible_active_sequences": admissible_count,
            "admitted_request_ids": admitted_request_ids,
            "rejected_request_ids": rejected_request_ids,
            "admitted_request_count": (
                len(admitted_request_ids)
                if admitted_request_ids is not None
                else None
            ),
            "rejected_request_count": (
                len(rejected_request_ids)
                if rejected_request_ids is not None
                else None
            ),
            "available_slots": None,
            "available_slots_reason": (
                "future_request_shapes_unavailable"
            ),
            "fits_configured_limit": fits_configured_limit,
            "fits_memory": fits_memory,
            "requested_fits": requested_fits,
            "limiting_factor": limiting_factor,
        },
        requested=requested,
        requested_for_report=requested_for_report,
        speculative=speculative,
        mode_unmodeled_components=[
            "request_arrival_timing_and_batch_turnover",
            "scheduler_queueing_preemption_and_reordering",
        ],
        notes=[
            (
                "Each request keeps its own prompt and generation lengths; "
                "shared KV storage is limited to matching prefix groups."
            ),
            (
                "Named prefix groups are modeled as resident cache hits; "
                "population, lookup, and eviction behavior is unmodeled."
            ),
            (
                "Chunked-prefill memory uses the largest component-wise "
                "transient envelope at the configured concurrency."
            ),
            (
                "Admission accepts the longest fitting prefix of the "
                "manifest and does not reorder requests."
            ),
            (
                "Capacity admission does not predict latency, throughput, "
                "queue time, or cache hit probability."
            ),
        ],
        vllm_note=(
            "vLLM mode profiles the configured manifest prefix, reserves "
            "the resulting paged KV pool, and admits the longest request "
            "prefix whose logical cache demand fits that pool."
        ),
        speculative_note=(
            "Speculative decoding applies one draft configuration to every "
            "request while retaining each request's effective "
            "generation-length bound."
        ),
    )


def _build_serving_report(
    *,
    schema_version: str,
    workload_signature: str,
    method_base: str,
    timeline_source_base: str,
    accuracy_reason: str,
    model: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    weight_storage: Mapping[str, Any],
    inputs: Mapping[str, Any],
    runtime: str,
    runtime_report: Mapping[str, Any] | None,
    target_profile: Mapping[str, Any] | None,
    capacity_source: str,
    selected_capacity_bytes: int | None,
    memory_utilization: float,
    usable_capacity_bytes: int | None,
    scheduler: Mapping[str, Any],
    requested: Mapping[str, Any],
    requested_for_report: Mapping[str, Any],
    speculative: Mapping[str, Any] | None,
    mode_unmodeled_components: Sequence[str],
    notes: Sequence[str],
    vllm_note: str,
    speculative_note: str,
) -> dict[str, Any]:
    """Build the shared report envelope for both serving-plan modes."""
    speculative_suffix = (
        "_with_draft_model_speculative_decoding"
        if speculative is not None
        else ""
    )
    runtime_suffix = (
        "_with_vllm_runtime_budget" if runtime == "vllm" else ""
    )
    kv_cache_report = dict(requested["kv_cache"])
    if runtime_report is not None:
        kv_cache_report["runtime_reservation"] = dict(
            runtime_report["kv_cache"]
        )
    runtime_output = (
        runtime_report
        if runtime_report is not None
        else {
            "engine": "generic",
            "method": "logical_kv_growth_memory_model",
        }
    )
    unmodeled_components = [
        "cuda_context_and_loaded_modules",
        "caching_allocator_fragmentation",
        "backend_kernel_and_attention_workspaces",
        *mode_unmodeled_components,
        "paged_cache_block_table_metadata",
        "prefix_cache_lookup_eviction_and_copy_cost",
        "network_and_tokenization_buffers",
        *(
            [
                "speculative_acceptance_distribution_and_dynamic_draft_length",
                "speculative_kv_rollback_and_scheduler_workspace",
                "target_draft_kernel_latency_and_overlap",
                "draft_target_tokenizer_identity",
            ]
            if speculative is not None
            else ["speculative_draft_model_weights_and_kv_cache"]
        ),
        "tensor_parallel_communication_and_rank_imbalance",
    ]
    report_notes = list(notes)
    if runtime == "vllm":
        report_notes.append(vllm_note)
    if speculative is not None:
        report_notes.append(speculative_note)

    return {
        "schema_version": schema_version,
        "workload_signature": workload_signature,
        "method": method_base + speculative_suffix + runtime_suffix,
        "validation_status": "Modeled",
        "accuracy": {
            "status": "uncalibrated",
            "prediction_interval_bytes": None,
            "error_percent": None,
            "reason": accuracy_reason,
        },
        "model": model,
        "checkpoint": checkpoint,
        "weight_storage": weight_storage,
        "inputs": inputs,
        "runtime": runtime_output,
        "target": {
            "profile": target_profile,
            "capacity_source": capacity_source,
            "device_capacity_bytes": selected_capacity_bytes,
            "memory_utilization": float(memory_utilization),
            "usable_capacity_bytes": usable_capacity_bytes,
        },
        "scheduler": dict(scheduler),
        "kv_cache": kv_cache_report,
        "speculative_decoding": requested["speculative_decoding"],
        "optimizations": requested["optimizations"],
        "memory": requested_for_report["memory"],
        "memory_timeline": {
            "unit": "bytes",
            "source": timeline_source_base + speculative_suffix + runtime_suffix,
            "phases": requested_for_report["phases"],
            "peak_bytes": requested_for_report["peak_bytes"],
            "peak_phase": requested_for_report["peak_phase"],
            "usable_capacity_headroom_bytes": (
                usable_capacity_bytes
                - int(requested_for_report["peak_bytes"])
                if usable_capacity_bytes is not None
                else None
            ),
        },
        "unmodeled_components": unmodeled_components,
        "notes": report_notes,
    }


def _speculative_input_report(
    *,
    speculative: Mapping[str, Any] | None,
    speculative_tokens: int,
    acceptance_rate: float | None,
) -> dict[str, Any]:
    return {
        "enabled": speculative is not None,
        "draft_model_dir": (
            speculative["model_dir"]
            if speculative is not None
            else None
        ),
        "draft_dtype": (
            speculative["dtype"] if speculative is not None else None
        ),
        "proposal_tokens_per_step": (
            speculative_tokens if speculative is not None else None
        ),
        "acceptance_rate": (
            float(acceptance_rate)
            if speculative is not None and acceptance_rate is not None
            else None
        ),
    }


def _serving_workload_signature(
    *,
    report_schema_version: str,
    model: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    weight_storage: Mapping[str, Any],
    inputs: Mapping[str, Any],
    speculative: Mapping[str, Any] | None,
) -> str:
    portable_inputs = dict(inputs)
    speculative_inputs = portable_inputs.get("speculative_decoding")
    if isinstance(speculative_inputs, Mapping):
        portable_speculative_inputs = dict(speculative_inputs)
        portable_speculative_inputs.pop("draft_model_dir", None)
        portable_inputs["speculative_decoding"] = (
            portable_speculative_inputs
        )

    draft_identity = None
    if speculative is not None:
        draft_identity = {
            "model": _portable_model_identity(speculative["model"]),
            "checkpoint": dict(speculative["checkpoint"]),
            "weight_storage": _portable_weight_storage(
                speculative["weight_storage"]
            ),
        }
    identity = {
        "signature_schema_version": WORKLOAD_SIGNATURE_SCHEMA_VERSION,
        "report_schema_version": report_schema_version,
        "target": {
            "model": _portable_model_identity(model),
            "checkpoint": dict(checkpoint),
            "weight_storage": _portable_weight_storage(weight_storage),
        },
        "inputs": portable_inputs,
        "draft": draft_identity,
    }
    canonical = json.dumps(
        identity,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return (
        f"{WORKLOAD_SIGNATURE_SCHEMA_VERSION}:sha256:{digest}"
    )


def _portable_model_identity(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in model.items()
        if key != "path"
    }


def _portable_weight_storage(
    weight_storage: Mapping[str, Any],
) -> dict[str, Any]:
    portable = {
        key: value
        for key, value in weight_storage.items()
        if key != "adapters"
    }
    adapters = weight_storage.get("adapters")
    if isinstance(adapters, list):
        portable["adapters"] = [
            {
                key: value
                for key, value in adapter.items()
                if key != "path"
            }
            for adapter in adapters
            if isinstance(adapter, Mapping)
        ]
    return portable


def _validate_speculative_inputs(
    *,
    speculative_tokens: int,
    acceptance_rate: float | None,
) -> None:
    _positive_integer(speculative_tokens, "speculative_tokens")
    if acceptance_rate is None:
        return
    if (
        not isinstance(acceptance_rate, (int, float))
        or isinstance(acceptance_rate, bool)
        or not math.isfinite(float(acceptance_rate))
        or not 0 <= float(acceptance_rate) <= 1
    ):
        raise ServingPlanError(
            "speculative_acceptance_rate must be finite and in the "
            "interval [0, 1]"
        )


def _serving_runtime_configuration(
    *,
    runtime: str,
    memory_utilization: float | None,
    kv_cache_strategy: str,
    draft_model_dir: str | Path | None,
    vllm_kv_cache_memory_bytes: int | None,
    vllm_non_kv_cache_memory_bytes: int | None,
) -> tuple[str, float]:
    normalized_runtime = str(runtime).strip().lower()
    if normalized_runtime not in SERVING_RUNTIMES:
        choices = ", ".join(sorted(SERVING_RUNTIMES))
        raise ServingPlanError(
            f"runtime must be one of: {choices}"
        )
    resolved_utilization = (
        VLLM_DEFAULT_GPU_MEMORY_UTILIZATION
        if memory_utilization is None and normalized_runtime == "vllm"
        else DEFAULT_MEMORY_UTILIZATION
        if memory_utilization is None
        else memory_utilization
    )
    if (
        not isinstance(resolved_utilization, (int, float))
        or isinstance(resolved_utilization, bool)
        or not math.isfinite(float(resolved_utilization))
        or not 0 < float(resolved_utilization) <= 1
    ):
        raise ServingPlanError(
            "memory_utilization must be finite and in the interval (0, 1]"
        )

    vllm_values = {
        "vllm_kv_cache_memory_bytes": vllm_kv_cache_memory_bytes,
        "vllm_non_kv_cache_memory_bytes": (
            vllm_non_kv_cache_memory_bytes
        ),
    }
    if normalized_runtime != "vllm":
        configured = [
            name for name, value in vllm_values.items() if value is not None
        ]
        if configured:
            raise ServingPlanError(
                ", ".join(configured) + " require runtime='vllm'"
            )
        return normalized_runtime, float(resolved_utilization)

    if kv_cache_strategy != "paged":
        raise ServingPlanError(
            "runtime='vllm' requires kv_cache_strategy='paged'"
        )
    if draft_model_dir is not None:
        raise ServingPlanError(
            "runtime='vllm' does not yet model speculative decoding"
        )
    if vllm_kv_cache_memory_bytes is not None:
        _positive_integer(
            vllm_kv_cache_memory_bytes,
            "vllm_kv_cache_memory_bytes",
        )
    if vllm_non_kv_cache_memory_bytes is not None:
        _nonnegative_integer(
            vllm_non_kv_cache_memory_bytes,
            "vllm_non_kv_cache_memory_bytes",
        )
    return normalized_runtime, float(resolved_utilization)


def _load_speculative_draft(
    *,
    draft_model_dir: str | Path | None,
    draft_dtype: str,
    prompt_tokens: int,
    attention_implementation: str,
    target_dimensions: Mapping[str, Any],
    target_parameter_bytes: int,
    speculative_tokens: int,
    acceptance_rate: float | None,
) -> dict[str, Any] | None:
    if draft_model_dir is None:
        return None
    draft = estimate_decoder_inference(
        draft_model_dir,
        batch_size=1,
        prompt_tokens=prompt_tokens,
        generated_tokens=1,
        dtype=draft_dtype,
        use_cache=False,
        attention_implementation=attention_implementation,
        runtime_overhead_bytes=0,
    )
    dimensions = dict(draft["model"])
    target_vocab_size = int(target_dimensions["vocab_size"])
    draft_vocab_size = int(dimensions["vocab_size"])
    if target_vocab_size != draft_vocab_size:
        raise ServingPlanError(
            "draft and target vocab_size must match for shared-tokenizer "
            "speculative decoding"
        )
    parameter_bytes = int(draft["memory"]["parameter_bytes"])
    return {
        "model_dir": str(Path(draft_model_dir).expanduser().resolve()),
        "model": dimensions,
        "checkpoint": draft["checkpoint"],
        "weight_storage": draft["weight_storage"],
        "dtype": draft["inputs"]["dtype"],
        "element_bytes": int(draft["inputs"]["element_bytes"]),
        "parameter_bytes": parameter_bytes,
        "target_parameter_bytes": target_parameter_bytes,
        "weight_bytes_ratio_to_target": (
            parameter_bytes / target_parameter_bytes
            if target_parameter_bytes
            else None
        ),
        "speculative_tokens": speculative_tokens,
        "acceptance_rate": (
            float(acceptance_rate)
            if acceptance_rate is not None
            else None
        ),
    }


def _speculative_acceptance_metrics(
    proposal_tokens: int,
    acceptance_rate: float,
) -> dict[str, Any]:
    expected_accepted = sum(
        acceptance_rate**position
        for position in range(1, proposal_tokens + 1)
    )
    expected_output = 1.0 + expected_accepted
    return {
        "assumption": "independent_constant_per_token_acceptance",
        "rate": acceptance_rate,
        "proposal_tokens_per_step": proposal_tokens,
        "probability_all_draft_tokens_accepted": (
            acceptance_rate**proposal_tokens
        ),
        "expected_accepted_draft_tokens_per_step": expected_accepted,
        "expected_output_tokens_per_target_step": expected_output,
        "analytical_target_forward_reduction_factor": expected_output,
        "interpretation": (
            "target_forward_calls_only_not_end_to_end_speedup"
        ),
    }


def _select_larger_transient(
    target: Mapping[str, Any],
    draft: Mapping[str, Any],
) -> dict[str, Any]:
    if int(target["peak_bytes"]) >= int(draft["peak_bytes"]):
        return {**target, "source": "target"}
    return {**draft, "source": "draft"}


def _speculative_report(
    *,
    speculative: Mapping[str, Any] | None,
    effective_proposal_tokens: Mapping[str, int],
    target_kv_cache: Mapping[str, Any] | None,
    draft_kv_cache: Mapping[str, Any] | None,
    target_prefill_transient: Mapping[str, Any] | None,
    draft_prefill_transient: Mapping[str, Any] | None,
    target_verification_transient: Mapping[str, Any] | None,
    draft_proposal_transient: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if speculative is None:
        return {"enabled": False}
    if (
        target_kv_cache is None
        or draft_kv_cache is None
        or target_prefill_transient is None
        or draft_prefill_transient is None
        or target_verification_transient is None
        or draft_proposal_transient is None
    ):
        raise ServingPlanError(
            "incomplete speculative decoding memory components"
        )
    effective_values = list(effective_proposal_tokens.values())
    configured_tokens = int(speculative["speculative_tokens"])
    target_parameter_bytes = int(speculative["target_parameter_bytes"])
    draft_parameter_bytes = int(speculative["parameter_bytes"])
    acceptance_rate = speculative["acceptance_rate"]
    acceptance = (
        {
            "status": "assumed",
            **_speculative_acceptance_metrics(
                max(effective_values),
                float(acceptance_rate),
            ),
        }
        if acceptance_rate is not None
        else {
            "status": "not_provided",
            "assumption": None,
            "rate": None,
            "proposal_tokens_per_step": max(effective_values),
            "probability_all_draft_tokens_accepted": None,
            "expected_accepted_draft_tokens_per_step": None,
            "expected_output_tokens_per_target_step": None,
            "analytical_target_forward_reduction_factor": None,
            "interpretation": (
                "acceptance_observation_required_for_target_call_estimate"
            ),
        }
    )
    return {
        "enabled": True,
        "method": "independent_autoregressive_draft_model",
        "proposal_tokens_per_step": configured_tokens,
        "effective_proposal_tokens": {
            "minimum": min(effective_values),
            "maximum": max(effective_values),
            "by_request": dict(effective_proposal_tokens),
        },
        "acceptance": acceptance,
        "compatibility": {
            "vocabulary_size_matches": True,
            "tokenizer_identity": "unverified",
            "tokenizer_mode": "shared_tokenizer_required",
            "draft_weight_bytes_smaller_than_target": (
                draft_parameter_bytes < target_parameter_bytes
            ),
        },
        "target": {
            "parameter_bytes": target_parameter_bytes,
            "verification_query_tokens": max(effective_values),
            "kv_cache": target_kv_cache,
            "prefill_transient": target_prefill_transient,
            "verification_transient": target_verification_transient,
        },
        "draft": {
            "model_dir": speculative["model_dir"],
            "model": speculative["model"],
            "checkpoint": speculative["checkpoint"],
            "weight_storage": speculative["weight_storage"],
            "dtype": speculative["dtype"],
            "parameter_bytes": draft_parameter_bytes,
            "weight_bytes_ratio_to_target": speculative[
                "weight_bytes_ratio_to_target"
            ],
            "proposal_query_tokens": 1,
            "kv_cache": draft_kv_cache,
            "prefill_transient": draft_prefill_transient,
            "proposal_transient": draft_proposal_transient,
        },
        "memory": {
            "combined_parameter_bytes": (
                target_parameter_bytes + draft_parameter_bytes
            ),
            "additional_parameter_bytes": draft_parameter_bytes,
            "combined_prefill_kv_cache_bytes": (
                int(target_kv_cache["prefill"]["allocated_bytes"])
                + int(draft_kv_cache["prefill"]["allocated_bytes"])
            ),
            "combined_decode_kv_cache_bytes": (
                int(target_kv_cache["decode"]["allocated_bytes"])
                + int(draft_kv_cache["decode"]["allocated_bytes"])
            ),
            "acceptance_rate_changes_peak_bytes": False,
        },
        "performance_status": (
            "analytical_target_call_reduction_only"
            if acceptance_rate is not None
            else "not_estimated_acceptance_not_provided"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__),
        description=(
            "Estimate online LLM serving memory and continuous-batching "
            "admission without loading checkpoint tensors."
        ),
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--requests",
        help=(
            "JSON, TOML, or YAML heterogeneous request manifest. "
            "Cannot be combined with homogeneous request-shape flags."
        ),
    )
    parser.add_argument("--active-sequences", type=int)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--generated-tokens", type=int)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--attention-implementation",
        choices=["eager", "sdpa"],
        default="sdpa",
    )
    parser.add_argument("--prefill-chunk-tokens", type=int)
    parser.add_argument("--prefill-concurrency", type=int, default=1)
    parser.add_argument("--shared-prefix-tokens", type=int)
    parser.add_argument(
        "--kv-cache-strategy",
        choices=sorted(KV_CACHE_STRATEGIES),
        default="paged",
    )
    parser.add_argument(
        "--kv-cache-bits",
        type=int,
        choices=[2, 4, 8],
        default=4,
    )
    parser.add_argument(
        "--kv-cache-residual-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--kv-cache-block-tokens",
        type=int,
        default=16,
    )
    parser.add_argument("--kv-cache-max-tokens", type=int)
    parser.add_argument("--kv-cache-window-tokens", type=int)
    parser.add_argument("--runtime-overhead-bytes", type=int, default=0)
    parser.add_argument(
        "--scheduler-overhead-bytes-per-sequence",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--adapter-dir",
        action="append",
        default=[],
        help="PEFT/LoRA adapter directory; may be repeated.",
    )
    parser.add_argument(
        "--runtime",
        choices=sorted(SERVING_RUNTIMES),
        default="generic",
        help=(
            "Serving memory policy. vllm models executor budgeting and "
            "preallocated paged KV cache."
        ),
    )
    parser.add_argument(
        "--vllm-kv-cache-memory-bytes",
        type=int,
        help=(
            "Explicit per-GPU vLLM KV cache size; overrides utilization-"
            "based automatic sizing."
        ),
    )
    parser.add_argument(
        "--vllm-non-kv-cache-memory-bytes",
        type=int,
        help=(
            "Measured vLLM profile result for weights, peak activations, "
            "non-torch allocations, and CUDA graphs."
        ),
    )
    parser.add_argument(
        "--draft-model-dir",
        help=(
            "Smaller decoder checkpoint for draft-model speculative "
            "decoding."
        ),
    )
    parser.add_argument(
        "--draft-dtype",
        help="Draft-model compute dtype (default: auto).",
    )
    parser.add_argument(
        "--speculative-tokens",
        type=int,
        help=(
            "Draft proposal length per target verification step "
            f"(default: {DEFAULT_SPECULATIVE_TOKENS})."
        ),
    )
    parser.add_argument(
        "--speculative-acceptance-rate",
        type=float,
        help=(
            "Optional measured or assumed per-token draft acceptance in "
            "[0, 1]."
        ),
    )
    capacity_group = parser.add_mutually_exclusive_group()
    capacity_group.add_argument("--target-profile")
    capacity_group.add_argument(
        "--device-memory-gib",
        type=float,
        help="Explicit device memory capacity in binary GiB.",
    )
    parser.add_argument(
        "--memory-utilization",
        type=float,
        help=(
            "Device budget fraction (default: 0.9 for generic, 0.92 for "
            "vllm)."
        ),
    )
    add_json_path_argument(parser)
    args = parser.parse_args(argv)

    device_capacity_bytes = None
    if args.device_memory_gib is not None:
        if (
            not math.isfinite(args.device_memory_gib)
            or args.device_memory_gib <= 0
        ):
            parser.error("--device-memory-gib must be finite and positive")
        device_capacity_bytes = math.floor(
            args.device_memory_gib * 2**30
        )

    homogeneous_shape_flags = {
        "--active-sequences": args.active_sequences,
        "--prompt-tokens": args.prompt_tokens,
        "--generated-tokens": args.generated_tokens,
        "--shared-prefix-tokens": args.shared_prefix_tokens,
    }
    if args.requests:
        conflicting_flags = [
            flag
            for flag, value in homogeneous_shape_flags.items()
            if value is not None
        ]
        if conflicting_flags:
            parser.error(
                "--requests cannot be combined with "
                + ", ".join(conflicting_flags)
            )
    else:
        missing_flags = [
            flag
            for flag in ("--active-sequences", "--prompt-tokens")
            if homogeneous_shape_flags[flag] is None
        ]
        if missing_flags:
            parser.error(
                "homogeneous mode requires "
                + " and ".join(missing_flags)
            )
        if args.prefill_concurrency != 1:
            parser.error(
                "--prefill-concurrency is available only with --requests"
            )

    speculative_flags = {
        "--draft-dtype": args.draft_dtype,
        "--speculative-tokens": args.speculative_tokens,
        "--speculative-acceptance-rate": (
            args.speculative_acceptance_rate
        ),
    }
    if args.draft_model_dir is None:
        configured_speculative_flags = [
            flag
            for flag, value in speculative_flags.items()
            if value is not None
        ]
        if configured_speculative_flags:
            parser.error(
                ", ".join(configured_speculative_flags)
                + " require --draft-model-dir"
            )
    speculative_tokens = (
        DEFAULT_SPECULATIVE_TOKENS
        if args.speculative_tokens is None
        else args.speculative_tokens
    )
    speculative_acceptance_rate = args.speculative_acceptance_rate

    try:
        if args.requests:
            request_manifest = load_serving_requests(args.requests)
            report = estimate_serving_request_set(
                args.model_dir,
                request_manifest["requests"],
                max_batch_size=args.max_batch_size,
                dtype=args.dtype,
                attention_implementation=(
                    args.attention_implementation
                ),
                prefill_chunk_tokens=args.prefill_chunk_tokens,
                prefill_concurrency=args.prefill_concurrency,
                kv_cache_strategy=args.kv_cache_strategy,
                kv_cache_bits=args.kv_cache_bits,
                kv_cache_residual_tokens=(
                    args.kv_cache_residual_tokens
                ),
                kv_cache_block_tokens=args.kv_cache_block_tokens,
                kv_cache_max_tokens=args.kv_cache_max_tokens,
                kv_cache_window_tokens=(
                    args.kv_cache_window_tokens
                ),
                runtime_overhead_bytes=args.runtime_overhead_bytes,
                scheduler_overhead_bytes_per_sequence=(
                    args.scheduler_overhead_bytes_per_sequence
                ),
                adapter_dirs=args.adapter_dir,
                draft_model_dir=args.draft_model_dir,
                draft_dtype=args.draft_dtype or "auto",
                speculative_tokens=speculative_tokens,
                speculative_acceptance_rate=(
                    speculative_acceptance_rate
                ),
                target_profile=args.target_profile,
                device_capacity_bytes=device_capacity_bytes,
                memory_utilization=args.memory_utilization,
                runtime=args.runtime,
                vllm_kv_cache_memory_bytes=(
                    args.vllm_kv_cache_memory_bytes
                ),
                vllm_non_kv_cache_memory_bytes=(
                    args.vllm_non_kv_cache_memory_bytes
                ),
            )
            report["inputs"]["request_manifest"] = (
                request_manifest["source"]
            )
        else:
            report = estimate_serving_plan(
                args.model_dir,
                active_sequences=args.active_sequences,
                max_batch_size=args.max_batch_size,
                prompt_tokens=args.prompt_tokens,
                generated_tokens=(
                    1
                    if args.generated_tokens is None
                    else args.generated_tokens
                ),
                dtype=args.dtype,
                attention_implementation=(
                    args.attention_implementation
                ),
                prefill_chunk_tokens=args.prefill_chunk_tokens,
                shared_prefix_tokens=(
                    0
                    if args.shared_prefix_tokens is None
                    else args.shared_prefix_tokens
                ),
                kv_cache_strategy=args.kv_cache_strategy,
                kv_cache_bits=args.kv_cache_bits,
                kv_cache_residual_tokens=(
                    args.kv_cache_residual_tokens
                ),
                kv_cache_block_tokens=args.kv_cache_block_tokens,
                kv_cache_max_tokens=args.kv_cache_max_tokens,
                kv_cache_window_tokens=(
                    args.kv_cache_window_tokens
                ),
                runtime_overhead_bytes=args.runtime_overhead_bytes,
                scheduler_overhead_bytes_per_sequence=(
                    args.scheduler_overhead_bytes_per_sequence
                ),
                adapter_dirs=args.adapter_dir,
                draft_model_dir=args.draft_model_dir,
                draft_dtype=args.draft_dtype or "auto",
                speculative_tokens=speculative_tokens,
                speculative_acceptance_rate=(
                    speculative_acceptance_rate
                ),
                target_profile=args.target_profile,
                device_capacity_bytes=device_capacity_bytes,
                memory_utilization=args.memory_utilization,
                runtime=args.runtime,
                vllm_kv_cache_memory_bytes=(
                    args.vllm_kv_cache_memory_bytes
                ),
                vllm_non_kv_cache_memory_bytes=(
                    args.vllm_non_kv_cache_memory_bytes
                ),
            )
    except (
            OSError,
            ValueError,
           ) as exc:
        usage_error(parser, exc)

    if args.json_path:
        output = emit_json(args.json_path, report)
        if output is not None:
            print(f"Serving plan: {output}")

    scheduler = report["scheduler"]
    timeline = report["memory_timeline"]
    print("FakeGPU LLM serving plan")
    print(
        "  active sequences: "
        f"{scheduler['requested_active_sequences']}"
    )
    print(f"  peak memory: {_format_bytes(timeline['peak_bytes'])}")
    print(f"  peak phase: {timeline['peak_phase']}")
    print(
        "  KV cache: "
        f"{report['kv_cache']['strategy']} "
        f"({_format_bytes(report['kv_cache']['decode']['allocated_bytes'])})"
    )
    runtime_report = report["runtime"]
    if runtime_report["engine"] == "vllm":
        runtime_kv_cache = runtime_report["kv_cache"]
        if runtime_kv_cache["allocatable_bytes"] is None:
            print("  vLLM reserved KV cache: device capacity unavailable")
        else:
            print(
                "  vLLM reserved KV cache: "
                f"{_format_bytes(runtime_kv_cache['allocatable_bytes'])} "
                f"({runtime_kv_cache['num_gpu_blocks']} blocks)"
            )
    speculative_report = report["speculative_decoding"]
    if speculative_report["enabled"]:
        print(
            "  speculative draft: "
            f"{_format_bytes(speculative_report['draft']['parameter_bytes'])}, "
            f"{speculative_report['proposal_tokens_per_step']} tokens"
        )
        expected_tokens = speculative_report["acceptance"][
            "expected_output_tokens_per_target_step"
        ]
        if expected_tokens is None:
            print("  expected tokens per target step: acceptance unavailable")
        else:
            print(
                "  expected tokens per target step: "
                f"{expected_tokens:.3f}"
            )
    if scheduler["admissible_active_sequences"] is None:
        print("  admission: device capacity unavailable")
    else:
        print(
            "  admissible sequences: "
            f"{scheduler['admissible_active_sequences']}"
        )
        print(
            "  requested fit: "
            f"{'yes' if scheduler['requested_fits'] else 'no'}"
        )
    print("  accuracy: uncalibrated")
    return 0


def _serving_batch_memory(
    *,
    dimensions: Mapping[str, Any],
    parameter_bytes: int,
    active_sequences: int,
    prompt_tokens: int,
    generated_tokens: int,
    element_bytes: int,
    attention_implementation: str,
    prefill_chunk_tokens: int | None,
    shared_prefix_tokens: int,
    kv_cache_strategy: str,
    kv_cache_bits: int,
    kv_cache_residual_tokens: int,
    kv_cache_block_tokens: int,
    kv_cache_max_tokens: int | None,
    kv_cache_window_tokens: int | None,
    runtime_overhead_bytes: int,
    scheduler_overhead_bytes_per_sequence: int,
    speculative: Mapping[str, Any] | None,
) -> dict[str, Any]:
    effective_speculative_tokens = (
        min(int(speculative["speculative_tokens"]), generated_tokens)
        if speculative is not None
        else 0
    )
    cache_generation_tokens = (
        generated_tokens + effective_speculative_tokens
    )
    kv_cache = estimate_serving_kv_pool(
        num_hidden_layers=int(dimensions["num_hidden_layers"]),
        num_key_value_heads=int(dimensions["num_key_value_heads"]),
        head_dim=int(dimensions["head_dim"]),
        active_sequences=active_sequences,
        prompt_tokens=prompt_tokens,
        generated_tokens=cache_generation_tokens,
        element_bytes=element_bytes,
        strategy=kv_cache_strategy,
        shared_prefix_tokens=shared_prefix_tokens,
        quantized_bits=kv_cache_bits,
        quantized_residual_tokens=kv_cache_residual_tokens,
        block_tokens=kv_cache_block_tokens,
        max_cache_tokens=kv_cache_max_tokens,
        window_tokens=kv_cache_window_tokens,
        elements_per_token_per_layer=int(
            dimensions["kv_cache_elements_per_token_per_layer"]
        ),
        cache_layout=str(dimensions["kv_cache_layout"]),
    )
    draft_kv_cache = None
    draft_dimensions = None
    draft_element_bytes = None
    if speculative is not None:
        draft_dimensions = dict(speculative["model"])
        draft_element_bytes = int(speculative["element_bytes"])
        draft_kv_cache = estimate_serving_kv_pool(
            num_hidden_layers=int(
                draft_dimensions["num_hidden_layers"]
            ),
            num_key_value_heads=int(
                draft_dimensions["num_key_value_heads"]
            ),
            head_dim=int(draft_dimensions["head_dim"]),
            active_sequences=active_sequences,
            prompt_tokens=prompt_tokens,
            generated_tokens=cache_generation_tokens,
            element_bytes=draft_element_bytes,
            strategy=kv_cache_strategy,
            shared_prefix_tokens=shared_prefix_tokens,
            quantized_bits=kv_cache_bits,
            quantized_residual_tokens=kv_cache_residual_tokens,
            block_tokens=kv_cache_block_tokens,
            max_cache_tokens=kv_cache_max_tokens,
            window_tokens=kv_cache_window_tokens,
            elements_per_token_per_layer=int(
                draft_dimensions[
                    "kv_cache_elements_per_token_per_layer"
                ]
            ),
            cache_layout=str(draft_dimensions["kv_cache_layout"]),
        )
    uncached_prompt_tokens = prompt_tokens - shared_prefix_tokens
    effective_chunk_tokens = (
        min(prefill_chunk_tokens, uncached_prompt_tokens)
        if prefill_chunk_tokens is not None
        else uncached_prompt_tokens
    )
    prefill_key_tokens = _effective_tokens(
        prompt_tokens,
        window_tokens=kv_cache_window_tokens,
    )
    decode_key_tokens = _effective_tokens(
        prompt_tokens
        + max(0, generated_tokens - 1)
        + effective_speculative_tokens,
        window_tokens=kv_cache_window_tokens,
    )
    target_prefill_transient = _transient_bytes(
        dimensions,
        batch_size=active_sequences,
        query_tokens=effective_chunk_tokens,
        key_tokens=prefill_key_tokens,
        element_bytes=element_bytes,
        attention_implementation=attention_implementation,
    )
    unchunked_prefill_transient = (
        target_prefill_transient
        if prefill_chunk_tokens is None
        else _transient_bytes(
            dimensions,
            batch_size=active_sequences,
            query_tokens=uncached_prompt_tokens,
            key_tokens=prefill_key_tokens,
            element_bytes=element_bytes,
            attention_implementation=attention_implementation,
        )
    )
    draft_prefill_transient = None
    draft_unchunked_prefill_transient = None
    target_verification_transient = None
    draft_proposal_transient = None
    if speculative is not None:
        assert draft_dimensions is not None
        assert draft_element_bytes is not None
        draft_prefill_transient = _transient_bytes(
            draft_dimensions,
            batch_size=active_sequences,
            query_tokens=effective_chunk_tokens,
            key_tokens=prefill_key_tokens,
            element_bytes=draft_element_bytes,
            attention_implementation=attention_implementation,
        )
        draft_unchunked_prefill_transient = _transient_bytes(
            draft_dimensions,
            batch_size=active_sequences,
            query_tokens=uncached_prompt_tokens,
            key_tokens=prefill_key_tokens,
            element_bytes=draft_element_bytes,
            attention_implementation=attention_implementation,
        )
        prefill_transient = _select_larger_transient(
            target_prefill_transient,
            draft_prefill_transient,
        )
        unchunked_prefill_transient = _select_larger_transient(
            unchunked_prefill_transient,
            draft_unchunked_prefill_transient,
        )
        target_verification_transient = _transient_bytes(
            dimensions,
            batch_size=active_sequences,
            query_tokens=effective_speculative_tokens,
            key_tokens=decode_key_tokens,
            element_bytes=element_bytes,
            attention_implementation=attention_implementation,
        )
        draft_proposal_transient = _transient_bytes(
            draft_dimensions,
            batch_size=active_sequences,
            query_tokens=1,
            key_tokens=decode_key_tokens,
            element_bytes=draft_element_bytes,
            attention_implementation=attention_implementation,
        )
        decode_transient = _select_larger_transient(
            target_verification_transient,
            draft_proposal_transient,
        )
    else:
        prefill_transient = target_prefill_transient
        decode_transient = _transient_bytes(
            dimensions,
            batch_size=active_sequences,
            query_tokens=1,
            key_tokens=decode_key_tokens,
            element_bytes=element_bytes,
            attention_implementation=attention_implementation,
        )
    scheduler_overhead_bytes = (
        scheduler_overhead_bytes_per_sequence * active_sequences
    )
    fixed_overhead_bytes = (
        runtime_overhead_bytes + scheduler_overhead_bytes
    )
    prefill_input_bytes = (
        active_sequences * effective_chunk_tokens * 8
    )
    decode_input_bytes = (
        active_sequences
        * (effective_speculative_tokens or 1)
        * 8
    )
    draft_parameter_bytes = (
        int(speculative["parameter_bytes"])
        if speculative is not None
        else 0
    )
    combined_parameter_bytes = parameter_bytes + draft_parameter_bytes
    draft_prefill_kv_bytes = (
        int(draft_kv_cache["prefill"]["allocated_bytes"])
        if draft_kv_cache is not None
        else 0
    )
    draft_decode_kv_bytes = (
        int(draft_kv_cache["decode"]["allocated_bytes"])
        if draft_kv_cache is not None
        else 0
    )
    target_prefill_kv_bytes = int(
        kv_cache["prefill"]["allocated_bytes"]
    )
    target_decode_kv_bytes = int(
        kv_cache["decode"]["allocated_bytes"]
    )
    combined_prefill_kv_bytes = (
        target_prefill_kv_bytes + draft_prefill_kv_bytes
    )
    combined_decode_kv_bytes = (
        target_decode_kv_bytes + draft_decode_kv_bytes
    )
    prefill_peak = (
        combined_parameter_bytes
        + combined_prefill_kv_bytes
        + int(prefill_transient["peak_bytes"])
        + prefill_input_bytes
        + fixed_overhead_bytes
    )
    decode_peak = (
        combined_parameter_bytes
        + combined_decode_kv_bytes
        + int(decode_transient["peak_bytes"])
        + decode_input_bytes
        + fixed_overhead_bytes
    )
    phases = [
        {
            "phase": "prefill",
            "peak_bytes": prefill_peak,
            "components": {
                "parameters": combined_parameter_bytes,
                "target_parameters": parameter_bytes,
                "draft_parameters": draft_parameter_bytes,
                "inputs": prefill_input_bytes,
                "kv_cache": combined_prefill_kv_bytes,
                "target_kv_cache": target_prefill_kv_bytes,
                "draft_kv_cache": draft_prefill_kv_bytes,
                "transient": int(prefill_transient["peak_bytes"]),
                "runtime_overhead": runtime_overhead_bytes,
                "scheduler_overhead": scheduler_overhead_bytes,
            },
        },
        {
            "phase": "decode",
            "peak_bytes": decode_peak,
            "components": {
                "parameters": combined_parameter_bytes,
                "target_parameters": parameter_bytes,
                "draft_parameters": draft_parameter_bytes,
                "inputs": decode_input_bytes,
                "kv_cache": combined_decode_kv_bytes,
                "target_kv_cache": target_decode_kv_bytes,
                "draft_kv_cache": draft_decode_kv_bytes,
                "transient": int(decode_transient["peak_bytes"]),
                "runtime_overhead": runtime_overhead_bytes,
                "scheduler_overhead": scheduler_overhead_bytes,
            },
        },
    ]
    peak_phase = "prefill" if prefill_peak >= decode_peak else "decode"
    speculative_report = _speculative_report(
        speculative=speculative,
        effective_proposal_tokens=(
            {"homogeneous_pool": effective_speculative_tokens}
            if speculative is not None
            else {}
        ),
        target_kv_cache=kv_cache if speculative is not None else None,
        draft_kv_cache=draft_kv_cache,
        target_prefill_transient=(
            target_prefill_transient
            if speculative is not None
            else None
        ),
        draft_prefill_transient=draft_prefill_transient,
        target_verification_transient=(
            target_verification_transient
        ),
        draft_proposal_transient=draft_proposal_transient,
    )
    return {
        "kv_cache": kv_cache,
        "speculative_decoding": speculative_report,
        "optimizations": {
            "continuous_batching": {
                "active_sequences": active_sequences,
            },
            "chunked_prefill": {
                "enabled": (
                    prefill_chunk_tokens is not None
                    and effective_chunk_tokens < uncached_prompt_tokens
                ),
                "requested_chunk_tokens": prefill_chunk_tokens,
                "uncached_prompt_tokens": uncached_prompt_tokens,
                "effective_query_tokens": effective_chunk_tokens,
                "last_chunk_key_tokens": prefill_key_tokens,
                "transient_bytes": int(
                    prefill_transient["peak_bytes"]
                ),
                "without_chunking_transient_bytes": int(
                    unchunked_prefill_transient["peak_bytes"]
                ),
                "transient_savings_bytes": (
                    int(unchunked_prefill_transient["peak_bytes"])
                    - int(prefill_transient["peak_bytes"])
                ),
            },
            "prefix_cache": {
                "enabled": shared_prefix_tokens > 0,
                "shared_prefix_tokens": shared_prefix_tokens,
                "prompt_token_hit_percent": (
                    shared_prefix_tokens / prompt_tokens * 100
                ),
                "prefill_savings_bytes": int(
                    kv_cache["prefill"]["prefix_cache_savings_bytes"]
                ),
                "decode_savings_bytes": int(
                    kv_cache["decode"]["prefix_cache_savings_bytes"]
                ),
            },
        },
        "memory": {
            "parameter_bytes": combined_parameter_bytes,
            "target_parameter_bytes": parameter_bytes,
            "draft_parameter_bytes": draft_parameter_bytes,
            "runtime_overhead_bytes": runtime_overhead_bytes,
            "scheduler_overhead_bytes": scheduler_overhead_bytes,
            "prefill_transient": prefill_transient,
            "decode_transient": decode_transient,
            "target_prefill_transient": target_prefill_transient,
            "draft_prefill_transient": draft_prefill_transient,
            "target_verification_transient": (
                target_verification_transient
            ),
            "draft_proposal_transient": draft_proposal_transient,
            "estimated_prefill_peak_bytes": prefill_peak,
            "estimated_decode_peak_bytes": decode_peak,
            "estimated_process_peak_bytes": max(
                prefill_peak,
                decode_peak,
            ),
        },
        "phases": phases,
        "peak_bytes": max(prefill_peak, decode_peak),
        "peak_phase": peak_phase,
    }


def _serving_request_set_memory(
    *,
    dimensions: Mapping[str, Any],
    parameter_bytes: int,
    requests: Sequence[Mapping[str, Any]],
    element_bytes: int,
    attention_implementation: str,
    prefill_chunk_tokens: int | None,
    prefill_concurrency: int,
    kv_cache_strategy: str,
    kv_cache_bits: int,
    kv_cache_residual_tokens: int,
    kv_cache_block_tokens: int,
    kv_cache_max_tokens: int | None,
    kv_cache_window_tokens: int | None,
    runtime_overhead_bytes: int,
    scheduler_overhead_bytes_per_sequence: int,
    speculative: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized = _normalize_serving_requests(requests)
    effective_proposal_tokens = {
        str(request["id"]): min(
            int(speculative["speculative_tokens"]),
            int(request["generated_tokens"]),
        )
        for request in normalized
    } if speculative is not None else {}
    cache_requests = (
        [
            {
                **request,
                "generated_tokens": (
                    int(request["generated_tokens"])
                    + effective_proposal_tokens[str(request["id"])]
                ),
            }
            for request in normalized
        ]
        if speculative is not None
        else normalized
    )
    kv_cache = estimate_serving_request_kv_pool(
        cache_requests,
        num_hidden_layers=int(dimensions["num_hidden_layers"]),
        num_key_value_heads=int(dimensions["num_key_value_heads"]),
        head_dim=int(dimensions["head_dim"]),
        element_bytes=element_bytes,
        strategy=kv_cache_strategy,
        quantized_bits=kv_cache_bits,
        quantized_residual_tokens=kv_cache_residual_tokens,
        block_tokens=kv_cache_block_tokens,
        max_cache_tokens=kv_cache_max_tokens,
        window_tokens=kv_cache_window_tokens,
        elements_per_token_per_layer=int(
            dimensions["kv_cache_elements_per_token_per_layer"]
        ),
        cache_layout=str(dimensions["kv_cache_layout"]),
    )
    draft_kv_cache = None
    draft_dimensions = None
    draft_element_bytes = None
    if speculative is not None:
        draft_dimensions = dict(speculative["model"])
        draft_element_bytes = int(speculative["element_bytes"])
        draft_kv_cache = estimate_serving_request_kv_pool(
            cache_requests,
            num_hidden_layers=int(
                draft_dimensions["num_hidden_layers"]
            ),
            num_key_value_heads=int(
                draft_dimensions["num_key_value_heads"]
            ),
            head_dim=int(draft_dimensions["head_dim"]),
            element_bytes=draft_element_bytes,
            strategy=kv_cache_strategy,
            quantized_bits=kv_cache_bits,
            quantized_residual_tokens=kv_cache_residual_tokens,
            block_tokens=kv_cache_block_tokens,
            max_cache_tokens=kv_cache_max_tokens,
            window_tokens=kv_cache_window_tokens,
            elements_per_token_per_layer=int(
                draft_dimensions[
                    "kv_cache_elements_per_token_per_layer"
                ]
            ),
            cache_layout=str(draft_dimensions["kv_cache_layout"]),
        )

    request_details: list[dict[str, Any]] = []
    for request in normalized:
        request_id = str(request["id"])
        request_speculative_tokens = effective_proposal_tokens.get(
            request_id,
            0,
        )
        uncached_prompt_tokens = (
            request["prompt_tokens"]
            - request["shared_prefix_tokens"]
        )
        effective_chunk_tokens = (
            min(prefill_chunk_tokens, uncached_prompt_tokens)
            if prefill_chunk_tokens is not None
            else uncached_prompt_tokens
        )
        prefill_key_tokens = _effective_tokens(
            request["prompt_tokens"],
            window_tokens=kv_cache_window_tokens,
        )
        decode_key_tokens = _effective_tokens(
            request["prompt_tokens"]
            + max(0, request["generated_tokens"] - 1)
            + request_speculative_tokens,
            window_tokens=kv_cache_window_tokens,
        )
        detail = {
            **request,
            "effective_speculative_tokens": request_speculative_tokens,
            "uncached_prompt_tokens": uncached_prompt_tokens,
            "effective_prefill_query_tokens": effective_chunk_tokens,
            "prefill_key_tokens": prefill_key_tokens,
            "decode_key_tokens": decode_key_tokens,
            "target_prefill_transient": _transient_bytes(
                dimensions,
                batch_size=1,
                query_tokens=effective_chunk_tokens,
                key_tokens=prefill_key_tokens,
                element_bytes=element_bytes,
                attention_implementation=attention_implementation,
            ),
            "target_unchunked_prefill_transient": _transient_bytes(
                dimensions,
                batch_size=1,
                query_tokens=uncached_prompt_tokens,
                key_tokens=prefill_key_tokens,
                element_bytes=element_bytes,
                attention_implementation=attention_implementation,
            ),
            "target_decode_transient": _transient_bytes(
                dimensions,
                batch_size=1,
                query_tokens=(request_speculative_tokens or 1),
                key_tokens=decode_key_tokens,
                element_bytes=element_bytes,
                attention_implementation=attention_implementation,
            ),
        }
        if speculative is not None:
            assert draft_dimensions is not None
            assert draft_element_bytes is not None
            detail.update(
                draft_prefill_transient=_transient_bytes(
                    draft_dimensions,
                    batch_size=1,
                    query_tokens=effective_chunk_tokens,
                    key_tokens=prefill_key_tokens,
                    element_bytes=draft_element_bytes,
                    attention_implementation=attention_implementation,
                ),
                draft_unchunked_prefill_transient=_transient_bytes(
                    draft_dimensions,
                    batch_size=1,
                    query_tokens=uncached_prompt_tokens,
                    key_tokens=prefill_key_tokens,
                    element_bytes=draft_element_bytes,
                    attention_implementation=attention_implementation,
                ),
                draft_proposal_transient=_transient_bytes(
                    draft_dimensions,
                    batch_size=1,
                    query_tokens=1,
                    key_tokens=decode_key_tokens,
                    element_bytes=draft_element_bytes,
                    attention_implementation=attention_implementation,
                ),
            )
        request_details.append(detail)

    effective_prefill_concurrency = min(
        prefill_concurrency,
        len(normalized),
    )
    target_prefill_transient = _worst_concurrent_transient(
        request_details,
        field="target_prefill_transient",
        concurrency=effective_prefill_concurrency,
    )
    target_unchunked_prefill_transient = _worst_concurrent_transient(
        request_details,
        field="target_unchunked_prefill_transient",
        concurrency=effective_prefill_concurrency,
    )
    target_decode_transient = _sum_request_transients(
        request_details,
        field="target_decode_transient",
    )
    draft_prefill_transient = None
    draft_unchunked_prefill_transient = None
    draft_proposal_transient = None
    if speculative is not None:
        draft_prefill_transient = _worst_concurrent_transient(
            request_details,
            field="draft_prefill_transient",
            concurrency=effective_prefill_concurrency,
        )
        draft_unchunked_prefill_transient = (
            _worst_concurrent_transient(
                request_details,
                field="draft_unchunked_prefill_transient",
                concurrency=effective_prefill_concurrency,
            )
        )
        draft_proposal_transient = _sum_request_transients(
            request_details,
            field="draft_proposal_transient",
        )
        prefill_transient = _select_larger_transient(
            target_prefill_transient,
            draft_prefill_transient,
        )
        unchunked_prefill_transient = _select_larger_transient(
            target_unchunked_prefill_transient,
            draft_unchunked_prefill_transient,
        )
        decode_transient = _select_larger_transient(
            target_decode_transient,
            draft_proposal_transient,
        )
    else:
        prefill_transient = target_prefill_transient
        unchunked_prefill_transient = (
            target_unchunked_prefill_transient
        )
        decode_transient = target_decode_transient

    prefill_input_requests = sorted(
        request_details,
        key=lambda item: int(
            item["effective_prefill_query_tokens"]
        ),
        reverse=True,
    )[:effective_prefill_concurrency]
    prefill_input_bytes = sum(
        int(request["effective_prefill_query_tokens"]) * 8
        for request in prefill_input_requests
    )
    decode_input_bytes = sum(
        effective_proposal_tokens.get(str(request["id"]), 1)
        for request in normalized
    ) * 8
    scheduler_overhead_bytes = (
        scheduler_overhead_bytes_per_sequence * len(normalized)
    )
    fixed_overhead_bytes = (
        runtime_overhead_bytes + scheduler_overhead_bytes
    )
    draft_parameter_bytes = (
        int(speculative["parameter_bytes"])
        if speculative is not None
        else 0
    )
    combined_parameter_bytes = parameter_bytes + draft_parameter_bytes
    target_prefill_kv_bytes = int(
        kv_cache["prefill"]["allocated_bytes"]
    )
    target_decode_kv_bytes = int(
        kv_cache["decode"]["allocated_bytes"]
    )
    draft_prefill_kv_bytes = (
        int(draft_kv_cache["prefill"]["allocated_bytes"])
        if draft_kv_cache is not None
        else 0
    )
    draft_decode_kv_bytes = (
        int(draft_kv_cache["decode"]["allocated_bytes"])
        if draft_kv_cache is not None
        else 0
    )
    combined_prefill_kv_bytes = (
        target_prefill_kv_bytes + draft_prefill_kv_bytes
    )
    combined_decode_kv_bytes = (
        target_decode_kv_bytes + draft_decode_kv_bytes
    )
    prefill_peak = (
        combined_parameter_bytes
        + combined_prefill_kv_bytes
        + int(prefill_transient["peak_bytes"])
        + prefill_input_bytes
        + fixed_overhead_bytes
    )
    decode_peak = (
        combined_parameter_bytes
        + combined_decode_kv_bytes
        + int(decode_transient["peak_bytes"])
        + decode_input_bytes
        + fixed_overhead_bytes
    )
    phases = [
        {
            "phase": "prefill",
            "peak_bytes": prefill_peak,
            "components": {
                "parameters": combined_parameter_bytes,
                "target_parameters": parameter_bytes,
                "draft_parameters": draft_parameter_bytes,
                "inputs": prefill_input_bytes,
                "kv_cache": combined_prefill_kv_bytes,
                "target_kv_cache": target_prefill_kv_bytes,
                "draft_kv_cache": draft_prefill_kv_bytes,
                "transient": int(prefill_transient["peak_bytes"]),
                "runtime_overhead": runtime_overhead_bytes,
                "scheduler_overhead": scheduler_overhead_bytes,
            },
        },
        {
            "phase": "decode",
            "peak_bytes": decode_peak,
            "components": {
                "parameters": combined_parameter_bytes,
                "target_parameters": parameter_bytes,
                "draft_parameters": draft_parameter_bytes,
                "inputs": decode_input_bytes,
                "kv_cache": combined_decode_kv_bytes,
                "target_kv_cache": target_decode_kv_bytes,
                "draft_kv_cache": draft_decode_kv_bytes,
                "transient": int(decode_transient["peak_bytes"]),
                "runtime_overhead": runtime_overhead_bytes,
                "scheduler_overhead": scheduler_overhead_bytes,
            },
        },
    ]
    prompt_lengths = [
        int(request["prompt_tokens"]) for request in normalized
    ]
    generation_lengths = [
        int(request["generated_tokens"]) for request in normalized
    ]
    prefix_groups = {
        str(request["prefix_group"])
        for request in normalized
        if request["prefix_group"] is not None
    }
    chunking_enabled = any(
        prefill_chunk_tokens is not None
        and int(request["effective_prefill_query_tokens"])
        < int(request["uncached_prompt_tokens"])
        for request in request_details
    )
    peak_phase = "prefill" if prefill_peak >= decode_peak else "decode"
    speculative_report = _speculative_report(
        speculative=speculative,
        effective_proposal_tokens=effective_proposal_tokens,
        target_kv_cache=kv_cache if speculative is not None else None,
        draft_kv_cache=draft_kv_cache,
        target_prefill_transient=(
            target_prefill_transient
            if speculative is not None
            else None
        ),
        draft_prefill_transient=draft_prefill_transient,
        target_verification_transient=(
            target_decode_transient
            if speculative is not None
            else None
        ),
        draft_proposal_transient=draft_proposal_transient,
    )
    return {
        "kv_cache": kv_cache,
        "speculative_decoding": speculative_report,
        "optimizations": {
            "continuous_batching": {
                "active_sequences": len(normalized),
                "request_ids": [
                    request["id"] for request in normalized
                ],
                "request_shapes": {
                    "prompt_tokens": {
                        "minimum": min(prompt_lengths),
                        "maximum": max(prompt_lengths),
                        "total": sum(prompt_lengths),
                    },
                    "generated_tokens": {
                        "minimum": min(generation_lengths),
                        "maximum": max(generation_lengths),
                        "total": sum(generation_lengths),
                    },
                },
            },
            "chunked_prefill": {
                "enabled": chunking_enabled,
                "requested_chunk_tokens": prefill_chunk_tokens,
                "requested_concurrency": prefill_concurrency,
                "effective_concurrency": (
                    effective_prefill_concurrency
                ),
                "input_request_ids": [
                    request["id"]
                    for request in prefill_input_requests
                ],
                "transient_bytes": int(
                    prefill_transient["peak_bytes"]
                ),
                "without_chunking_transient_bytes": int(
                    unchunked_prefill_transient["peak_bytes"]
                ),
                "transient_savings_bytes": (
                    int(unchunked_prefill_transient["peak_bytes"])
                    - int(prefill_transient["peak_bytes"])
                ),
            },
            "prefix_cache": {
                "enabled": bool(prefix_groups),
                "group_count": len(prefix_groups),
                "group_ids": sorted(prefix_groups),
                "request_count": sum(
                    request["prefix_group"] is not None
                    for request in normalized
                ),
                "prefill_savings_bytes": int(
                    kv_cache["prefill"]["prefix_cache_savings_bytes"]
                ),
                "decode_savings_bytes": int(
                    kv_cache["decode"]["prefix_cache_savings_bytes"]
                ),
            },
        },
        "memory": {
            "parameter_bytes": combined_parameter_bytes,
            "target_parameter_bytes": parameter_bytes,
            "draft_parameter_bytes": draft_parameter_bytes,
            "runtime_overhead_bytes": runtime_overhead_bytes,
            "scheduler_overhead_bytes": scheduler_overhead_bytes,
            "prefill_transient": prefill_transient,
            "decode_transient": decode_transient,
            "target_prefill_transient": target_prefill_transient,
            "draft_prefill_transient": draft_prefill_transient,
            "target_verification_transient": (
                target_decode_transient
                if speculative is not None
                else None
            ),
            "draft_proposal_transient": draft_proposal_transient,
            "request_details": request_details,
            "estimated_prefill_peak_bytes": prefill_peak,
            "estimated_decode_peak_bytes": decode_peak,
            "estimated_process_peak_bytes": max(
                prefill_peak,
                decode_peak,
            ),
        },
        "phases": phases,
        "peak_bytes": max(prefill_peak, decode_peak),
        "peak_phase": peak_phase,
    }


def _maximum_fitting_sequences(
    *,
    max_batch_size: int,
    usable_capacity_bytes: int,
    batch_memory: Any,
) -> int:
    return _maximum_fitting(
        max_count=max_batch_size,
        usable_capacity_bytes=usable_capacity_bytes,
        memory_for_count=batch_memory,
        metric="peak_bytes",
    )


def _maximum_fitting_request_prefix(
    *,
    max_request_count: int,
    usable_capacity_bytes: int,
    request_memory: Any,
) -> int:
    return _maximum_fitting(
        max_count=max_request_count,
        usable_capacity_bytes=usable_capacity_bytes,
        memory_for_count=request_memory,
        metric="peak_bytes",
    )


def _maximum_fitting_vllm_kv_count(
    *,
    max_count: int,
    kv_cache_capacity_bytes: int,
    memory_for_count: Any,
) -> int:
    return _maximum_fitting(
        max_count=max_count,
        usable_capacity_bytes=kv_cache_capacity_bytes,
        memory_for_count=memory_for_count,
        metric=None,
        value_fn=_logical_kv_peak_bytes,
    )


def _maximum_fitting(
    *,
    max_count: int,
    usable_capacity_bytes: int,
    memory_for_count: Any,
    metric: str | None,
    value_fn: Any | None = None,
) -> int:
    """Return the largest monotonically fitting count via one binary search."""
    if value_fn is None:
        if metric is None:  # pragma: no cover - internal misuse guard
            raise ValueError("metric or value_fn is required")

        def value_fn(memory: Mapping[str, Any]) -> int:
            return int(memory[metric])
    low = 0
    high = int(max_count)
    while low < high:
        midpoint = (low + high + 1) // 2
        if int(value_fn(memory_for_count(midpoint))) <= int(
            usable_capacity_bytes
        ):
            low = midpoint
        else:
            high = midpoint - 1
    return low


def _logical_kv_peak_bytes(memory: Mapping[str, Any]) -> int:
    return max(
        int(phase["components"]["kv_cache"])
        for phase in memory["phases"]
    )


def _non_kv_phase_bytes(
    memory: Mapping[str, Any],
) -> dict[str, int]:
    phase_bytes: dict[str, int] = {}
    for phase in memory["phases"]:
        phase_name = str(phase["phase"])
        peak_bytes = int(phase["peak_bytes"])
        kv_cache_bytes = int(phase["components"]["kv_cache"])
        non_kv_bytes = peak_bytes - kv_cache_bytes
        if non_kv_bytes < 0:
            raise ServingPlanError(
                f"{phase_name} peak is smaller than its KV cache"
            )
        phase_bytes[phase_name] = non_kv_bytes
    return phase_bytes


def _build_vllm_runtime_report(
    *,
    requested_memory: Mapping[str, Any],
    profiled_memory: Mapping[str, Any],
    selected_capacity_bytes: int | None,
    gpu_memory_utilization: float,
    kv_cache_memory_bytes: int | None,
    non_kv_cache_memory_bytes: int | None,
    max_model_len_tokens: int,
    profiled_sequence_count: int,
    profile_scope: str,
) -> dict[str, Any]:
    kv_cache = requested_memory["kv_cache"]
    block_tokens = int(kv_cache["block_tokens"])
    bytes_per_token = int(
        kv_cache["bytes_per_token_per_sequence_at_compute_dtype"]
    )
    block_bytes = block_tokens * bytes_per_token
    modeled_non_kv_phases = _non_kv_phase_bytes(profiled_memory)
    modeled_non_kv_peak = max(modeled_non_kv_phases.values())
    selected_non_kv_peak = (
        modeled_non_kv_peak
        if non_kv_cache_memory_bytes is None
        else int(non_kv_cache_memory_bytes)
    )
    non_kv_source = (
        "fakegpu_profile_shape_model"
        if non_kv_cache_memory_bytes is None
        else "user_supplied_vllm_profile_result"
    )
    requested_executor_memory = (
        math.ceil(
            int(selected_capacity_bytes) * gpu_memory_utilization
        )
        if selected_capacity_bytes is not None
        else None
    )
    explicit_kv_cache = kv_cache_memory_bytes is not None
    if explicit_kv_cache:
        available_before_rounding = int(kv_cache_memory_bytes)
        kv_cache_source = "kv_cache_memory_bytes_override"
        memory_limit_bytes = selected_capacity_bytes
    elif requested_executor_memory is not None:
        available_before_rounding = max(
            0,
            requested_executor_memory - selected_non_kv_peak,
        )
        kv_cache_source = (
            "gpu_memory_utilization_minus_profiled_non_kv_memory"
        )
        memory_limit_bytes = requested_executor_memory
    else:
        available_before_rounding = None
        kv_cache_source = "device_capacity_unavailable"
        memory_limit_bytes = None

    num_gpu_blocks = (
        available_before_rounding // block_bytes
        if available_before_rounding is not None
        else None
    )
    allocatable_kv_cache_bytes = (
        num_gpu_blocks * block_bytes
        if num_gpu_blocks is not None
        else None
    )
    block_rounding_tail_bytes = (
        available_before_rounding - allocatable_kv_cache_bytes
        if available_before_rounding is not None
        and allocatable_kv_cache_bytes is not None
        else None
    )
    kv_cache_size_tokens = (
        num_gpu_blocks * block_tokens
        if num_gpu_blocks is not None
        else None
    )
    logical_requested_kv_cache_bytes = _logical_kv_peak_bytes(
        requested_memory
    )
    requested_kv_cache_fits = (
        logical_requested_kv_cache_bytes
        <= allocatable_kv_cache_bytes
        if allocatable_kv_cache_bytes is not None
        else None
    )
    initialization_required_bytes = (
        selected_non_kv_peak + allocatable_kv_cache_bytes
        if allocatable_kv_cache_bytes is not None
        else None
    )
    if memory_limit_bytes is None:
        initialization_fits = None
    elif not explicit_kv_cache and (
        selected_non_kv_peak > memory_limit_bytes
    ):
        initialization_fits = False
    else:
        initialization_fits = (
            initialization_required_bytes <= memory_limit_bytes
            if initialization_required_bytes is not None
            else None
        )
    if requested_kv_cache_fits is False or initialization_fits is False:
        requested_fits_memory = False
    elif requested_kv_cache_fits is True and initialization_fits is True:
        requested_fits_memory = True
    else:
        requested_fits_memory = None

    return {
        "schema_version": "fakegpu.vllm_runtime_memory.v1",
        "engine": "vllm",
        "method": "vllm_v1_profiled_non_kv_memory_budget_model",
        "configuration": {
            "gpu_memory_utilization": gpu_memory_utilization,
            "gpu_memory_utilization_default": (
                VLLM_DEFAULT_GPU_MEMORY_UTILIZATION
            ),
            "gpu_memory_utilization_ignored": explicit_kv_cache,
            "kv_cache_memory_bytes": kv_cache_memory_bytes,
            "block_tokens": block_tokens,
            "max_model_len_tokens": max_model_len_tokens,
            "profiled_sequence_count": profiled_sequence_count,
            "profile_scope": profile_scope,
        },
        "memory": {
            "device_capacity_bytes": selected_capacity_bytes,
            "requested_model_executor_memory_bytes": (
                requested_executor_memory
            ),
            "memory_limit_bytes": memory_limit_bytes,
            "modeled_non_kv_cache_memory_bytes": (
                modeled_non_kv_peak
            ),
            "modeled_non_kv_phase_bytes": modeled_non_kv_phases,
            "non_kv_cache_memory_bytes": selected_non_kv_peak,
            "non_kv_cache_memory_source": non_kv_source,
            "initialization_required_bytes": (
                initialization_required_bytes
            ),
            "initialization_fits": initialization_fits,
        },
        "kv_cache": {
            "source": kv_cache_source,
            "bytes_per_token": bytes_per_token,
            "block_bytes": block_bytes,
            "available_before_block_rounding_bytes": (
                available_before_rounding
            ),
            "allocatable_bytes": allocatable_kv_cache_bytes,
            "block_rounding_tail_bytes": block_rounding_tail_bytes,
            "num_gpu_blocks": num_gpu_blocks,
            "size_tokens": kv_cache_size_tokens,
            "max_model_len_concurrency": (
                kv_cache_size_tokens / max_model_len_tokens
                if kv_cache_size_tokens is not None
                else None
            ),
            "logical_requested_peak_bytes": (
                logical_requested_kv_cache_bytes
            ),
            "logical_requested_fits": requested_kv_cache_fits,
            "headroom_bytes": (
                allocatable_kv_cache_bytes
                - logical_requested_kv_cache_bytes
                if allocatable_kv_cache_bytes is not None
                else None
            ),
        },
        "requested_fits_memory": requested_fits_memory,
        "notes": [
            (
                "Automatic mode follows vLLM V1's requested-memory minus "
                "profiled non-KV memory policy, then rounds down to whole "
                "paged-cache blocks."
            ),
            (
                "kv_cache_memory_bytes overrides utilization-based KV "
                "sizing, matching vLLM configuration semantics."
            ),
            (
                "A supplied non-KV value should come from the matching vLLM "
                "startup profile and include weights, activations, non-torch "
                "allocations, and CUDA graph memory."
            ),
        ],
    }


def _apply_vllm_kv_reservation(
    memory: Mapping[str, Any],
    runtime_report: Mapping[str, Any],
) -> dict[str, Any]:
    allocatable_bytes = runtime_report["kv_cache"]["allocatable_bytes"]
    if allocatable_bytes is None:
        return dict(memory)
    reserved_kv_cache_bytes = int(allocatable_bytes)
    phases = []
    for phase in memory["phases"]:
        components = dict(phase["components"])
        logical_kv_cache_bytes = int(components["kv_cache"])
        logical_target_kv_cache_bytes = int(
            components["target_kv_cache"]
        )
        components.update(
            {
                "logical_kv_cache": logical_kv_cache_bytes,
                "logical_target_kv_cache": (
                    logical_target_kv_cache_bytes
                ),
                "kv_cache": reserved_kv_cache_bytes,
                "target_kv_cache": reserved_kv_cache_bytes,
                "vllm_reserved_kv_cache": reserved_kv_cache_bytes,
            }
        )
        phase_peak = (
            int(phase["peak_bytes"])
            - logical_kv_cache_bytes
            + reserved_kv_cache_bytes
        )
        phases.append(
            {
                **phase,
                "peak_bytes": phase_peak,
                "components": components,
            }
        )
    peak_phase_report = max(
        phases,
        key=lambda item: int(item["peak_bytes"]),
    )
    memory_report = dict(memory["memory"])
    memory_report.update(
        {
            "logical_estimated_prefill_peak_bytes": memory_report[
                "estimated_prefill_peak_bytes"
            ],
            "logical_estimated_decode_peak_bytes": memory_report[
                "estimated_decode_peak_bytes"
            ],
            "logical_estimated_process_peak_bytes": memory_report[
                "estimated_process_peak_bytes"
            ],
            "vllm_reserved_kv_cache_bytes": reserved_kv_cache_bytes,
            "estimated_prefill_peak_bytes": phases[0]["peak_bytes"],
            "estimated_decode_peak_bytes": phases[1]["peak_bytes"],
            "estimated_process_peak_bytes": peak_phase_report[
                "peak_bytes"
            ],
        }
    )
    return {
        **memory,
        "memory": memory_report,
        "phases": phases,
        "peak_bytes": int(peak_phase_report["peak_bytes"]),
        "peak_phase": str(peak_phase_report["phase"]),
    }


_TRANSIENT_COMPONENTS = (
    "attention_bytes",
    "mlp_bytes",
    "dense_mlp_bytes",
    "routed_mlp_bytes",
    "router_bytes",
    "logits_bytes",
)


def _worst_concurrent_transient(
    requests: Sequence[Mapping[str, Any]],
    *,
    field: str,
    concurrency: int,
) -> dict[str, Any]:
    totals: dict[str, int] = {}
    contributors: dict[str, list[str]] = {}
    for component in _TRANSIENT_COMPONENTS:
        ranked = heapq.nlargest(
            concurrency,
            requests,
            key=lambda request: int(request[field][component]),
        )
        totals[component] = sum(
            int(request[field][component]) for request in ranked
        )
        contributors[component] = [
            str(request["id"]) for request in ranked
        ]
    totals["peak_bytes"] = max(
        totals["attention_bytes"],
        totals["mlp_bytes"],
        totals["logits_bytes"],
    )
    return {
        **totals,
        "concurrency": concurrency,
        "contributing_request_ids": contributors,
    }


def _sum_request_transients(
    requests: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> dict[str, Any]:
    totals = {
        component: sum(
            int(request[field][component]) for request in requests
        )
        for component in _TRANSIENT_COMPONENTS
    }
    totals["peak_bytes"] = max(
        totals["attention_bytes"],
        totals["mlp_bytes"],
        totals["logits_bytes"],
    )
    return {
        **totals,
        "concurrency": len(requests),
        "contributing_request_ids": {
            component: [
                str(request["id"]) for request in requests
            ]
            for component in _TRANSIENT_COMPONENTS
        },
    }


def _normalize_serving_requests(
    requests: Any,
) -> list[dict[str, Any]]:
    if (
        not isinstance(requests, Sequence)
        or isinstance(requests, (str, bytes, bytearray))
        or not requests
    ):
        raise ServingPlanError(
            "requests must be a non-empty array of request objects"
        )

    normalized: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    group_prefix_tokens: dict[str, int] = {}
    allowed_fields = {
        "id",
        "prompt_tokens",
        "generated_tokens",
        "prefix_group",
        "shared_prefix_tokens",
    }
    for index, raw_request in enumerate(requests):
        if not isinstance(raw_request, Mapping):
            raise ServingPlanError(
                f"requests[{index}] must be an object"
            )
        unknown_fields = sorted(
            str(field)
            for field in set(raw_request) - allowed_fields
        )
        if unknown_fields:
            joined = ", ".join(unknown_fields)
            raise ServingPlanError(
                f"requests[{index}] has unsupported fields: {joined}"
            )

        raw_id = raw_request.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ServingPlanError(
                f"requests[{index}].id must be a non-empty string"
            )
        request_id = raw_id.strip()
        if request_id in request_ids:
            raise ServingPlanError(
                f"duplicate serving request id: {request_id!r}"
            )
        request_ids.add(request_id)

        prompt_tokens = _positive_integer(
            raw_request.get("prompt_tokens"),
            f"requests[{index}].prompt_tokens",
        )
        generated_tokens = _positive_integer(
            raw_request.get("generated_tokens", 1),
            f"requests[{index}].generated_tokens",
        )
        shared_prefix_tokens = _nonnegative_integer(
            raw_request.get("shared_prefix_tokens", 0),
            f"requests[{index}].shared_prefix_tokens",
        )
        if shared_prefix_tokens > prompt_tokens:
            raise ServingPlanError(
                f"requests[{index}].shared_prefix_tokens must not exceed "
                "prompt_tokens"
            )

        raw_group = raw_request.get("prefix_group")
        if raw_group is None:
            prefix_group = None
        elif isinstance(raw_group, str) and raw_group.strip():
            prefix_group = raw_group.strip()
        else:
            raise ServingPlanError(
                f"requests[{index}].prefix_group must be a non-empty "
                "string or null"
            )
        if shared_prefix_tokens and prefix_group is None:
            raise ServingPlanError(
                f"requests[{index}].shared_prefix_tokens requires "
                "prefix_group"
            )
        if prefix_group is not None and not shared_prefix_tokens:
            raise ServingPlanError(
                f"requests[{index}].prefix_group requires positive "
                "shared_prefix_tokens"
            )
        if prefix_group is not None:
            known_prefix_tokens = group_prefix_tokens.setdefault(
                prefix_group,
                shared_prefix_tokens,
            )
            if known_prefix_tokens != shared_prefix_tokens:
                raise ServingPlanError(
                    f"prefix group {prefix_group!r} must use one "
                    "shared_prefix_tokens value"
                )

        normalized.append(
            {
                "id": request_id,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "prefix_group": prefix_group,
                "shared_prefix_tokens": shared_prefix_tokens,
            }
        )
    return normalized


def _retained_prefix_tokens(
    *,
    shared_prefix_tokens: int,
    logical_tokens: int,
    effective_tokens: int,
) -> int:
    evicted_tokens = max(0, logical_tokens - effective_tokens)
    return max(0, shared_prefix_tokens - evicted_tokens)


def _cache_segment(
    *,
    tokens: int,
    sequence_count: int,
    bytes_per_token_per_sequence: int,
    strategy: str,
    block_tokens: int,
) -> dict[str, int]:
    if tokens <= 0:
        return _empty_cache_segment()
    allocated_tokens_per_sequence = (
        math.ceil(tokens / block_tokens) * block_tokens
        if strategy == "paged"
        else tokens
    )
    return {
        "sequence_count": sequence_count,
        "logical_tokens_per_sequence": tokens,
        "allocated_tokens_per_sequence": (
            allocated_tokens_per_sequence
        ),
        "allocated_bytes": (
            sequence_count
            * allocated_tokens_per_sequence
            * bytes_per_token_per_sequence
        ),
    }


def _empty_cache_segment() -> dict[str, int]:
    return {
        "sequence_count": 0,
        "logical_tokens_per_sequence": 0,
        "allocated_tokens_per_sequence": 0,
        "allocated_bytes": 0,
    }


def _effective_tokens(
    tokens: int,
    *,
    window_tokens: int | None,
) -> int:
    return min(tokens, window_tokens) if window_tokens is not None else tokens


def _transient_bytes(
    dimensions: Mapping[str, Any],
    *,
    batch_size: int,
    query_tokens: int,
    key_tokens: int,
    element_bytes: int,
    attention_implementation: str,
) -> dict[str, int]:
    if query_tokens == 0:
        return {
            "attention_bytes": 0,
            "mlp_bytes": 0,
            "dense_mlp_bytes": 0,
            "routed_mlp_bytes": 0,
            "router_bytes": 0,
            "logits_bytes": 0,
            "peak_bytes": 0,
        }
    return _forward_transient_bytes(
        dimensions,
        batch_size=batch_size,
        query_tokens=query_tokens,
        key_tokens=key_tokens,
        element_bytes=element_bytes,
        attention_implementation=attention_implementation,
    )


def _positive_integer(value: Any, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ServingPlanError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ServingPlanError(
            f"{name} must be a non-negative integer"
        )
    return value


def _format_bytes(value: int) -> str:
    return f"{int(value) / 2**30:.3f} GiB"


if __name__ == "__main__":
    raise SystemExit(main())
