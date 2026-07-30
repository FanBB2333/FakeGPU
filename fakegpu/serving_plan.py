from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .llm_estimator import (
    KV_CACHE_STRATEGIES,
    _forward_transient_bytes,
    estimate_decoder_inference,
    estimate_kv_cache_memory,
)
from .profile_catalog import get_profile
from .structured_io import load_mapping


SCHEMA_VERSION = "fakegpu.llm_serving_plan.v1"
REQUEST_SET_SCHEMA_VERSION = "fakegpu.llm_serving_request_set_plan.v1"
REQUEST_MANIFEST_SCHEMA_VERSION = "fakegpu.serving_requests.v1"
_PREFIX_CACHE_STRATEGIES = frozenset({"dynamic", "paged"})


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
    target_profile: str | None = None,
    device_capacity_bytes: int | None = None,
    memory_utilization: float = 0.9,
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
    if (
        not isinstance(memory_utilization, (int, float))
        or isinstance(memory_utilization, bool)
        or not math.isfinite(float(memory_utilization))
        or not 0 < float(memory_utilization) <= 1
    ):
        raise ServingPlanError(
            "memory_utilization must be finite and in the interval (0, 1]"
        )
    if device_capacity_bytes is not None:
        _positive_integer(
            device_capacity_bytes,
            "device_capacity_bytes",
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

    def batch_memory(sequence_count: int) -> dict[str, Any]:
        return _serving_batch_memory(
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
        )

    requested = batch_memory(active_sequences)
    usable_capacity_bytes = (
        math.floor(
            int(selected_capacity_bytes) * float(memory_utilization)
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
    admissible_sequences = (
        min(max_batch_size, memory_limited_sequences)
        if memory_limited_sequences is not None
        else None
    )
    fits_configured_limit = active_sequences <= max_batch_size
    fits_memory = (
        int(requested["peak_bytes"]) <= usable_capacity_bytes
        if usable_capacity_bytes is not None
        else None
    )
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
    elif fits_memory is False:
        limiting_factor = "usable_device_memory"
    elif fits_memory is None:
        limiting_factor = "device_capacity_unavailable"
    elif admissible_sequences == max_batch_size:
        limiting_factor = "configured_max_batch_size"
    else:
        limiting_factor = "usable_device_memory"

    return {
        "schema_version": SCHEMA_VERSION,
        "method": (
            "safetensors_headers_plus_decoder_shape_and_serving_pool_model"
        ),
        "validation_status": "Modeled",
        "accuracy": {
            "status": "uncalibrated",
            "prediction_interval_bytes": None,
            "error_percent": None,
            "reason": (
                "No matching online-serving GPU observation was supplied."
            ),
        },
        "model": dimensions,
        "checkpoint": base["checkpoint"],
        "weight_storage": base["weight_storage"],
        "inputs": {
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
                kv_cache_bits
                if kv_cache_strategy == "quantized"
                else None
            ),
            "kv_cache_residual_tokens": (
                kv_cache_residual_tokens
                if kv_cache_strategy == "quantized"
                else None
            ),
            "kv_cache_block_tokens": (
                kv_cache_block_tokens
                if kv_cache_strategy == "paged"
                else None
            ),
            "kv_cache_max_tokens": kv_cache_max_tokens,
            "kv_cache_window_tokens": kv_cache_window_tokens,
            "runtime_overhead_bytes": runtime_overhead_bytes,
            "scheduler_overhead_bytes_per_sequence": (
                scheduler_overhead_bytes_per_sequence
            ),
        },
        "target": {
            "profile": target,
            "capacity_source": capacity_source,
            "device_capacity_bytes": selected_capacity_bytes,
            "memory_utilization": float(memory_utilization),
            "usable_capacity_bytes": usable_capacity_bytes,
        },
        "scheduler": {
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
        "kv_cache": requested["kv_cache"],
        "optimizations": requested["optimizations"],
        "memory": requested["memory"],
        "memory_timeline": {
            "unit": "bytes",
            "source": "decoder_shape_and_serving_pool_model",
            "phases": requested["phases"],
            "peak_bytes": requested["peak_bytes"],
            "peak_phase": requested["peak_phase"],
            "usable_capacity_headroom_bytes": (
                usable_capacity_bytes - int(requested["peak_bytes"])
                if usable_capacity_bytes is not None
                else None
            ),
        },
        "unmodeled_components": [
            "cuda_context_and_loaded_modules",
            "caching_allocator_fragmentation",
            "backend_kernel_and_attention_workspaces",
            "request_length_distribution_and_batch_turnover",
            "scheduler_queueing_and_preemption",
            "paged_cache_block_table_metadata",
            "prefix_cache_lookup_eviction_and_copy_cost",
            "network_and_tokenization_buffers",
            "speculative_draft_model_weights_and_kv_cache",
            "tensor_parallel_communication_and_rank_imbalance",
        ],
        "notes": [
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
    }


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
    target_profile: str | None = None,
    device_capacity_bytes: int | None = None,
    memory_utilization: float = 0.9,
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
    if (
        not isinstance(memory_utilization, (int, float))
        or isinstance(memory_utilization, bool)
        or not math.isfinite(float(memory_utilization))
        or not 0 < float(memory_utilization) <= 1
    ):
        raise ServingPlanError(
            "memory_utilization must be finite and in the interval (0, 1]"
        )
    if device_capacity_bytes is not None:
        _positive_integer(
            device_capacity_bytes,
            "device_capacity_bytes",
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
        )
        memory_by_request_count[request_count] = result
        return result

    requested_count = len(normalized)
    requested = request_memory(requested_count)
    usable_capacity_bytes = (
        math.floor(
            int(selected_capacity_bytes) * float(memory_utilization)
        )
        if selected_capacity_bytes is not None
        else None
    )
    candidate_count = min(max_batch_size, requested_count)
    memory_limited_count = None
    if usable_capacity_bytes is not None:
        memory_limited_count = _maximum_fitting_request_prefix(
            max_request_count=candidate_count,
            usable_capacity_bytes=usable_capacity_bytes,
            request_memory=request_memory,
        )
    admissible_count = memory_limited_count
    fits_configured_limit = requested_count <= max_batch_size
    fits_memory = (
        int(requested["peak_bytes"]) <= usable_capacity_bytes
        if usable_capacity_bytes is not None
        else None
    )
    requested_fits = (
        fits_configured_limit and fits_memory
        if fits_memory is not None
        else False
        if not fits_configured_limit
        else None
    )
    if not fits_configured_limit:
        limiting_factor = "configured_max_batch_size"
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

    return {
        "schema_version": REQUEST_SET_SCHEMA_VERSION,
        "method": (
            "safetensors_headers_plus_decoder_shape_and_heterogeneous_"
            "serving_request_model"
        ),
        "validation_status": "Modeled",
        "accuracy": {
            "status": "uncalibrated",
            "prediction_interval_bytes": None,
            "error_percent": None,
            "reason": (
                "No matching heterogeneous online-serving GPU "
                "observation was supplied."
            ),
        },
        "model": dimensions,
        "checkpoint": base["checkpoint"],
        "weight_storage": base["weight_storage"],
        "inputs": {
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
                kv_cache_bits
                if kv_cache_strategy == "quantized"
                else None
            ),
            "kv_cache_residual_tokens": (
                kv_cache_residual_tokens
                if kv_cache_strategy == "quantized"
                else None
            ),
            "kv_cache_block_tokens": (
                kv_cache_block_tokens
                if kv_cache_strategy == "paged"
                else None
            ),
            "kv_cache_max_tokens": kv_cache_max_tokens,
            "kv_cache_window_tokens": kv_cache_window_tokens,
            "runtime_overhead_bytes": runtime_overhead_bytes,
            "scheduler_overhead_bytes_per_sequence": (
                scheduler_overhead_bytes_per_sequence
            ),
        },
        "target": {
            "profile": target,
            "capacity_source": capacity_source,
            "device_capacity_bytes": selected_capacity_bytes,
            "memory_utilization": float(memory_utilization),
            "usable_capacity_bytes": usable_capacity_bytes,
        },
        "scheduler": {
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
        "kv_cache": requested["kv_cache"],
        "optimizations": requested["optimizations"],
        "memory": requested["memory"],
        "memory_timeline": {
            "unit": "bytes",
            "source": (
                "decoder_shape_and_heterogeneous_serving_request_model"
            ),
            "phases": requested["phases"],
            "peak_bytes": requested["peak_bytes"],
            "peak_phase": requested["peak_phase"],
            "usable_capacity_headroom_bytes": (
                usable_capacity_bytes - int(requested["peak_bytes"])
                if usable_capacity_bytes is not None
                else None
            ),
        },
        "unmodeled_components": [
            "cuda_context_and_loaded_modules",
            "caching_allocator_fragmentation",
            "backend_kernel_and_attention_workspaces",
            "request_arrival_timing_and_batch_turnover",
            "scheduler_queueing_preemption_and_reordering",
            "paged_cache_block_table_metadata",
            "prefix_cache_lookup_eviction_and_copy_cost",
            "network_and_tokenization_buffers",
            "speculative_draft_model_weights_and_kv_cache",
            "tensor_parallel_communication_and_rank_imbalance",
        ],
        "notes": [
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
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu plan-serving",
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
    capacity_group = parser.add_mutually_exclusive_group()
    capacity_group.add_argument("--target-profile")
    capacity_group.add_argument(
        "--device-memory-gib",
        type=float,
        help="Explicit device memory capacity in binary GiB.",
    )
    parser.add_argument("--memory-utilization", type=float, default=0.9)
    parser.add_argument("--json", dest="json_path")
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
                target_profile=args.target_profile,
                device_capacity_bytes=device_capacity_bytes,
                memory_utilization=args.memory_utilization,
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
                target_profile=args.target_profile,
                device_capacity_bytes=device_capacity_bytes,
                memory_utilization=args.memory_utilization,
            )
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        parser.exit(2, f"fakegpu plan-serving: {exc}\n")

    if args.json_path:
        path = Path(args.json_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Serving plan: {path}")

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
) -> dict[str, Any]:
    kv_cache = estimate_serving_kv_pool(
        num_hidden_layers=int(dimensions["num_hidden_layers"]),
        num_key_value_heads=int(dimensions["num_key_value_heads"]),
        head_dim=int(dimensions["head_dim"]),
        active_sequences=active_sequences,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        element_bytes=element_bytes,
        strategy=kv_cache_strategy,
        shared_prefix_tokens=shared_prefix_tokens,
        quantized_bits=kv_cache_bits,
        quantized_residual_tokens=kv_cache_residual_tokens,
        block_tokens=kv_cache_block_tokens,
        max_cache_tokens=kv_cache_max_tokens,
        window_tokens=kv_cache_window_tokens,
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
        prompt_tokens + max(0, generated_tokens - 1),
        window_tokens=kv_cache_window_tokens,
    )
    prefill_transient = _transient_bytes(
        dimensions,
        batch_size=active_sequences,
        query_tokens=effective_chunk_tokens,
        key_tokens=prefill_key_tokens,
        element_bytes=element_bytes,
        attention_implementation=attention_implementation,
    )
    unchunked_prefill_transient = _transient_bytes(
        dimensions,
        batch_size=active_sequences,
        query_tokens=uncached_prompt_tokens,
        key_tokens=prefill_key_tokens,
        element_bytes=element_bytes,
        attention_implementation=attention_implementation,
    )
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
    decode_input_bytes = active_sequences * 8
    prefill_peak = (
        parameter_bytes
        + int(kv_cache["prefill"]["allocated_bytes"])
        + int(prefill_transient["peak_bytes"])
        + prefill_input_bytes
        + fixed_overhead_bytes
    )
    decode_peak = (
        parameter_bytes
        + int(kv_cache["decode"]["allocated_bytes"])
        + int(decode_transient["peak_bytes"])
        + decode_input_bytes
        + fixed_overhead_bytes
    )
    phases = [
        {
            "phase": "prefill",
            "peak_bytes": prefill_peak,
            "components": {
                "parameters": parameter_bytes,
                "inputs": prefill_input_bytes,
                "kv_cache": int(
                    kv_cache["prefill"]["allocated_bytes"]
                ),
                "transient": int(prefill_transient["peak_bytes"]),
                "runtime_overhead": runtime_overhead_bytes,
                "scheduler_overhead": scheduler_overhead_bytes,
            },
        },
        {
            "phase": "decode",
            "peak_bytes": decode_peak,
            "components": {
                "parameters": parameter_bytes,
                "inputs": decode_input_bytes,
                "kv_cache": int(
                    kv_cache["decode"]["allocated_bytes"]
                ),
                "transient": int(decode_transient["peak_bytes"]),
                "runtime_overhead": runtime_overhead_bytes,
                "scheduler_overhead": scheduler_overhead_bytes,
            },
        },
    ]
    peak_phase = "prefill" if prefill_peak >= decode_peak else "decode"
    return {
        "kv_cache": kv_cache,
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
            "parameter_bytes": parameter_bytes,
            "runtime_overhead_bytes": runtime_overhead_bytes,
            "scheduler_overhead_bytes": scheduler_overhead_bytes,
            "prefill_transient": prefill_transient,
            "decode_transient": decode_transient,
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
) -> dict[str, Any]:
    normalized = _normalize_serving_requests(requests)
    kv_cache = estimate_serving_request_kv_pool(
        normalized,
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
    )

    request_details: list[dict[str, Any]] = []
    for request in normalized:
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
            + max(0, request["generated_tokens"] - 1),
            window_tokens=kv_cache_window_tokens,
        )
        request_details.append(
            {
                **request,
                "uncached_prompt_tokens": uncached_prompt_tokens,
                "effective_prefill_query_tokens": (
                    effective_chunk_tokens
                ),
                "prefill_key_tokens": prefill_key_tokens,
                "decode_key_tokens": decode_key_tokens,
                "prefill_transient": _transient_bytes(
                    dimensions,
                    batch_size=1,
                    query_tokens=effective_chunk_tokens,
                    key_tokens=prefill_key_tokens,
                    element_bytes=element_bytes,
                    attention_implementation=(
                        attention_implementation
                    ),
                ),
                "unchunked_prefill_transient": _transient_bytes(
                    dimensions,
                    batch_size=1,
                    query_tokens=uncached_prompt_tokens,
                    key_tokens=prefill_key_tokens,
                    element_bytes=element_bytes,
                    attention_implementation=(
                        attention_implementation
                    ),
                ),
                "decode_transient": _transient_bytes(
                    dimensions,
                    batch_size=1,
                    query_tokens=1,
                    key_tokens=decode_key_tokens,
                    element_bytes=element_bytes,
                    attention_implementation=(
                        attention_implementation
                    ),
                ),
            }
        )

    effective_prefill_concurrency = min(
        prefill_concurrency,
        len(normalized),
    )
    prefill_transient = _worst_concurrent_transient(
        request_details,
        field="prefill_transient",
        concurrency=effective_prefill_concurrency,
    )
    unchunked_prefill_transient = _worst_concurrent_transient(
        request_details,
        field="unchunked_prefill_transient",
        concurrency=effective_prefill_concurrency,
    )
    decode_transient = _sum_request_transients(
        request_details,
        field="decode_transient",
    )

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
    decode_input_bytes = len(normalized) * 8
    scheduler_overhead_bytes = (
        scheduler_overhead_bytes_per_sequence * len(normalized)
    )
    fixed_overhead_bytes = (
        runtime_overhead_bytes + scheduler_overhead_bytes
    )
    prefill_peak = (
        parameter_bytes
        + int(kv_cache["prefill"]["allocated_bytes"])
        + int(prefill_transient["peak_bytes"])
        + prefill_input_bytes
        + fixed_overhead_bytes
    )
    decode_peak = (
        parameter_bytes
        + int(kv_cache["decode"]["allocated_bytes"])
        + int(decode_transient["peak_bytes"])
        + decode_input_bytes
        + fixed_overhead_bytes
    )
    phases = [
        {
            "phase": "prefill",
            "peak_bytes": prefill_peak,
            "components": {
                "parameters": parameter_bytes,
                "inputs": prefill_input_bytes,
                "kv_cache": int(
                    kv_cache["prefill"]["allocated_bytes"]
                ),
                "transient": int(prefill_transient["peak_bytes"]),
                "runtime_overhead": runtime_overhead_bytes,
                "scheduler_overhead": scheduler_overhead_bytes,
            },
        },
        {
            "phase": "decode",
            "peak_bytes": decode_peak,
            "components": {
                "parameters": parameter_bytes,
                "inputs": decode_input_bytes,
                "kv_cache": int(
                    kv_cache["decode"]["allocated_bytes"]
                ),
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
    return {
        "kv_cache": kv_cache,
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
            "parameter_bytes": parameter_bytes,
            "runtime_overhead_bytes": runtime_overhead_bytes,
            "scheduler_overhead_bytes": scheduler_overhead_bytes,
            "prefill_transient": prefill_transient,
            "decode_transient": decode_transient,
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
    low = 0
    high = max_batch_size
    while low < high:
        midpoint = (low + high + 1) // 2
        if int(batch_memory(midpoint)["peak_bytes"]) <= (
            usable_capacity_bytes
        ):
            low = midpoint
        else:
            high = midpoint - 1
    return low


def _maximum_fitting_request_prefix(
    *,
    max_request_count: int,
    usable_capacity_bytes: int,
    request_memory: Any,
) -> int:
    low = 0
    high = max_request_count
    while low < high:
        midpoint = (low + high + 1) // 2
        if int(request_memory(midpoint)["peak_bytes"]) <= (
            usable_capacity_bytes
        ):
            low = midpoint
        else:
            high = midpoint - 1
    return low


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
        ranked = sorted(
            requests,
            key=lambda request: int(
                request[field][component]
            ),
            reverse=True,
        )[:concurrency]
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
