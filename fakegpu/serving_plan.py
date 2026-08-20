from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._cli import (
    add_json_path_argument,
    command_prog,
    usage_error,
)
from ._serving_kv import (
    REQUEST_MANIFEST_SCHEMA_VERSION,
    _effective_tokens,
    _normalize_serving_requests,
    _transient_bytes,
    estimate_serving_kv_pool,
    estimate_serving_request_kv_pool,
    load_serving_requests,
)
from ._serving_speculative import (
    DEFAULT_SPECULATIVE_TOKENS,
    _load_speculative_draft,
    _select_larger_transient,
    _speculative_input_report,
    _speculative_report,
    _validate_speculative_inputs,
)
from ._serving_types import (
    ServingPlanError,
    _nonnegative_integer,
    _positive_integer,
)
from ._serving_vllm import (
    VLLM_DEFAULT_GPU_MEMORY_UTILIZATION,
    _apply_vllm_kv_reservation,
    _build_vllm_runtime_report,
    _logical_kv_peak_bytes,
)
from .llm_estimator import (
    KV_CACHE_STRATEGIES,
    estimate_decoder_inference,
)
from .profile_catalog import get_profile
from .structured_io import emit_json


__all__ = [
    "REQUEST_MANIFEST_SCHEMA_VERSION",
    "REQUEST_SET_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "ServingPlanError",
    "estimate_serving_kv_pool",
    "estimate_serving_plan",
    "estimate_serving_request_kv_pool",
    "estimate_serving_request_set",
    "load_serving_requests",
    "main",
]

SCHEMA_VERSION = "fakegpu.llm_serving_plan.v1"
REQUEST_SET_SCHEMA_VERSION = "fakegpu.llm_serving_request_set_plan.v1"
WORKLOAD_SIGNATURE_SCHEMA_VERSION = "fakegpu.serving_workload.v1"
SERVING_RUNTIMES = frozenset({"generic", "vllm"})
DEFAULT_MEMORY_UTILIZATION = 0.9
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
    capacity = _resolve_serving_capacity(
        attention_implementation=attention_implementation,
        kv_cache_strategy=kv_cache_strategy,
        runtime=runtime,
        memory_utilization=memory_utilization,
        draft_model_dir=draft_model_dir,
        vllm_kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
        vllm_non_kv_cache_memory_bytes=vllm_non_kv_cache_memory_bytes,
        device_capacity_bytes=device_capacity_bytes,
        speculative_tokens=speculative_tokens,
        speculative_acceptance_rate=speculative_acceptance_rate,
        target_profile=target_profile,
    )
    runtime = capacity.runtime
    memory_utilization = capacity.memory_utilization

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
    admission = _serving_admission(
        runtime=runtime,
        requested=requested,
        memory_for_count=batch_memory,
        candidate_count=max_batch_size,
        selected_capacity_bytes=capacity.selected_capacity_bytes,
        memory_utilization=memory_utilization,
        vllm_kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
        vllm_non_kv_cache_memory_bytes=vllm_non_kv_cache_memory_bytes,
        max_model_len_tokens=prompt_tokens + generated_tokens,
        profile_scope="configured_max_batch_size",
    )
    runtime_report = admission.runtime_report
    requested_for_report = admission.requested_for_report
    usable_capacity_bytes = admission.usable_capacity_bytes
    memory_limited_sequences = admission.memory_limited_count
    fits_memory = admission.fits_memory
    admissible_sequences = (
        min(max_batch_size, memory_limited_sequences)
        if memory_limited_sequences is not None
        else None
    )
    fits_configured_limit = active_sequences <= max_batch_size
    requested_fits = _requested_fits(
        fits_configured_limit=fits_configured_limit,
        fits_memory=fits_memory,
    )
    available_slots = (
        max(0, admissible_sequences - active_sequences)
        if admissible_sequences is not None
        else None
    )
    limiting_factor = _serving_limiting_factor(
        fits_configured_limit=fits_configured_limit,
        runtime=runtime,
        runtime_report=runtime_report,
        fits_memory=fits_memory,
        admissible_count=admissible_sequences,
        candidate_count=max_batch_size,
        exhausted_factor="configured_max_batch_size",
    )

    input_report = _serving_input_report(
        mode_inputs={
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "shared_prefix_tokens": shared_prefix_tokens,
        },
        active_sequences=active_sequences,
        max_batch_size=max_batch_size,
        dtype=base["inputs"]["dtype"],
        element_bytes=element_bytes,
        attention_implementation=attention_implementation,
        prefill_chunk_tokens=prefill_chunk_tokens,
        kv_cache_strategy=kv_cache_strategy,
        kv_cache_bits=kv_cache_bits,
        kv_cache_residual_tokens=kv_cache_residual_tokens,
        kv_cache_block_tokens=kv_cache_block_tokens,
        kv_cache_max_tokens=kv_cache_max_tokens,
        kv_cache_window_tokens=kv_cache_window_tokens,
        runtime=runtime,
        memory_utilization=memory_utilization,
        vllm_kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
        vllm_non_kv_cache_memory_bytes=vllm_non_kv_cache_memory_bytes,
        runtime_overhead_bytes=runtime_overhead_bytes,
        scheduler_overhead_bytes_per_sequence=(
            scheduler_overhead_bytes_per_sequence
        ),
        speculative=speculative,
        speculative_tokens=speculative_tokens,
        speculative_acceptance_rate=speculative_acceptance_rate,
    )
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
        target_profile=capacity.target,
        capacity_source=capacity.capacity_source,
        selected_capacity_bytes=capacity.selected_capacity_bytes,
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
    capacity = _resolve_serving_capacity(
        attention_implementation=attention_implementation,
        kv_cache_strategy=kv_cache_strategy,
        runtime=runtime,
        memory_utilization=memory_utilization,
        draft_model_dir=draft_model_dir,
        vllm_kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
        vllm_non_kv_cache_memory_bytes=vllm_non_kv_cache_memory_bytes,
        device_capacity_bytes=device_capacity_bytes,
        speculative_tokens=speculative_tokens,
        speculative_acceptance_rate=speculative_acceptance_rate,
        target_profile=target_profile,
    )
    runtime = capacity.runtime
    memory_utilization = capacity.memory_utilization

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
    admission = _serving_admission(
        runtime=runtime,
        requested=requested,
        memory_for_count=request_memory,
        candidate_count=candidate_count,
        selected_capacity_bytes=capacity.selected_capacity_bytes,
        memory_utilization=memory_utilization,
        vllm_kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
        vllm_non_kv_cache_memory_bytes=vllm_non_kv_cache_memory_bytes,
        max_model_len_tokens=max(
            int(request["prompt_tokens"]) + int(request["generated_tokens"])
            for request in normalized[:candidate_count]
        ),
        profile_scope="configured_request_manifest_prefix",
    )
    runtime_report = admission.runtime_report
    requested_for_report = admission.requested_for_report
    usable_capacity_bytes = admission.usable_capacity_bytes
    memory_limited_count = admission.memory_limited_count
    fits_memory = admission.fits_memory
    admissible_count = memory_limited_count
    fits_configured_limit = requested_count <= max_batch_size
    requested_fits = _requested_fits(
        fits_configured_limit=fits_configured_limit,
        fits_memory=fits_memory,
    )
    limiting_factor = _serving_limiting_factor(
        fits_configured_limit=fits_configured_limit,
        runtime=runtime,
        runtime_report=runtime_report,
        fits_memory=fits_memory,
        admissible_count=admissible_count,
        candidate_count=candidate_count,
        exhausted_factor="request_manifest_exhausted",
    )

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

    input_report = _serving_input_report(
        mode_inputs={
            "mode": "heterogeneous_request_set",
            "requests": normalized,
            "prefill_concurrency": prefill_concurrency,
        },
        active_sequences=requested_count,
        max_batch_size=max_batch_size,
        dtype=base["inputs"]["dtype"],
        element_bytes=element_bytes,
        attention_implementation=attention_implementation,
        prefill_chunk_tokens=prefill_chunk_tokens,
        kv_cache_strategy=kv_cache_strategy,
        kv_cache_bits=kv_cache_bits,
        kv_cache_residual_tokens=kv_cache_residual_tokens,
        kv_cache_block_tokens=kv_cache_block_tokens,
        kv_cache_max_tokens=kv_cache_max_tokens,
        kv_cache_window_tokens=kv_cache_window_tokens,
        runtime=runtime,
        memory_utilization=memory_utilization,
        vllm_kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
        vllm_non_kv_cache_memory_bytes=vllm_non_kv_cache_memory_bytes,
        runtime_overhead_bytes=runtime_overhead_bytes,
        scheduler_overhead_bytes_per_sequence=(
            scheduler_overhead_bytes_per_sequence
        ),
        speculative=speculative,
        speculative_tokens=speculative_tokens,
        speculative_acceptance_rate=speculative_acceptance_rate,
    )
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
        target_profile=capacity.target,
        capacity_source=capacity.capacity_source,
        selected_capacity_bytes=capacity.selected_capacity_bytes,
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


@dataclass(frozen=True, slots=True)
class _ServingCapacity:
    """Runtime selection and device capacity shared by both planners."""

    runtime: str
    memory_utilization: float
    target: dict[str, Any] | None
    selected_capacity_bytes: int | None
    capacity_source: str


@dataclass(frozen=True, slots=True)
class _ServingAdmission:
    """How much of the requested workload the device budget admits."""

    runtime_report: dict[str, Any] | None
    requested_for_report: Mapping[str, Any]
    usable_capacity_bytes: int | None
    memory_limited_count: int | None
    fits_memory: bool | None


def _resolve_serving_capacity(
    *,
    attention_implementation: str,
    kv_cache_strategy: str,
    runtime: str,
    memory_utilization: float | None,
    draft_model_dir: str | Path | None,
    vllm_kv_cache_memory_bytes: int | None,
    vllm_non_kv_cache_memory_bytes: int | None,
    device_capacity_bytes: int | None,
    speculative_tokens: int,
    speculative_acceptance_rate: float | None,
    target_profile: str | None,
) -> _ServingCapacity:
    """Validate the inputs both planners share and resolve the capacity."""

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
        vllm_non_kv_cache_memory_bytes=vllm_non_kv_cache_memory_bytes,
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
    return _ServingCapacity(
        runtime=runtime,
        memory_utilization=memory_utilization,
        target=target,
        selected_capacity_bytes=(
            device_capacity_bytes
            if device_capacity_bytes is not None
            else profile_capacity_bytes
        ),
        capacity_source=(
            "explicit_device_capacity"
            if device_capacity_bytes is not None
            else "gpu_profile"
            if profile_capacity_bytes is not None
            else "unavailable"
        ),
    )


def _serving_admission(
    *,
    runtime: str,
    requested: Mapping[str, Any],
    memory_for_count: Any,
    candidate_count: int,
    selected_capacity_bytes: int | None,
    memory_utilization: float,
    vllm_kv_cache_memory_bytes: int | None,
    vllm_non_kv_cache_memory_bytes: int | None,
    max_model_len_tokens: int,
    profile_scope: str,
) -> _ServingAdmission:
    """Admit as many of the first ``candidate_count`` sequences as fit.

    ``memory_for_count`` reports the pool memory for a sequence or request
    count; the homogeneous and heterogeneous planners differ only in what
    that callable measures.
    """

    if runtime == "vllm":
        runtime_report = _build_vllm_runtime_report(
            requested_memory=requested,
            profiled_memory=memory_for_count(candidate_count),
            selected_capacity_bytes=selected_capacity_bytes,
            gpu_memory_utilization=memory_utilization,
            kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
            non_kv_cache_memory_bytes=vllm_non_kv_cache_memory_bytes,
            max_model_len_tokens=max_model_len_tokens,
            profiled_sequence_count=candidate_count,
            profile_scope=profile_scope,
        )
        kv_cache_capacity_bytes = runtime_report["kv_cache"][
            "allocatable_bytes"
        ]
        if kv_cache_capacity_bytes is None:
            memory_limited_count = None
        elif runtime_report["memory"]["initialization_fits"] is False:
            memory_limited_count = 0
        else:
            memory_limited_count = _maximum_fitting(
                max_count=candidate_count,
                usable_capacity_bytes=int(kv_cache_capacity_bytes),
                memory_for_count=memory_for_count,
                metric=None,
                value_fn=_logical_kv_peak_bytes,
            )
        return _ServingAdmission(
            runtime_report=runtime_report,
            requested_for_report=_apply_vllm_kv_reservation(
                requested,
                runtime_report,
            ),
            usable_capacity_bytes=runtime_report["memory"][
                "memory_limit_bytes"
            ],
            memory_limited_count=memory_limited_count,
            fits_memory=runtime_report["requested_fits_memory"],
        )

    usable_capacity_bytes = (
        math.floor(int(selected_capacity_bytes) * memory_utilization)
        if selected_capacity_bytes is not None
        else None
    )
    if usable_capacity_bytes is None:
        return _ServingAdmission(
            runtime_report=None,
            requested_for_report=requested,
            usable_capacity_bytes=None,
            memory_limited_count=None,
            fits_memory=None,
        )
    return _ServingAdmission(
        runtime_report=None,
        requested_for_report=requested,
        usable_capacity_bytes=usable_capacity_bytes,
        memory_limited_count=_maximum_fitting(
            max_count=candidate_count,
            usable_capacity_bytes=usable_capacity_bytes,
            memory_for_count=memory_for_count,
            metric="peak_bytes",
        ),
        fits_memory=int(requested["peak_bytes"]) <= usable_capacity_bytes,
    )


def _requested_fits(
    *,
    fits_configured_limit: bool,
    fits_memory: bool | None,
) -> bool | None:
    if fits_memory is not None:
        return fits_configured_limit and fits_memory
    return False if not fits_configured_limit else None


def _serving_limiting_factor(
    *,
    fits_configured_limit: bool,
    runtime: str,
    runtime_report: Mapping[str, Any] | None,
    fits_memory: bool | None,
    admissible_count: int | None,
    candidate_count: int,
    exhausted_factor: str,
) -> str:
    """Name the constraint that bounds admission.

    ``exhausted_factor`` is what limits a plan that already admits every
    candidate: the configured batch size for a homogeneous pool, the end of
    the manifest for a request set.
    """

    if not fits_configured_limit:
        return "configured_max_batch_size"
    if runtime == "vllm" and runtime_report is not None:
        if runtime_report["memory"]["initialization_fits"] is False:
            return "vllm_model_executor_budget"
        if fits_memory is False:
            return "vllm_kv_cache_capacity"
        if fits_memory is None:
            return "device_capacity_unavailable"
        if admissible_count == candidate_count:
            return exhausted_factor
        return "vllm_kv_cache_capacity"
    if fits_memory is False:
        return "usable_device_memory"
    if fits_memory is None:
        return "device_capacity_unavailable"
    if admissible_count == candidate_count:
        return exhausted_factor
    return "usable_device_memory"


def _serving_input_report(
    *,
    mode_inputs: Mapping[str, Any],
    active_sequences: int,
    max_batch_size: int,
    dtype: str,
    element_bytes: int,
    attention_implementation: str,
    prefill_chunk_tokens: int | None,
    kv_cache_strategy: str,
    kv_cache_bits: int,
    kv_cache_residual_tokens: int,
    kv_cache_block_tokens: int,
    kv_cache_max_tokens: int | None,
    kv_cache_window_tokens: int | None,
    runtime: str,
    memory_utilization: float,
    vllm_kv_cache_memory_bytes: int | None,
    vllm_non_kv_cache_memory_bytes: int | None,
    runtime_overhead_bytes: int,
    scheduler_overhead_bytes_per_sequence: int,
    speculative: Mapping[str, Any] | None,
    speculative_tokens: int,
    speculative_acceptance_rate: float | None,
) -> dict[str, Any]:
    """Echo the inputs both planners report, plus the mode-specific ones."""

    return {
        **mode_inputs,
        "active_sequences": active_sequences,
        "max_batch_size": max_batch_size,
        "dtype": dtype,
        "element_bytes": element_bytes,
        "attention_implementation": attention_implementation,
        "prefill_chunk_tokens": prefill_chunk_tokens,
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
                "kv_cache_memory_bytes": vllm_kv_cache_memory_bytes,
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


@dataclass(frozen=True, slots=True)
class _ServingMemoryBreakdown:
    """Phase peaks and the memory section both pool models report."""

    prefill_peak: int
    decode_peak: int
    peak_phase: str
    phases: list[dict[str, Any]]
    memory: dict[str, Any]

    @property
    def peak_bytes(self) -> int:
        return max(self.prefill_peak, self.decode_peak)


def _serving_memory_breakdown(
    *,
    kv_cache: Mapping[str, Any],
    draft_kv_cache: Mapping[str, Any] | None,
    speculative: Mapping[str, Any] | None,
    parameter_bytes: int,
    sequence_count: int,
    runtime_overhead_bytes: int,
    scheduler_overhead_bytes_per_sequence: int,
    prefill_input_bytes: int,
    decode_input_bytes: int,
    prefill_transient: Mapping[str, Any],
    decode_transient: Mapping[str, Any],
    target_prefill_transient: Mapping[str, Any] | None,
    draft_prefill_transient: Mapping[str, Any] | None,
    target_verification_transient: Mapping[str, Any] | None,
    draft_proposal_transient: Mapping[str, Any] | None,
    memory_extra: Mapping[str, Any] | None = None,
) -> _ServingMemoryBreakdown:
    """Add up one phase peak per phase from the resident and transient parts.

    Both pool models resolve their own KV pool, transients, and input bytes;
    from there the residency arithmetic and the reported breakdown are the
    same, so only memory_extra (the per-request details of the
    heterogeneous model) differs in the result.
    """

    scheduler_overhead_bytes = (
        scheduler_overhead_bytes_per_sequence * sequence_count
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
    target_prefill_kv_bytes = int(kv_cache["prefill"]["allocated_bytes"])
    target_decode_kv_bytes = int(kv_cache["decode"]["allocated_bytes"])
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
    return _ServingMemoryBreakdown(
        prefill_peak=prefill_peak,
        decode_peak=decode_peak,
        peak_phase=(
            "prefill" if prefill_peak >= decode_peak else "decode"
        ),
        phases=phases,
        memory={
            "parameter_bytes": combined_parameter_bytes,
            "target_parameter_bytes": parameter_bytes,
            "draft_parameter_bytes": draft_parameter_bytes,
            "runtime_overhead_bytes": runtime_overhead_bytes,
            "scheduler_overhead_bytes": scheduler_overhead_bytes,
            "prefill_transient": prefill_transient,
            "decode_transient": decode_transient,
            "target_prefill_transient": target_prefill_transient,
            "draft_prefill_transient": draft_prefill_transient,
            "target_verification_transient": target_verification_transient,
            "draft_proposal_transient": draft_proposal_transient,
            **(memory_extra or {}),
            "estimated_prefill_peak_bytes": prefill_peak,
            "estimated_decode_peak_bytes": decode_peak,
            "estimated_process_peak_bytes": max(
                prefill_peak,
                decode_peak,
            ),
        },
    )


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
    breakdown = _serving_memory_breakdown(
        kv_cache=kv_cache,
        draft_kv_cache=draft_kv_cache,
        speculative=speculative,
        parameter_bytes=parameter_bytes,
        sequence_count=active_sequences,
        runtime_overhead_bytes=runtime_overhead_bytes,
        scheduler_overhead_bytes_per_sequence=(
            scheduler_overhead_bytes_per_sequence
        ),
        prefill_input_bytes=active_sequences * effective_chunk_tokens * 8,
        decode_input_bytes=(
            active_sequences * (effective_speculative_tokens or 1) * 8
        ),
        prefill_transient=prefill_transient,
        decode_transient=decode_transient,
        target_prefill_transient=target_prefill_transient,
        draft_prefill_transient=draft_prefill_transient,
        target_verification_transient=target_verification_transient,
        draft_proposal_transient=draft_proposal_transient,
    )
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
        "memory": breakdown.memory,
        "phases": breakdown.phases,
        "peak_bytes": breakdown.peak_bytes,
        "peak_phase": breakdown.peak_phase,
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
    breakdown = _serving_memory_breakdown(
        kv_cache=kv_cache,
        draft_kv_cache=draft_kv_cache,
        speculative=speculative,
        parameter_bytes=parameter_bytes,
        sequence_count=len(normalized),
        runtime_overhead_bytes=runtime_overhead_bytes,
        scheduler_overhead_bytes_per_sequence=(
            scheduler_overhead_bytes_per_sequence
        ),
        prefill_input_bytes=prefill_input_bytes,
        decode_input_bytes=decode_input_bytes,
        prefill_transient=prefill_transient,
        decode_transient=decode_transient,
        target_prefill_transient=target_prefill_transient,
        draft_prefill_transient=draft_prefill_transient,
        target_verification_transient=(
            target_decode_transient if speculative is not None else None
        ),
        draft_proposal_transient=draft_proposal_transient,
        memory_extra={"request_details": request_details},
    )
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
        "memory": breakdown.memory,
        "phases": breakdown.phases,
        "peak_bytes": breakdown.peak_bytes,
        "peak_phase": breakdown.peak_phase,
    }


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


def _format_bytes(value: int) -> str:
    return f"{int(value) / 2**30:.3f} GiB"


if __name__ == "__main__":
    raise SystemExit(main())
