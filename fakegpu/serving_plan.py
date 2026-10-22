from __future__ import annotations

import argparse
import functools
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
    emit_json_report,
    usage_error,
)
from ._serving_kv import (
    REQUEST_MANIFEST_SCHEMA_VERSION,
    _normalize_serving_requests,
    estimate_serving_kv_pool,
    estimate_serving_request_kv_pool,
    load_serving_requests,
    serving_kv_model,
)
from ._serving_memory import _serving_batch_memory, _serving_request_set_memory
from ._serving_speculative import (
    DEFAULT_SPECULATIVE_TOKENS,
    _load_speculative_draft,
    _speculative_input_report,
    _validate_speculative_inputs,
)
from ._serving_types import (
    ServingPlanError,
    _nonnegative_integer,
    _positive_integer,
)
from ._serving_vllm import (
    VLLM_DEFAULT_GPU_MEMORY_UTILIZATION,
    VLLM_KV_CACHE_DTYPE_BYTES,
    VLLM_USAGE_CONTEXTS,
    _apply_vllm_kv_reservation,
    _build_vllm_runtime_report,
    _logical_kv_peak_bytes,
    resolve_vllm_batch_limits,
    vllm_default_block_tokens,
    vllm_kv_cache_groups,
    vllm_profile_memory,
)
from .llm_estimator import (
    DEFAULT_KV_QUANTIZATION_GROUP_SIZE,
    KV_CACHE_STRATEGIES,
    MLA_KV_CACHE_LAYOUTS,
    VISION_ESTIMATE_WARNING,
    VISION_UNMODELED_COMPONENTS,
    estimate_decoder_inference,
)
from .profile_catalog import get_profile


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
DEFAULT_MAX_BATCH_SIZE = 256
DEFAULT_KV_CACHE_BLOCK_TOKENS = 16


def estimate_serving_plan(
    model_dir: str | Path,
    *,
    active_sequences: int,
    max_batch_size: int | None = None,
    prompt_tokens: int,
    generated_tokens: int = 1,
    dtype: str = "auto",
    attention_implementation: str = "sdpa",
    prefill_chunk_tokens: int | None = None,
    shared_prefix_tokens: int = 0,
    kv_cache_strategy: str = "paged",
    kv_cache_bits: int = 4,
    kv_cache_residual_tokens: int = 128,
    kv_cache_block_tokens: int | None = None,
    kv_cache_max_tokens: int | None = None,
    kv_cache_window_tokens: int | None = None,
    kv_cache_group_size: int = DEFAULT_KV_QUANTIZATION_GROUP_SIZE,
    mla_kv_cache_layout: str = "auto",
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
    vllm_max_num_batched_tokens: int | None = None,
    vllm_kv_cache_dtype: str | None = None,
    vllm_usage_context: str | None = None,
) -> dict[str, Any]:
    """Plan memory admission for a homogeneous continuous-batching pool.

    With ``runtime="vllm"``, an unset ``max_batch_size`` (vLLM's
    max_num_seqs) and ``vllm_max_num_batched_tokens`` take vLLM's defaults
    for the target device and ``vllm_usage_context`` ("api_server", the
    ``vllm serve`` defaults, unless set to "llm_class").
    """

    _positive_integer(active_sequences, "active_sequences")
    if max_batch_size is not None:
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
        vllm_max_num_batched_tokens=vllm_max_num_batched_tokens,
        vllm_kv_cache_dtype=vllm_kv_cache_dtype,
        vllm_usage_context=vllm_usage_context,
        kv_cache_block_tokens=kv_cache_block_tokens,
        kv_cache_group_size=kv_cache_group_size,
        mla_kv_cache_layout=mla_kv_cache_layout,
        device_capacity_bytes=device_capacity_bytes,
        speculative_tokens=speculative_tokens,
        speculative_acceptance_rate=speculative_acceptance_rate,
        target_profile=target_profile,
    )
    runtime = capacity.runtime
    memory_utilization = capacity.memory_utilization

    base = _estimate_serving_decoder(
        model_dir,
        prompt_tokens=prompt_tokens,
        dtype=dtype,
        runtime=runtime,
        target=capacity.target,
        attention_implementation=attention_implementation,
        adapter_dirs=adapter_dirs,
    )
    dimensions = dict(base["model"])
    element_bytes = int(base["inputs"]["element_bytes"])
    parameter_bytes = int(base["memory"]["parameter_bytes"])
    layout = _resolve_serving_layout(
        capacity=capacity,
        dimensions=dimensions,
        element_bytes=element_bytes,
        kv_cache_strategy=kv_cache_strategy,
        kv_cache_block_tokens=kv_cache_block_tokens,
        kv_cache_window_tokens=kv_cache_window_tokens,
        mla_kv_cache_layout=mla_kv_cache_layout,
        max_batch_size=max_batch_size,
        max_model_len_tokens=prompt_tokens + generated_tokens,
        vllm_max_num_batched_tokens=vllm_max_num_batched_tokens,
        vllm_kv_cache_dtype=vllm_kv_cache_dtype,
        vllm_usage_context=vllm_usage_context,
    )
    max_batch_size = layout.max_batch_size
    kv_cache_block_tokens = layout.block_tokens
    prefill_chunk_tokens = _runtime_prefill_chunk_tokens(
        prefill_chunk_tokens,
        layout,
    )
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

    @functools.cache
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
            speculative=speculative,
            kv_model=layout.kv_model,
            mla_kv_cache_layout=mla_kv_cache_layout,
            kv_cache_group_size=kv_cache_group_size,
        )

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
        layout=layout,
        dimensions=dimensions,
        parameter_bytes=parameter_bytes,
        element_bytes=element_bytes,
        sm_count=capacity.sm_count,
        runtime_overhead_bytes=runtime_overhead_bytes,
        scheduler_overhead_bytes_per_sequence=(
            scheduler_overhead_bytes_per_sequence
        ),
    )
    runtime_report = admission.runtime_report
    requested_for_report = admission.requested_for_report
    usable_capacity_bytes = admission.usable_capacity_bytes
    memory_limited_sequences = admission.memory_limited_count
    fits_memory = admission.fits_memory
    admissible_sequences = memory_limited_sequences
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
        kv_cache_group_size=kv_cache_group_size,
        layout=layout,
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
    return _build_serving_report(
        schema_version=SCHEMA_VERSION,
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
                "Prefix caching shares retained prefix KV; a paged cache "
                "shares only full prefix blocks and keeps the partial last "
                "prefix block in each sequence's private segment."
            ),
            (
                "Admission uses the configured memory-utilization fraction "
                "and does not predict latency, throughput, or queue time."
            ),
        ],
        vllm_note=_VLLM_NOTE,
        speculative_note=(
            "Speculative decoding keeps target and draft weights and KV "
            "caches resident; a supplied acceptance assumption affects "
            "expected target calls but not the conservative memory peak."
        ),
    )


_VLLM_NOTE = (
    "vLLM mode models the startup profile run (one forward of "
    "max_num_batched_tokens tokens and a sampler pass over one row per "
    "profiled request), reserves the remaining executor budget as whole "
    "paged-cache blocks, chunks prefill at max_num_batched_tokens unless "
    "prefill_chunk_tokens is set, and admits by logical KV demand."
)


def estimate_serving_request_set(
    model_dir: str | Path,
    requests: Sequence[Mapping[str, Any]],
    *,
    max_batch_size: int | None = None,
    dtype: str = "auto",
    attention_implementation: str = "sdpa",
    prefill_chunk_tokens: int | None = None,
    prefill_concurrency: int = 1,
    kv_cache_strategy: str = "paged",
    kv_cache_bits: int = 4,
    kv_cache_residual_tokens: int = 128,
    kv_cache_block_tokens: int | None = None,
    kv_cache_max_tokens: int | None = None,
    kv_cache_window_tokens: int | None = None,
    kv_cache_group_size: int = DEFAULT_KV_QUANTIZATION_GROUP_SIZE,
    mla_kv_cache_layout: str = "auto",
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
    vllm_max_num_batched_tokens: int | None = None,
    vllm_kv_cache_dtype: str | None = None,
    vllm_usage_context: str | None = None,
) -> dict[str, Any]:
    """Plan ordered admission for a heterogeneous serving request set."""

    normalized = _normalize_serving_requests(requests)
    if max_batch_size is not None:
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
        vllm_max_num_batched_tokens=vllm_max_num_batched_tokens,
        vllm_kv_cache_dtype=vllm_kv_cache_dtype,
        vllm_usage_context=vllm_usage_context,
        kv_cache_block_tokens=kv_cache_block_tokens,
        kv_cache_group_size=kv_cache_group_size,
        mla_kv_cache_layout=mla_kv_cache_layout,
        device_capacity_bytes=device_capacity_bytes,
        speculative_tokens=speculative_tokens,
        speculative_acceptance_rate=speculative_acceptance_rate,
        target_profile=target_profile,
    )
    runtime = capacity.runtime
    memory_utilization = capacity.memory_utilization

    def max_model_len(count: int) -> int:
        return max(
            int(request["prompt_tokens"]) + int(request["generated_tokens"])
            for request in normalized[:count]
        )

    max_prompt_tokens = max(
        request["prompt_tokens"] for request in normalized
    )
    base = _estimate_serving_decoder(
        model_dir,
        prompt_tokens=max_prompt_tokens,
        dtype=dtype,
        runtime=runtime,
        target=capacity.target,
        attention_implementation=attention_implementation,
        adapter_dirs=adapter_dirs,
    )
    dimensions = dict(base["model"])
    element_bytes = int(base["inputs"]["element_bytes"])
    parameter_bytes = int(base["memory"]["parameter_bytes"])
    layout_inputs = dict(
        capacity=capacity,
        dimensions=dimensions,
        element_bytes=element_bytes,
        kv_cache_strategy=kv_cache_strategy,
        kv_cache_block_tokens=kv_cache_block_tokens,
        kv_cache_window_tokens=kv_cache_window_tokens,
        mla_kv_cache_layout=mla_kv_cache_layout,
        max_batch_size=max_batch_size,
        vllm_max_num_batched_tokens=vllm_max_num_batched_tokens,
        vllm_kv_cache_dtype=vllm_kv_cache_dtype,
        vllm_usage_context=vllm_usage_context,
    )
    # vLLM's max_num_seqs default does not depend on max_model_len, so a
    # first pass fixes the candidate prefix whose longest request then sets
    # max_model_len (and the max_num_batched_tokens default).
    candidate_count = min(
        _resolve_serving_layout(
            **layout_inputs,
            max_model_len_tokens=max_model_len(len(normalized)),
        ).max_batch_size,
        len(normalized),
    )
    layout = _resolve_serving_layout(
        **layout_inputs,
        max_model_len_tokens=max_model_len(candidate_count),
    )
    max_batch_size = layout.max_batch_size
    kv_cache_block_tokens = layout.block_tokens
    prefill_chunk_tokens = _runtime_prefill_chunk_tokens(
        prefill_chunk_tokens,
        layout,
    )
    speculative = _load_speculative_draft(
        draft_model_dir=draft_model_dir,
        draft_dtype=draft_dtype,
        prompt_tokens=max_prompt_tokens,
        attention_implementation=attention_implementation,
        target_dimensions=dimensions,
        target_parameter_bytes=parameter_bytes,
        speculative_tokens=speculative_tokens,
        acceptance_rate=speculative_acceptance_rate,
    )

    @functools.cache
    def request_memory(request_count: int) -> dict[str, Any]:
        return _serving_request_set_memory(
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
            kv_model=layout.kv_model,
            mla_kv_cache_layout=mla_kv_cache_layout,
            kv_cache_group_size=kv_cache_group_size,
        )

    requested_count = len(normalized)
    requested = request_memory(requested_count)
    admission = _serving_admission(
        runtime=runtime,
        requested=requested,
        memory_for_count=request_memory,
        candidate_count=candidate_count,
        selected_capacity_bytes=capacity.selected_capacity_bytes,
        memory_utilization=memory_utilization,
        vllm_kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
        vllm_non_kv_cache_memory_bytes=vllm_non_kv_cache_memory_bytes,
        max_model_len_tokens=max_model_len(candidate_count),
        layout=layout,
        dimensions=dimensions,
        parameter_bytes=parameter_bytes,
        element_bytes=element_bytes,
        sm_count=capacity.sm_count,
        runtime_overhead_bytes=runtime_overhead_bytes,
        scheduler_overhead_bytes_per_sequence=(
            scheduler_overhead_bytes_per_sequence
        ),
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
        runtime_report=runtime_report,
        fits_memory=fits_memory,
        admissible_count=admissible_count,
        candidate_count=candidate_count,
        exhausted_factor="request_manifest_exhausted",
    )

    if admissible_count is None:
        admitted_request_ids = rejected_request_ids = None
        admitted_request_count = rejected_request_count = None
    else:
        admitted_request_ids = [
            request["id"] for request in normalized[:admissible_count]
        ]
        rejected_request_ids = [
            request["id"] for request in normalized[admissible_count:]
        ]
        admitted_request_count = len(admitted_request_ids)
        rejected_request_count = len(rejected_request_ids)

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
        kv_cache_group_size=kv_cache_group_size,
        layout=layout,
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
    return _build_serving_report(
        schema_version=REQUEST_SET_SCHEMA_VERSION,
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
            "admitted_request_count": admitted_request_count,
            "rejected_request_count": rejected_request_count,
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
            _VLLM_NOTE[:-1]
            + "; it admits the longest request prefix whose logical cache "
            "demand fits that pool."
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
    workload_signature = _serving_workload_signature(
        report_schema_version=schema_version,
        model=model,
        checkpoint=checkpoint,
        weight_storage=weight_storage,
        inputs=inputs,
        speculative=speculative,
    )
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
    if model.get("has_vision_encoder"):
        unmodeled_components.extend(VISION_UNMODELED_COMPONENTS)
        report_notes.append(VISION_ESTIMATE_WARNING)
    if speculative is not None and speculative["model"].get("has_vision_encoder"):
        unmodeled_components.extend(
            f"draft_{component}" for component in VISION_UNMODELED_COMPONENTS
        )
        report_notes.append(f"Draft model: {VISION_ESTIMATE_WARNING}")
    if runtime == "vllm":
        unmodeled_components.extend(
            [
                "vllm_cuda_graph_capture_memory",
                "vllm_torch_compile_profile_activation_changes",
                "vllm_attention_backend_profile_workspaces",
            ]
        )
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


def _estimate_serving_decoder(
    model_dir: str | Path,
    *,
    prompt_tokens: int,
    dtype: str,
    runtime: str,
    target: Mapping[str, Any] | None,
    attention_implementation: str,
    adapter_dirs: Sequence[str | Path] | None,
) -> dict[str, Any]:
    """Read the decoder shape at the dtype the serving runtime computes in."""

    def estimate(selected_dtype: str) -> dict[str, Any]:
        return estimate_decoder_inference(
            model_dir,
            batch_size=1,
            prompt_tokens=prompt_tokens,
            generated_tokens=1,
            dtype=selected_dtype,
            use_cache=False,
            attention_implementation=attention_implementation,
            runtime_overhead_bytes=0,
            adapter_dirs=adapter_dirs,
        )

    base = estimate(dtype)
    if (
        runtime == "vllm"
        and dtype == "auto"
        and base["inputs"]["dtype"] == "float32"
    ):
        # vLLM's dtype="auto" downcasts float32 checkpoints to the
        # platform's first supported dtype: bfloat16 from compute
        # capability 8.0, float16 before it.
        capability = (target or {}).get("compute_capability")
        major = int(str(capability).split(".")[0]) if capability else 8
        base = estimate("bfloat16" if major >= 8 else "float16")
    return base


@dataclass(frozen=True, slots=True)
class _ServingCapacity:
    """Runtime selection and device capacity shared by both planners."""

    runtime: str
    memory_utilization: float
    target: dict[str, Any] | None
    selected_capacity_bytes: int | None
    capacity_source: str
    sm_count: int | None = None
    device_name: str | None = None


@dataclass(frozen=True, slots=True)
class _ServingLayout:
    """Cache layout and scheduler limits resolved for one plan.

    ``kv_model`` describes the cache the runtime keeps; in vLLM mode the
    layer counts include the padding of hybrid KV cache groups and the
    element size follows the cache dtype.
    """

    block_tokens: int
    kv_model: dict[str, Any]
    kv_groups: dict[str, Any] | None
    kv_cache_dtype: str | None
    batch_limits: dict[str, Any] | None
    max_batch_size: int


def _resolve_serving_layout(
    *,
    capacity: _ServingCapacity,
    dimensions: Mapping[str, Any],
    element_bytes: int,
    kv_cache_strategy: str,
    kv_cache_block_tokens: int | None,
    kv_cache_window_tokens: int | None,
    mla_kv_cache_layout: str,
    max_batch_size: int | None,
    max_model_len_tokens: int,
    vllm_max_num_batched_tokens: int | None,
    vllm_kv_cache_dtype: str | None,
    vllm_usage_context: str | None,
) -> _ServingLayout:
    """Resolve block size, cache layout and batch limits for a plan."""

    if capacity.runtime != "vllm":
        return _ServingLayout(
            block_tokens=(
                DEFAULT_KV_CACHE_BLOCK_TOKENS
                if kv_cache_block_tokens is None
                else kv_cache_block_tokens
            ),
            kv_model=serving_kv_model(
                dimensions,
                element_bytes=element_bytes,
                kv_cache_strategy=kv_cache_strategy,
                mla_kv_cache_layout=mla_kv_cache_layout,
            ),
            kv_groups=None,
            kv_cache_dtype=None,
            batch_limits=None,
            max_batch_size=(
                DEFAULT_MAX_BATCH_SIZE
                if max_batch_size is None
                else max_batch_size
            ),
        )

    kv_cache_dtype = vllm_kv_cache_dtype or "auto"
    kv_element_bytes = VLLM_KV_CACHE_DTYPE_BYTES[kv_cache_dtype]
    kv_model = serving_kv_model(
        dimensions,
        element_bytes=(
            element_bytes if kv_element_bytes is None else kv_element_bytes
        ),
        kv_cache_strategy=kv_cache_strategy,
        mla_kv_cache_layout=mla_kv_cache_layout,
    )
    if kv_cache_window_tokens is not None:
        # The explicit window applies to every layer, so one group holds
        # them all.
        kv_model["sliding_window_layers"] = 0
        kv_model["sliding_window_tokens"] = None
    kv_groups = vllm_kv_cache_groups(kv_model)
    # Every block id of a hybrid model spends one page in group_size layers
    # of each group, so the padded layer counts set the cache demand.
    kv_model["num_hidden_layers"] = int(kv_groups["padded_layers"])
    kv_model["sliding_window_layers"] = int(
        kv_groups["padded_sliding_layers"]
    )
    batch_limits = resolve_vllm_batch_limits(
        max_num_batched_tokens=vllm_max_num_batched_tokens,
        max_num_seqs=max_batch_size,
        max_model_len_tokens=max_model_len_tokens,
        device_memory_bytes=capacity.selected_capacity_bytes,
        device_name=capacity.device_name,
        usage_context=vllm_usage_context or "api_server",
    )
    return _ServingLayout(
        block_tokens=(
            vllm_default_block_tokens(
                dimensions,
                compute_capability=(capacity.target or {}).get(
                    "compute_capability"
                ),
            )
            if kv_cache_block_tokens is None
            else kv_cache_block_tokens
        ),
        kv_model=kv_model,
        kv_groups=kv_groups,
        kv_cache_dtype=kv_cache_dtype,
        batch_limits=batch_limits,
        max_batch_size=int(batch_limits["max_num_seqs"]),
    )


def _runtime_prefill_chunk_tokens(
    prefill_chunk_tokens: int | None,
    layout: _ServingLayout,
) -> int | None:
    """vLLM V1 always chunks prefill at the max_num_batched_tokens budget."""

    if prefill_chunk_tokens is not None or layout.batch_limits is None:
        return prefill_chunk_tokens
    return int(layout.batch_limits["max_num_batched_tokens"])


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
    vllm_max_num_batched_tokens: int | None,
    vllm_kv_cache_dtype: str | None,
    vllm_usage_context: str | None,
    kv_cache_block_tokens: int | None,
    kv_cache_group_size: int,
    mla_kv_cache_layout: str,
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
        vllm_max_num_batched_tokens=vllm_max_num_batched_tokens,
        vllm_kv_cache_dtype=vllm_kv_cache_dtype,
        vllm_usage_context=vllm_usage_context,
    )
    if kv_cache_block_tokens is not None:
        _positive_integer(kv_cache_block_tokens, "kv_cache_block_tokens")
    _positive_integer(kv_cache_group_size, "kv_cache_group_size")
    if mla_kv_cache_layout not in MLA_KV_CACHE_LAYOUTS:
        choices = ", ".join(sorted(MLA_KV_CACHE_LAYOUTS))
        raise ServingPlanError(
            f"mla_kv_cache_layout must be one of: {choices}"
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
    sm_count = None
    device_name = None
    if target_profile:
        profile = get_profile(target_profile)
        profile_capacity_bytes = profile.memory_bytes
        sm_count = int(profile.sm_count) or None
        device_name = profile.torch_name
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
        sm_count=sm_count,
        device_name=device_name,
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
    layout: _ServingLayout,
    dimensions: Mapping[str, Any],
    parameter_bytes: int,
    element_bytes: int,
    sm_count: int | None,
    runtime_overhead_bytes: int,
    scheduler_overhead_bytes_per_sequence: int,
) -> _ServingAdmission:
    """Admit as many of the first ``candidate_count`` sequences as fit.

    ``memory_for_count`` reports the pool memory for a sequence or request
    count; the homogeneous and heterogeneous planners differ only in what
    that callable measures.
    """

    if runtime == "vllm":
        batch_limits = layout.batch_limits or {}
        max_num_seqs = int(batch_limits["max_num_seqs"])
        profile = vllm_profile_memory(
            dimensions,
            parameter_bytes=parameter_bytes,
            element_bytes=element_bytes,
            max_num_batched_tokens=int(
                batch_limits["max_num_batched_tokens"]
            ),
            max_num_seqs=max_num_seqs,
            sm_count=sm_count,
            runtime_overhead_bytes=runtime_overhead_bytes,
            scheduler_overhead_bytes=(
                scheduler_overhead_bytes_per_sequence * max_num_seqs
            ),
        )
        runtime_report = _build_vllm_runtime_report(
            requested_memory=requested,
            profile=profile,
            selected_capacity_bytes=selected_capacity_bytes,
            gpu_memory_utilization=memory_utilization,
            kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
            non_kv_cache_memory_bytes=vllm_non_kv_cache_memory_bytes,
            max_model_len_tokens=max_model_len_tokens,
            batch_limits=batch_limits,
            kv_model=layout.kv_model,
            kv_groups=layout.kv_groups or {},
            kv_cache_dtype=str(layout.kv_cache_dtype),
        )
        kv_cache_capacity_bytes = runtime_report["kv_cache"][
            "usable_bytes"
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

    if selected_capacity_bytes is None:
        return _ServingAdmission(
            runtime_report=None,
            requested_for_report=requested,
            usable_capacity_bytes=None,
            memory_limited_count=None,
            fits_memory=None,
        )
    usable_capacity_bytes = math.floor(
        int(selected_capacity_bytes) * memory_utilization
    )
    return _ServingAdmission(
        runtime_report=None,
        requested_for_report=requested,
        usable_capacity_bytes=usable_capacity_bytes,
        memory_limited_count=_maximum_fitting(
            max_count=candidate_count,
            usable_capacity_bytes=usable_capacity_bytes,
            memory_for_count=memory_for_count,
            value_fn=lambda memory: int(memory["peak_bytes"]),
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
    memory_factor = "usable_device_memory"
    if runtime_report is not None:
        if runtime_report["memory"]["initialization_fits"] is False:
            return "vllm_model_executor_budget"
        memory_factor = "vllm_kv_cache_capacity"
    if fits_memory is False:
        return memory_factor
    if fits_memory is None:
        return "device_capacity_unavailable"
    if admissible_count == candidate_count:
        return exhausted_factor
    return memory_factor


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
    kv_cache_group_size: int,
    layout: _ServingLayout,
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
        "kv_cache_group_size": (
            kv_cache_group_size if kv_cache_strategy == "quantized" else None
        ),
        "mla_kv_cache_layout": layout.kv_model["mla_layout"],
        "runtime": runtime,
        "vllm": (
            {
                "gpu_memory_utilization": memory_utilization,
                "kv_cache_memory_bytes": vllm_kv_cache_memory_bytes,
                "non_kv_cache_memory_bytes": (
                    vllm_non_kv_cache_memory_bytes
                ),
                "kv_cache_dtype": layout.kv_cache_dtype,
                **{
                    key: (layout.batch_limits or {})[key]
                    for key in (
                        "usage_context",
                        "max_num_batched_tokens",
                        "max_num_seqs",
                    )
                },
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
    portable_inputs = {
        **inputs,
        "speculative_decoding": {
            key: value
            for key, value in inputs["speculative_decoding"].items()
            if key != "draft_model_dir"
        },
    }

    draft_identity = None
    if speculative is not None:
        draft_identity = {
            "model": _portable_model_identity(speculative["model"]),
            "checkpoint": speculative["checkpoint"],
            "weight_storage": _portable_weight_storage(
                speculative["weight_storage"]
            ),
        }
    identity = {
        "signature_schema_version": WORKLOAD_SIGNATURE_SCHEMA_VERSION,
        "report_schema_version": report_schema_version,
        "target": {
            "model": _portable_model_identity(model),
            "checkpoint": checkpoint,
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
    return {
        **weight_storage,
        "adapters": [
            {
                key: value
                for key, value in adapter.items()
                if key != "path"
            }
            for adapter in weight_storage["adapters"]
        ],
    }


def _serving_runtime_configuration(
    *,
    runtime: str,
    memory_utilization: float | None,
    kv_cache_strategy: str,
    draft_model_dir: str | Path | None,
    vllm_kv_cache_memory_bytes: int | None,
    vllm_non_kv_cache_memory_bytes: int | None,
    vllm_max_num_batched_tokens: int | None = None,
    vllm_kv_cache_dtype: str | None = None,
    vllm_usage_context: str | None = None,
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
        "vllm_max_num_batched_tokens": vllm_max_num_batched_tokens,
        "vllm_kv_cache_dtype": vllm_kv_cache_dtype,
        "vllm_usage_context": vllm_usage_context,
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
    if vllm_max_num_batched_tokens is not None:
        _positive_integer(
            vllm_max_num_batched_tokens,
            "vllm_max_num_batched_tokens",
        )
    if (
        vllm_kv_cache_dtype is not None
        and vllm_kv_cache_dtype not in VLLM_KV_CACHE_DTYPE_BYTES
    ):
        choices = ", ".join(VLLM_KV_CACHE_DTYPE_BYTES)
        raise ServingPlanError(
            f"vllm_kv_cache_dtype must be one of: {choices}"
        )
    if (
        vllm_usage_context is not None
        and vllm_usage_context not in VLLM_USAGE_CONTEXTS
    ):
        choices = ", ".join(VLLM_USAGE_CONTEXTS)
        raise ServingPlanError(
            f"vllm_usage_context must be one of: {choices}"
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
    parser.add_argument(
        "--max-batch-size",
        type=int,
        help=(
            "Concurrent sequence limit (vLLM max_num_seqs). Default: "
            f"{DEFAULT_MAX_BATCH_SIZE}, or vLLM's default for the target "
            "with --runtime vllm."
        ),
    )
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
        help=(
            "Quantized-cache residual length: generated tokens are kept "
            "in full precision until this many accumulate, then the "
            "cache is re-quantized (default: 128)."
        ),
    )
    parser.add_argument(
        "--kv-cache-group-size",
        type=int,
        default=DEFAULT_KV_QUANTIZATION_GROUP_SIZE,
        help=(
            "Quantized-cache q_group_size; each group stores a scale and "
            "a zero point in the compute dtype "
            f"(default: {DEFAULT_KV_QUANTIZATION_GROUP_SIZE})."
        ),
    )
    parser.add_argument(
        "--kv-cache-block-tokens",
        type=int,
        help=(
            "Paged-cache block size in tokens. Default: "
            f"{DEFAULT_KV_CACHE_BLOCK_TOKENS}; with --runtime vllm, the "
            "attention backend's default for the model and target."
        ),
    )
    parser.add_argument("--kv-cache-max-tokens", type=int)
    parser.add_argument(
        "--kv-cache-window-tokens",
        type=int,
        help=(
            "Override: apply one attention window to every layer instead "
            "of the model's configured sliding-window layers."
        ),
    )
    parser.add_argument(
        "--mla-kv-cache-layout",
        choices=sorted(MLA_KV_CACHE_LAYOUTS),
        default="auto",
        help=(
            "Cache layout for multi-head latent attention models: latent "
            "(vLLM: compressed latent plus rope key) or decompressed "
            "(Hugging Face transformers: per-head keys and values). auto "
            "picks latent for the paged strategy, decompressed otherwise."
        ),
    )
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
        "--vllm-max-num-batched-tokens",
        type=int,
        help=(
            "vLLM max_num_batched_tokens; sets the profile-run forward "
            "size. Default: vLLM's default for the target and usage "
            "context."
        ),
    )
    parser.add_argument(
        "--vllm-kv-cache-dtype",
        choices=list(VLLM_KV_CACHE_DTYPE_BYTES),
        help="vLLM --kv-cache-dtype (default: auto, the model dtype).",
    )
    parser.add_argument(
        "--vllm-usage-context",
        choices=list(VLLM_USAGE_CONTEXTS),
        help=(
            "Which vLLM entry point sets the batch defaults: api_server "
            "(vllm serve, default) or llm_class (the offline LLM class)."
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
    common = dict(
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        prefill_chunk_tokens=args.prefill_chunk_tokens,
        kv_cache_strategy=args.kv_cache_strategy,
        kv_cache_bits=args.kv_cache_bits,
        kv_cache_residual_tokens=args.kv_cache_residual_tokens,
        kv_cache_block_tokens=args.kv_cache_block_tokens,
        kv_cache_max_tokens=args.kv_cache_max_tokens,
        kv_cache_window_tokens=args.kv_cache_window_tokens,
        kv_cache_group_size=args.kv_cache_group_size,
        mla_kv_cache_layout=args.mla_kv_cache_layout,
        runtime_overhead_bytes=args.runtime_overhead_bytes,
        scheduler_overhead_bytes_per_sequence=(
            args.scheduler_overhead_bytes_per_sequence
        ),
        adapter_dirs=args.adapter_dir,
        draft_model_dir=args.draft_model_dir,
        draft_dtype=args.draft_dtype or "auto",
        speculative_tokens=speculative_tokens,
        speculative_acceptance_rate=args.speculative_acceptance_rate,
        target_profile=args.target_profile,
        device_capacity_bytes=device_capacity_bytes,
        memory_utilization=args.memory_utilization,
        runtime=args.runtime,
        vllm_kv_cache_memory_bytes=args.vllm_kv_cache_memory_bytes,
        vllm_non_kv_cache_memory_bytes=(
            args.vllm_non_kv_cache_memory_bytes
        ),
        vllm_max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        vllm_kv_cache_dtype=args.vllm_kv_cache_dtype,
        vllm_usage_context=args.vllm_usage_context,
    )

    try:
        if args.requests:
            request_manifest = load_serving_requests(args.requests)
            report = estimate_serving_request_set(
                args.model_dir,
                request_manifest["requests"],
                max_batch_size=args.max_batch_size,
                prefill_concurrency=args.prefill_concurrency,
                **common,
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
                shared_prefix_tokens=(
                    0
                    if args.shared_prefix_tokens is None
                    else args.shared_prefix_tokens
                ),
                **common,
            )
    except (
            OSError,
            ValueError,
           ) as exc:
        usage_error(parser, exc)

    if args.json_path:
        emit_json_report(args.json_path, report, "Serving plan")

    scheduler = report["scheduler"]
    timeline = report["memory_timeline"]
    print("FakeGPU LLM serving plan")
    if "vision_encoder_activations" in report["unmodeled_components"]:
        print(f"  warning: {VISION_ESTIMATE_WARNING}")
    if "draft_vision_encoder_activations" in report["unmodeled_components"]:
        print(f"  warning: draft model: {VISION_ESTIMATE_WARNING}")
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


def _maximum_fitting(
    *,
    max_count: int,
    usable_capacity_bytes: int,
    memory_for_count: Any,
    value_fn: Any,
) -> int:
    """Return the largest monotonically fitting count via one binary search."""
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


def _format_bytes(value: int) -> str:
    return f"{int(value) / 2**30:.3f} GiB"


if __name__ == "__main__":
    raise SystemExit(main())
