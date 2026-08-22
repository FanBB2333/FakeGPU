"""Serving memory phase models shared by homogeneous and request-set plans.

This module contains only the pool-memory arithmetic. Public plan assembly
and admission decisions remain in :mod:`fakegpu.serving_plan`.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._serving_kv import (
    _effective_tokens,
    _normalize_serving_requests,
    _transient_bytes,
    estimate_serving_kv_pool,
    estimate_serving_request_kv_pool,
)
from ._serving_speculative import _select_larger_transient, _speculative_report

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


