"""Serving memory phase models shared by homogeneous and request-set plans.

This module contains only the pool-memory arithmetic. Public plan assembly
and admission decisions remain in :mod:`fakegpu.serving_plan`.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from typing import Any

from ._serving_kv import (
    _transient_bytes,
    estimate_serving_kv_pool,
    estimate_serving_request_kv_pool,
    serving_kv_model,
)
from ._serving_speculative import _select_larger_transient, _speculative_report
from .llm_estimator import DEFAULT_KV_QUANTIZATION_GROUP_SIZE
from .llm_estimator import _effective_context_tokens as _effective_tokens


def _prefix_hit_tokens(
    *,
    shared_prefix_tokens: int,
    prompt_tokens: int,
    kv_cache_strategy: str,
    block_tokens: int,
) -> int:
    """Prompt tokens a prefix-cache hit lets the prefill skip.

    A paged cache hits whole blocks only, and like vLLM's
    ``get_computed_blocks`` it caps the hit at prompt_tokens - 1 so the last
    prompt token is recomputed for its logits.
    """

    if kv_cache_strategy != "paged" or not shared_prefix_tokens:
        return shared_prefix_tokens
    capped = min(shared_prefix_tokens, prompt_tokens - 1)
    return capped // block_tokens * block_tokens


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
) -> dict[str, Any]:
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
    phases = []
    for phase, input_bytes, transient in (
        ("prefill", prefill_input_bytes, prefill_transient),
        ("decode", decode_input_bytes, decode_transient),
    ):
        target_kv_bytes = int(kv_cache[phase]["allocated_bytes"])
        draft_kv_bytes = (
            int(draft_kv_cache[phase]["allocated_bytes"])
            if draft_kv_cache is not None
            else 0
        )
        transient_bytes = int(transient["peak_bytes"])
        phases.append(
            {
                "phase": phase,
                "peak_bytes": (
                    combined_parameter_bytes
                    + target_kv_bytes
                    + draft_kv_bytes
                    + transient_bytes
                    + input_bytes
                    + fixed_overhead_bytes
                ),
                "components": {
                    "parameters": combined_parameter_bytes,
                    "target_parameters": parameter_bytes,
                    "draft_parameters": draft_parameter_bytes,
                    "inputs": input_bytes,
                    "kv_cache": target_kv_bytes + draft_kv_bytes,
                    "target_kv_cache": target_kv_bytes,
                    "draft_kv_cache": draft_kv_bytes,
                    "transient": transient_bytes,
                    "runtime_overhead": runtime_overhead_bytes,
                    "scheduler_overhead": scheduler_overhead_bytes,
                },
            }
        )
    prefill_peak = phases[0]["peak_bytes"]
    decode_peak = phases[1]["peak_bytes"]
    return {
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
            "target_verification_transient": target_verification_transient,
            "draft_proposal_transient": draft_proposal_transient,
            **(memory_extra or {}),
            "estimated_prefill_peak_bytes": prefill_peak,
            "estimated_decode_peak_bytes": decode_peak,
            "estimated_process_peak_bytes": max(prefill_peak, decode_peak),
        },
        "phases": phases,
        "peak_bytes": max(prefill_peak, decode_peak),
        "peak_phase": "prefill" if prefill_peak >= decode_peak else "decode",
    }


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
    kv_model: Mapping[str, Any] | None = None,
    mla_kv_cache_layout: str = "auto",
    kv_cache_group_size: int = DEFAULT_KV_QUANTIZATION_GROUP_SIZE,
) -> dict[str, Any]:
    effective_speculative_tokens = (
        min(int(speculative["speculative_tokens"]), generated_tokens)
        if speculative is not None
        else 0
    )
    cache_generation_tokens = (
        generated_tokens + effective_speculative_tokens
    )

    def kv_pool(kv: Mapping[str, Any]) -> dict[str, Any]:
        return estimate_serving_kv_pool(
            num_hidden_layers=int(kv["num_hidden_layers"]),
            num_key_value_heads=int(kv["num_key_value_heads"]),
            head_dim=int(kv["head_dim"]),
            active_sequences=active_sequences,
            prompt_tokens=prompt_tokens,
            generated_tokens=cache_generation_tokens,
            element_bytes=int(kv["element_bytes"]),
            strategy=kv_cache_strategy,
            shared_prefix_tokens=shared_prefix_tokens,
            quantized_bits=kv_cache_bits,
            quantized_residual_tokens=kv_cache_residual_tokens,
            block_tokens=kv_cache_block_tokens,
            max_cache_tokens=kv_cache_max_tokens,
            window_tokens=kv_cache_window_tokens,
            elements_per_token_per_layer=int(
                kv["elements_per_token_per_layer"]
            ),
            cache_layout=str(kv["cache_layout"]),
            sliding_window_layers=int(kv["sliding_window_layers"]),
            sliding_window_tokens=kv["sliding_window_tokens"],
            prefill_chunk_tokens=prefill_chunk_tokens,
            quantized_group_size=kv_cache_group_size,
        )

    def kv_description(
        model: Mapping[str, Any],
        model_element_bytes: int,
    ) -> Mapping[str, Any]:
        return serving_kv_model(
            model,
            element_bytes=model_element_bytes,
            kv_cache_strategy=kv_cache_strategy,
            mla_kv_cache_layout=mla_kv_cache_layout,
        )

    def transient(
        model: Mapping[str, Any],
        model_element_bytes: int,
        query_tokens: int,
        key_tokens: int,
    ) -> dict[str, int]:
        return _transient_bytes(
            model,
            batch_size=active_sequences,
            query_tokens=query_tokens,
            key_tokens=key_tokens,
            element_bytes=model_element_bytes,
            attention_implementation=attention_implementation,
        )

    kv_cache = kv_pool(
        kv_model
        if kv_model is not None
        else kv_description(dimensions, element_bytes)
    )
    prefix_hit_tokens = _prefix_hit_tokens(
        shared_prefix_tokens=shared_prefix_tokens,
        prompt_tokens=prompt_tokens,
        kv_cache_strategy=kv_cache_strategy,
        block_tokens=kv_cache_block_tokens,
    )
    uncached_prompt_tokens = prompt_tokens - prefix_hit_tokens
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
        prompt_tokens + generated_tokens - 1 + effective_speculative_tokens,
        window_tokens=kv_cache_window_tokens,
    )
    target_prefill_transient = transient(
        dimensions,
        element_bytes,
        effective_chunk_tokens,
        prefill_key_tokens,
    )
    unchunked_prefill_transient = (
        target_prefill_transient
        if prefill_chunk_tokens is None
        else transient(
            dimensions,
            element_bytes,
            uncached_prompt_tokens,
            prefill_key_tokens,
        )
    )
    if speculative is None:
        draft_kv_cache = None
        draft_prefill_transient = None
        target_verification_transient = None
        draft_proposal_transient = None
        prefill_transient = target_prefill_transient
        decode_transient = transient(
            dimensions,
            element_bytes,
            1,
            decode_key_tokens,
        )
        speculative_report = {"enabled": False}
    else:
        draft_dimensions = speculative["model"]
        draft_element_bytes = int(speculative["element_bytes"])
        draft_kv_cache = kv_pool(
            kv_description(draft_dimensions, draft_element_bytes)
        )
        draft_prefill_transient = transient(
            draft_dimensions,
            draft_element_bytes,
            effective_chunk_tokens,
            prefill_key_tokens,
        )
        target_verification_transient = transient(
            dimensions,
            element_bytes,
            effective_speculative_tokens,
            decode_key_tokens,
        )
        draft_proposal_transient = transient(
            draft_dimensions,
            draft_element_bytes,
            1,
            decode_key_tokens,
        )
        prefill_transient = _select_larger_transient(
            target_prefill_transient,
            draft_prefill_transient,
        )
        unchunked_prefill_transient = _select_larger_transient(
            unchunked_prefill_transient,
            transient(
                draft_dimensions,
                draft_element_bytes,
                uncached_prompt_tokens,
                prefill_key_tokens,
            ),
        )
        decode_transient = _select_larger_transient(
            target_verification_transient,
            draft_proposal_transient,
        )
        speculative_report = _speculative_report(
            speculative=speculative,
            effective_proposal_tokens={
                "homogeneous_pool": effective_speculative_tokens,
            },
            target_kv_cache=kv_cache,
            draft_kv_cache=draft_kv_cache,
            target_prefill_transient=target_prefill_transient,
            draft_prefill_transient=draft_prefill_transient,
            target_verification_transient=target_verification_transient,
            draft_proposal_transient=draft_proposal_transient,
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
                "hit_tokens": prefix_hit_tokens,
                "prompt_token_hit_percent": (
                    prefix_hit_tokens / prompt_tokens * 100
                ),
                "prefill_savings_bytes": int(
                    kv_cache["prefill"]["prefix_cache_savings_bytes"]
                ),
                "decode_savings_bytes": int(
                    kv_cache["decode"]["prefix_cache_savings_bytes"]
                ),
            },
        },
        **breakdown,
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
    kv_model: Mapping[str, Any] | None = None,
    mla_kv_cache_layout: str = "auto",
    kv_cache_group_size: int = DEFAULT_KV_QUANTIZATION_GROUP_SIZE,
) -> dict[str, Any]:
    effective_proposal_tokens = {
        request["id"]: min(
            int(speculative["speculative_tokens"]),
            request["generated_tokens"],
        )
        for request in requests
    } if speculative is not None else {}
    cache_requests = (
        [
            {
                **request,
                "generated_tokens": (
                    request["generated_tokens"]
                    + effective_proposal_tokens[request["id"]]
                ),
            }
            for request in requests
        ]
        if speculative is not None
        else requests
    )

    def kv_pool(kv: Mapping[str, Any]) -> dict[str, Any]:
        return estimate_serving_request_kv_pool(
            cache_requests,
            num_hidden_layers=int(kv["num_hidden_layers"]),
            num_key_value_heads=int(kv["num_key_value_heads"]),
            head_dim=int(kv["head_dim"]),
            element_bytes=int(kv["element_bytes"]),
            strategy=kv_cache_strategy,
            quantized_bits=kv_cache_bits,
            quantized_residual_tokens=kv_cache_residual_tokens,
            block_tokens=kv_cache_block_tokens,
            max_cache_tokens=kv_cache_max_tokens,
            window_tokens=kv_cache_window_tokens,
            elements_per_token_per_layer=int(
                kv["elements_per_token_per_layer"]
            ),
            cache_layout=str(kv["cache_layout"]),
            sliding_window_layers=int(kv["sliding_window_layers"]),
            sliding_window_tokens=kv["sliding_window_tokens"],
            prefill_chunk_tokens=prefill_chunk_tokens,
            quantized_group_size=kv_cache_group_size,
        )

    def kv_description(
        model: Mapping[str, Any],
        model_element_bytes: int,
    ) -> Mapping[str, Any]:
        return serving_kv_model(
            model,
            element_bytes=model_element_bytes,
            kv_cache_strategy=kv_cache_strategy,
            mla_kv_cache_layout=mla_kv_cache_layout,
        )

    def transient(
        model: Mapping[str, Any],
        model_element_bytes: int,
        query_tokens: int,
        key_tokens: int,
    ) -> dict[str, int]:
        return _transient_bytes(
            model,
            batch_size=1,
            query_tokens=query_tokens,
            key_tokens=key_tokens,
            element_bytes=model_element_bytes,
            attention_implementation=attention_implementation,
        )

    kv_cache = kv_pool(
        kv_model
        if kv_model is not None
        else kv_description(dimensions, element_bytes)
    )
    if speculative is not None:
        draft_dimensions = speculative["model"]
        draft_element_bytes = int(speculative["element_bytes"])

    request_details: list[dict[str, Any]] = []
    for request in requests:
        request_speculative_tokens = effective_proposal_tokens.get(
            request["id"],
            0,
        )
        uncached_prompt_tokens = request["prompt_tokens"] - _prefix_hit_tokens(
            shared_prefix_tokens=request["shared_prefix_tokens"],
            prompt_tokens=request["prompt_tokens"],
            kv_cache_strategy=kv_cache_strategy,
            block_tokens=kv_cache_block_tokens,
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
            + request["generated_tokens"]
            - 1
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
            "target_prefill_transient": transient(
                dimensions,
                element_bytes,
                effective_chunk_tokens,
                prefill_key_tokens,
            ),
            "target_unchunked_prefill_transient": transient(
                dimensions,
                element_bytes,
                uncached_prompt_tokens,
                prefill_key_tokens,
            ),
            "target_decode_transient": transient(
                dimensions,
                element_bytes,
                request_speculative_tokens or 1,
                decode_key_tokens,
            ),
        }
        if speculative is not None:
            detail.update(
                draft_prefill_transient=transient(
                    draft_dimensions,
                    draft_element_bytes,
                    effective_chunk_tokens,
                    prefill_key_tokens,
                ),
                draft_unchunked_prefill_transient=transient(
                    draft_dimensions,
                    draft_element_bytes,
                    uncached_prompt_tokens,
                    prefill_key_tokens,
                ),
                draft_proposal_transient=transient(
                    draft_dimensions,
                    draft_element_bytes,
                    1,
                    decode_key_tokens,
                ),
            )
        request_details.append(detail)

    effective_prefill_concurrency = min(
        prefill_concurrency,
        len(requests),
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
    if speculative is None:
        draft_kv_cache = None
        draft_prefill_transient = None
        draft_proposal_transient = None
        prefill_transient = target_prefill_transient
        unchunked_prefill_transient = (
            target_unchunked_prefill_transient
        )
        decode_transient = target_decode_transient
        speculative_report = {"enabled": False}
    else:
        draft_kv_cache = kv_pool(
            kv_description(draft_dimensions, draft_element_bytes)
        )
        draft_prefill_transient = _worst_concurrent_transient(
            request_details,
            field="draft_prefill_transient",
            concurrency=effective_prefill_concurrency,
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
            _worst_concurrent_transient(
                request_details,
                field="draft_unchunked_prefill_transient",
                concurrency=effective_prefill_concurrency,
            ),
        )
        decode_transient = _select_larger_transient(
            target_decode_transient,
            draft_proposal_transient,
        )
        speculative_report = _speculative_report(
            speculative=speculative,
            effective_proposal_tokens=effective_proposal_tokens,
            target_kv_cache=kv_cache,
            draft_kv_cache=draft_kv_cache,
            target_prefill_transient=target_prefill_transient,
            draft_prefill_transient=draft_prefill_transient,
            target_verification_transient=target_decode_transient,
            draft_proposal_transient=draft_proposal_transient,
        )

    prefill_input_requests = sorted(
        request_details,
        key=lambda item: item["effective_prefill_query_tokens"],
        reverse=True,
    )[:effective_prefill_concurrency]
    prefill_input_bytes = sum(
        request["effective_prefill_query_tokens"] * 8
        for request in prefill_input_requests
    )
    decode_input_bytes = sum(
        effective_proposal_tokens.get(request["id"], 1)
        for request in requests
    ) * 8
    breakdown = _serving_memory_breakdown(
        kv_cache=kv_cache,
        draft_kv_cache=draft_kv_cache,
        speculative=speculative,
        parameter_bytes=parameter_bytes,
        sequence_count=len(requests),
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
    prompt_lengths = [request["prompt_tokens"] for request in requests]
    generation_lengths = [
        request["generated_tokens"] for request in requests
    ]
    prefix_groups = {
        request["prefix_group"]
        for request in requests
        if request["prefix_group"] is not None
    }
    chunking_enabled = any(
        prefill_chunk_tokens is not None
        and request["effective_prefill_query_tokens"]
        < request["uncached_prompt_tokens"]
        for request in request_details
    )
    return {
        "kv_cache": kv_cache,
        "speculative_decoding": speculative_report,
        "optimizations": {
            "continuous_batching": {
                "active_sequences": len(requests),
                "request_ids": [
                    request["id"] for request in requests
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
                    for request in requests
                ),
                "prefill_savings_bytes": int(
                    kv_cache["prefill"]["prefix_cache_savings_bytes"]
                ),
                "decode_savings_bytes": int(
                    kv_cache["decode"]["prefix_cache_savings_bytes"]
                ),
            },
        },
        **breakdown,
    }


