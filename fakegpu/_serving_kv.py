"""KV-pool sizing and the request manifest a serving plan is built from.

Split out of ``serving_plan`` unchanged: how many bytes a homogeneous pool
or a set of heterogeneous requests reserves for its KV cache under each
strategy, how a shared prefix is split from private cache segments, the
per-phase transient bytes an attention implementation needs, and the
normalization of a ``fakegpu.serving_requests.v1`` manifest.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._serving_types import (
    ServingPlanError,
    _nonnegative_integer,
    _positive_integer,
)
from .llm_estimator import (
    KV_CACHE_STRATEGIES,
    _forward_transient_bytes,
    estimate_kv_cache_memory,
)
from .structured_io import load_mapping


REQUEST_MANIFEST_SCHEMA_VERSION = "fakegpu.serving_requests.v1"


_PREFIX_CACHE_STRATEGIES = frozenset({"dynamic", "paged"})


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


def _retained_prefix_tokens(
    *,
    shared_prefix_tokens: int,
    logical_tokens: int,
    effective_tokens: int,
) -> int:
    evicted_tokens = max(0, logical_tokens - effective_tokens)
    return max(0, shared_prefix_tokens - evicted_tokens)


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
