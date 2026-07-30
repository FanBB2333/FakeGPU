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


SCHEMA_VERSION = "fakegpu.llm_serving_plan.v1"
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu plan-serving",
        description=(
            "Estimate online LLM serving memory and continuous-batching "
            "admission without loading checkpoint tensors."
        ),
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--active-sequences", type=int, required=True)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--generated-tokens", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--attention-implementation",
        choices=["eager", "sdpa"],
        default="sdpa",
    )
    parser.add_argument("--prefill-chunk-tokens", type=int)
    parser.add_argument("--shared-prefix-tokens", type=int, default=0)
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

    try:
        report = estimate_serving_plan(
            args.model_dir,
            active_sequences=args.active_sequences,
            max_batch_size=args.max_batch_size,
            prompt_tokens=args.prompt_tokens,
            generated_tokens=args.generated_tokens,
            dtype=args.dtype,
            attention_implementation=args.attention_implementation,
            prefill_chunk_tokens=args.prefill_chunk_tokens,
            shared_prefix_tokens=args.shared_prefix_tokens,
            kv_cache_strategy=args.kv_cache_strategy,
            kv_cache_bits=args.kv_cache_bits,
            kv_cache_residual_tokens=args.kv_cache_residual_tokens,
            kv_cache_block_tokens=args.kv_cache_block_tokens,
            kv_cache_max_tokens=args.kv_cache_max_tokens,
            kv_cache_window_tokens=args.kv_cache_window_tokens,
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
