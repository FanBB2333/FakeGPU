"""vLLM's memory budget: profile non-KV memory, reserve the rest as KV blocks.

Split out of ``serving_plan`` unchanged. vLLM sizes its paged KV pool once
at startup from a profiling run, so a plan for ``runtime="vllm"`` reports
the same split: the executor budget the model and its transients take, and
the whole blocks left over for the cache.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ._serving_types import ServingPlanError


VLLM_DEFAULT_GPU_MEMORY_UTILIZATION = 0.92


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


def _logical_kv_peak_bytes(memory: Mapping[str, Any]) -> int:
    return max(
        int(phase["components"]["kv_cache"])
        for phase in memory["phases"]
    )


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
