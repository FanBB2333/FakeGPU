"""vLLM's memory budget: profile non-KV memory, reserve the rest as KV blocks.

vLLM sizes its paged KV pool once at startup from a profiling run, so a plan
for ``runtime="vllm"`` reports the same split: the executor budget the model
and its profile-run transients take, and the whole blocks left over for the
cache. The scheduler defaults, the profile run, the KV cache groups of
hybrid sliding-window models and the startup log values follow the vLLM
0.19.1 sources (``engine/arg_utils.py``, ``v1/worker/gpu_model_runner.py``,
``v1/worker/gpu_worker.py``, ``v1/core/kv_cache_utils.py``,
``v1/kv_cache_interface.py``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ._serving_types import ServingPlanError
from .llm_estimator import _forward_transient_bytes


VLLM_DEFAULT_GPU_MEMORY_UTILIZATION = 0.92
VLLM_REFERENCE_VERSION = "0.19.1"
VLLM_USAGE_CONTEXTS = ("api_server", "llm_class")
# CacheConfig.cache_dtype choices and their storage bytes (fp8 variants are
# stored as uint8); "auto" keeps the model dtype.
VLLM_KV_CACHE_DTYPE_BYTES: dict[str, int | None] = {
    "auto": None,
    "bfloat16": 2,
    "float16": 2,
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
}
# Modeled default, not a vLLM constant: the non-torch memory growth vLLM's
# profiler counted as non-KV memory in single-GPU vLLM 0.19.1 startups on an
# Ampere GPU (sm_80), 106-128 MiB across four models. It covers library
# handles and loaded kernel modules and varies with the software stack.
VLLM_MODELED_NON_TORCH_MEMORY_BYTES = 112 * 2**20
_GIB = 2**30
# Below this many requests vLLM's top-k/top-p falls back to a PyTorch sort.
_TRITON_TOP_K_TOP_P_MIN_ROWS = 8


def resolve_vllm_batch_limits(
    *,
    max_num_batched_tokens: int | None,
    max_num_seqs: int | None,
    max_model_len_tokens: int,
    device_memory_bytes: int | None,
    device_name: str | None,
    usage_context: str,
) -> dict[str, Any]:
    """Resolve vLLM's max_num_batched_tokens and max_num_seqs defaults.

    ``EngineArgs.get_batch_defaults`` picks the larger table at 70 GiB of
    device memory unless the device name contains "a100";
    ``_set_default_max_num_seqs_and_batched_tokens_args`` then caps the
    token budget at max_num_seqs * max_model_len (chunked prefill on) and
    the sequence limit at the token budget.
    """

    if usage_context not in VLLM_USAGE_CONTEXTS:
        choices = ", ".join(VLLM_USAGE_CONTEXTS)
        raise ServingPlanError(f"vllm_usage_context must be one of: {choices}")
    large_device = (
        device_memory_bytes is not None
        and device_memory_bytes >= 70 * _GIB
        and "a100" not in str(device_name or "").lower()
    )
    table_tokens = {
        "llm_class": 16384 if large_device else 8192,
        "api_server": 8192 if large_device else 2048,
    }[usage_context]
    table_seqs = 1024 if large_device else 256
    seqs = table_seqs if max_num_seqs is None else int(max_num_seqs)
    if max_num_batched_tokens is None:
        tokens = min(seqs * max_model_len_tokens, table_tokens)
    else:
        tokens = int(max_num_batched_tokens)
    if max_num_seqs is None:
        seqs = min(seqs, tokens)
    if tokens < seqs:
        raise ServingPlanError(
            f"vllm max_num_batched_tokens ({tokens}) must be greater than "
            f"or equal to max_num_seqs ({seqs})"
        )
    return {
        "usage_context": usage_context,
        "large_device_defaults": large_device,
        "max_num_batched_tokens": tokens,
        "max_num_batched_tokens_source": (
            "vllm_default" if max_num_batched_tokens is None else "configured"
        ),
        "max_num_seqs": seqs,
        "max_num_seqs_source": (
            "vllm_default" if max_num_seqs is None else "configured"
        ),
    }


def vllm_default_block_tokens(
    dimensions: Mapping[str, Any],
    *,
    compute_capability: str | None,
) -> int:
    """Block size vLLM picks when --block-size is not given.

    CacheConfig.DEFAULT_BLOCK_SIZE is 16 and the attention backends keep it,
    except on compute capability 10.x where MLA models select FlashInfer MLA
    (block 32, needs qk_nope_head_dim of 64, 128 or 192) or else CUTLASS MLA
    (block 128) (platforms/cuda.py backend priorities and each backend's
    supported kernel block sizes).
    """

    if dimensions.get("attention_kind") != "multi_head_latent":
        return 16
    major = (
        int(str(compute_capability).split(".")[0])
        if compute_capability
        else None
    )
    if major != 10:
        return 16
    if int(dimensions.get("qk_nope_head_dim") or 0) in {64, 128, 192}:
        return 32
    return 128


def vllm_kv_cache_groups(kv_model: Mapping[str, Any]) -> dict[str, Any]:
    """Group layers into vLLM KV cache groups of equal size.

    Uniform models keep one group of every layer. Hybrid full/sliding models
    split each attention type into groups of ``group_size`` layers, padding
    the last group of a type, and every block id spends ``group_size`` layer
    pages (``_get_kv_cache_groups_uniform_page_size``).
    """

    layers = int(kv_model["num_hidden_layers"])
    sliding = int(kv_model.get("sliding_window_layers", 0))
    full = layers - sliding
    if not sliding or not full:
        attention = "sliding" if sliding else "full"
        return {
            "kind": "uniform",
            "group_size": layers,
            "full_groups": 1 if full else 0,
            "sliding_groups": 1 if sliding else 0,
            "padding_layers": 0,
            "padded_layers": layers,
            "padded_sliding_layers": sliding,
            "attention": attention,
        }
    smallest = min(full, sliding)
    largest = max(full, sliding)
    group_size = largest if largest < smallest * 1.5 else smallest
    full_groups = math.ceil(full / group_size)
    sliding_groups = math.ceil(sliding / group_size)
    padded_full = full_groups * group_size
    padded_sliding = sliding_groups * group_size
    return {
        "kind": "hybrid",
        "group_size": group_size,
        "full_groups": full_groups,
        "sliding_groups": sliding_groups,
        "padding_layers": padded_full + padded_sliding - layers,
        "padded_layers": padded_full + padded_sliding,
        "padded_sliding_layers": padded_sliding,
        "attention": "hybrid",
    }


def _triton_top_k_top_p_rows(rows: int, sm_count: int | None) -> int:
    # apply_top_k_top_p_triton caches a (min(next_pow2(programs), SMs), vocab)
    # float32 buffer with programs = min(SMs, rows). Without an SM count the
    # next power of two of the row count bounds it from above.
    if sm_count is None or sm_count <= 0:
        return 1 << max(0, rows - 1).bit_length()
    programs = min(sm_count, rows)
    return min(1 << max(0, programs - 1).bit_length(), sm_count)


def vllm_profile_memory(
    dimensions: Mapping[str, Any],
    *,
    parameter_bytes: int,
    element_bytes: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    sm_count: int | None,
    runtime_overhead_bytes: int,
    scheduler_overhead_bytes: int,
) -> dict[str, Any]:
    """Model vLLM's ``profile_run`` peak on top of the loaded weights.

    ``_dummy_run(max_num_batched_tokens)`` runs one forward of T tokens
    spread over R = min(T, max_num_seqs) requests, and
    ``_dummy_sampler_run`` samples one row per request: model-dtype logits,
    a float32 copy, the softmax probabilities, the exponential noise and the
    cached Triton top-k/top-p buffer, while the forward's hidden states
    are still held.
    """

    tokens = int(max_num_batched_tokens)
    rows = min(tokens, int(max_num_seqs))
    hidden = int(dimensions["hidden_size"])
    vocab = int(dimensions["vocab_size"])
    forward = _forward_transient_bytes(
        dimensions,
        batch_size=1,
        query_tokens=tokens,
        key_tokens=tokens,
        element_bytes=element_bytes,
        attention_implementation="sdpa",
        logits_to_keep=rows,
    )
    forward_bytes = max(
        int(forward["attention_bytes"]),
        int(forward["mlp_bytes"]),
    )
    logits_elements = rows * vocab
    if rows >= _TRITON_TOP_K_TOP_P_MIN_ROWS:
        top_k_top_p_rows = _triton_top_k_top_p_rows(rows, sm_count)
        sampler_components = {
            "hidden_states": tokens * hidden * element_bytes,
            "sampled_hidden_states": 2 * rows * hidden * element_bytes,
            "logits": logits_elements * element_bytes,
            "float32_logits_probs_and_noise": 3 * logits_elements * 4,
            "top_k_top_p_workspace": top_k_top_p_rows * vocab * 4,
        }
    else:
        # apply_top_k_top_p_pytorch sorts: float32 values, int64 indices,
        # float32 sorted probabilities and two boolean masks.
        top_k_top_p_rows = rows
        sampler_components = {
            "hidden_states": tokens * hidden * element_bytes,
            "sampled_hidden_states": 2 * rows * hidden * element_bytes,
            "logits": logits_elements * element_bytes,
            "float32_logits_probs_and_noise": logits_elements * 4,
            "top_k_top_p_workspace": logits_elements * (4 + 8 + 4 + 1 + 1),
        }
    sampler_bytes = sum(sampler_components.values())
    activation_bytes = max(forward_bytes, sampler_bytes)
    non_torch_bytes = VLLM_MODELED_NON_TORCH_MEMORY_BYTES
    fixed_bytes = (
        parameter_bytes
        + non_torch_bytes
        + runtime_overhead_bytes
        + scheduler_overhead_bytes
    )
    return {
        "max_num_batched_tokens": tokens,
        "profile_requests": rows,
        "forward_activation_bytes": forward_bytes,
        "sampler_bytes": sampler_bytes,
        "sampler_components": sampler_components,
        "top_k_top_p_rows": top_k_top_p_rows,
        "sm_count": sm_count,
        "peak_activation_bytes": activation_bytes,
        "peak_activation_phase": (
            "sampler" if sampler_bytes >= forward_bytes else "forward"
        ),
        "parameter_bytes": parameter_bytes,
        "non_torch_memory_bytes": non_torch_bytes,
        "non_torch_memory_source": "fakegpu_modeled_default",
        "runtime_overhead_bytes": runtime_overhead_bytes,
        "scheduler_overhead_bytes": scheduler_overhead_bytes,
        "phase_bytes": {
            "profile_forward": fixed_bytes + forward_bytes,
            "profile_sampler": fixed_bytes + sampler_bytes,
        },
        "non_kv_cache_memory_bytes": fixed_bytes + activation_bytes,
    }


def _logical_kv_peak_bytes(memory: Mapping[str, Any]) -> int:
    return max(
        int(phase["components"]["kv_cache"])
        for phase in memory["phases"]
    )


def _max_model_len_blocks(
    *,
    groups: Mapping[str, Any],
    max_model_len_tokens: int,
    block_tokens: int,
    sliding_window_tokens: int | None,
    max_num_batched_tokens: int,
) -> int:
    # Blocks one max_model_len request needs over all groups
    # (_max_memory_usage_bytes_from_groups). A sliding-window group keeps the
    # last window-1 tokens plus one scheduling step's tokens, plus a block
    # for a window that does not start on a block boundary
    # (SlidingWindowSpec.max_memory_usage_bytes).
    full_blocks = math.ceil(max_model_len_tokens / block_tokens)
    sliding_blocks = 0
    if groups["sliding_groups"]:
        window_tokens = min(
            int(sliding_window_tokens) - 1 + max_num_batched_tokens,
            max_model_len_tokens,
        )
        sliding_blocks = math.ceil(window_tokens / block_tokens) + 1
    return (
        groups["full_groups"] * full_blocks
        + groups["sliding_groups"] * sliding_blocks
    )


def _build_vllm_runtime_report(
    *,
    requested_memory: Mapping[str, Any],
    profile: Mapping[str, Any],
    selected_capacity_bytes: int | None,
    gpu_memory_utilization: float,
    kv_cache_memory_bytes: int | None,
    non_kv_cache_memory_bytes: int | None,
    max_model_len_tokens: int,
    batch_limits: Mapping[str, Any],
    kv_model: Mapping[str, Any],
    kv_groups: Mapping[str, Any],
    kv_cache_dtype: str,
) -> dict[str, Any]:
    kv_cache = requested_memory["kv_cache"]
    block_tokens = int(kv_cache["block_tokens"])
    page_bytes = (
        block_tokens
        * int(kv_model["elements_per_token_per_layer"])
        * int(kv_model["element_bytes"])
    )
    layers_per_block = int(kv_groups["group_size"])
    block_bytes = layers_per_block * page_bytes
    num_groups = int(kv_groups["full_groups"]) + int(
        kv_groups["sliding_groups"]
    )
    modeled_non_kv_peak = int(profile["non_kv_cache_memory_bytes"])
    selected_non_kv_peak = (
        modeled_non_kv_peak
        if non_kv_cache_memory_bytes is None
        else int(non_kv_cache_memory_bytes)
    )
    non_kv_source = (
        "fakegpu_vllm_profile_run_model"
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

    logical_requested_kv_cache_bytes = _logical_kv_peak_bytes(
        requested_memory
    )
    max_model_len_blocks = _max_model_len_blocks(
        groups=kv_groups,
        max_model_len_tokens=max_model_len_tokens,
        block_tokens=block_tokens,
        sliding_window_tokens=kv_model.get("sliding_window_tokens"),
        max_num_batched_tokens=int(batch_limits["max_num_batched_tokens"]),
    )
    if available_before_rounding is None:
        num_gpu_blocks = None
        allocatable_kv_cache_bytes = None
        usable_kv_cache_bytes = None
        block_rounding_tail_bytes = None
        kv_cache_size_tokens = None
        logged_size_tokens = None
        requested_kv_cache_fits = None
        max_model_len_fits = None
        initialization_required_bytes = None
        max_model_len_concurrency = None
        headroom_bytes = None
    else:
        num_gpu_blocks = available_before_rounding // block_bytes
        allocatable_kv_cache_bytes = num_gpu_blocks * block_bytes
        # vLLM's BlockPool permanently holds block 0 back as the null
        # block, so requests can only use the remaining blocks.
        usable_kv_cache_bytes = max(0, num_gpu_blocks - 1) * block_bytes
        block_rounding_tail_bytes = (
            available_before_rounding - allocatable_kv_cache_bytes
        )
        kv_cache_size_tokens = num_gpu_blocks * block_tokens
        # "GPU KV cache size: N tokens" divides the blocks among the groups.
        logged_size_tokens = num_gpu_blocks // num_groups * block_tokens
        requested_kv_cache_fits = (
            logical_requested_kv_cache_bytes <= usable_kv_cache_bytes
        )
        # vLLM refuses to start unless one request at max_model_len fits
        # in the cache after the null block is held back.
        max_model_len_fits = num_gpu_blocks - 1 >= max_model_len_blocks
        initialization_required_bytes = (
            selected_non_kv_peak + allocatable_kv_cache_bytes
        )
        # vLLM divides the block count by the whole blocks one
        # max_model_len request occupies across all groups.
        max_model_len_concurrency = num_gpu_blocks / max_model_len_blocks
        headroom_bytes = (
            usable_kv_cache_bytes - logical_requested_kv_cache_bytes
        )
    if (
        available_before_rounding is not None
        and available_before_rounding <= block_bytes
    ):
        initialization_failure = "no_available_kv_cache_memory"
    elif max_model_len_fits is False:
        initialization_failure = "max_model_len_exceeds_kv_cache"
    elif (
        memory_limit_bytes is not None
        and initialization_required_bytes > memory_limit_bytes
    ):
        initialization_failure = "device_memory_exceeded"
    else:
        initialization_failure = None
    if initialization_failure is not None:
        initialization_fits = False
    elif memory_limit_bytes is None:
        initialization_fits = None
    else:
        initialization_fits = True
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
        "reference_version": VLLM_REFERENCE_VERSION,
        "configuration": {
            "gpu_memory_utilization": gpu_memory_utilization,
            "gpu_memory_utilization_default": (
                VLLM_DEFAULT_GPU_MEMORY_UTILIZATION
            ),
            "gpu_memory_utilization_ignored": explicit_kv_cache,
            "kv_cache_memory_bytes": kv_cache_memory_bytes,
            "kv_cache_dtype": kv_cache_dtype,
            "block_tokens": block_tokens,
            "max_model_len_tokens": max_model_len_tokens,
            **dict(batch_limits),
            "profiled_sequence_count": int(profile["profile_requests"]),
            "profile_scope": "vllm_profile_run_max_num_batched_tokens",
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
            "modeled_non_kv_phase_bytes": dict(profile["phase_bytes"]),
            "profile": {
                key: value
                for key, value in profile.items()
                if key != "phase_bytes"
            },
            "non_kv_cache_memory_bytes": selected_non_kv_peak,
            "non_kv_cache_memory_source": non_kv_source,
            "initialization_required_bytes": (
                initialization_required_bytes
            ),
            "initialization_fits": initialization_fits,
            "initialization_failure": initialization_failure,
        },
        "kv_cache": {
            "source": kv_cache_source,
            "bytes_per_token": block_bytes // block_tokens,
            "page_bytes_per_layer": page_bytes,
            "layers_per_block": layers_per_block,
            "groups": dict(kv_groups),
            "block_bytes": block_bytes,
            "available_before_block_rounding_bytes": (
                available_before_rounding
            ),
            "allocatable_bytes": allocatable_kv_cache_bytes,
            "usable_bytes": usable_kv_cache_bytes,
            "null_block_count": 1,
            "block_rounding_tail_bytes": block_rounding_tail_bytes,
            "num_gpu_blocks": num_gpu_blocks,
            "size_tokens": kv_cache_size_tokens,
            "logged_size_tokens": logged_size_tokens,
            "max_model_len_blocks": max_model_len_blocks,
            "max_model_len_fits": max_model_len_fits,
            "max_model_len_concurrency": max_model_len_concurrency,
            "logical_requested_peak_bytes": (
                logical_requested_kv_cache_bytes
            ),
            "logical_requested_fits": requested_kv_cache_fits,
            "headroom_bytes": headroom_bytes,
        },
        "requested_fits_memory": requested_fits_memory,
        "notes": [
            (
                "Automatic mode follows vLLM V1's requested-memory minus "
                "profiled non-KV memory policy, then rounds down to whole "
                "paged-cache blocks."
            ),
            (
                "The non-KV estimate models vLLM's profile run: one forward "
                "of max_num_batched_tokens tokens and a sampler pass over "
                "min(max_num_batched_tokens, max_num_seqs) rows, plus the "
                "weights and a modeled non-torch allowance."
            ),
            (
                "kv_cache_memory_bytes overrides utilization-based KV "
                "sizing, matching vLLM configuration semantics."
            ),
            (
                "Like vLLM's block pool, one block is held back as the null "
                "block; startup fails unless one max_model_len request fits "
                "in the remaining blocks."
            ),
            (
                "CUDA graphs are captured after the KV cache is allocated; "
                "vLLM 0.19.1 does not subtract their memory from the budget "
                "unless VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1, and "
                "torch.compile can change the profiled forward peak. Supply "
                "a measured non-KV value to cover either."
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
