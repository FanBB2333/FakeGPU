"""Speculative decoding: the resident draft model and its acceptance math.

Split out of ``serving_plan`` unchanged. A speculative plan keeps both the
target and draft weights and KV caches resident, so this module loads the
draft's shape, reports the combined residency, and turns an assumed
acceptance rate into expected target calls — which change the expected
work, not the conservative memory peak.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._serving_types import ServingPlanError, _positive_integer
from .llm_estimator import estimate_decoder_inference


DEFAULT_SPECULATIVE_TOKENS = 5


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
