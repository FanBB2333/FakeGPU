from __future__ import annotations

from copy import deepcopy

import pytest

from fakegpu._finetune_pipeline_plan import pipeline_plan_from_run
from fakegpu.finetune_estimator import FinetuneEstimateError
from fakegpu.finetune_memory_estimator import (
    compare_finetuning_memory,
    estimate_finetuning_memory,
)
from test_finetune_pipeline_estimator import _pipeline_run
from test_finetune_run_estimator import native_run as native_run
from test_finetune_window_estimator import _batch


@pytest.fixture
def runs(native_run):
    result = []
    for number, epochs, allocated, reserved in (
        (1, 24, 1000, 2000),
        (2, 28, 1200, 2400),
        (3, 32, 1100, 2200),
        (4, 32, 1300, 2600),
        (5, 32, 900, 1900),
    ):
        run = _pipeline_run(native_run, number, epochs=epochs, factor=number)
        run["observation"].update(
            peak_allocated_bytes=allocated, peak_reserved_bytes=reserved
        )
        result.append(run)
    return result


def test_whole_process_peak_envelope_never_sums_windows_or_runs(runs):
    before = deepcopy(runs)
    plan = pipeline_plan_from_run(runs[2])
    plan["source"] = "cpu_native_schedule"
    result = estimate_finetuning_memory(runs[:2], plan)
    assert result["memory"] == {
        "peak_allocated_bytes": {"min": 1000, "max": 1200},
        "peak_reserved_bytes": {"min": 2000, "max": 2400},
    }
    assert result["target_plan"]["source"] == "cpu_native_schedule"
    assert result["coverage"]["includes_warmup_and_first_occurrences"]
    assert result["coverage"]["complete"]
    assert {group["kind"] for group in result["component_groups"]} == {
        "setup",
        "trainer_other",
        "finalization",
        "optimizer_update",
        "evaluation",
        "checkpoint",
    }
    assert result["budget"]["status"] == "not_requested"
    assert not result["budget"]["oom_safety_verified"]
    assert "comparison" not in result
    assert runs == before
    result["signature"]["profile"]["id"] = "changed"
    result["component_groups"][0]["descriptor"]["kind"] = "changed"
    assert runs == before


def test_five_independent_observations_report_every_holdout_including_exceedance(runs):
    results = [compare_finetuning_memory(runs[:2], target) for target in runs[2:]]
    assert len({run["run_identity"]["uuid"] for run in runs}) == 5
    assert [
        result["comparison"]["peak_reserved_bytes"]["exceeds_observed_max"]
        for result in results
    ] == [False, True, False]
    assert [
        result["comparison"]["peak_reserved_bytes"]["within_observed_envelope"]
        for result in results
    ] == [True, False, False]
    assert results[1]["comparison"]["peak_reserved_bytes"][
        "underprediction_percent"
    ] == pytest.approx(100 / 13)
    assert all(
        result["estimate"]["memory"] == results[0]["estimate"]["memory"]
        for result in results
    )


def test_target_peaks_are_used_only_for_comparison(runs):
    original = compare_finetuning_memory(runs[:2], runs[2])
    changed = deepcopy(runs[2])
    changed["observation"].update(peak_allocated_bytes=9000, peak_reserved_bytes=10000)
    result = compare_finetuning_memory(runs[:2], changed)
    assert result["estimate"] == original["estimate"]
    assert result["comparison"]["peak_reserved_bytes"]["actual"] == 10000
    assert result["comparison"]["peak_reserved_bytes"]["exceeds_observed_max"]


@pytest.mark.parametrize(
    "budget,reserve,status",
    [
        (2500, 100, "observed_envelope_fits_budget"),
        (2500, 101, "observed_envelope_exceeds_budget"),
        (1000, 0, "observed_envelope_exceeds_budget"),
    ],
)
def test_budget_result_is_arithmetic_not_a_claim_of_actual_oom(
    runs, budget, reserve, status
):
    result = estimate_finetuning_memory(
        runs[:2],
        pipeline_plan_from_run(runs[2]),
        memory_budget_bytes=budget,
        reserve_bytes=reserve,
    )
    assert result["budget"] == {
        "memory_budget_bytes": budget,
        "reserve_bytes": reserve,
        "recorded_allocator_limit_bytes": runs[0]["observation"]["hardware"][
            "total_memory_bytes"
        ],
        "observed_reserved_peak_plus_reserve_bytes": 2400 + reserve,
        "fits_recorded_allocator_cap": True,
        "status": status,
        "oom_safety_verified": False,
    }
    assert any(
        "not a minimum memory requirement" in limitation
        for limitation in result["limitations"]
    )


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("memory_budget_bytes", 0),
        ("memory_budget_bytes", -1),
        ("memory_budget_bytes", True),
        ("memory_budget_bytes", 2500.0),
        ("reserve_bytes", -1),
        ("reserve_bytes", True),
        ("reserve_bytes", 1.5),
    ],
)
def test_byte_budgets_are_explicit_non_boolean_integers(runs, keyword, value):
    with pytest.raises(FinetuneEstimateError, match=keyword):
        estimate_finetuning_memory(
            runs[:2], pipeline_plan_from_run(runs[2]), **{keyword: value}
        )


def test_budget_cannot_name_more_memory_than_recorded_hardware(runs):
    plan = pipeline_plan_from_run(runs[2])
    with pytest.raises(FinetuneEstimateError, match="physical GPU"):
        estimate_finetuning_memory(
            runs[:2],
            plan,
            memory_budget_bytes=plan["hardware"]["total_memory_bytes"] + 1,
        )


def test_requested_budget_cannot_exceed_active_allocator_fraction(runs):
    for run in runs:
        run["observation"]["memory_allocator"]["per_process_memory_fraction"] = 0.5
    plan = pipeline_plan_from_run(runs[2])
    cap = plan["hardware"]["total_memory_bytes"] // 2
    with pytest.raises(FinetuneEstimateError, match="allocator cap"):
        estimate_finetuning_memory(runs[:2], plan, memory_budget_bytes=cap + 1)
    result = estimate_finetuning_memory(runs[:2], plan, memory_budget_bytes=cap)
    assert result["budget"]["recorded_allocator_limit_bytes"] == cap
    assert result["budget"]["status"] == "observed_envelope_fits_budget"
    result = estimate_finetuning_memory(runs[:2], plan, reserve_bytes=cap)
    assert not result["budget"]["fits_recorded_allocator_cap"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda plan: plan["signature"].update(optimizer="different"), "signature"),
        (lambda plan: plan["hardware"].update(name="different"), "hardware"),
        (
            lambda plan: plan["cache_regime"].update(mode="empty_kernel_cache"),
            "cache_regime",
        ),
        (
            lambda plan: plan["setup_workload"].update(dataset_num_samples=6),
            "setup_workload",
        ),
        (
            lambda plan: plan["memory_allocator"].update(
                per_process_memory_fraction=0.5
            ),
            "memory_allocator",
        ),
        (
            lambda plan: plan["memory_allocator"]["environment"].update(
                PYTORCH_ALLOC_CONF="expandable_segments:True"
            ),
            "memory_allocator",
        ),
        (
            lambda plan: plan["execution_policy"].update(save_total_limit=100),
            "execution_policy",
        ),
    ],
)
def test_model_execution_hardware_dataset_and_cache_cohorts_must_match(
    runs, mutation, match
):
    plan = pipeline_plan_from_run(runs[2])
    mutation(plan)
    with pytest.raises(FinetuneEstimateError, match=match):
        estimate_finetuning_memory(runs[:2], plan)


@pytest.mark.parametrize("step", [0, 2, 30])
def test_unseen_shape_is_rejected_even_during_warmup_or_first_occurrence(runs, step):
    plan = pipeline_plan_from_run(runs[2])
    plan["updates"][step]["batches"][0].update(_batch(3))
    with pytest.raises(FinetuneEstimateError, match="uncovered component"):
        estimate_finetuning_memory(runs[:2], plan)


def test_unseen_event_workload_and_its_window_context_are_rejected(runs):
    plan = pipeline_plan_from_run(runs[2])
    for root in plan["root_events"]:
        if root["descriptor"]["kind"] == "evaluation":
            root["descriptor"]["workload"]["eval_batch_size"] += 1
    with pytest.raises(FinetuneEstimateError, match="uncovered component"):
        estimate_finetuning_memory(runs[:2], plan)


def test_unseen_ending_context_is_not_replaced_with_a_known_shape(runs, native_run):
    target = _pipeline_run(native_run, 6, epochs=33, factor=1)
    with pytest.raises(FinetuneEstimateError, match="uncovered component"):
        compare_finetuning_memory(runs[:2], target)


def test_each_calibration_must_cover_the_complete_target(runs, native_run):
    short = _pipeline_run(native_run, 6, epochs=8, factor=1)
    with pytest.raises(FinetuneEstimateError, match="uncovered component"):
        estimate_finetuning_memory([runs[0], short], pipeline_plan_from_run(runs[2]))


def test_independent_runs_and_independent_holdouts_are_required(runs):
    plan = pipeline_plan_from_run(runs[2])
    with pytest.raises(FinetuneEstimateError, match="two independent"):
        estimate_finetuning_memory(runs[:1], plan)
    with pytest.raises(FinetuneEstimateError, match="UUID"):
        estimate_finetuning_memory([runs[0], runs[0]], plan)
    copied = deepcopy(runs[0])
    copied["run_identity"] = deepcopy(runs[1]["run_identity"])
    with pytest.raises(FinetuneEstimateError, match="observation"):
        estimate_finetuning_memory([runs[0], copied], plan)
    copied = deepcopy(runs[1])
    copied["run_identity"].update(
        pid=runs[0]["run_identity"]["pid"],
        started_at_utc=runs[0]["run_identity"]["started_at_utc"],
    )
    with pytest.raises(FinetuneEstimateError, match="process identity"):
        estimate_finetuning_memory([runs[0], copied], plan)
    with pytest.raises(FinetuneEstimateError, match="independent native run"):
        compare_finetuning_memory(runs[:2], runs[0])


@pytest.mark.parametrize(
    "field,value",
    [
        ("peak_allocated_bytes", 1.5),
        ("peak_allocated_bytes", True),
        ("peak_reserved_bytes", 999),
        ("peak_reserved_bytes", 2**60),
    ],
)
def test_invalid_peak_evidence_is_rejected(runs, field, value):
    runs[0]["observation"][field] = value
    with pytest.raises(FinetuneEstimateError):
        estimate_finetuning_memory(runs[:2], pipeline_plan_from_run(runs[2]))


def test_missing_peak_evidence_is_rejected(runs):
    del runs[0]["observation"]["peak_reserved_bytes"]
    with pytest.raises(FinetuneEstimateError, match="peak_reserved_bytes"):
        estimate_finetuning_memory(runs[:2], pipeline_plan_from_run(runs[2]))


def test_complete_timeline_is_required_for_whole_run_memory_coverage(runs):
    del runs[0]["observation"]["timeline"]
    with pytest.raises((FinetuneEstimateError, ValueError), match="timeline"):
        estimate_finetuning_memory(runs[:2], pipeline_plan_from_run(runs[2]))
