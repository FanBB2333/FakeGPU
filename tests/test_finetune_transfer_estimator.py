"""The cost engine is tested independently of native sampler construction.

Shared-plan and native timeline validation have their own integration tests.
These fixtures are already parsed component evidence, so tests can deliberately
vary individual coverage pools and verify their arithmetic without training.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from fakegpu import finetune_transfer_estimator as estimator
from fakegpu.finetune_estimator import FinetuneEstimateError
from test_finetune_window_estimator import _batch


def _plan(*, dataset="target", repeats=4):
    result = {
        "source": "cpu_native_schedule",
        "signature": {
            "data_signature": dataset,
            "native_execution_signature": "native-data-configuration",
        },
        "transfer_identity": {"model": "same-model", "backend": "same-backend"},
        "hardware": {"total_memory_bytes": 10000},
        "memory_allocator": {"per_process_memory_fraction": 1.0},
        "execution_policy": {"logging_strategy": "steps"},
        "cache_regime": {"mode": "empty_kernel_cache"},
        "setup_workload": {
            "dataset_content_sha256": dataset,
            "dataset_num_samples": 100 if dataset == "target" else 10,
            "preprocessing_policy": {"cache": "fixed"},
            "sampling_plan_horizon": 8,
        },
        "warmup": {"optimizer_steps": 1},
        "workload": {
            "optimizer_steps": repeats + 1,
            "microsteps": repeats + 1,
            "samples": repeats + 1,
        },
        "updates": [{"batches": [_batch(4)]} for _ in range(repeats + 1)],
    }
    specifications = [
        ("setup", "setup", 1),
        ("optimizer_update", "first", 1),
        *[("optimizer_update", "steady", 2)] * repeats,
        *[("evaluation", "eval", 2)] * 2,
        *[("checkpoint", "save", 2)] * 2,
        *[("trainer_other", "gap", 2)] * 2,
        ("finalization", "final", 1),
    ]
    components = []
    step = 0
    for kind, group, minimum in specifications:
        reference = {"kind": kind}
        if kind == "optimizer_update":
            step += 1
            reference["optimizer_step"] = step
        else:
            reference["position"] = len(components)
        components.append(
            {
                "component_id": len(components) + 1,
                "kind": kind,
                "group_id": group,
                "descriptor": {"kind": kind, "group": group},
                "minimum_samples_per_run": minimum,
                "reference": reference,
            }
        )
    result["_components"] = components
    return result


def _evidence(number, *, dataset="target", factor=1.0, repeats=4, groups=None):
    plan = _plan(dataset=dataset, repeats=repeats)
    components = deepcopy(plan["_components"])
    for component in components:
        component["observed"] = {
            "wall_seconds": 2 * factor,
            "instrumentation_seconds": 0.5 * factor,
            "seconds": 1.5 * factor,
        }
    if groups is not None:
        components = [row for row in components if row["group_id"] in groups]
    grouped = {}
    for component in components:
        grouped.setdefault(component["group_id"], []).append(component)
    return {
        "plan": plan,
        "identity": {
            "uuid": f"run-{number}",
            "pid": number,
            "started_at_utc": "2026-10-14T00:00:00Z",
        },
        "evidence_sha256": f"timing-{number}",
        "components": components,
        "groups": grouped,
        "peaks": {
            "peak_allocated_bytes": 1000 + number,
            "peak_reserved_bytes": 2000 + number,
        },
    }


@pytest.fixture(autouse=True)
def parsed_component_boundary(monkeypatch):
    monkeypatch.setattr(estimator, "validate_transfer_plan", deepcopy)
    monkeypatch.setattr(
        estimator,
        "transfer_signature",
        lambda plan: deepcopy(plan["transfer_identity"]),
    )
    monkeypatch.setattr(
        estimator,
        "describe_transfer_components",
        lambda plan: deepcopy(plan["_components"]),
    )
    monkeypatch.setattr(estimator, "_sample", deepcopy)


@pytest.fixture
def evidence():
    work = [
        _evidence(1, dataset="small", factor=1),
        _evidence(2, dataset="small", factor=3),
    ]
    setup = [
        _evidence(3, factor=5, groups={"setup", "first", "final"}),
        _evidence(4, factor=7, groups={"setup", "first", "final"}),
    ]
    return work, setup


def _estimate(evidence, plan=None):
    work, setup = evidence
    return estimator.estimate_finetuning_transfer(
        work, plan or _plan(), setup_calibration_runs=setup
    )


def test_target_setup_is_measured_once_and_never_scaled_from_other_dataset(evidence):
    result = _estimate(evidence)
    by_kind = result["timing"]["by_kind"]
    assert by_kind["setup"]["wall_seconds"] == {"min": 10, "expected": 12, "max": 14}
    assert by_kind["finalization"]["wall_seconds"]["expected"] == 12
    groups = {group["group_id"]: group for group in result["component_groups"]}
    assert groups["steady"]["per_component"]["wall_seconds"]["expected"] == 4
    assert {row["run_uuid"] for row in groups["setup"]["calibration_runs"]} == {
        "run-3",
        "run-4",
    }
    assert result["coverage"]["excluded_components"] == 0
    assert set(by_kind) == {
        "setup",
        "optimizer_update",
        "evaluation",
        "checkpoint",
        "trainer_other",
        "finalization",
    }
    timing = result["timing"]
    assert (
        timing["wall_seconds"]["expected"]
        == timing["seconds"]["expected"] + timing["instrumentation_seconds"]["expected"]
    )


def test_complementary_processes_can_cover_different_groups(evidence):
    work, setup = evidence
    work += [
        _evidence(5, dataset="small", groups={"eval", "save"}),
        _evidence(6, dataset="small", groups={"eval", "save"}),
    ]
    for run in work[:2]:
        for group in ("eval", "save"):
            del run["groups"][group]
    result = _estimate(evidence)
    for group in result["component_groups"]:
        assert len(group["calibration_runs"]) >= 2
    assert result["coverage"]["complete"]


def test_each_group_uses_process_means_not_pooled_samples():
    runs = [_evidence(1, factor=1, repeats=2), _evidence(2, factor=3, repeats=20)]
    result = estimator.estimate_finetuning_transfer(
        runs, _plan(), setup_calibration_runs=runs
    )
    steady = next(
        group for group in result["component_groups"] if group["group_id"] == "steady"
    )
    assert steady["per_component"]["wall_seconds"] == {
        "min": 2,
        "expected": 4,
        "max": 6,
    }
    assert len(result["calibration_runs"]) == 2


def test_empty_pools_return_requirements_and_refuse_a_budget():
    coverage = estimator.inspect_finetuning_transfer_coverage(
        [], _plan(), setup_calibration_runs=[]
    )
    assert not coverage["complete"]
    assert coverage["covered_groups"] == 0
    assert all(
        group["missing_independent_runs"] == 2 for group in coverage["missing_groups"]
    )
    with pytest.raises(FinetuneEstimateError, match="insufficient transfer coverage"):
        estimator.estimate_finetuning_transfer([], _plan(), setup_calibration_runs=[])


def test_repeated_group_needs_two_samples_in_each_of_two_processes(evidence):
    evidence[0][1]["groups"]["steady"] = evidence[0][1]["groups"]["steady"][:1]
    coverage = estimator.inspect_finetuning_transfer_coverage(
        evidence[0], _plan(), setup_calibration_runs=evidence[1]
    )
    missing = coverage["missing_groups"]
    assert len(missing) == 1
    assert missing[0]["group_id"] == "steady"
    assert missing[0]["eligible_independent_runs"] == 1
    with pytest.raises(FinetuneEstimateError, match="steady"):
        _estimate(evidence)


def test_final_save_requires_target_dataset_evidence_even_though_kind_is_checkpoint(
    evidence,
):
    plan = _plan()
    for component in plan["_components"]:
        if component["group_id"] == "save":
            component["descriptor"]["event"] = {"inside_train": False, "phase": "final"}
    coverage = estimator.inspect_finetuning_transfer_coverage(
        evidence[0], plan, setup_calibration_runs=evidence[1]
    )
    assert len(coverage["missing_groups"]) == 1
    missing = coverage["missing_groups"][0]
    assert missing["group_id"] == "save"
    assert missing["evidence_role"] == "target_dataset"
    assert missing["eligible_independent_runs"] == 0


def test_shape_pooling_retains_different_supervision_and_mask_statistics(evidence):
    changed = evidence[0][1]["plan"]["updates"][1]["batches"][0]
    changed.update(
        row_nonpadding_tokens=[3],
        nonpadding_tokens=3,
        padding_tokens=1,
        row_supervised_tokens=[2],
        supervised_tokens=2,
        row_causal_supervised_tokens=[1],
        causal_supervised_tokens=1,
    )
    result = _estimate(evidence)
    assert result["empirical_approximation"]["enabled"]
    assert result["empirical_approximation"]["exact_token_distribution_match"] is False
    steady = next(
        group for group in result["component_groups"] if group["group_id"] == "steady"
    )
    second = steady["calibration_runs"][1]["batch_descriptor_distribution"]
    assert len(second) == 2
    assert sum(row["optimizer_updates"] for row in second) == 4
    assert {row["batches"][0]["supervised_tokens"] for row in second} == {2, 4}
    assert len(steady["target_batch_descriptor_distribution"]) == 1


@pytest.mark.parametrize(
    "field",
    [
        "dataset_content_sha256",
        "dataset_num_samples",
        "preprocessing_policy",
        "sampling_plan_horizon",
    ],
)
def test_setup_pool_cannot_borrow_a_different_target_preparation(evidence, field):
    evidence[1][0]["plan"]["setup_workload"][field] = "changed"
    with pytest.raises(FinetuneEstimateError, match="complete target dataset"):
        _estimate(evidence)


@pytest.mark.parametrize(
    "field",
    [
        "hardware",
        "memory_allocator",
        "execution_policy",
        "cache_regime",
        "transfer_identity",
    ],
)
def test_work_pool_requires_identical_execution_conditions(evidence, field):
    evidence[0][0]["plan"][field] = {"different": True}
    with pytest.raises(FinetuneEstimateError, match="requires matching"):
        _estimate(evidence)


@pytest.mark.parametrize("kind", ["UUID", "process", "evidence"])
def test_process_evidence_must_be_independent_across_both_pools(evidence, kind):
    source, target = evidence[0][0], evidence[1][0]
    if kind == "UUID":
        target["identity"]["uuid"] = source["identity"]["uuid"]
    elif kind == "process":
        target["identity"]["pid"] = source["identity"]["pid"]
    else:
        target["evidence_sha256"] = source["evidence_sha256"]
    with pytest.raises(FinetuneEstimateError, match="different reports|repeats"):
        _estimate(evidence)


def test_same_report_may_have_both_roles_but_not_appear_twice_in_one_pool():
    runs = [_evidence(1), _evidence(2)]
    result = estimator.estimate_finetuning_transfer(
        runs, _plan(), setup_calibration_runs=deepcopy(runs)
    )
    assert len(result["calibration_runs"]) == 2
    assert all(
        len(group["calibration_runs"]) == 2 for group in result["component_groups"]
    )
    with pytest.raises(FinetuneEstimateError, match="repeats a run UUID"):
        estimator.estimate_finetuning_transfer(
            [*runs, runs[0]], _plan(), setup_calibration_runs=runs
        )


def test_all_inputs_are_preserved_and_output_does_not_share_mutable_values(evidence):
    before = deepcopy(evidence)
    plan = _plan()
    original = deepcopy(plan)
    result = _estimate(evidence, plan)
    result["component_groups"][0]["descriptor"]["kind"] = "changed"
    result["execution_signature"]["model"] = "changed"
    assert evidence == before
    assert plan == original


def test_holdout_measurements_never_influence_prediction(evidence):
    target = _evidence(10, factor=9)
    original = estimator.compare_finetuning_transfer(
        evidence[0], target, setup_calibration_runs=evidence[1]
    )
    target["evidence_sha256"] = "changed"
    for component in target["components"]:
        for field in component["observed"]:
            component["observed"][field] *= 2
    changed = estimator.compare_finetuning_transfer(
        evidence[0], target, setup_calibration_runs=evidence[1]
    )
    for field in ("wall_seconds", "instrumentation_seconds", "seconds"):
        assert original["timing"][field] == changed["timing"][field]
        assert (
            changed["timing"]["comparison"][field]["actual"]
            == original["timing"]["comparison"][field]["actual"] * 2
        )


def test_holdout_cannot_reuse_a_calibration_process(evidence):
    target = deepcopy(evidence[0][0])
    target["plan"] = _plan()
    with pytest.raises(FinetuneEstimateError, match="repeats"):
        estimator.compare_finetuning_transfer(
            evidence[0], target, setup_calibration_runs=evidence[1]
        )


def test_memory_uses_full_process_peak_extrema_and_explicit_limits(evidence):
    result = estimator.estimate_finetuning_transfer_memory(
        evidence[0],
        _plan(),
        setup_calibration_runs=evidence[1],
        memory_budget_bytes=2100,
        reserve_bytes=96,
    )
    assert result["memory"] == {
        "peak_allocated_bytes": {"min": 1001, "max": 1004},
        "peak_reserved_bytes": {"min": 2001, "max": 2004},
    }
    assert result["budget"]["observed_reserved_peak_plus_reserve_bytes"] == 2100
    assert result["budget"]["status"] == "observed_envelope_fits_budget"
    assert result["budget"]["oom_safety_verified"] is False
    assert "combined target" in result["limitations"][1]


@pytest.mark.parametrize("value", [0, -1, True, 2.5, 10001])
def test_memory_budget_must_fit_recorded_hardware(evidence, value):
    with pytest.raises(
        FinetuneEstimateError, match="memory budget|memory_budget_bytes"
    ):
        estimator.estimate_finetuning_transfer_memory(
            evidence[0],
            _plan(),
            setup_calibration_runs=evidence[1],
            memory_budget_bytes=value,
        )


def test_memory_budget_cannot_exceed_allocator_cap(evidence):
    plan = _plan()
    plan["memory_allocator"]["per_process_memory_fraction"] = 0.5
    for run in [*evidence[0], *evidence[1]]:
        run["plan"]["memory_allocator"]["per_process_memory_fraction"] = 0.5
    with pytest.raises(FinetuneEstimateError, match="allocator cap"):
        estimator.estimate_finetuning_transfer_memory(
            evidence[0],
            plan,
            setup_calibration_runs=evidence[1],
            memory_budget_bytes=6000,
        )


def test_memory_refuses_incomplete_timing_component_coverage(evidence):
    del evidence[0][0]["groups"]["steady"]
    with pytest.raises(FinetuneEstimateError, match="complete component coverage"):
        estimator.estimate_finetuning_transfer_memory(
            evidence[0], _plan(), setup_calibration_runs=evidence[1]
        )


def test_holdout_memory_does_not_recalibrate_the_envelope(evidence):
    target = _evidence(10)
    original = estimator.compare_finetuning_transfer_memory(
        evidence[0], target, setup_calibration_runs=evidence[1]
    )
    target["peaks"] = {"peak_allocated_bytes": 3000, "peak_reserved_bytes": 4000}
    changed = estimator.compare_finetuning_transfer_memory(
        evidence[0], target, setup_calibration_runs=evidence[1]
    )
    assert original["estimate"] == changed["estimate"]
    assert changed["comparison"]["peak_reserved_bytes"]["exceeds_observed_max"]
    assert changed["comparison"]["peak_reserved_bytes"][
        "underprediction_percent"
    ] == pytest.approx(49.9)
