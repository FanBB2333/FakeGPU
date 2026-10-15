from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from uuid import uuid4

import pytest

from fakegpu._finetune_allocator import ALLOCATOR_VARIABLES
from fakegpu._finetune_workflow_store import _digest
from fakegpu.phase_memory import (
    SAMPLE_SCHEMA,
    estimate_phase_memory,
    validate_phase_memory,
)


def sample(index, *, timestamp=None, allocated=100, reserved=140):
    plan = {
        "schema_version": "test",
        "memory_timeline": {
            "phases": [
                {"phase": "prefill", "peak_bytes": 90},
                {"phase": "decode", "peak_bytes": 80},
            ]
        },
    }
    identity = {
        "allocator": {
            "schema_version": "fakegpu.finetune_allocator.v1",
            "backend": "native",
            "per_process_memory_fraction": 1.0,
            "environment": dict.fromkeys(ALLOCATOR_VARIABLES),
        },
        "environment": {
            "simulated": False,
            "backend": "cuda",
            "gpu_uuid": "physical-1",
        },
        "model_files": {"weights": {"sha256": "content"}},
        "implementation": {"runner": "source"},
        "execution": {"cache_regime": "uncontrolled"},
    }
    stamp = timestamp or (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    return {
        "schema_version": SAMPLE_SCHEMA,
        "run_id": str(uuid4()),
        "prediction": plan,
        "prediction_sha256": _digest(plan),
        "identity": identity,
        "identity_sha256": _digest(identity),
        "process": {"pid": index + 1, "start_ticks": str(index + 10)},
        "started_at_utc": stamp,
        "finished_at_utc": stamp,
        "phases": {
            p: {"allocated": allocated, "reserved": reserved}
            for p in ("prefill", "decode")
        },
    }


def test_separate_metrics_native_maxima_and_independent_holdout():
    samples = [sample(i, allocated=100 + i, reserved=140 + i) for i in range(5)]
    original = deepcopy(samples)
    budget = estimate_phase_memory(samples)
    assert budget["phases"]["decode"]["allocated"]["max"] == 104
    assert budget["phases"]["decode"]["reserved"]["max"] == 144
    assert budget["peak_bytes"] == {"allocated": 104, "reserved": 144}
    assert samples == original
    holdout = sample(
        9,
        timestamp=(datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
        allocated=105,
        reserved=145,
    )
    result = validate_phase_memory(budget, holdout)
    assert result["passed"] and not result["oom_safety_verified"]
    assert result["comparison"]["decode"]["reserved"]["exceeds_observed_max"]
    assert not validate_phase_memory(budget, holdout, max_underestimate_percent=0)[
        "passed"
    ]


@pytest.mark.parametrize(
    "change", ["gpu", "source", "allocator", "plan", "model", "process"]
)
def test_calibration_identity_mismatch_rejected(change):
    samples = [sample(i) for i in range(5)]
    altered = samples[-1]
    if change == "process":
        altered["process"] = deepcopy(samples[0]["process"])
    elif change == "plan":
        altered["prediction"]["memory_timeline"]["phases"][0]["peak_bytes"] = 99
        altered["prediction_sha256"] = _digest(altered["prediction"])
    else:
        if change == "gpu":
            altered["identity"]["environment"]["gpu_uuid"] = "another"
        if change == "source":
            altered["identity"]["implementation"]["runner"] = "another"
        if change == "model":
            altered["identity"]["model_files"]["weights"]["sha256"] = "another"
        if change == "allocator":
            altered["identity"]["allocator"]["per_process_memory_fraction"] = 0.5
        altered["identity_sha256"] = _digest(altered["identity"])
    with pytest.raises(ValueError):
        estimate_phase_memory(samples)


def test_insufficient_duplicate_tampered_and_late_budget():
    samples = [sample(i) for i in range(5)]
    with pytest.raises(ValueError, match="five"):
        estimate_phase_memory(samples[:4])
    with pytest.raises(ValueError, match="reused"):
        estimate_phase_memory(samples[:4] + [samples[0]])
    budget = estimate_phase_memory(samples)
    with pytest.raises(ValueError, match="before"):
        validate_phase_memory(budget, sample(9))
    future = sample(
        9, timestamp=(datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()
    )
    budget["phases"]["decode"]["reserved"]["max"] = 999
    with pytest.raises(ValueError, match="seal"):
        validate_phase_memory(budget, future)


def test_cli_offline_budget_and_failed_gate(tmp_path):
    from fakegpu.calibration import main

    paths = []
    for i in range(5):
        path = tmp_path / f"cal-{i}.json"
        path.write_text(json.dumps(sample(i)))
        paths.extend(["--calibration-sample", str(path)])
    budget_path = tmp_path / "budget.json.gz"
    assert main(["phase-estimate", *paths, "--json", str(budget_path)]) == 0
    holdout = tmp_path / "holdout.json"
    holdout.write_text(
        json.dumps(
            sample(
                99,
                timestamp=(
                    datetime.now(timezone.utc) + timedelta(seconds=2)
                ).isoformat(),
                allocated=1000,
                reserved=1500,
            )
        )
    )
    assert (
        main(
            [
                "phase-validate",
                "--budget",
                str(budget_path),
                "--holdout-sample",
                str(holdout),
                "--json",
                str(tmp_path / "result.json"),
            ]
        )
        == 1
    )
    with pytest.raises(SystemExit) as error:
        main(["phase-estimate", *paths, "--json", str(budget_path)])
    assert error.value.code == 2


def test_diffusion_sampler_rejects_unverified_model_path_override(tmp_path):
    from test_diffusion_estimator import _write_pipeline

    from fakegpu.diffusion_estimator import estimate_diffusion_generation
    from fakegpu.phase_memory_runner import _model_inputs

    root = tmp_path / "verified-pipeline"
    _write_pipeline(root, transformer=False)
    prediction = estimate_diffusion_generation(model_dir=root, dtype="bfloat16")
    _, workload, verified_root, paths = _model_inputs(prediction)
    assert workload["model_dir"] == verified_root == root
    assert all(path.is_relative_to(root) for path in paths)

    prediction["inputs"]["model_dir"] = str(tmp_path / "unverified-pipeline")
    with pytest.raises(ValueError, match="model_dir"):
        _model_inputs(prediction)


@pytest.mark.parametrize("source_changed", [False, True])
def test_phase_sampler_checks_implementation_after_execution(
    tmp_path, monkeypatch, source_changed
):
    import sys
    from types import SimpleNamespace

    from fakegpu import phase_memory_runner as runner

    def noop(*args, **kwargs):
        return None

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            is_initialized=lambda: False,
            set_device=noop,
        ),
        set_num_threads=noop,
        manual_seed=noop,
        backends=SimpleNamespace(
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(allow_tf32=False),
                flash_sdp_enabled=lambda: True,
                mem_efficient_sdp_enabled=lambda: True,
                math_sdp_enabled=lambda: True,
            ),
            cudnn=SimpleNamespace(
                allow_tf32=False,
                benchmark=False,
                deterministic=False,
                version=lambda: 123,
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="test"))
    monkeypatch.setattr(runner, "_reject_simulated_cuda_environment", noop)
    base = sample(0)
    monkeypatch.setattr(
        runner, "_real_cuda_environment", lambda *a, **k: base["identity"]["environment"]
    )
    monkeypatch.setattr(
        runner, "capture_memory_allocator", lambda _: base["identity"]["allocator"]
    )
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"fixed-model-content")
    monkeypatch.setattr(
        runner, "_model_inputs", lambda _: ("transformers", {}, tmp_path, [weights])
    )
    before = {"runner.py": {"sha256": "before"}}
    after = {"runner.py": {"sha256": "after"}} if source_changed else deepcopy(before)
    identities = iter([before, after])
    monkeypatch.setattr(runner, "_source_identity", lambda: next(identities))
    monkeypatch.setattr(runner, "_process_marker", lambda _: {"start_ticks": "1"})

    def execute(torch, workload, peaks):
        peaks.phases = deepcopy(base["phases"])
        return {"finite_check": "stubbed native execution"}

    monkeypatch.setattr(runner, "_serving", execute)
    if source_changed:
        with pytest.raises(ValueError, match="implementation changed"):
            runner.sample_phase_memory(base["prediction"])
    else:
        result = runner.sample_phase_memory(base["prediction"])
        assert result["identity"]["implementation"] == before
        assert result["phases"] == base["phases"]
