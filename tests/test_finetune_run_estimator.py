from __future__ import annotations

from copy import deepcopy

import pytest

from fakegpu.finetune_config import normalize_finetune_config
from fakegpu.finetune_data import profile_tokenized_records
from fakegpu.finetune_estimator import finetune_workload_signature
from fakegpu.finetune_run_estimator import estimate_finetuning_run
from fakegpu.profile_catalog import get_profile


@pytest.fixture
def native_run():
    model = {
        "model_type": "llama",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "vocab_size": 64,
        "tie_word_embeddings": True,
    }
    training = normalize_finetune_config(
        {
            "model_name_or_path": "/models/local",
            "finetuning_type": "lora",
            "lora_target": "q_proj,v_proj",
            "lora_rank": 4,
            "cutoff_len": 4,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "bf16": True,
            "flash_attn": "sdpa",
            "num_samples": 12,
            "max_steps": 4,
        },
        framework="llamafactory",
    )
    dataset = profile_tokenized_records(
        [{"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]}] * 12,
        sequence_length=4,
        source_framework="llamafactory",
    )
    training.update(
        data_signature=dataset["provenance"]["content_sha256"],
        native_execution_signature="b" * 64,
        checkpoint_sha256="a" * 64,
        software_versions={"torch": "fixture", "llamafactory": "fixture"},
    )
    profile = get_profile("a100")
    updates = [
        {
            "optimizer_step": index + 1,
            "seconds": seconds,
            "microsteps": 2,
            "samples": 2,
            "losses": [1.0, 0.9],
            "warmup": index < 2,
        }
        for index, seconds in enumerate([2.0, 2.0, 0.8, 1.2])
    ]
    workload = {
        "kind": "native_framework_processed_text",
        "fixed_shape": True,
        "sequence_length": 4,
        "optimizer_steps": 4,
        "microsteps": 8,
        "samples": 8,
        "warmup_optimizer_steps": 2,
        "measured_optimizer_steps": 2,
        "warmup_microsteps": 4,
        "measured_microsteps": 4,
        "shape_counts": [{"shape": [1, 4], "microsteps": 8}],
        "sha256": training["data_signature"],
    }
    observation = {
        "schema_version": "fakegpu.finetune_observation.v1",
        "source": "fakegpu.native_framework_cuda_run",
        "native_framework": "llamafactory",
        "signature": finetune_workload_signature(model, training, profile_id="a100"),
        "seconds_per_microstep": [0.4, 0.6],
        "optimizer_update_seconds": [0.8, 1.2],
        "updates": updates,
        "peak_allocated_bytes": 1024**2,
        "peak_reserved_bytes": 2 * 1024**2,
        "checkpoint_identity": {
            "algorithm": "sha256_content_manifest_v1",
            "sha256": "a" * 64,
            "model_dir": "/models/local",
        },
        "hardware": {
            "name": profile.name,
            "total_memory_bytes": profile.memory_bytes,
            "compute_capability": profile.compute_capability_text,
        },
        "software_versions": training["software_versions"],
        "workload": deepcopy(workload),
    }
    return {
        "schema_version": "fakegpu.finetune_run.v1",
        "framework": "llamafactory",
        "profile_id": "a100",
        "model_config": model,
        "training_config": training,
        "dataset": dataset,
        "observation": observation,
        "workload": workload,
        "timing": {
            "total_wall_seconds": 20.0,
            "training_loop_seconds": 6.0,
            "optimizer_update_sample_seconds": 6.0,
            "warmup_training_seconds": 4.0,
            "trainer_other_seconds": 0.0,
        },
        "schedule": {"eval_strategy": "no", "save_strategy": "no"},
    }


def test_native_epoch_projection_keeps_measured_fixed_cost_once(native_run):
    result = estimate_finetuning_run(native_run, epochs=3, max_steps=-1)
    assert result["workload"]["optimizer_steps"] == 18
    assert result["timing"]["training_loop_seconds"]["expected"] == 18
    pipeline = result["timing"]["native_pipeline"]
    assert pipeline["available"] is True
    assert pipeline["fixed_cost_seconds"] == 16
    assert pipeline["seconds"]["expected"] == 34
    assert result["timing"]["evidence"] == "matched_real_measurement_extrapolation"
    assert native_run["training_config"]["max_steps"] == 4


def test_same_length_native_projection_reproduces_observed_wall(native_run):
    result = estimate_finetuning_run(native_run)
    assert result["timing"]["native_pipeline"]["seconds"]["expected"] == 20


def test_different_dataset_size_only_projects_training(native_run):
    result = estimate_finetuning_run(native_run, num_samples=24, epochs=2, max_steps=-1)
    assert result["timing"]["training_loop_seconds"]["expected"] == 24
    assert result["timing"]["native_pipeline"]["available"] is False
    assert "dataset size" in result["timing"]["native_pipeline"]["reasons"][0]


@pytest.mark.parametrize("field", ["eval_strategy", "save_strategy"])
def test_periodic_event_costs_are_not_falsely_extrapolated(native_run, field):
    native_run["schedule"][field] = "steps"
    result = estimate_finetuning_run(native_run, max_steps=100)
    assert result["timing"]["native_pipeline"]["available"] is False


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r["workload"].update(fixed_shape=False), "fixed full batches"),
        (
            lambda r: r["training_config"].update(data_signature="c" * 64),
            "dataset identity",
        ),
        (
            lambda r: r["observation"].update(native_framework="axolotl"),
            "native framework",
        ),
        (
            lambda r: r["observation"]["workload"].update(sha256="c" * 64),
            "workload differs",
        ),
        (lambda r: r["timing"].update(total_wall_seconds=1), "wall time"),
        (lambda r: r["timing"].update(total_wall_seconds=float("nan")), "finite"),
        (
            lambda r: r["training_config"]["software_versions"].update(torch="changed"),
            "signature mismatch",
        ),
        (
            lambda r: r["training_config"].update(native_execution_signature="changed"),
            "signature mismatch",
        ),
    ],
)
def test_native_identity_and_measurement_checks(native_run, mutate, match):
    # Detach all aliases as JSON would, so changed versions do not rewrite the evidence.
    import json

    native_run = json.loads(json.dumps(native_run))
    mutate(native_run)
    with pytest.raises(ValueError, match=match):
        estimate_finetuning_run(native_run)


def test_native_calibration_rejects_different_gpu(native_run):
    with pytest.raises(ValueError, match="signature mismatch"):
        estimate_finetuning_run(native_run, profile_id="h100")


def test_old_probe_signature_does_not_gain_optional_native_fields(native_run):
    config = deepcopy(native_run["training_config"])
    config.pop("data_signature")
    config.pop("native_execution_signature")
    signature = finetune_workload_signature(
        native_run["model_config"], config, profile_id="a100"
    )
    assert "data_signature" not in signature
    assert "native_execution_signature" not in signature
    assert "batch_observation_version" not in signature


def _attach_batch_evidence(run):
    from types import SimpleNamespace

    from fakegpu._finetune_batches import describe_batch, summarize_batch_updates

    run["training_config"]["batch_observation_version"] = "fakegpu.finetune_batches.v1"
    observation = run["observation"]
    observation["signature"] = finetune_workload_signature(
        run["model_config"], run["training_config"], profile_id="a100"
    )
    descriptor = describe_batch({"input_ids": SimpleNamespace(shape=(1, 4))})
    for update in observation["updates"]:
        update["batches"] = [deepcopy(descriptor) for _ in range(update["microsteps"])]
    observation["batch_profile"] = summarize_batch_updates(observation["updates"])


def test_batch_observation_signature_preserves_legacy_and_new_run_reuse(native_run):
    legacy = estimate_finetuning_run(native_run, epochs=3, max_steps=-1)
    old_signature = deepcopy(native_run["observation"]["signature"])
    _attach_batch_evidence(native_run)
    result = estimate_finetuning_run(native_run, epochs=3, max_steps=-1)
    assert result["timing"] == legacy["timing"]
    assert old_signature != native_run["observation"]["signature"]
    assert result["signature"]["batch_observation_version"] == "fakegpu.finetune_batches.v1"


def test_legacy_run_without_kernel_metadata_keeps_fixed_projection(native_run):
    expected = estimate_finetuning_run(native_run, epochs=3, max_steps=-1)
    # Early native reports recorded a collator but no kernels or TF32 backend.
    native_run["native_config"] = {"seed": 42}
    native_run["observation"]["observed_model"] = {
        "collator_class": "fixture.TextCollator"
    }
    assert estimate_finetuning_run(native_run, epochs=3, max_steps=-1) == expected


def _attach_native_backend(run):
    from fakegpu._finetune_model import (
        native_config_signature,
        native_execution_signature,
    )

    run["native_config"] = {"seed": 42}
    observed = {
        "kernel_implementations": {},
        "collator_class": "fixture.TextCollator",
        "tf32_matmul": False,
        "tf32_cudnn": True,
    }
    run["observation"]["observed_model"] = observed
    run["training_config"]["native_execution_signature"] = native_execution_signature(
        native_config_signature(run["native_config"]), observed
    )
    _attach_batch_evidence(run)


@pytest.mark.parametrize(
    "field", ["kernel_implementations", "collator_class", "tf32_matmul", "tf32_cudnn"]
)
def test_batch_protocol_cannot_use_legacy_missing_backend_exception(native_run, field):
    _attach_native_backend(native_run)
    estimate_finetuning_run(native_run)
    native_run["observation"]["observed_model"].pop(field)
    with pytest.raises(ValueError, match=f"lacks backend field '{field}'"):
        estimate_finetuning_run(native_run)


def test_native_config_must_reproduce_recorded_execution_signature(native_run):
    _attach_native_backend(native_run)
    estimate_finetuning_run(native_run)
    native_run["native_config"]["seed"] = 100
    with pytest.raises(ValueError, match="execution signature differs"):
        estimate_finetuning_run(native_run)


@pytest.mark.parametrize("direct_observation", [False, True])
@pytest.mark.parametrize("mutation", ["token_count", "remove_evidence", "sample_seconds"])
def test_public_estimators_validate_batch_evidence(native_run, direct_observation, mutation):
    from fakegpu.finetune_estimator import FinetuneEstimateError, estimate_finetuning

    _attach_batch_evidence(native_run)
    observation = native_run["observation"]
    if mutation == "token_count":
        observation["updates"][0]["batches"][0]["token_slots"] += 1
    elif mutation == "remove_evidence":
        observation.pop("batch_profile")
        for update in observation["updates"]:
            update.pop("batches")
    else:
        observation["seconds_per_microstep"][0] *= 2
    with pytest.raises(FinetuneEstimateError, match="native batch"):
        if direct_observation:
            estimate_finetuning(
                native_run["model_config"], native_run["training_config"],
                profile_id="a100", observation=observation,
            )
        else:
            estimate_finetuning_run(native_run)


def _set_workload(run, field, value):
    run["workload"][field] = deepcopy(value)
    run["observation"]["workload"][field] = deepcopy(value)


@pytest.mark.parametrize(
    "field", ["microsteps", "samples", "optimizer_steps", "sequence_length"]
)
def test_outer_workload_cannot_override_observation_evidence(native_run, field):
    native_run["workload"][field] += 1
    with pytest.raises(ValueError, match="workload differs from observation"):
        estimate_finetuning_run(native_run, epochs=3, max_steps=-1)


@pytest.mark.parametrize("schedule", [None, [], "no", 0])
def test_schedule_must_be_a_mapping(native_run, schedule):
    native_run["schedule"] = schedule
    with pytest.raises(ValueError, match="requires schedule"):
        estimate_finetuning_run(native_run)


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("microsteps", 7, "microstep/sample totals"),
        ("samples", 7, "microstep/sample totals"),
        ("optimizer_steps", 5, "optimizer counts"),
        ("warmup_optimizer_steps", 1, "optimizer counts"),
        ("measured_optimizer_steps", 1, "at least two measured"),
        ("warmup_microsteps", 3, "microstep/sample totals"),
        ("measured_microsteps", 3, "microstep/sample totals"),
        ("microsteps", True, "positive integer"),
        ("kind", "synthetic", "processed text"),
        ("shape_counts", [{"shape": [1, 4], "microsteps": 7}], "shape_counts totals"),
        ("shape_counts", [{"shape": [1, 3], "microsteps": 8}], "fixed_shape flag"),
        ("shape_counts", [{"shape": [2, 4], "microsteps": 8}], "shape_counts totals"),
        (
            "shape_counts",
            [{"shape": [1, 4], "microsteps": 4}, {"shape": [1, 4], "microsteps": 4}],
            "duplicate shapes",
        ),
        ("shape_counts", [{"shape": [1, True], "microsteps": 8}], "shape dimension"),
        ("shape_counts", [], "requires shape_counts"),
    ],
)
def test_workload_totals_must_match_recorded_updates(native_run, field, value, match):
    _set_workload(native_run, field, value)
    with pytest.raises(ValueError, match=match):
        estimate_finetuning_run(native_run)


@pytest.mark.parametrize(
    "index, field, value, match",
    [
        (0, "optimizer_step", 2, "consecutive order"),
        (0, "optimizer_step", True, "positive integer"),
        (0, "warmup", False, "warmup flags"),
        (2, "warmup", True, "warmup flags"),
        (0, "microsteps", 3, "gradient_accumulation_steps"),
        (0, "microsteps", 0, "positive integer"),
        (0, "samples", 1, "full microbatches"),
        (0, "seconds", 0, "must be positive"),
        (0, "seconds", float("nan"), "finite"),
        (0, "losses", [1.0], "one finite loss"),
        (0, "losses", [1.0, float("inf")], "one finite loss"),
        (0, "losses", [1.0, True], "one finite loss"),
    ],
)
def test_optimizer_updates_are_validated(native_run, index, field, value, match):
    native_run["observation"]["updates"][index][field] = value
    with pytest.raises(ValueError, match=match):
        estimate_finetuning_run(native_run)


@pytest.mark.parametrize(
    "field, values",
    [
        ("optimizer_update_seconds", [0.8, 1.3]),
        ("seconds_per_microstep", [0.4, 0.7]),
        ("seconds_per_microstep", [0.4]),
        ("optimizer_update_seconds", [0.8]),
    ],
)
def test_sampled_timing_must_match_measured_update_windows(native_run, field, values):
    native_run["observation"][field] = values
    with pytest.raises(ValueError, match=field):
        estimate_finetuning_run(native_run)


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("training_loop_seconds", 5.0, "cover its recorded optimizer"),
        ("optimizer_update_sample_seconds", 5.0, "optimizer_update_sample_seconds"),
        ("warmup_training_seconds", 3.0, "warmup_training_seconds"),
        ("trainer_other_seconds", 1.0, "trainer_other_seconds"),
    ],
)
def test_loop_breakdown_agrees_with_recorded_updates(native_run, field, value, match):
    native_run["timing"][field] = value
    with pytest.raises(ValueError, match=match):
        estimate_finetuning_run(native_run)


def test_partial_accumulation_windows_are_valid_during_warmup_and_measurement(
    native_run,
):
    # Three rows and accumulation=2 yield one full and one partial update per epoch.
    dataset = profile_tokenized_records(
        [{"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]}] * 3,
        sequence_length=4,
        source_framework="llamafactory",
    )
    native_run["dataset"] = dataset
    digest = dataset["provenance"]["content_sha256"]
    native_run["training_config"].update(num_samples=3, data_signature=digest)
    _set_workload(native_run, "sha256", digest)
    observation = native_run["observation"]
    observation["signature"] = finetune_workload_signature(
        native_run["model_config"],
        native_run["training_config"],
        profile_id="a100",
    )
    for index, seconds in ((1, 1.0), (3, 0.6)):
        observation["updates"][index].update(
            microsteps=1,
            samples=1,
            losses=[0.9],
            seconds=seconds,
        )
    for field, value in (
        ("microsteps", 6),
        ("samples", 6),
        ("warmup_microsteps", 3),
        ("measured_microsteps", 3),
        ("shape_counts", [{"shape": [1, 4], "microsteps": 6}]),
    ):
        _set_workload(native_run, field, value)
    observation["optimizer_update_seconds"] = [0.8, 0.6]
    native_run["timing"].update(
        training_loop_seconds=5.0,
        optimizer_update_sample_seconds=4.4,
        warmup_training_seconds=3.0,
        trainer_other_seconds=0.6,
    )
    report = estimate_finetuning_run(native_run)
    assert report["workload"]["microsteps"] == 6
    assert report["timing"]["native_pipeline"]["seconds"]["expected"] == 20.0


def test_pipeline_discloses_repeated_overhead_in_fixed_residual(native_run):
    report = estimate_finetuning_run(native_run, max_steps=1000)
    limitations = report["timing"]["native_pipeline"]["limitations"]
    assert any(
        "Logging and observer work" in item and "held-out run validation" in item
        for item in limitations
    )
