"""CPU synthetic protocol fixtures; these reports are not GPU measurements."""

from copy import deepcopy
import hashlib
import json
import struct
import shutil
import subprocess
import sys
from uuid import uuid4

import pytest

from fakegpu.diffusion_training import (
    RUN_SCHEMA,
    _digest,
    compare_diffusion_training,
    estimate_diffusion_training,
    plan_diffusion_training,
    project_diffusion_training,
    training_windows,
    validate_training_plan,
)
from fakegpu.diffusion_training_cli import plan_main
from fakegpu.structured_io import write_json, load_mapping
from fakegpu._finetune_cache import capture_cache_regime
from fakegpu._finetune_allocator import ALLOCATOR_VARIABLES


def safefile(path, tensors):
    offset = 0
    header = {}
    for name, (dtype, shape) in tensors.items():
        size = 8 if dtype == "I64" else 4
        for dimension in shape:
            size *= dimension
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


@pytest.fixture
def inputs(tmp_path):
    model = tmp_path / "model"
    (model / "unet").mkdir(parents=True)
    (model / "scheduler").mkdir()
    (model / "unet/config.json").write_text(
        json.dumps(
            {
                "_class_name": "UNet2DConditionModel",
                "in_channels": 4,
                "out_channels": 4,
                "cross_attention_dim": 4,
            }
        )
    )
    (model / "scheduler/scheduler_config.json").write_text(
        json.dumps({"num_train_timesteps": 1000, "prediction_type": "epsilon"})
    )
    safefile(
        model / "unet/diffusion_pytorch_model.safetensors",
        {
            "attn." + name + ".weight": ("F32", [4, 4])
            for name in ("to_q", "to_k", "to_v", "to_out.0")
        },
    )
    data = tmp_path / "cached.safetensors"
    safefile(
        data,
        {
            "latents": ("F32", [5, 4, 8, 8]),
            "noise": ("F32", [5, 4, 8, 8]),
            "timesteps": ("I64", [5]),
            "encoder_hidden_states": ("F32", [5, 3, 4]),
        },
    )
    return model, data


def plan(inputs, **changes):
    return plan_diffusion_training(
        *inputs,
        batch_size=2,
        gradient_accumulation_steps=2,
        epochs=3,
        lora_rank=2,
        **changes,
    )


def native(source, number, factor):
    updates = []
    for window in training_windows(source):
        group = window["group"]
        seconds = (
            10
            if group["stage"] == "first_update"
            else 2
            if group["microbatch_rows"] == [2, 2]
            else 1
        ) * factor
        updates.append(
            {
                **window,
                "seconds": seconds,
                "loss": 1.0,
                "input_sha256": [
                    hashlib.sha256(str(row).encode()).hexdigest()
                    for row in window["row_indices"]
                ],
            }
        )
    timing = {"setup_seconds": 3 * factor, "finalization_seconds": 2 * factor}
    timing["total_seconds"] = sum(timing.values()) + sum(
        row["seconds"] for row in updates
    )
    s = source["signature"]
    return {
        "schema_version": RUN_SCHEMA,
        "plan": deepcopy(source),
        "identity": {
            "uuid": str(uuid4()),
            "pid": number + 100,
            "started_at_utc": "2026-10-15T00:00:00+00:00",
        },
        "environment": {
            "backend": "cuda",
            "simulated": False,
            "gpu_uuid": "synthetic-test-fixture",
            "total_memory_bytes": 2**30,
            "software": {"framework": "diffusers", "framework_version": "test"},
        },
        "implementation_sha256": "f" * 64,
        "cache_regime": capture_cache_regime(),
        "memory_allocator": {
            "schema_version": "fakegpu.finetune_allocator.v1",
            "backend": "native",
            "per_process_memory_fraction": 1.0,
            "environment": {name: None for name in ALLOCATOR_VARIABLES},
        },
        "observed_model": {
            "base_parameters": s["base_parameters"],
            "trainable_parameters": s["trainable_parameters"],
            "trainable_dtypes": {"torch.float32": s["trainable_parameters"]},
            "gradient_checkpointing": s["gradient_checkpointing"],
            "gradient_bytes": source["static_memory"]["components"]["gradients"],
            "optimizer_moment_bytes": source["static_memory"]["components"][
                "adamw_moments"
            ],
            "ema_bytes": source["static_memory"]["components"]["ema_parameters"],
        },
        "updates": updates,
        "timing": timing,
        "peak_allocated_bytes": int(1000000 * factor),
        "peak_reserved_bytes": int(1100000 * factor),
    }


def test_exact_static_storage_keeps_unknown_transients_explicit(inputs):
    value = plan(inputs)
    assert value["signature"]["base_parameters"] == 64
    assert value["signature"]["trainable_parameters"] == 64
    assert value["static_memory"]["bytes"] == 64 * 2 + 64 * 16
    assert value["static_memory"]["activation_bytes"] is None
    full = plan(inputs, method="full")
    assert full["static_memory"]["bytes"] == 64 * 16
    assert (
        plan(inputs, method="full", ema_decay=0.9)["static_memory"]["bytes"] == 64 * 20
    )
    assert plan(inputs, ema_decay=0.9)["static_memory"]["bytes"] == 64 * 2 + 64 * 20
    estimate = estimate_diffusion_training(value)
    assert (
        not estimate["coverage"]["complete"]
        and estimate["time"] is None
        and estimate["memory"] is None
    )


def test_tail_rows_and_accumulation_windows_are_never_dropped(inputs):
    windows = list(training_windows(plan(inputs)))
    assert [row["row_indices"] for row in windows] == [[[0, 1], [2, 3]], [[4]]] * 3
    assert [row["group"]["stage"] for row in windows] == ["first_update"] + [
        "repeated_update"
    ] * 5


def test_independent_process_means_predict_before_holdout_exists(inputs):
    calibration = plan(inputs)
    reports = [native(calibration, 1, 1), native(calibration, 2, 3)]
    before = deepcopy(reports)
    target = project_diffusion_training(calibration, epochs=5)
    prediction = estimate_diffusion_training(target, reports)
    assert prediction["coverage"]["complete"]
    assert prediction["time"]["updates_seconds"] == 46
    assert prediction["time"]["total_seconds"] == 56
    holdout = native(target, 3, 2.2)
    comparison = compare_diffusion_training(reports, holdout)
    assert comparison["prediction"] == prediction
    assert comparison["time_comparison"]["total_seconds"]["actual"] == pytest.approx(
        61.6
    )
    assert reports == before


def test_missing_repeated_tail_coverage_does_not_use_a_ratio(inputs):
    source = project_diffusion_training(plan(inputs), epochs=1)
    reports = [native(source, 1, 1), native(source, 2, 3)]
    target = project_diffusion_training(source, epochs=5)
    result = estimate_diffusion_training(target, reports)
    assert not result["coverage"]["complete"]
    assert result["time"] is None and result["memory"] is None


def test_first_tail_shape_initialization_is_charged_once(inputs):
    source = plan(inputs)
    reports = [native(source, 1, 1), native(source, 2, 3)]
    for report, factor in zip(reports, (1, 3)):
        report["updates"][1]["seconds"] = 20 * factor
        report["timing"]["total_seconds"] = (
            report["timing"]["setup_seconds"]
            + report["timing"]["finalization_seconds"]
            + sum(row["seconds"] for row in report["updates"])
        )
    target = project_diffusion_training(source, epochs=5)
    result = estimate_diffusion_training(target, reports)
    assert result["coverage"]["complete"]
    assert len(result["time"]["groups"]) == 4
    assert result["time"]["total_seconds"] == 94
    assert [
        row["group"]["first_window_shape_use"] for row in training_windows(source)
    ] == [True, True, False, False, False, False]


@pytest.mark.parametrize("change", ["uuid", "process", "cloned-timing"])
def test_duplicate_calibration_evidence_is_rejected(inputs, change):
    source = plan(inputs)
    first, second = native(source, 1, 1), native(source, 2, 2)
    if change == "uuid":
        second["identity"]["uuid"] = first["identity"]["uuid"]
    elif change == "process":
        second["identity"]["pid"] = first["identity"]["pid"]
    else:
        second["updates"] = deepcopy(first["updates"])
        second["timing"] = deepcopy(first["timing"])
    with pytest.raises(ValueError, match="repeats"):
        estimate_diffusion_training(source, [first, second])


@pytest.mark.parametrize(
    "change", ["row", "time", "optimizer", "allocator", "gpu", "implementation"]
)
def test_conflicting_native_measurements_cannot_be_pooled(inputs, change):
    source = plan(inputs)
    reports = [native(source, 1, 1), native(source, 2, 2)]
    run = reports[1]
    if change == "row":
        run["updates"][0]["row_indices"][0][0] = 4
    elif change == "time":
        run["timing"]["total_seconds"] += 1
    elif change == "optimizer":
        run["observed_model"]["optimizer_moment_bytes"] += 4
    elif change == "allocator":
        run["peak_reserved_bytes"] = 1
    elif change == "gpu":
        run["environment"]["gpu_uuid"] = "other"
    else:
        run["implementation_sha256"] = "e" * 64
    with pytest.raises(ValueError):
        estimate_diffusion_training(source, reports)


def test_changed_cached_payload_requires_a_new_calibration(inputs):
    source = plan(inputs)
    data = inputs[1]
    raw = bytearray(data.read_bytes())
    raw[-4:] = struct.pack("<f", 1)
    data.write_bytes(raw)
    changed = plan(inputs)
    assert changed["signature"]["inputs_sha256"] != source["signature"]["inputs_sha256"]
    with pytest.raises(ValueError, match="identical weights, data"):
        estimate_diffusion_training(
            changed, [native(source, 1, 1), native(source, 2, 2)]
        )


def test_holdout_cannot_reuse_a_calibration_process(inputs):
    source = plan(inputs)
    reports = [native(source, 1, 1), native(source, 2, 2)]
    with pytest.raises(ValueError, match="holdout repeats"):
        compare_diffusion_training(reports, reports[0])


@pytest.mark.parametrize(
    "field,value", [("epochs", True), ("epochs", 0), ("unknown", 1)]
)
def test_resealed_invalid_plans_are_rejected(inputs, field, value):
    source = plan(inputs)
    source[field] = value
    source["plan_sha256"] = _digest(
        {k: v for k, v in source.items() if k != "plan_sha256"}
    )
    with pytest.raises(ValueError):
        validate_training_plan(source)


@pytest.mark.parametrize("decay", [True, -0.1, 1, float("nan")])
def test_ema_decay_is_an_explicit_validated_policy(inputs, decay):
    with pytest.raises(ValueError, match="ema_decay"):
        plan(inputs, ema_decay=decay)


def test_text_time_config_without_time_embedding_width_is_a_usage_error(
    inputs, capsys
):
    model, data = inputs
    config_path = model / "unet/config.json"
    config = json.loads(config_path.read_text())
    config.update(
        addition_embed_type="text_time", projection_class_embeddings_input_dim=10
    )
    config_path.write_text(json.dumps(config))
    safefile(
        data,
        {
            "latents": ("F32", [5, 4, 8, 8]),
            "noise": ("F32", [5, 4, 8, 8]),
            "timesteps": ("I64", [5]),
            "encoder_hidden_states": ("F32", [5, 3, 4]),
            "pooled_text_embeds": ("F32", [5, 4]),
            "time_ids": ("F32", [5, 6]),
        },
    )
    with pytest.raises(SystemExit) as error:
        plan_main(["--model-dir", str(model), "--dataset", str(data), "--epochs", "1"])
    assert error.value.code == 2
    assert "six time IDs must match the UNet projection" in capsys.readouterr().err


def test_stdlib_cli_budgets_compressed_evidence_after_input_files_are_removed(
    inputs, tmp_path
):
    source = plan(inputs, ema_decay=0.9)
    reports = [native(source, 1, 1), native(source, 2, 3)]
    paths = [tmp_path / f"cal-{index}.json.gz" for index in range(2)]
    for path, report in zip(paths, reports):
        write_json(path, report)
    plan_path, output = tmp_path / "plan.json.gz", tmp_path / "budget.json.gz"
    write_json(plan_path, source)
    shutil.rmtree(inputs[0])
    inputs[1].unlink()
    args = [value for path in paths for value in ("--calibration-run", str(path))]
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "fakegpu",
            "estimate-diffusion-training",
            "--plan",
            str(plan_path),
            "--epochs",
            "5",
            *args,
            "--json",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert load_mapping(output)["time"]["total_seconds"] == 56


def test_native_cli_preserves_existing_report_before_loading_optional_runtime(tmp_path):
    output, directory = tmp_path / "run.json", tmp_path / "output"
    output.write_text("existing evidence")
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "fakegpu",
            "run-diffusion-training",
            "--plan",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(directory),
            "--json",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2 and "report already exists" in result.stderr
    assert output.read_text() == "existing evidence" and not directory.exists()


@pytest.mark.parametrize("limit,status", [(5, 1), (10, 0)])
def test_validation_cli_enforces_explicit_accuracy_limits(
    inputs, tmp_path, limit, status
):
    source = plan(inputs)
    reports = [native(source, 1, 1), native(source, 2, 3)]
    target = project_diffusion_training(source, epochs=5)
    holdout = native(target, 3, 2.2)
    paths = [tmp_path / f"c-{index}.json" for index in range(2)]
    for path, report in zip(paths, reports):
        write_json(path, report)
    actual, output = tmp_path / "holdout.json", tmp_path / "comparison.json"
    write_json(actual, holdout)
    args = [value for path in paths for value in ("--calibration-run", str(path))]
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "fakegpu",
            "validate-diffusion-training",
            *args,
            "--holdout-run",
            str(actual),
            "--max-total-time-error-percent",
            str(limit),
            "--max-memory-underestimate-percent",
            "0",
            "--json",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == status, result.stderr
    assert load_mapping(output)["validation"]["passed"] is (status == 0)
