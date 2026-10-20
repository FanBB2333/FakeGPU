"""Offline plans and measured budgets for cached-feature UNet training.

Latents and text embeddings are supplied explicitly. VAE/text-encoder training,
raw-image preprocessing, distributed execution and unsupported denoisers are
outside this protocol. Activation and allocator peaks require native evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
from datetime import datetime
from uuid import UUID

from ._diffusion_pipeline import _select_component_checkpoint
from .llm_estimator import _read_safetensors_header, inspect_safetensors_checkpoint
from ._finetune_workflow_store import _file as _file_identity
from ._finetune_cache import validate_cache_regime
from ._finetune_allocator import validate_memory_allocator

PLAN_SCHEMA = "fakegpu.diffusion_training_plan.v1"
RUN_SCHEMA = "fakegpu.diffusion_training_run.v1"
TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
_CACHED_TENSORS = frozenset(
    {"latents", "noise", "timesteps", "encoder_hidden_states"}
)
_TEXT_TIME_TENSORS = frozenset({"pooled_text_embeds", "time_ids"})


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a SHA256 hex digest")
    return value


def _file(path):
    return {"name": Path(path).name, **_file_identity(path)}


def _persistent_storage(s):
    trainable = s["trainable_parameters"]
    return {
        "frozen_parameters": s["base_parameters"] * 2 if s["method"] == "lora" else 0,
        "trainable_parameters": trainable * 4,
        "gradients": trainable * 4,
        "adamw_moments": trainable * 8,
        "ema_parameters": trainable * 4 if s["ema_decay"] is not None else 0,
    }


def _execution_context(run):
    return {
        "environment": run["environment"],
        "implementation_sha256": run["implementation_sha256"],
        "cache_regime": run["cache_regime"],
        "memory_allocator": run["memory_allocator"],
    }


def plan_diffusion_training(
    model_dir,
    dataset_path,
    *,
    batch_size=1,
    gradient_accumulation_steps=1,
    epochs=1,
    method="lora",
    lora_rank=8,
    gradient_checkpointing=False,
    ema_decay=None,
):
    """Inspect exact cached data/weights with stdlib; no transient memory fallback."""
    batch = _integer(batch_size, "batch_size")
    accum = _integer(gradient_accumulation_steps, "gradient_accumulation_steps")
    _integer(epochs, "epochs")
    _integer(lora_rank, "lora_rank")
    if ema_decay is not None:
        ema_decay = _finite(ema_decay, "ema_decay")
        if ema_decay >= 1:
            raise ValueError("ema_decay must be in [0, 1)")
    if method not in {"full", "lora"} or not isinstance(gradient_checkpointing, bool):
        raise ValueError(
            "method must be full/lora and gradient_checkpointing a boolean"
        )
    root = Path(model_dir).expanduser().resolve()
    data_path = Path(dataset_path).expanduser().resolve()
    config_path = root / "unet/config.json"
    config = json.loads(config_path.read_text())
    scheduler_path = root / "scheduler/scheduler_config.json"
    scheduler = json.loads(scheduler_path.read_text())
    if (
        config.get("_class_name") != "UNet2DConditionModel"
        or config.get("in_channels") != 4
        or config.get("out_channels") != 4
        or config.get("addition_embed_type") not in (None, "text_time")
        or config.get("class_embed_type") is not None
        or config.get("encoder_hid_dim_type") is not None
        or config.get("num_class_embeds") is not None
        or config.get("time_cond_proj_dim") is not None
        or config.get("dual_cross_attention", False)
        or config.get("attention_type", "default") != "default"
    ):
        raise ValueError(
            "cached training requires standard four-channel UNet2DConditionModel with optional text_time conditioning"
        )
    if scheduler.get("prediction_type", "epsilon") != "epsilon":
        raise ValueError("cached training requires epsilon prediction")
    horizon = _integer(scheduler.get("num_train_timesteps"), "num_train_timesteps")
    names, _, variant = _select_component_checkpoint(root / "unet", weight_variant=None)
    if variant is not None:
        raise ValueError("training requires the default unquantized checkpoint family")
    inspection = inspect_safetensors_checkpoint(root / "unet", files=names)
    base = inspection["parameter_count"]
    adapt = 0
    selected = {}
    for name in names:
        for key, meta in _read_safetensors_header(root / "unet" / name).items():
            if key == "__metadata__":
                continue
            if meta["dtype"] not in {"F32", "F16", "BF16"}:
                raise ValueError("training requires floating-point UNet weights")
            if any(key.endswith("." + target + ".weight") for target in TARGETS):
                shape = meta["shape"]
                if len(shape) != 2:
                    raise ValueError(
                        "attention adapters require two-dimensional projection weights"
                    )
                if key in selected:
                    raise ValueError("checkpoint repeats an attention projection")
                selected[key] = shape
                adapt += lora_rank * sum(shape)
    if not selected:
        raise ValueError(
            "UNet checkpoint has no supported attention projection targets"
        )
    header = {
        k: v
        for k, v in _read_safetensors_header(data_path).items()
        if k != "__metadata__"
    }
    text_time = config.get("addition_embed_type") == "text_time"
    required = _CACHED_TENSORS | _TEXT_TIME_TENSORS if text_time else _CACHED_TENSORS
    if set(header) != required:
        raise ValueError(
            "cached dataset tensors do not match the model conditioning contract"
        )
    # Validate all offsets, byte counts and dtypes with the shared safetensors reader.
    inspect_safetensors_checkpoint(data_path.parent, files=[data_path.name])
    latent = header["latents"]["shape"]
    if (
        len(latent) != 4
        or latent[1] != 4
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in latent)
        or latent[2] % 8
        or latent[3] % 8
    ):
        raise ValueError(
            "latents must be [rows,4,H,W] with positive spatial multiples of eight"
        )
    count, _, height, width = latent
    embedding = header["encoder_hidden_states"]["shape"]
    cross = config.get("cross_attention_dim")
    if not isinstance(cross, int) or isinstance(cross, bool) or cross < 1:
        raise ValueError("training requires one integer cross_attention_dim")
    if (
        len(embedding) != 3
        or embedding[0] != count
        or embedding[1] < 1
        or embedding[2] != cross
    ):
        raise ValueError(
            "encoder_hidden_states must match rows and cross_attention_dim"
        )
    if (
        header["noise"]["shape"] != latent
        or header["timesteps"]["shape"] != [count]
        or header["timesteps"]["dtype"] != "I64"
    ):
        raise ValueError("noise/timesteps do not match latent rows")
    for key, meta in header.items():
        if key != "timesteps" and meta["dtype"] != "F32":
            raise ValueError("cached floating tensors must be FP32")
    if text_time:
        pooled = header["pooled_text_embeds"]["shape"]
        time_ids = header["time_ids"]["shape"]
        time_dim = config.get("addition_time_embed_dim")
        projection = config.get("projection_class_embeddings_input_dim")
        if (
            len(pooled) != 2
            or pooled[0] != count
            or time_ids != [count, 6]
            or time_dim is None
            or projection is None
            or pooled[1] + 6 * time_dim != projection
        ):
            raise ValueError(
                "text_time pooled embeddings and six time IDs must match the UNet projection"
            )
    trainable = base if method == "full" else adapt
    manifest = {
        "model": [
            _file(config_path),
            _file(scheduler_path),
            *[_file(root / "unet" / n) for n in names],
        ],
        "dataset": _file(data_path),
    }
    signature = {
        "inputs_sha256": _digest(manifest),
        "method": method,
        "lora_rank": lora_rank if method == "lora" else None,
        "batch_size": batch,
        "gradient_accumulation_steps": accum,
        "gradient_checkpointing": gradient_checkpointing,
        "ema_decay": ema_decay,
        "optimizer": "adamw_torch_foreach_false",
        "learning_rate": 2e-5,
        "seed": 42,
        "precision": "fp32_trainable_bf16_autocast",
        "attention_backend": "sdpa",
        "save": "model_or_adapter_only",
        "dataset_rows": count,
        "latent_shape": latent[1:],
        "embedding_shape": embedding[1:],
        "tensor_shapes": {k: v["shape"] for k, v in header.items()},
        "scheduler_num_train_timesteps": horizon,
        "base_parameters": base,
        "trainable_parameters": trainable,
    }
    state = _persistent_storage(signature)
    plan = {
        "schema_version": PLAN_SCHEMA,
        "signature": signature,
        "model_dir": str(root),
        "dataset_path": str(data_path),
        "input_manifest": manifest,
        "epochs": epochs,
        "static_memory": {
            "kind": "persistent_tensor_storage_lower_bound",
            "components": state,
            "bytes": sum(state.values()),
            "activation_bytes": None,
            "allocator_reserved_bytes": None,
        },
        "limitations": [
            "Only cached-feature UNet training is included; VAE/text encoders, raw-image preparation and optimizer-state resume are excluded.",
            "Persistent storage is a lower bound, not a peak or an OOM safety claim. Native compatible processes are required for time and allocator envelopes.",
            "Calibration allocator ranges are observed process peaks, not physical OOM guarantees or bounds on longer-run growth. Kernel-cache identity covers only the explicitly recorded directories.",
        ],
    }
    plan["plan_sha256"] = _digest(plan)
    return validate_training_plan(plan)


def validate_training_plan(plan):
    fields = {
        "schema_version",
        "signature",
        "model_dir",
        "dataset_path",
        "input_manifest",
        "epochs",
        "static_memory",
        "limitations",
        "plan_sha256",
    }
    if (
        not isinstance(plan, Mapping)
        or set(plan) != fields
        or plan.get("schema_version") != PLAN_SCHEMA
    ):
        raise ValueError("unsupported diffusion training plan")
    if plan.get("plan_sha256") != _digest(
        {k: v for k, v in plan.items() if k != "plan_sha256"}
    ):
        raise ValueError("diffusion training plan digest differs from contents")
    _integer(plan["epochs"], "epochs")
    s = plan["signature"]
    signature_fields = {
        "inputs_sha256",
        "method",
        "lora_rank",
        "batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "ema_decay",
        "optimizer",
        "learning_rate",
        "seed",
        "precision",
        "attention_backend",
        "save",
        "dataset_rows",
        "latent_shape",
        "embedding_shape",
        "tensor_shapes",
        "scheduler_num_train_timesteps",
        "base_parameters",
        "trainable_parameters",
    }
    if not isinstance(s, Mapping) or set(s) != signature_fields:
        raise ValueError(
            "diffusion execution signature fields are incomplete or unsupported"
        )
    for name in ("model_dir", "dataset_path"):
        if not isinstance(plan[name], str) or not Path(plan[name]).is_absolute():
            raise ValueError(f"{name} must be an absolute local path")
    for key in (
        "dataset_rows",
        "batch_size",
        "gradient_accumulation_steps",
        "base_parameters",
        "trainable_parameters",
        "scheduler_num_train_timesteps",
    ):
        _integer(s[key], key)
    if (
        s["method"] not in {"full", "lora"}
        or s["optimizer"] != "adamw_torch_foreach_false"
        or s["precision"] != "fp32_trainable_bf16_autocast"
        or s["attention_backend"] != "sdpa"
        or s["save"] != "model_or_adapter_only"
        or s["learning_rate"] != 2e-5
        or s["seed"] != 42
        or not isinstance(s["gradient_checkpointing"], bool)
    ):
        raise ValueError("unsupported diffusion training execution policy")
    if s["method"] == "lora":
        _integer(s["lora_rank"], "lora_rank")
    elif (
        s["lora_rank"] is not None or s["base_parameters"] != s["trainable_parameters"]
    ):
        raise ValueError("full training parameter identity is inconsistent")
    if s["ema_decay"] is not None and _finite(s["ema_decay"], "ema_decay") >= 1:
        raise ValueError("ema_decay must be in [0, 1)")
    for name, width in (("latent_shape", 3), ("embedding_shape", 2)):
        shape = s[name]
        if not isinstance(shape, list) or len(shape) != width:
            raise ValueError(f"{name} has an invalid rank")
        for value in shape:
            _integer(value, name)
    if s["latent_shape"][0] != 4 or any(v % 8 for v in s["latent_shape"][1:]):
        raise ValueError(
            "latent dimensions must be four channels and spatial multiples of eight"
        )
    shapes = s["tensor_shapes"]
    if not isinstance(shapes, Mapping) or set(shapes) not in (
        _CACHED_TENSORS,
        _CACHED_TENSORS | _TEXT_TIME_TENSORS,
    ):
        raise ValueError("cached tensor shape fields are incomplete or unsupported")
    for shape in shapes.values():
        if not isinstance(shape, list) or not shape:
            raise ValueError("cached tensor shapes must be nonempty lists")
        for dimension in shape:
            _integer(dimension, "cached tensor dimension")
    rows = s["dataset_rows"]
    if (
        shapes["latents"] != [rows, *s["latent_shape"]]
        or shapes["noise"] != shapes["latents"]
        or shapes["timesteps"] != [rows]
        or shapes["encoder_hidden_states"] != [rows, *s["embedding_shape"]]
    ):
        raise ValueError("cached tensor shapes disagree with the training signature")
    if "time_ids" in shapes:
        pooled = shapes["pooled_text_embeds"]
        if (
            shapes["time_ids"] != [rows, 6]
            or len(pooled) != 2
            or pooled[0] != rows
        ):
            raise ValueError("cached text_time shapes are inconsistent")
    expected = _persistent_storage(s)
    memory = plan["static_memory"]
    if (
        not isinstance(memory, Mapping)
        or set(memory)
        != {
            "kind",
            "components",
            "bytes",
            "activation_bytes",
            "allocator_reserved_bytes",
        }
        or memory["kind"] != "persistent_tensor_storage_lower_bound"
        or memory["activation_bytes"] is not None
        or memory["allocator_reserved_bytes"] is not None
    ):
        raise ValueError(
            "static training memory must remain an explicit storage lower bound"
        )
    if memory["components"] != expected or memory["bytes"] != sum(expected.values()):
        raise ValueError("persistent training tensor storage is inconsistent")
    manifest = plan["input_manifest"]
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != {"model", "dataset"}
        or not isinstance(manifest["model"], list)
        or len(manifest["model"]) < 3
    ):
        raise ValueError("training input manifest is incomplete")
    for entry in [*manifest["model"], manifest["dataset"]]:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"name", "bytes", "sha256"}
            or not isinstance(entry["name"], str)
            or Path(entry["name"]).name != entry["name"]
        ):
            raise ValueError("training input file identity is incomplete")
        _integer(entry["bytes"], "input bytes")
        _sha(entry["sha256"], "input SHA256")
    if _digest(manifest) != s["inputs_sha256"]:
        raise ValueError("training input manifest identity is inconsistent")
    if not isinstance(plan["limitations"], list) or not all(
        isinstance(v, str) for v in plan["limitations"]
    ):
        raise ValueError("training limitations must be a string list")
    return deepcopy(dict(plan))


def project_diffusion_training(plan, *, epochs):
    """Change only the declared duration, without rereading local model weights."""
    result = validate_training_plan(plan)
    result["epochs"] = _integer(epochs, "epochs")
    result.pop("plan_sha256")
    result["plan_sha256"] = _digest(result)
    return result


def training_windows(plan):
    """Yield sequential native batches, including partial rows and accumulation tails."""
    s = plan["signature"]
    step = 0
    seen = set()
    for epoch in range(plan["epochs"]):
        window_rows = s["batch_size"] * s["gradient_accumulation_steps"]
        for start in range(0, s["dataset_rows"], window_rows):
            window_end = min(start + window_rows, s["dataset_rows"])
            window = [
                list(range(offset, min(offset + s["batch_size"], window_end)))
                for offset in range(start, window_end, s["batch_size"])
            ]
            step += 1
            shape = tuple(len(ids) for ids in window)
            first_shape = shape not in seen
            seen.add(shape)
            yield {
                "optimizer_step": step,
                "epoch_index": epoch + 1,
                "row_indices": window,
                "group": {
                    "stage": "first_update" if step == 1 else "repeated_update",
                    "microbatch_rows": list(shape),
                    "first_window_shape_use": first_shape,
                },
            }


def _finite(value, name, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or positive
        and value == 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _run(run):
    fields = {
        "schema_version",
        "plan",
        "identity",
        "environment",
        "updates",
        "timing",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "observed_model",
        "implementation_sha256",
        "cache_regime",
        "memory_allocator",
    }
    if (
        not isinstance(run, Mapping)
        or not fields.issubset(run)
        or run.get("schema_version") != RUN_SCHEMA
    ):
        raise ValueError("unsupported diffusion native run")
    plan = validate_training_plan(run["plan"])
    validate_cache_regime(run["cache_regime"])
    validate_memory_allocator(run["memory_allocator"])
    if run["memory_allocator"]["backend"] != "native":
        raise ValueError(
            "diffusion peak evidence requires the native caching allocator"
        )
    identity = run["identity"]
    environment = run["environment"]
    updates = run["updates"]
    if (
        not isinstance(environment, Mapping)
        or not isinstance(run.get("observed_model"), Mapping)
        or not isinstance(run.get("timing"), Mapping)
    ):
        raise ValueError(
            "native training requires environment, observed_model and timing mappings"
        )
    if (
        environment.get("backend") != "cuda"
        or environment.get("simulated") is not False
        or not environment.get("gpu_uuid")
        or not isinstance(environment.get("software"), Mapping)
    ):
        raise ValueError(
            "training evidence requires real CUDA UUID and software identity"
        )
    if "total_memory_bytes" not in environment:
        raise ValueError("native CUDA environment identity is incomplete")
    _integer(environment["total_memory_bytes"], "physical GPU memory")
    if (
        not isinstance(identity, Mapping)
        or not isinstance(identity.get("uuid"), str)
        or isinstance(identity.get("pid"), bool)
        or not isinstance(identity.get("pid"), int)
        or identity["pid"] < 1
        or not isinstance(identity.get("started_at_utc"), str)
    ):
        raise ValueError("native process identity is incomplete")
    if (
        str(UUID(identity["uuid"])) != identity["uuid"]
        or datetime.fromisoformat(identity["started_at_utc"]).utcoffset() is None
    ):
        raise ValueError(
            "native identity requires a canonical UUID and timezone-aware start"
        )
    s = plan["signature"]
    window_rows = s["batch_size"] * s["gradient_accumulation_steps"]
    per_epoch = (s["dataset_rows"] + window_rows - 1) // window_rows
    if not isinstance(updates, list) or len(updates) != per_epoch * plan["epochs"]:
        raise ValueError("native updates do not cover declared epochs")
    for actual, expected in zip(updates, training_windows(plan)):
        if (
            not isinstance(actual, Mapping)
            or not set(expected).issubset(actual)
            or not {"seconds", "loss", "input_sha256"}.issubset(actual)
        ):
            raise ValueError("native update evidence is incomplete")
        if {k: actual[k] for k in expected} != expected:
            raise ValueError("native row consumption differs from plan")
        _finite(actual["seconds"], "update seconds", True)
        _finite(actual["loss"], "training loss")
        hashes = actual["input_sha256"]
        if not isinstance(hashes, list) or len(hashes) != len(expected["row_indices"]):
            raise ValueError("native update requires every consumed microbatch hash")
        for digest in hashes:
            _sha(digest, "microbatch SHA256")
    phase = run["timing"]
    if not {"setup_seconds", "finalization_seconds", "total_seconds"}.issubset(phase):
        raise ValueError("native training phase timings are incomplete")
    for key in ("setup_seconds", "finalization_seconds", "total_seconds"):
        _finite(phase[key], key, True)
    total = (
        phase["setup_seconds"]
        + sum(u["seconds"] for u in updates)
        + phase["finalization_seconds"]
    )
    if not math.isclose(total, phase["total_seconds"], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("training timing does not conserve native wall time")
    allocated = _integer(run["peak_allocated_bytes"], "peak_allocated_bytes")
    reserved = _integer(run["peak_reserved_bytes"], "peak_reserved_bytes")
    if reserved < allocated or reserved > environment["total_memory_bytes"]:
        raise ValueError("native allocator peaks are inconsistent")
    if allocated < plan["static_memory"]["bytes"]:
        raise ValueError("native peak is below the required persistent tensor storage")
    model = run["observed_model"]
    if not {
        "base_parameters",
        "trainable_parameters",
        "trainable_dtypes",
        "gradient_checkpointing",
        "gradient_bytes",
        "optimizer_moment_bytes",
        "ema_bytes",
    }.issubset(model):
        raise ValueError("native model storage evidence is incomplete")
    _integer(model["base_parameters"], "observed base parameters")
    _integer(model["trainable_parameters"], "observed trainable parameters")
    if (
        model["base_parameters"] != plan["signature"]["base_parameters"]
        or model["trainable_parameters"] != plan["signature"]["trainable_parameters"]
        or model["trainable_dtypes"] != {"torch.float32": model["trainable_parameters"]}
        or model["gradient_checkpointing"] is not s["gradient_checkpointing"]
        or model["gradient_bytes"] != plan["static_memory"]["components"]["gradients"]
        or model["optimizer_moment_bytes"]
        != plan["static_memory"]["components"]["adamw_moments"]
        or model["ema_bytes"] != plan["static_memory"]["components"]["ema_parameters"]
    ):
        raise ValueError("native parameter storage differs from the training plan")
    _sha(run["implementation_sha256"], "native implementation SHA256")
    return plan


def estimate_diffusion_training(plan, calibration_runs=()):
    """Use two independent, matching native processes for complete training costs."""
    target = validate_training_plan(plan)
    counts, descriptors = Counter(), {}
    for window in training_windows(target):
        key = _digest(window["group"])
        counts[key] += 1
        descriptors[key] = window["group"]
    runs = list(calibration_runs)
    by_run = []
    identities = set()
    evidence = set()
    timings = set()
    common = None
    for run in runs:
        measured = _run(run)
        if measured["signature"] != target["signature"]:
            raise ValueError(
                "diffusion calibration requires identical weights, data, shapes and execution policy"
            )
        identity = run["identity"]
        pid = (identity["pid"], identity["started_at_utc"])
        timing_digest = _digest({"updates": run["updates"], "timing": run["timing"]})
        if (
            identity["uuid"] in identities
            or pid in evidence
            or timing_digest in timings
        ):
            raise ValueError("calibration repeats a native process")
        identities.add(identity["uuid"])
        evidence.add(pid)
        timings.add(timing_digest)
        context = _execution_context(run)
        if common is not None and context != common:
            raise ValueError(
                "calibration requires matching GPU, software and implementation"
            )
        common = context
        groups = defaultdict(list)
        for row in run["updates"]:
            groups[_digest(row["group"])].append(row["seconds"])
        by_run.append(groups)
    missing = []
    eligible = {}
    for key, descriptor in descriptors.items():
        minimum = 1 if descriptor["first_window_shape_use"] else 2
        eligible[key] = [g[key] for g in by_run if len(g[key]) >= minimum]
        if len(eligible[key]) < 2:
            missing.append(
                {
                    "group": descriptor,
                    "eligible_independent_runs": len(eligible[key]),
                    "required_independent_runs": 2,
                    "minimum_samples_per_run": minimum,
                }
            )
    result = {
        "schema_version": "fakegpu.diffusion_training_estimate.v1",
        "plan_sha256": target["plan_sha256"],
        "static_memory": target["static_memory"],
        "coverage": {"complete": not missing, "missing": missing},
        "time": None,
        "memory": None,
        "limitations": target["limitations"],
    }
    if missing:
        return result
    groups = []
    total = 0.0
    for key, descriptor in descriptors.items():
        means = [statistics.mean(samples) for samples in eligible[key]]
        cost = statistics.mean(means) * counts[key]
        total += cost
        groups.append(
            {
                "group": descriptor,
                "target_updates": counts[key],
                "per_process_means_seconds": means,
                "predicted_seconds": cost,
            }
        )
    setup = statistics.mean(run["timing"]["setup_seconds"] for run in runs)
    final = statistics.mean(run["timing"]["finalization_seconds"] for run in runs)
    result.update(
        time={
            "setup_seconds": setup,
            "updates_seconds": total,
            "finalization_seconds": final,
            "total_seconds": setup + total + final,
            "groups": groups,
        },
        memory={
            field: {
                "min": min(run[field] for run in runs),
                "max": max(run[field] for run in runs),
            }
            for field in ("peak_allocated_bytes", "peak_reserved_bytes")
        },
        calibration_identities=[deepcopy(run["identity"]) for run in runs],
        execution_context=common,
    )
    return result


def compare_diffusion_training(calibration_runs, holdout_run):
    """Compare an independent native run without fitting any holdout measurement."""
    plan = _run(holdout_run)
    runs = list(calibration_runs)
    prediction = estimate_diffusion_training(plan, runs)
    if not prediction["coverage"]["complete"]:
        raise ValueError(
            "diffusion holdout requires complete prior calibration coverage"
        )
    if _execution_context(holdout_run) != prediction["execution_context"]:
        raise ValueError("holdout requires the same GPU, software and implementation")
    actual_id = holdout_run["identity"]
    for run in runs:
        identity = run["identity"]
        if (
            actual_id["uuid"] == identity["uuid"]
            or (actual_id["pid"], actual_id["started_at_utc"])
            == (identity["pid"], identity["started_at_utc"])
            or (holdout_run["updates"], holdout_run["timing"])
            == (run["updates"], run["timing"])
        ):
            raise ValueError("holdout repeats calibration evidence")
    observed = {
        **holdout_run["timing"],
        "updates_seconds": sum(row["seconds"] for row in holdout_run["updates"]),
    }
    timing = {}
    for name in (
        "setup_seconds",
        "updates_seconds",
        "finalization_seconds",
        "total_seconds",
    ):
        expected, actual = prediction["time"][name], observed[name]
        timing[name] = {
            "predicted": expected,
            "actual": actual,
            "absolute_percentage_error": abs(expected - actual) / actual * 100,
            "underprediction_percent": max(0, actual - expected) / actual * 100,
        }
    memory = {}
    for field, interval in prediction["memory"].items():
        actual = holdout_run[field]
        memory[field] = {
            "actual": actual,
            "observed_max": interval["max"],
            "exceeds_observed_max": actual > interval["max"],
            "underprediction_percent": max(0, actual - interval["max"]) / actual * 100,
        }
    return {
        "schema_version": "fakegpu.diffusion_training_comparison.v1",
        "prediction": prediction,
        "time_comparison": timing,
        "memory_comparison": memory,
        "holdout_identity": deepcopy(actual_id),
        "oom_safety_verified": False,
    }
