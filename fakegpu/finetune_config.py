"""Read training settings without importing or executing a training framework.

Only supervised/causal language-model training is normalized. The result describes
configuration semantics, not proof that a framework or model ran successfully.

Configuration references:
https://huggingface.co/docs/transformers/main_classes/trainer
https://huggingface.co/docs/peft/main/en/package_reference/lora
https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html
https://docs.axolotl.ai/docs/config-reference.html
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .structured_io import load_mapping
from .training_plan import TrainingPlanError, normalize_training_config


SCHEMA_VERSION = "fakegpu.finetune_config.v1"
SUPPORTED_FRAMEWORKS = {"auto", "huggingface", "llamafactory", "axolotl"}
_FRAMEWORK_ALIASES = {
    "hf": "huggingface",
    "trainer": "huggingface",
    "peft": "huggingface",
    "llama-factory": "llamafactory",
    "llama_factory": "llamafactory",
}


class FinetuneConfigError(ValueError):
    pass


def load_finetune_config(
    path: str | Path, *, framework: str = "auto"
) -> dict[str, Any]:
    """Resolve local model, dataset and distributed paths against this file."""
    resolved = Path(path).expanduser().resolve()
    result = normalize_finetune_config(
        load_mapping(resolved), framework=framework, base_dir=resolved.parent
    )
    result["source_path"] = str(resolved)
    return result


def normalize_finetune_config(
    config: Mapping[str, Any],
    *,
    framework: str = "auto",
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize Trainer/PEFT, LLaMA-Factory or Axolotl configuration.

    ``num_samples`` means the effective training split *after* filtering and
    validation splitting. ``max_samples`` remains a cap, never a dataset count.
    Missing sequence length and sample count remain None for caller overrides.
    """
    if not isinstance(config, Mapping):
        raise FinetuneConfigError("training configuration must be a mapping")
    _check_finite(config)
    requested = _framework(framework)
    declared = _framework(config.get("framework", "auto"))
    if requested != "auto" and declared not in {"auto", requested}:
        raise FinetuneConfigError("framework argument conflicts with configuration")
    source = declared if requested == "auto" else requested
    if source == "auto":
        source = _detect_framework(config)
    root = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd()
    values = _flatten(config)
    peft = _mapping_alias(values, "peft", "peft_config")
    assumptions: list[str] = []
    _check_strategies(values, peft)

    model = _alias(values, "model_name_or_path", "base_model")
    if not isinstance(model, str) or not model.strip():
        raise FinetuneConfigError("model_name_or_path (or base_model) is required")
    revision = _alias(values, "model_revision", "revision")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise FinetuneConfigError("model_revision must be a non-empty string")
    checkpoint_sha256 = values.get("checkpoint_sha256")
    if checkpoint_sha256 is not None:
        if (
            not isinstance(checkpoint_sha256, str)
            or len(checkpoint_sha256) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in checkpoint_sha256)
        ):
            raise FinetuneConfigError(
                "checkpoint_sha256 must contain exactly 64 hexadecimal characters"
            )
        checkpoint_sha256 = checkpoint_sha256.lower()
    dtype = _precision(values, source)
    storage_dtype = _alias(values, "parameter_dtype", "torch_dtype")
    parameter_dtype = _dtype(storage_dtype) if storage_dtype is not None else "fp32"
    if storage_dtype is None:
        assumptions.append(
            "Parameter storage defaults to FP32; mixed precision controls computation. "
            "Set torch_dtype/parameter_dtype to match checkpoint loading for tighter estimates."
        )
    quant = _quantization_mapping(values)
    quantization = _quantization(values, quant, source)
    method, rank, targets = _adapter(values, peft, source, quantization)
    adapter_options = _adapter_options(values, peft, source, rank, assumptions)
    quantization_options = _quantization_options(
        values, quant, source, quantization, dtype
    )
    learning_rate = values.get("learning_rate")
    if learning_rate is None:
        learning_rate = None if source == "axolotl" else 5e-5
        if learning_rate is None:
            assumptions.append(
                "Axolotl requires learning_rate for execution; it is unspecified in this estimate."
            )
    if learning_rate is not None:
        # Axolotl explicitly accepts scientific-notation strings in YAML.
        if isinstance(learning_rate, str):
            try:
                learning_rate = float(learning_rate)
            except ValueError as exc:
                raise FinetuneConfigError(
                    "learning_rate must be a finite positive number"
                ) from exc
        learning_rate = _positive_float(learning_rate, "learning_rate")
    optimizer_options = _optimizer_options(values, source, assumptions)

    micro_batch = _integer(
        _alias(
            values,
            "micro_batch_size",
            "per_device_train_batch_size",
            default=1 if source == "axolotl" else 8,
        ),
        "micro_batch_size",
    )
    accumulation = _integer(
        values.get("gradient_accumulation_steps", 1), "gradient_accumulation_steps"
    )
    world_size = _integer(
        _alias(values, "world_size", "num_processes", default=1), "world_size"
    )
    epochs = _positive_float(
        _alias(
            values,
            "epochs",
            "num_train_epochs",
            "num_epochs",
            default=1.0 if source == "axolotl" else 3.0,
        ),
        "epochs",
    )
    max_steps = values.get("max_steps")
    if max_steps is not None:
        max_steps = _integer(max_steps, "max_steps", minimum=-1)
        if max_steps == -1:
            max_steps = None
        elif max_steps == 0:
            raise FinetuneConfigError("max_steps must be positive or -1 (epochs)")
    if max_steps is not None:
        assumptions.append("max_steps overrides epochs, as in the source trainer.")

    checkpoint = values.get("gradient_checkpointing")
    disabled = values.get("disable_gradient_checkpointing")
    if source == "llamafactory" and checkpoint is False and disabled is not True:
        raise FinetuneConfigError(
            "LLaMA-Factory enables checkpointing independently of gradient_checkpointing=false; "
            "set disable_gradient_checkpointing=true to disable it"
        )
    if disabled is not None:
        enabled = not _boolean(disabled, "disable_gradient_checkpointing")
        if (
            checkpoint is not None
            and _boolean(checkpoint, "gradient_checkpointing") != enabled
        ):
            raise FinetuneConfigError("gradient checkpointing settings conflict")
        checkpoint = enabled
    if checkpoint is None:
        checkpoint = source == "llamafactory"
    checkpoint = _boolean(checkpoint, "gradient_checkpointing")
    sequence = _optional_integer(
        _alias(
            values,
            "sequence_length",
            "max_seq_length",
            "seq_length",
            "sequence_len",
            "cutoff_len",
            "block_size",
            default=2048 if source == "llamafactory" else None,
        ),
        "sequence_length",
    )
    samples = _optional_integer(
        _alias(values, "num_samples", "dataset_num_samples", "dataset_samples"),
        "num_samples",
    )
    max_samples = _optional_integer(
        _alias(values, "max_samples", "max_train_samples"), "max_samples"
    )
    if samples is None:
        assumptions.append(
            "Provide the effective training sample count after filtering and splitting."
        )
    if sequence is None:
        assumptions.append(
            "Provide the padded training sequence length including prompt/template tokens."
        )
    stage = str(values.get("stage", "sft")).lower()
    packing = _boolean(
        _alias(
            values,
            "packing",
            "sample_packing",
            default=source == "llamafactory" and stage == "pt",
        ),
        "packing",
    )
    # LLaMA-Factory DataArguments.__post_init__ turns packing on for neat_packing.
    if (
        source == "llamafactory"
        and values.get("neat_packing") is not None
        and _boolean(values["neat_packing"], "neat_packing")
    ):
        packing = True
    if packing:
        assumptions.append(
            "With packing, sample count must describe packed training sequences."
        )
    if max_samples is not None:
        assumptions.append(
            "max_samples is a per-dataset cap, not a measured training dataset size."
        )
    optimizer = str(
        _alias(values, "optimizer", "optim", default="adamw_torch_fused")
    ).lower()
    if optimizer not in {
        "adam",
        "adamw",
        "adamw_torch",
        "adamw_torch_fused",
        "adamw_hf",
        "adamw_apex_fused",
        "sgd",
    }:
        raise FinetuneConfigError(
            f"unsupported optimizer {optimizer!r}; use Adam/AdamW or SGD"
        )
    if "batch_size" in values:
        batch = _integer(values["batch_size"], "batch_size")
        if batch != micro_batch * accumulation * world_size:
            raise FinetuneConfigError(
                "batch_size conflicts with micro batch, accumulation and world_size"
            )
    distributed = _distributed(
        values,
        root,
        world_size,
        accumulation,
        micro_batch,
        dtype,
        checkpoint,
        assumptions,
    )
    sources = _dataset_sources(values, root, source)
    software = values.get("software_versions", {})
    if not isinstance(software, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in software.items()
    ):
        raise FinetuneConfigError("software_versions must map names to version strings")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_framework": source,
        "model_name_or_path": _local_path(model.strip(), root),
        "model_revision": revision,
        "checkpoint_sha256": checkpoint_sha256,
        "method": method,
        "dtype": dtype,
        "parameter_dtype": parameter_dtype,
        "attention_implementation": _attention(values),
        "tf32": _boolean(values["tf32"], "tf32")
        if values.get("tf32") is not None
        else None,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "epochs": epochs,
        "max_steps": max_steps,
        "sequence_length": sequence,
        "num_samples": samples,
        "max_samples": max_samples,
        "world_size": world_size,
        "lora_rank": rank,
        "lora_target_modules": targets,
        **adapter_options,
        "gradient_checkpointing": distributed["activation_checkpointing"],
        **_checkpoint_options(values, source, distributed["activation_checkpointing"]),
        "quantization_bits": quantization,
        **quantization_options,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        **optimizer_options,
        "packing": packing,
        "dataloader_drop_last": _boolean(
            values.get("dataloader_drop_last", False), "dataloader_drop_last"
        ),
        "dataset_sources": sources,
        "distributed_config": distributed,
        "software_versions": dict(software),
        "assumptions": assumptions,
    }


def _framework(value: Any) -> str:
    if not isinstance(value, str):
        raise FinetuneConfigError("framework must be a string")
    result = value.strip().lower()
    result = _FRAMEWORK_ALIASES.get(result, result)
    if result not in SUPPORTED_FRAMEWORKS:
        raise FinetuneConfigError(
            "framework must be auto, huggingface, llamafactory or axolotl"
        )
    return result


def _detect_framework(config: Mapping[str, Any]) -> str:
    candidates = set()
    if any(
        k in config
        for k in ("base_model", "micro_batch_size", "sequence_len", "adapter")
    ):
        candidates.add("axolotl")
    if any(
        k in config
        for k in ("finetuning_type", "cutoff_len", "lora_target", "template")
    ):
        candidates.add("llamafactory")
    if any(k in config for k in ("training_args", "peft", "peft_config")):
        candidates.add("huggingface")
    if not candidates and any(
        k in config for k in ("model_name_or_path", "per_device_train_batch_size")
    ):
        candidates.add("huggingface")
    if len(candidates) != 1:
        raise FinetuneConfigError(
            "cannot unambiguously detect framework; specify framework explicitly"
        )
    return candidates.pop()


def _flatten(config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(config)
    for section in ("model_args", "training_args", "data_args"):
        if section not in result:
            continue
        nested = result.pop(section)
        if not isinstance(nested, Mapping):
            raise FinetuneConfigError(f"{section} must be a mapping")
        for key, value in nested.items():
            if key in result and result[key] != value:
                raise FinetuneConfigError(
                    f"conflicting {key!r} in {section} and outer configuration"
                )
            result[key] = value
    dataset = result.get("dataset")
    if isinstance(dataset, Mapping):
        for key in ("num_samples", "sequence_length"):
            if key in dataset:
                if key in result and result[key] != dataset[key]:
                    raise FinetuneConfigError(f"conflicting dataset.{key} and {key}")
                result[key] = dataset[key]
    return result


def _alias(values: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    supplied = [(key, values[key]) for key in keys if values.get(key) is not None]
    if not supplied:
        return default
    if any(value != supplied[0][1] for _, value in supplied[1:]):
        raise FinetuneConfigError(
            "conflicting values for " + ", ".join(k for k, _ in supplied)
        )
    return supplied[0][1]


def _mapping_alias(values: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    result = _alias(values, *keys, default={})
    if not isinstance(result, Mapping):
        raise FinetuneConfigError(f"{keys[0]} must be a mapping")
    return dict(result)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise FinetuneConfigError(
            f"{name} must be true or false; automatic/offload modes require explicit settings"
        )
    return value


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or int(value) != value
        or value < minimum
    ):
        raise FinetuneConfigError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _optional_integer(value: Any, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _positive_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise FinetuneConfigError(f"{name} must be a finite positive number")
    return float(value)


def _nonnegative_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise FinetuneConfigError(f"{name} must be a finite non-negative number")
    return float(value)


def _optimizer_options(
    values: Mapping[str, Any], source: str, assumptions: list[str]
) -> dict[str, Any]:
    scheduler = _alias(
        values,
        "lr_scheduler_type",
        "lr_scheduler",
        default="cosine" if source == "axolotl" else "linear",
    )
    if not isinstance(scheduler, str) or scheduler not in {
        "constant",
        "constant_with_warmup",
        "linear",
        "cosine",
    }:
        raise FinetuneConfigError(
            "lr_scheduler_type must be constant, constant_with_warmup, linear or cosine"
        )
    for key in (
        "optim_args",
        "optimizer_kwargs",
        "lr_scheduler_kwargs",
        "lr_quadratic_warmup",
        "cosine_min_lr_ratio",
        "cosine_constant_lr_ratio",
        "lr_groups",
        "embedding_lr",
        "embedding_lr_scale",
    ):
        if values.get(key):
            raise FinetuneConfigError(
                f"{key} is not supported by the fine-tuning estimator"
            )
    steps = values.get("warmup_steps")
    ratio = values.get("warmup_ratio")
    if source != "axolotl":
        steps = 0 if steps is None else steps
        ratio = 0.0 if ratio is None else ratio
    if steps is not None:
        steps = _integer(steps, "warmup_steps", minimum=0)
    if ratio is not None:
        ratio = _nonnegative_float(ratio, "warmup_ratio")
        if ratio > 1:
            raise FinetuneConfigError("warmup_ratio must be in [0, 1]")
    if steps and ratio:
        if source == "axolotl":
            raise FinetuneConfigError(
                "Axolotl warmup_steps and warmup_ratio are mutually exclusive"
            )
        assumptions.append(
            "warmup_steps takes precedence over warmup_ratio, as in Hugging Face Trainer."
        )
    if source == "axolotl" and steps is None and ratio is None:
        assumptions.append(
            "Axolotl warmup settings are unspecified; its runner selects the execution defaults."
        )
    decay = _nonnegative_float(
        _alias(values, "weight_decay", default=0.0), "weight_decay"
    )
    # Axolotl 0.18.0 forwards max_grad_norm only when it is set
    # (core/builders/base.py _set_base_training_args), so an unset value leaves
    # TrainingArguments' default of 1.0 and Trainer clips gradients.
    grad_norm = _nonnegative_float(
        _alias(values, "max_grad_norm", default=1.0), "max_grad_norm"
    )
    return {
        "lr_scheduler_type": scheduler,
        "warmup_steps": steps,
        "warmup_ratio": ratio,
        "weight_decay": decay,
        "max_grad_norm": grad_norm,
    }


def _check_finite(value: Any, path: str = "config") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FinetuneConfigError(f"{path} must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, f"{path}[{index}]")


def _dtype(value: Any) -> str:
    dtype = {
        "float32": "fp32",
        "float16": "fp16",
        "bfloat16": "bf16",
        "fp32": "fp32",
        "fp16": "fp16",
        "bf16": "bf16",
    }.get(str(value).lower())
    if dtype is None:
        raise FinetuneConfigError("dtype must explicitly be fp32, fp16 or bf16")
    return dtype


def _precision(values: Mapping[str, Any], source: str) -> str:
    selected = []
    for key, dtype in (
        ("bf16", "bf16"),
        ("bfloat16", "bf16"),
        ("pure_bf16", "bf16"),
        ("fp16", "fp16"),
        ("float16", "fp16"),
        ("float32", "fp32"),
    ):
        if values.get(key) is not None and _boolean(values[key], key):
            selected.append(dtype)
    if values.get("dtype") is not None:
        selected.append(_dtype(values["dtype"]))
    if not selected and values.get("torch_dtype") is not None:
        selected.append(_dtype(values["torch_dtype"]))
    if len(set(selected)) > 1:
        raise FinetuneConfigError("conflicting dtype/bf16/fp16 precision settings")
    if (
        not selected
        and source == "axolotl"
        and not any(k in values for k in ("bf16", "fp16", "float32"))
    ):
        raise FinetuneConfigError(
            "Axolotl defaults to automatic precision; set bf16/fp16 or dtype explicitly"
        )
    return selected[0] if selected else "fp32"


def _attention(values: Mapping[str, Any]) -> str:
    raw = _alias(
        values, "attn_implementation", "attention_implementation", "flash_attn"
    )
    selected = []
    if raw is not None:
        if not isinstance(raw, str):
            raise FinetuneConfigError("attention implementation must be a string")
        name = {
            "fa2": "flash_attention_2",
            "fa3": "flash_attention_3",
            "disabled": "eager",
        }.get(raw, raw)
        if name not in {
            "eager",
            "sdpa",
            "flash_attention_2",
            "flash_attention_3",
            "auto",
        }:
            raise FinetuneConfigError(f"unsupported attention implementation {raw!r}")
        selected.append(name)
    for key, name in (
        ("flash_attention", "flash_attention_2"),
        ("sdp_attention", "sdpa"),
    ):
        if values.get(key) is not None and _boolean(values[key], key):
            selected.append(name)
    if len(set(selected)) > 1:
        raise FinetuneConfigError("conflicting attention implementations")
    return selected[0] if selected else "auto"


def _quantization(
    values: Mapping[str, Any], quant: Mapping[str, Any], source: str
) -> int | None:
    bits = _optional_integer(
        _alias(values, "quantization_bits", "quantization_bit"), "quantization_bits"
    )
    flags = []
    for width in (4, 8):
        key = f"load_in_{width}bit"
        if bits == width and any(
            mapping.get(key) is False for mapping in (values, quant)
        ):
            raise FinetuneConfigError(
                f"{key}=false conflicts with quantization_bits={bits}"
            )
        fields = {"outer": values.get(key), "inner": quant.get(key)}
        enabled = _alias(fields, *fields, default=False)
        if _boolean(enabled, key):
            flags.append(width)
    if len(flags) > 1 or (flags and bits is not None and bits != flags[0]):
        raise FinetuneConfigError("conflicting quantization widths")
    bits = flags[0] if flags else bits
    if bits not in {None, 4, 8}:
        raise FinetuneConfigError("quantization_bits must be 4 or 8")
    if (
        bits is not None
        and source == "llamafactory"
        and values.get("quantization_method") not in (None, "bnb")
    ):
        raise FinetuneConfigError(
            "LLaMA-Factory requires quantization_method=bnb; other spellings "
            "do not activate its bitsandbytes loader"
        )
    fields = {
        "outer": values.get("quantization_method"),
        "inner": quant.get("quant_method"),
    }
    method = _alias(fields, *fields, default="bitsandbytes")
    if bits is not None and (
        not isinstance(method, str) or method not in {"bitsandbytes", "bnb"}
    ):
        raise FinetuneConfigError(f"unsupported quantization method {method!r}")
    if quant and bits is None:
        raise FinetuneConfigError(
            "quantization_config needs load_in_4bit or load_in_8bit"
        )
    return bits


def _quantization_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    result = _mapping_alias(values, "quantization_config")
    for key, value in _mapping_alias(values, "bnb_config_kwargs").items():
        if key in result and result[key] != value:
            raise FinetuneConfigError(
                f"conflicting quantization_config and bnb_config_kwargs.{key}"
            )
        result[key] = value
    return result


def _quantization_options(
    values: Mapping[str, Any],
    quant: Mapping[str, Any],
    source: str,
    bits: int | None,
    dtype: str,
) -> dict[str, Any]:
    allowed = {
        "load_in_4bit",
        "load_in_8bit",
        "quant_method",
        "bnb_4bit_quant_type",
        "bnb_4bit_compute_dtype",
        "bnb_4bit_use_double_quant",
        "bnb_4bit_quant_storage",
        "llm_int8_threshold",
        "llm_int8_skip_modules",
        "llm_int8_enable_fp32_cpu_offload",
        "llm_int8_has_fp16_weight",
    }
    unknown = set(quant) - allowed
    if unknown:
        raise FinetuneConfigError(
            "unsupported quantization settings: " + ", ".join(sorted(unknown))
        )

    def option(*keys: str, default: Any = None) -> Any:
        fields = {
            f"{prefix}{key}": mapping[key]
            for prefix, mapping in (("", values), ("quantization_config.", quant))
            for key in keys
            if mapping.get(key) is not None
        }
        return _alias(fields, *fields, default=default)

    for key in (
        "llm_int8_enable_fp32_cpu_offload",
        "llm_int8_has_fp16_weight",
        "llm_int8_skip_modules",
    ):
        if option(key):
            raise FinetuneConfigError(
                f"{key} is not supported by the fine-tuning estimator"
            )
    raw_type = option("quantization_type", "bnb_4bit_quant_type")
    raw_double = option("double_quantization", "bnb_4bit_use_double_quant")
    raw_compute = option("quantization_compute_dtype", "bnb_4bit_compute_dtype")
    raw_storage = option("quantization_storage", "bnb_4bit_quant_storage")
    if bits != 4:
        if any(
            value is not None
            for value in (raw_type, raw_double, raw_compute, raw_storage)
        ):
            raise FinetuneConfigError(
                "4-bit quantization options require 4-bit base weights"
            )
        return {
            "quantization_type": None,
            "double_quantization": None,
            "quantization_compute_dtype": None,
            "quantization_storage": None,
        }
    quant_type = (
        raw_type
        if raw_type is not None
        else ("fp4" if source == "huggingface" else "nf4")
    )
    if not isinstance(quant_type, str) or quant_type not in {"nf4", "fp4"}:
        raise FinetuneConfigError("quantization_type must be nf4 or fp4")
    double_quant = _boolean(
        raw_double if raw_double is not None else source != "huggingface",
        "double_quantization",
    )
    compute = _dtype(
        raw_compute
        if raw_compute is not None
        else ("fp32" if source == "huggingface" else dtype)
    )
    default_storage = {"llamafactory": dtype, "axolotl": "bf16"}.get(source, "uint8")
    storage = raw_storage if raw_storage is not None else default_storage
    if storage != "uint8":
        storage = _dtype(storage)
    return {
        "quantization_type": quant_type,
        "double_quantization": double_quant,
        "quantization_compute_dtype": compute,
        "quantization_storage": storage,
    }


def _adapter_options(
    values: Mapping[str, Any],
    peft: Mapping[str, Any],
    source: str,
    rank: int | None,
    assumptions: list[str],
) -> dict[str, Any]:
    if rank is None:
        return {"lora_alpha": None, "lora_dropout": 0.0, "lora_bias": "none"}
    if source == "axolotl" and all(
        mapping.get(key) is None
        for mapping, key in ((values, "lora_rank"), (values, "lora_r"), (peft, "r"))
    ):
        # Axolotl 0.18.0 has no lora_r default and passes LoraConfig(r=None),
        # which PEFT rejects when it builds the adapter layers.
        assumptions.append(
            f"Axolotl lora_r is unspecified; the estimate assumes r={rank}. "
            "Provide it before execution."
        )
    fields = {
        "lora_alpha": values.get("lora_alpha"),
        "peft.lora_alpha": peft.get("lora_alpha"),
    }
    alpha = _alias(
        fields,
        *fields,
        default=rank * 2
        if source == "llamafactory"
        else 8
        if source == "huggingface"
        else None,
    )
    if alpha is not None:
        alpha = _positive_float(alpha, "lora_alpha")
    else:
        assumptions.append(
            "Axolotl lora_alpha is unspecified; provide it before execution."
        )
    fields = {
        "lora_dropout": values.get("lora_dropout"),
        "peft.lora_dropout": peft.get("lora_dropout"),
    }
    dropout = _alias(fields, *fields, default=0.0)
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(dropout)
        or not 0 <= dropout < 1
    ):
        raise FinetuneConfigError("lora_dropout must be finite and in [0, 1)")
    fields = {"lora_bias": values.get("lora_bias"), "peft.bias": peft.get("bias")}
    bias = _alias(fields, *fields, default="none")
    if bias != "none":
        raise FinetuneConfigError("only lora_bias=none is supported")
    return {"lora_alpha": alpha, "lora_dropout": float(dropout), "lora_bias": bias}


def _adapter(
    values: Mapping[str, Any], peft: Mapping[str, Any], source: str, bits: int | None
) -> tuple[str, int | None, list[str]]:
    peft_requested = any(values.get(key) is not None for key in ("peft", "peft_config"))
    default = "lora" if source == "llamafactory" or peft_requested else "full"
    method = str(
        _alias(values, "method", "finetuning_type", "adapter", default=default)
    ).lower()
    if method not in {"full", "lora", "qlora"}:
        raise FinetuneConfigError(f"unsupported fine-tuning method {method!r}")
    if peft and str(peft.get("peft_type", "LORA")).upper() != "LORA":
        raise FinetuneConfigError("only LoRA PEFT adapters are supported")
    if peft.get("task_type") not in {None, "CAUSAL_LM"}:
        raise FinetuneConfigError("only CAUSAL_LM PEFT training is supported")
    if method == "full":
        if peft_requested or bits is not None:
            raise FinetuneConfigError(
                "full fine-tuning cannot use PEFT or frozen quantized base weights"
            )
        return method, None, []
    if method == "qlora" and bits != 4:
        raise FinetuneConfigError("QLoRA requires explicit 4-bit quantization")
    if bits == 4:
        method = "qlora"
    fields = {
        "lora_rank": values.get("lora_rank"),
        "lora_r": values.get("lora_r"),
        "peft.r": peft.get("r"),
    }
    rank = _integer(_alias(fields, *fields, default=8), "lora_rank")
    fields = {
        "lora_target_modules": values.get("lora_target_modules"),
        "lora_target": values.get("lora_target"),
        "peft.target_modules": peft.get("target_modules"),
    }
    target = _alias(
        fields, *fields, default="all" if source == "llamafactory" else None
    )
    if values.get("lora_target_linear") is not None and _boolean(
        values["lora_target_linear"], "lora_target_linear"
    ):
        if target is not None and target not in ("all", "all-linear"):
            raise FinetuneConfigError(
                "lora_target_linear conflicts with explicit LoRA targets"
            )
        target = "all-linear"
    if target is None:
        raise FinetuneConfigError(
            "provide explicit LoRA target modules or all-linear; model-dependent defaults are unresolved"
        )
    if isinstance(target, str):
        if target in {"all", "all-linear"}:
            targets = ["all-linear"]
        elif "lora_target" in values:
            targets = [item.strip() for item in target.split(",")]
        else:
            raise FinetuneConfigError(
                "PEFT target_modules strings are regular expressions; use an explicit list or all-linear"
            )
    elif isinstance(target, list):
        targets = target.copy()
    else:
        raise FinetuneConfigError("LoRA targets must be a non-empty list or all-linear")
    if not targets or any(
        not isinstance(item, str) or not item.strip() for item in targets
    ):
        raise FinetuneConfigError("LoRA targets must contain non-empty module names")
    if len(set(targets)) != len(targets):
        raise FinetuneConfigError("LoRA target modules must be unique")
    return method, rank, targets


def _checkpoint_options(
    values: Mapping[str, Any], source: str, enabled: bool
) -> dict[str, Any]:
    options = values.get("gradient_checkpointing_kwargs")
    alias = values.get("use_reentrant_gc")
    if options is not None:
        if not isinstance(options, Mapping) or set(options) - {"use_reentrant"}:
            raise FinetuneConfigError(
                "gradient_checkpointing_kwargs supports only use_reentrant"
            )
        if source == "llamafactory" and options:
            raise FinetuneConfigError(
                "LLaMA-Factory controls checkpointing with use_reentrant_gc; "
                "set that option instead of gradient_checkpointing_kwargs"
            )
    if alias is not None and source != "llamafactory":
        raise FinetuneConfigError("use_reentrant_gc is a LLaMA-Factory option")
    value = alias if alias is not None else (options or {}).get("use_reentrant")
    if value is None:
        if options and "use_reentrant" in options:
            raise FinetuneConfigError("use_reentrant must be a boolean")
        return {}
    value = _boolean(value, "use_reentrant")
    return (
        {"gradient_checkpointing_kwargs": {"use_reentrant": value}} if enabled else {}
    )


def _check_strategies(values: Mapping[str, Any], peft: Mapping[str, Any]) -> None:
    if values.get("stage", "sft") not in {"sft", "pt"}:
        raise FinetuneConfigError("only sft and pt training stages are supported")
    if values.get("device_map") is not None:
        raise FinetuneConfigError(
            "device_map is not supported; this estimator requires one GPU without model-level placement or offload"
        )
    for key in (
        "rl",
        "trl",
        "reward_model",
        "use_dora",
        "use_galore",
        "use_apollo",
        "use_badam",
        "use_adam_mini",
        "use_llama_pro",
        "train_mm_proj_only",
        "fp8",
        "fp8_config",
        "use_unsloth",
        "use_unsloth_gc",
        "loraplus_lr_ratio",
        "relora_steps",
        "modules_to_save",
        "additional_target",
        "layer_replication",
        "plugins",
        "adapter_name_or_path",
        "lora_model_dir",
        "resume_from_checkpoint",
        "lora_target_parameters",
        "lora_modules_to_save",
        "lora_rank_pattern",
        "lora_alpha_pattern",
        "peft_layers_to_transform",
        "peft_layer_replication",
        "peft_use_dora",
        "xformers_attention",
        "relora",
        "gptq",
        "use_rslora",
        "peft_use_rslora",
        "lora_fan_in_fan_out",
        "lora_exclude_modules",
        "peft_trainable_token_indices",
        "torch_compile",
        "enable_liger_kernel",
        "use_liger_kernel",
        "model_quantization_config",
        "quantize_moe_experts",
    ):
        if values.get(key):
            raise FinetuneConfigError(
                f"{key} is not supported by the fine-tuning estimator"
            )
    for key in (
        "use_dora",
        "rank_pattern",
        "alpha_pattern",
        "layers_to_transform",
        "layer_replication",
        "modules_to_save",
        "trainable_token_indices",
        "target_parameters",
        "exclude_modules",
        "lora_bias",
        "use_qalora",
        "alora_invocation_tokens",
        "inference_mode",
        "use_rslora",
        "fan_in_fan_out",
        "loftq_config",
    ):
        if peft.get(key):
            raise FinetuneConfigError(
                f"PEFT {key} is not supported by the fine-tuning estimator"
            )
    if peft.get("bias", "none") != "none":
        raise FinetuneConfigError("PEFT bias training is not supported")
    for key in (
        "tensor_parallel_size",
        "tensor_parallel_degree",
        "pipeline_parallel_size",
        "context_parallel_size",
        "sequence_parallel_degree",
    ):
        if values.get(key) is not None and _integer(values[key], key) != 1:
            raise FinetuneConfigError(f"{key} > 1 is not supported by this adapter")


def _local_path(value: str, root: Path, *, force: bool = False) -> str:
    candidate = Path(value).expanduser()
    if (
        force
        or candidate.is_absolute()
        or value.startswith((".", "~"))
        or (root / candidate).exists()
    ):
        return str((root / candidate).resolve())
    return value


def _dataset_sources(values: Mapping[str, Any], root: Path, source: str) -> list[str]:
    raw = values.get("datasets", values.get("dataset", values.get("train_file", [])))
    entries = raw if isinstance(raw, list) else [raw]
    result = []
    for item in entries:
        if isinstance(item, Mapping):
            item = item.get("path", item.get("name"))
        if item is None:
            continue
        if not isinstance(item, str):
            raise FinetuneConfigError(
                "dataset sources must be names, paths or mappings with path/name"
            )
        if source == "llamafactory":
            result.extend(name.strip() for name in item.split(",") if name.strip())
        else:
            result.append(
                _local_path(
                    item,
                    root,
                    force=Path(item).suffix
                    in {".json", ".jsonl", ".csv", ".parquet", ".txt"},
                )
            )
    return result


def _config_mapping(value: Any, root: Path, name: str) -> dict[str, Any]:
    if isinstance(value, str):
        return load_mapping(_local_path(value, root, force=True))
    if not isinstance(value, Mapping):
        raise FinetuneConfigError(
            f"{name} must be a mapping or a configuration file path"
        )
    return dict(value)


def _distributed(
    values: Mapping[str, Any],
    root: Path,
    world: int,
    accumulation: int,
    micro: int,
    dtype: str,
    checkpoint: bool,
    assumptions: list[str],
) -> dict[str, Any]:
    ds = values.get("deepspeed")
    fsdp = values.get("fsdp")
    has_fsdp = bool(fsdp) or bool(values.get("fsdp_config"))
    if ds and has_fsdp:
        raise FinetuneConfigError("DeepSpeed and FSDP cannot both be enabled")
    common = {
        "world_size": world,
        "gradient_accumulation_steps": accumulation,
        "mixed_precision": "no" if dtype == "fp32" else dtype,
        "activation_checkpointing": checkpoint,
    }
    if not ds and not has_fsdp:
        return {
            "framework": "ddp" if world > 1 else "single",
            "zero_stage": 0,
            "sharding_strategy": "replicated",
            "parameter_offload_device": "none",
            "optimizer_offload_device": "none",
            **common,
        }
    if ds:
        config = _deepspeed_settings(
            ds, root, world, accumulation, micro, dtype, assumptions
        )
        framework = "deepspeed"
    else:
        config, framework = _fsdp_settings(
            values, fsdp, root, world, accumulation, dtype
        )
    try:
        normalized = normalize_training_config(config, framework=framework)
    except TrainingPlanError as exc:
        raise FinetuneConfigError(str(exc)) from exc
    common["activation_checkpointing"] = (
        checkpoint or normalized["activation_checkpointing"]
    )
    normalized.update(common)
    return normalized


def _deepspeed_settings(
    ds: Any,
    root: Path,
    world: int,
    accumulation: int,
    micro: int,
    dtype: str,
    assumptions: list[str],
) -> dict[str, Any]:
    config = _config_mapping(ds, root, "deepspeed")
    _check_finite(config, "deepspeed")
    for key, expected in (
        ("gradient_accumulation_steps", accumulation),
        ("train_micro_batch_size_per_gpu", micro),
        ("train_batch_size", world * micro * accumulation),
        ("world_size", world),
    ):
        value = config.get(key, "auto")
        if value != "auto" and _integer(value, f"deepspeed.{key}") != expected:
            raise FinetuneConfigError(
                f"deepspeed.{key} conflicts with trainer settings"
            )
        config[key] = expected
    zero = config.get("zero_optimization", {})
    if not isinstance(zero, Mapping):
        raise FinetuneConfigError("zero_optimization must be a mapping")
    zero = dict(zero)
    stage = _integer(zero.get("stage", 0), "zero_optimization.stage", minimum=0)
    if stage > 3:
        raise FinetuneConfigError("DeepSpeed stage must be 0, 1, 2 or 3")
    for key in ("offload_param", "offload_optimizer"):
        if zero.get(key) is not None:
            offload = zero[key]
            if not isinstance(offload, Mapping) or offload.get(
                "device", "none"
            ) not in {"none", "cpu", "nvme"}:
                raise FinetuneConfigError(f"unsupported DeepSpeed {key} device")
    for key in (
        "reduce_bucket_size",
        "allgather_bucket_size",
        "stage3_prefetch_bucket_size",
    ):
        value = zero.get(key, 0)
        if value == "auto":
            assumptions.append(
                f"DeepSpeed {key}=auto is model-dependent; its bucket memory is not resolved."
            )
        else:
            _integer(value, f"deepspeed.{key}", minimum=0)
    config["zero_optimization"] = zero
    for key in ("bf16", "fp16"):
        precision = config.get(key, {})
        if not isinstance(precision, Mapping):
            raise FinetuneConfigError(f"deepspeed.{key} must be a mapping")
        enabled = precision.get("enabled", "auto")
        if enabled != "auto" and _boolean(enabled, f"deepspeed.{key}.enabled") != (
            dtype == key
        ):
            raise FinetuneConfigError(
                f"deepspeed.{key} conflicts with trainer precision"
            )
        config[key] = {**precision, "enabled": dtype == key}
    return config


def _fsdp_settings(
    values: Mapping[str, Any],
    fsdp: Any,
    root: Path,
    world: int,
    accumulation: int,
    dtype: str,
) -> tuple[dict[str, Any], str]:
    config = _config_mapping(values.get("fsdp_config", {}), root, "fsdp_config")
    _check_finite(config, "fsdp_config")
    if (
        "world_size" in config
        and _integer(config["world_size"], "fsdp_config.world_size") != world
    ):
        raise FinetuneConfigError(
            "fsdp_config.world_size conflicts with trainer settings"
        )
    if fsdp is True:
        tokens = ["full_shard"]
    elif isinstance(fsdp, str):
        tokens = fsdp.lower().split()
    elif isinstance(fsdp, list) and all(isinstance(token, str) for token in fsdp):
        tokens = [token.lower() for token in fsdp]
    elif fsdp in (None, False):
        tokens = []
    else:
        raise FinetuneConfigError("fsdp must be a strategy string, list or boolean")
    strategies = {"full_shard", "shard_grad_op", "no_shard", "hybrid_shard"}
    if set(tokens) - strategies - {"offload", "auto_wrap"}:
        raise FinetuneConfigError("unsupported FSDP options")
    selected = set(tokens) & strategies
    if len(selected) > 1:
        raise FinetuneConfigError("multiple FSDP sharding strategies configured")
    if selected:
        strategy = selected.pop()
        declared = _alias(config, "sharding_strategy", "fsdp_sharding_strategy")
        if declared is not None and str(declared).lower() != strategy:
            raise FinetuneConfigError("FSDP sharding strategies conflict")
        config["sharding_strategy"] = strategy
    if "offload" in tokens:
        config["fsdp_offload_params"] = True
    for key in (
        "offload_params",
        "activation_checkpointing",
        "reshard_after_forward",
    ):
        if config.get(key) is not None:
            _boolean(config[key], f"fsdp_config.{key}")
    if "offload_params" in config:
        if "offload" in tokens and not config["offload_params"]:
            raise FinetuneConfigError("FSDP offload settings conflict")
        config["fsdp_offload_params"] = config["offload_params"]
    precision = _alias(
        config,
        "mixed_precision",
        "fsdp_mixed_precision",
        "mixed_precision_policy",
    )
    if (
        precision is not None
        and ("fp32" if precision == "no" else _dtype(precision)) != dtype
    ):
        raise FinetuneConfigError(
            "FSDP mixed precision conflicts with trainer settings"
        )
    fields = {
        "outer": values.get("fsdp_version"),
        "inner": config.get("fsdp_version"),
    }
    version = _integer(
        _alias(fields, *fields, default=2 if values.get("base_model") else 1),
        "fsdp_version",
    )
    if version not in {1, 2}:
        raise FinetuneConfigError("fsdp_version must be 1 or 2")
    if version == 2 and config.get("reshard_after_forward") is False:
        config["sharding_strategy"] = "shard_grad_op"
    config.update({"world_size": world, "gradient_accumulation_steps": accumulation})
    return config, "fsdp2" if version == 2 else "fsdp"


__all__ = [
    "FinetuneConfigError",
    "SUPPORTED_FRAMEWORKS",
    "load_finetune_config",
    "normalize_finetune_config",
]
