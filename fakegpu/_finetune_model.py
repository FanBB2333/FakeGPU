"""Inspect effective Trainer and model settings without trusting input defaults.

Torch and bitsandbytes remain optional: only the runtime inspection entry point
imports Torch, and packed parameter inspection uses the loaded module classes.
"""

from __future__ import annotations

from collections import Counter
from functools import partial
import hashlib
import inspect
import json
import math


_DTYPES = {"torch.float32": "fp32", "torch.float16": "fp16", "torch.bfloat16": "bf16"}
_STORAGE_DTYPES = {**_DTYPES, "torch.uint8": "uint8"}


def _bnb_type(value, name):
    """Recognize loaded bnb classes, including subclasses, without importing bnb."""
    return any(
        cls.__name__ == name and cls.__module__.startswith("bitsandbytes.")
        for cls in type(value).__mro__
    )


def _dtype(value, *, packed=False):
    name = str(value)
    choices = _STORAGE_DTYPES if packed else _DTYPES
    if name not in choices:
        raise ValueError(f"actual parameter dtype {name!r} is unsupported")
    return choices[name]


def _adapter_settings(model, *, quantized):
    adapters = getattr(model, "peft_config", {})
    if not adapters:
        if quantized:
            raise ValueError(
                "actual quantized model must have a trainable LoRA adapter"
            )
        return {"method": "full"}
    if len(adapters) != 1:
        raise ValueError("actual LoRA model must contain exactly one adapter")
    adapter = next(iter(adapters.values()))
    peft_type = getattr(adapter.peft_type, "value", adapter.peft_type)
    if peft_type != "LORA":
        raise ValueError(f"actual adapter type {peft_type!r} is unsupported")
    for option in (
        "use_dora",
        "use_rslora",
        "rank_pattern",
        "alpha_pattern",
        "modules_to_save",
        "target_parameters",
        "lora_bias",
        "inference_mode",
    ):
        if getattr(adapter, option, False):
            raise ValueError(f"actual adapter option {option} is unsupported")
    if adapter.bias != "none":
        raise ValueError("actual LoRA adapter must use bias=none")
    targets = adapter.target_modules
    if not targets:
        raise ValueError("actual LoRA adapter has no target modules")
    return {
        "method": "qlora" if quantized else "lora",
        "lora_rank": adapter.r,
        "lora_alpha": adapter.lora_alpha,
        "lora_dropout": adapter.lora_dropout,
        "lora_bias": adapter.bias,
        "lora_target_modules": [targets]
        if isinstance(targets, str)
        else sorted(targets),
    }


def _validate_adapter_parameters(parameters):
    for name, parameter in parameters:
        is_adapter = bool({"lora_A", "lora_B"} & set(name.split(".")))
        if parameter.requires_grad != is_adapter:
            raise ValueError(
                "actual LoRA training requires frozen base weights and trainable lora_A/lora_B tensors"
            )


def _quantization_metadata_bytes(parameters):
    """Count retained CUDA storages once, including shared quantization maps."""
    storages = {}
    states = [p.quant_state for _, p in parameters if _bnb_type(p, "Params4bit")]
    seen = set()
    while states:
        state = states.pop()
        if id(state) in seen:
            continue
        seen.add(id(state))
        names = ("absmax", "code", "offset") if state.nested else ("absmax", "code")
        for name in names:
            tensor = getattr(state, name, None)
            if tensor is None or tensor.device.type != "cuda":
                raise ValueError("actual quantization metadata must reside on CUDA")
            storage = tensor.untyped_storage()
            # Storage addresses are runtime-only deduplication keys, never signatures.
            key = (str(tensor.device), storage.data_ptr())
            storages[key] = storage.nbytes()
        if state.nested:
            nested_state = getattr(state, "state2", None)
            if nested_state is None:
                raise ValueError("actual nested quantization state is missing")
            states.append(nested_state)
    return sum(storages.values())


def _quantized_parameters(model, parameters):
    """Describe logical 4-bit matrices separately from their packed tensors."""
    packed = {id(p): (name, p) for name, p in parameters if _bnb_type(p, "Params4bit")}
    if not packed:
        if getattr(model, "is_quantized", False) or getattr(
            model, "is_loaded_in_4bit", False
        ):
            raise ValueError(
                "actual quantized model has no supported bitsandbytes 4-bit weights"
            )
        return {}
    if getattr(model, "is_loaded_in_8bit", False):
        raise ValueError("actual 8-bit model is unsupported")
    quantizer = getattr(model, "hf_quantizer", None)
    if quantizer is not None and getattr(quantizer, "pre_quantized", False):
        raise ValueError(
            "actual quantized model must originate from unquantized weights"
        )
    method = getattr(model, "quantization_method", "bitsandbytes")
    if getattr(method, "value", method) != "bitsandbytes":
        raise ValueError("actual quantization backend must be bitsandbytes")

    result = {}
    for module in model.modules():
        if not _bnb_type(module, "Linear4bit"):
            continue
        parameter = module.weight
        if id(parameter) not in packed:
            raise ValueError(
                "actual bitsandbytes linear weight is not a Params4bit tensor"
            )
        name, parameter = packed[id(parameter)]
        state = getattr(parameter, "quant_state", None)
        if state is None or not getattr(parameter, "bnb_quantized", False):
            raise ValueError("actual 4-bit weight has not been quantized on CUDA")
        if parameter.requires_grad:
            raise ValueError("actual packed base weights must be frozen")
        shape = tuple(state.shape)
        if (
            len(shape) != 2
            or any(
                not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in shape
            )
            or shape != (module.out_features, module.in_features)
        ):
            raise ValueError(
                "actual 4-bit logical shape disagrees with its linear module"
            )
        quant_type = state.quant_type
        if quant_type not in {"nf4", "fp4"} or parameter.quant_type != quant_type:
            raise ValueError(
                "actual 4-bit quantization type must consistently be nf4 or fp4"
            )
        nested = state.nested
        if (
            not isinstance(nested, bool)
            or bool(parameter.compress_statistics) != nested
        ):
            raise ValueError(
                "actual double-quantization flags disagree with quantization state"
            )
        storage = _dtype(parameter.dtype, packed=True)
        if str(parameter.quant_storage) != str(parameter.dtype):
            raise ValueError(
                "actual packed storage dtype disagrees with Params4bit metadata"
            )
        logical_count = math.prod(shape)
        packed_bytes = parameter.numel() * parameter.element_size()
        required_bytes = math.ceil(logical_count / 2)
        # Floating storage containers can have a final partial element.
        if (
            not required_bytes
            <= packed_bytes
            < required_bytes + parameter.element_size()
        ):
            raise ValueError(
                "actual 4-bit storage size disagrees with its logical shape"
            )
        if state.blocksize != parameter.blocksize or state.blocksize not in {
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
        }:
            raise ValueError(
                "actual quantization block size is unsupported or inconsistent"
            )
        result[name] = {
            "shape": list(shape),
            "logical_parameter_count": logical_count,
            "packed_element_count": parameter.numel(),
            "packed_bytes": packed_bytes,
            "quantization_type": quant_type,
            "double_quantization": nested,
            "block_size": state.blocksize,
            "compute_dtype": _dtype(module.compute_dtype),
            "storage_dtype": storage,
            "original_dtype": _dtype(state.dtype),
        }
    if set(result) != {name for name, _ in packed.values()}:
        raise ValueError(
            "actual 4-bit parameters must belong to supported Linear4bit modules"
        )
    return result


def _checkpointing_kwargs(model):
    choices = set()
    for module in model.modules():
        if not getattr(module, "gradient_checkpointing", False):
            continue
        function = getattr(module, "_gradient_checkpointing_func", None)
        if function is None:
            continue
        # LLaMA-Factory wraps the HF partial with functools.wraps.
        function = inspect.unwrap(function)
        if not isinstance(function, partial):
            raise ValueError(
                "actual gradient checkpointing function has no inspectable kwargs"
            )
        kwargs = function.keywords
        if set(kwargs) - {"use_reentrant"}:
            raise ValueError(
                "actual gradient checkpointing only supports use_reentrant"
            )
        use_reentrant = kwargs.get("use_reentrant", True)
        if not isinstance(use_reentrant, bool):
            raise ValueError(
                "actual gradient checkpointing use_reentrant must be boolean"
            )
        choices.add(use_reentrant)
    if len(choices) != 1:
        raise ValueError(
            "actual gradient checkpointing kwargs must be inspectable and consistent"
        )
    return {"use_reentrant": choices.pop()}


def _parameter_inventory(model, parameters):
    quantization = _quantized_parameters(model, parameters)
    dtype_counts = Counter()
    trainable_dtypes = Counter()
    frozen_dtypes = Counter()
    total = 0
    for name, parameter in parameters:
        count = parameter.numel()
        dtype_counts[str(parameter.dtype)] += count
        if name in quantization:
            total += quantization[name]["logical_parameter_count"]
        else:
            _dtype(parameter.dtype)
            total += count
            (trainable_dtypes if parameter.requires_grad else frozen_dtypes)[
                str(parameter.dtype)
            ] += count
    if not total or not trainable_dtypes:
        raise ValueError("actual model must contain trainable parameters")
    base_dtypes = frozen_dtypes or trainable_dtypes
    storage_dtype = max(base_dtypes, key=base_dtypes.get)
    return (
        total,
        storage_dtype,
        dtype_counts,
        trainable_dtypes,
        frozen_dtypes,
        quantization,
    )


def actual_training_config(trainer, normalized):
    import torch

    args, model = trainer.args, trainer.model
    if (
        args.world_size != 1
        or getattr(trainer, "is_deepspeed_enabled", False)
        or getattr(trainer, "is_fsdp_enabled", False)
    ):
        raise ValueError("actual Trainer enabled unsupported distributed training")
    parameters = list(model.named_parameters())
    if any(parameter.device.type != "cuda" for _, parameter in parameters):
        raise ValueError(
            "actual model parameters must all reside on the selected CUDA GPU"
        )
    (
        total,
        storage_dtype,
        dtype_counts,
        trainable_dtypes,
        frozen_dtypes,
        quantization,
    ) = _parameter_inventory(model, parameters)
    adapter_settings = _adapter_settings(model, quantized=bool(quantization))
    if adapter_settings["method"] != normalized["method"]:
        raise ValueError(
            "actual fine-tuning method disagrees with the requested configuration"
        )
    if adapter_settings["method"] in {"lora", "qlora"}:
        _validate_adapter_parameters(parameters)
    result = dict(normalized)
    result.update(adapter_settings)
    result.update(
        {
            "parameter_dtype": _DTYPES[storage_dtype],
            "dtype": "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32",
            "micro_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "epochs": args.num_train_epochs,
            "max_steps": args.max_steps if args.max_steps > 0 else None,
            "world_size": args.world_size,
            "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "gradient_checkpointing": bool(
                getattr(model, "is_gradient_checkpointing", False)
            ),
            "optimizer": getattr(args.optim, "value", args.optim),
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": getattr(
                args.lr_scheduler_type, "value", args.lr_scheduler_type
            ),
            "warmup_steps": args.warmup_steps,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "dataloader_drop_last": args.dataloader_drop_last,
            "trainable_parameter_count": sum(trainable_dtypes.values()),
        }
    )
    result["parameter_count"] = (
        total - result["trainable_parameter_count"]
        if result["method"] in {"lora", "qlora"}
        else total
    )
    result["distributed_config"] = dict(
        result["distributed_config"],
        activation_checkpointing=result["gradient_checkpointing"],
    )
    if result["gradient_checkpointing"]:
        result["gradient_checkpointing_kwargs"] = _checkpointing_kwargs(model)
    else:
        result.pop("gradient_checkpointing_kwargs", None)
    if quantization:
        for key, field in (
            ("quantization_type", "quantization_type"),
            ("double_quantization", "double_quantization"),
            ("quantization_compute_dtype", "compute_dtype"),
            ("quantization_storage", "storage_dtype"),
        ):
            choices = {entry[field] for entry in quantization.values()}
            if len(choices) != 1:
                raise ValueError(f"actual quantized modules have inconsistent {field}")
            result[key] = choices.pop()
        result["quantization_bits"] = 4
        result["actual_quantized_storage"] = {
            "quantized_parameters": sum(
                entry["logical_parameter_count"] for entry in quantization.values()
            ),
            "packed_weight_bytes": sum(
                entry["packed_bytes"] for entry in quantization.values()
            ),
            "metadata_bytes": _quantization_metadata_bytes(parameters),
            "unquantized_weight_bytes": sum(
                parameter.numel() * parameter.element_size()
                for name, parameter in parameters
                if name not in quantization and not parameter.requires_grad
            ),
        }
        result["actual_model_signature"] = hashlib.sha256(
            json.dumps(
                {
                    "quantized_modules": quantization,
                    "frozen_dtype_counts": dict(frozen_dtypes),
                    "trainable_dtype_counts": dict(trainable_dtypes),
                    "storage": result["actual_quantized_storage"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    attention = getattr(model.config, "_attn_implementation", None)
    if attention not in {"eager", "sdpa"}:
        raise ValueError(f"actual attention backend {attention!r} is unsupported")
    result["attention_implementation"] = attention
    result["execution_backend"] = "native_framework_trainer"
    result["assumptions"] = [
        "Configuration values are reconciled with the actual native Trainer and model."
    ]
    kernel_choices = {}
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
    ):
        choices = set()
        for module in model.modules():
            if hasattr(module, name):
                implementation = getattr(module, name)
                choices.add(
                    None
                    if implementation is None
                    else (
                        getattr(
                            implementation,
                            "__module__",
                            type(implementation).__module__,
                        )
                        + "."
                        + getattr(
                            implementation,
                            "__qualname__",
                            type(implementation).__qualname__,
                        )
                    )
                )
        if choices:
            kernel_choices[name] = sorted(choices, key=str)
    native_backend = {
        "kernel_implementations": kernel_choices,
        "collator_class": type(trainer.data_collator).__module__
        + "."
        + type(trainer.data_collator).__name__,
        "tf32_matmul": result["tf32"],
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    result["native_execution_signature"] = hashlib.sha256(
        json.dumps(
            {
                "native_config_signature": normalized["native_execution_signature"],
                "backend": native_backend,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    details = {
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "trainer_class": type(trainer).__module__ + "." + type(trainer).__name__,
        "collator_class": type(trainer.data_collator).__module__
        + "."
        + type(trainer.data_collator).__name__,
        "parameter_count": result["parameter_count"],
        "total_parameter_count": total,
        "trainable_parameter_count": result["trainable_parameter_count"],
        "parameter_dtype_counts": dict(dtype_counts),
        "trainable_dtype_counts": dict(trainable_dtypes),
        "gradient_checkpointing": result["gradient_checkpointing"],
        "attention_implementation": attention,
        "training_arguments": args.to_dict(),
        **native_backend,
    }
    if result["gradient_checkpointing"]:
        details["gradient_checkpointing_kwargs"] = result[
            "gradient_checkpointing_kwargs"
        ]
    if quantization:
        details["quantization"] = {
            "backend": "bitsandbytes",
            "bits": 4,
            "modules": quantization,
            "logical_parameter_count": sum(
                q["logical_parameter_count"] for q in quantization.values()
            ),
            "packed_bytes": sum(q["packed_bytes"] for q in quantization.values()),
            "frozen_unquantized_dtype_counts": dict(frozen_dtypes),
            "stored_parameter_elements": sum(dtype_counts.values()),
        }
        details["actual_model_signature"] = result["actual_model_signature"]
        details["actual_quantized_storage"] = result["actual_quantized_storage"]
    return result, details
