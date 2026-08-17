from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .structured_io import load_mapping, write_json


SCHEMA_VERSION = "fakegpu.training_plan.v1"
SUPPORTED_FRAMEWORKS = {"auto", "deepspeed", "accelerate", "fsdp", "fsdp2"}


class TrainingPlanError(ValueError):
    pass


def normalize_training_config(
    config: Mapping[str, Any],
    *,
    framework: str = "auto",
) -> dict[str, Any]:
    """Normalize DeepSpeed, Accelerate, FSDP, and FSDP2 settings."""

    requested = framework.strip().lower()
    if requested not in SUPPORTED_FRAMEWORKS:
        raise TrainingPlanError(
            "framework must be one of: "
            + ", ".join(sorted(SUPPORTED_FRAMEWORKS))
        )
    detected = _detect_framework(config) if requested == "auto" else requested
    if detected == "deepspeed":
        normalized = _normalize_deepspeed(config)
    elif detected == "accelerate":
        normalized = _normalize_accelerate(config)
    elif detected in {"fsdp", "fsdp2"}:
        normalized = _normalize_fsdp(config, framework=detected)
    else:
        raise TrainingPlanError(
            "could not detect DeepSpeed, Accelerate, FSDP, or FSDP2 settings"
        )
    normalized["source_framework"] = detected
    return normalized


def estimate_training_plan(
    config: Mapping[str, Any],
    *,
    parameter_bytes: int,
    activation_bytes: int,
    world_size: int | None = None,
    optimizer: str = "adamw",
    parameter_element_bytes: int = 2,
) -> dict[str, Any]:
    """Estimate rank-local training memory from a normalized configuration.

    The estimate separates persistent tensor storage, communication workspaces,
    phase peaks, and CPU-offloaded bytes. It intentionally exposes every
    formula so real measurements can replace individual terms.
    """

    for name, value in (
        ("parameter_bytes", parameter_bytes),
        ("activation_bytes", activation_bytes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise TrainingPlanError(f"{name} must be a non-negative integer")
    if (
        not isinstance(parameter_element_bytes, int)
        or isinstance(parameter_element_bytes, bool)
        or parameter_element_bytes <= 0
    ):
        raise TrainingPlanError(
            "parameter_element_bytes must be a positive integer"
        )
    if optimizer not in {"adam", "adamw", "sgd", "sgd_momentum"}:
        raise TrainingPlanError(
            "optimizer must be adam, adamw, sgd, or sgd_momentum"
        )

    normalized = dict(config)
    configured_world_size = int(normalized.get("world_size", 1) or 1)
    ranks = int(world_size if world_size is not None else configured_world_size)
    if ranks <= 0:
        raise TrainingPlanError("world_size must be greater than zero")

    strategy = str(normalized.get("sharding_strategy") or "replicated")
    zero_stage = int(normalized.get("zero_stage", 0) or 0)
    full_shard = strategy in {"full_shard", "hybrid_shard"} or zero_stage >= 3
    shard_gradients = (
        full_shard or strategy == "shard_grad_op" or zero_stage >= 2
    )
    shard_optimizer = (
        full_shard or strategy == "shard_grad_op" or zero_stage >= 1
    )

    rank_parameter_storage_bytes = (
        _ceil_div(parameter_bytes, ranks) if full_shard else parameter_bytes
    )
    gradient_bytes = parameter_bytes
    gpu_gradient_bytes = (
        _ceil_div(gradient_bytes, ranks) if shard_gradients else gradient_bytes
    )
    if optimizer in {"adam", "adamw"}:
        optimizer_state_bytes = parameter_bytes * 2
    elif optimizer == "sgd_momentum":
        optimizer_state_bytes = parameter_bytes
    else:
        optimizer_state_bytes = 0
    rank_optimizer_state_bytes = (
        _ceil_div(optimizer_state_bytes, ranks)
        if shard_optimizer
        else optimizer_state_bytes
    )

    mixed_precision = str(normalized.get("mixed_precision") or "no")
    master_parameter_bytes = (
        math.ceil(parameter_bytes * 4 / parameter_element_bytes)
        if mixed_precision in {"fp16", "bf16", "fp8"}
        and optimizer in {"adam", "adamw", "sgd_momentum"}
        else 0
    )
    rank_master_parameter_bytes = (
        _ceil_div(master_parameter_bytes, ranks)
        if shard_optimizer
        else master_parameter_bytes
    )
    checkpointing = bool(normalized.get("activation_checkpointing", False))
    checkpoint_ratio = float(
        normalized.get(
            "activation_retention_factor",
            0.35 if checkpointing else 1.0,
        )
    )
    if not 0 < checkpoint_ratio <= 1:
        raise TrainingPlanError(
            "activation_retention_factor must be in (0, 1]"
        )
    retained_activation_bytes = math.ceil(
        activation_bytes * checkpoint_ratio
    )

    parameter_offload_device = str(
        normalized.get("parameter_offload_device") or "none"
    )
    optimizer_offload_device = str(
        normalized.get("optimizer_offload_device") or "none"
    )
    offload_parameters = parameter_offload_device in {"cpu", "nvme"}
    offload_optimizer = optimizer_offload_device in {"cpu", "nvme"}
    gpu_parameter_bytes = (
        0 if offload_parameters else rank_parameter_storage_bytes
    )
    cpu_offloaded_parameter_bytes = (
        rank_parameter_storage_bytes
        if parameter_offload_device == "cpu"
        else 0
    )
    cpu_offloaded_optimizer_bytes = (
        rank_optimizer_state_bytes + rank_master_parameter_bytes
        if optimizer_offload_device == "cpu"
        else 0
    )
    nvme_offloaded_parameter_bytes = (
        rank_parameter_storage_bytes
        if parameter_offload_device == "nvme"
        else 0
    )
    nvme_offloaded_optimizer_bytes = (
        rank_optimizer_state_bytes + rank_master_parameter_bytes
        if optimizer_offload_device == "nvme"
        else 0
    )
    gpu_optimizer_state_bytes = (
        0 if offload_optimizer else rank_optimizer_state_bytes
    )
    gpu_master_parameter_bytes = (
        0 if offload_optimizer else rank_master_parameter_bytes
    )

    reduce_bucket = int(normalized.get("reduce_bucket_bytes", 0) or 0)
    allgather_bucket = int(normalized.get("allgather_bucket_bytes", 0) or 0)
    if reduce_bucket < 0 or allgather_bucket < 0:
        raise TrainingPlanError("communication bucket sizes must be non-negative")
    default_bucket = min(
        parameter_bytes,
        500_000_000,
    )
    if shard_gradients and reduce_bucket == 0:
        reduce_bucket = default_bucket
    if (full_shard or zero_stage >= 3) and allgather_bucket == 0:
        allgather_bucket = default_bucket
    communication_workspace_bytes = max(reduce_bucket, allgather_bucket)

    gathered_parameter_window = (
        min(
            parameter_bytes,
            max(allgather_bucket, rank_parameter_storage_bytes),
        )
        if full_shard
        else rank_parameter_storage_bytes
        if offload_parameters
        else 0
    )
    forward_peak = (
        gpu_parameter_bytes
        + retained_activation_bytes
        + gathered_parameter_window
        + communication_workspace_bytes
    )
    backward_peak = (
        gpu_parameter_bytes
        + retained_activation_bytes
        + gpu_gradient_bytes
        + gathered_parameter_window
        + communication_workspace_bytes
    )
    optimizer_temporary_bytes = (
        gpu_gradient_bytes
        if optimizer in {"adam", "adamw"} and not offload_optimizer
        else 0
    )
    optimizer_peak = (
        gpu_parameter_bytes
        + gpu_gradient_bytes
        + gpu_optimizer_state_bytes
        + gpu_master_parameter_bytes
        + optimizer_temporary_bytes
        + (reduce_bucket if shard_gradients else 0)
    )
    model_load_peak = gpu_parameter_bytes + (
        gathered_parameter_window if offload_parameters else 0
    )
    peak = max(model_load_peak, forward_peak, backward_peak, optimizer_peak)
    peak_phase = max(
        (
            ("model_load", model_load_peak),
            ("forward", forward_peak),
            ("backward", backward_peak),
            ("optimizer_step", optimizer_peak),
        ),
        key=lambda item: item[1],
    )[0]

    timeline = [
        _phase(
            "model_load",
            model_load_peak,
            {
                "parameters": gpu_parameter_bytes,
                "parameter_staging": (
                    gathered_parameter_window if offload_parameters else 0
                ),
            },
        ),
        _phase(
            "forward",
            forward_peak,
            {
                "parameters": gpu_parameter_bytes,
                "activations": retained_activation_bytes,
                "allgathered_parameters": gathered_parameter_window,
                "communication_workspace": communication_workspace_bytes,
            },
        ),
        _phase(
            "backward",
            backward_peak,
            {
                "parameters": gpu_parameter_bytes,
                "activations": retained_activation_bytes,
                "gradients": gpu_gradient_bytes,
                "allgathered_parameters": gathered_parameter_window,
                "communication_workspace": communication_workspace_bytes,
            },
        ),
        _phase(
            "optimizer_step",
            optimizer_peak,
            {
                "parameters": gpu_parameter_bytes,
                "gradients": gpu_gradient_bytes,
                "optimizer_state": gpu_optimizer_state_bytes,
                "master_parameters": gpu_master_parameter_bytes,
                "optimizer_temporary": optimizer_temporary_bytes,
                "communication_workspace": (
                    reduce_bucket if shard_gradients else 0
                ),
            },
        ),
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "method": "configuration_aware_rank_local_storage_and_phase_model",
        "configuration": normalized,
        "inputs": {
            "parameter_bytes": parameter_bytes,
            "activation_bytes": activation_bytes,
            "parameter_element_bytes": parameter_element_bytes,
            "world_size": ranks,
            "optimizer": optimizer,
        },
        "sharding": {
            "strategy": strategy,
            "zero_stage": zero_stage,
            "parameters_sharded": full_shard,
            "gradients_sharded": shard_gradients,
            "optimizer_state_sharded": shard_optimizer,
        },
        "rank_local_storage": {
            "parameter_bytes": gpu_parameter_bytes,
            "gradient_bytes": gpu_gradient_bytes,
            "optimizer_state_bytes": gpu_optimizer_state_bytes,
            "master_parameter_bytes": gpu_master_parameter_bytes,
            "retained_activation_bytes": retained_activation_bytes,
            "communication_workspace_bytes": communication_workspace_bytes,
        },
        "cpu_offload": {
            "parameter_bytes": cpu_offloaded_parameter_bytes,
            "optimizer_and_master_parameter_bytes": (
                cpu_offloaded_optimizer_bytes
            ),
            "total_bytes": (
                cpu_offloaded_parameter_bytes
                + cpu_offloaded_optimizer_bytes
            ),
        },
        "storage_offload": {
            "parameter_device": parameter_offload_device,
            "optimizer_device": optimizer_offload_device,
            "cpu_bytes": (
                cpu_offloaded_parameter_bytes
                + cpu_offloaded_optimizer_bytes
            ),
            "nvme_bytes": (
                nvme_offloaded_parameter_bytes
                + nvme_offloaded_optimizer_bytes
            ),
        },
        "memory_timeline": {
            "unit": "bytes",
            "phases": timeline,
            "peak_phase": peak_phase,
            "peak_bytes": peak,
        },
        "estimated_rank_peak_bytes": peak,
        "communication": {
            "reduce_bucket_bytes": reduce_bucket,
            "allgather_bucket_bytes": allgather_bucket,
            "overlap_communication": bool(
                normalized.get("overlap_communication", False)
            ),
            "gradient_accumulation_steps": int(
                normalized.get("gradient_accumulation_steps", 1) or 1
            ),
        },
        "unmodeled_components": [
            "cuda_context_and_loaded_modules",
            "allocator_fragmentation",
            "backend_operator_workspaces",
            "prefetch_schedule_specific_overlap",
            "fused_optimizer_temporaries",
        ],
        "notes": [
            "Gradient accumulation retains one gradient buffer; it does not multiply gradient storage by the accumulation count.",
            "Activation checkpointing uses the configured retention factor and does not predict recomputation time.",
            "ZeRO and FSDP prefetch windows depend on module granularity; bucket sizes provide a conservative communication workspace.",
            "Use a matching real-GPU calibration to replace unmodeled runtime and allocator terms.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu plan-training",
        description=(
            "Normalize DeepSpeed/Accelerate/FSDP configuration and estimate "
            "rank-local memory phases."
        ),
    )
    parser.add_argument("config")
    parser.add_argument(
        "--framework",
        choices=sorted(SUPPORTED_FRAMEWORKS),
        default="auto",
    )
    parser.add_argument("--parameter-bytes", type=int, required=True)
    parser.add_argument("--activation-bytes", type=int, required=True)
    parser.add_argument("--world-size", type=int)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "sgd", "sgd_momentum"],
        default="adamw",
    )
    parser.add_argument("--parameter-element-bytes", type=int, default=2)
    parser.add_argument(
        "--json",
        dest="json_path",
        nargs="?",
        const="-",
        help="Write JSON to PATH, or stdout when PATH is omitted.",
    )
    args = parser.parse_args(argv)
    try:
        raw = load_mapping(args.config)
        normalized = normalize_training_config(
            raw,
            framework=args.framework,
        )
        report = estimate_training_plan(
            normalized,
            parameter_bytes=args.parameter_bytes,
            activation_bytes=args.activation_bytes,
            world_size=args.world_size,
            optimizer=args.optimizer,
            parameter_element_bytes=args.parameter_element_bytes,
        )
        if args.json_path:
            payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
            if args.json_path == "-":
                print(payload, end="")
            else:
                output = write_json(args.json_path, report)
                print(f"Training plan: {output}")
        else:
            _print_report(report)
        return 0
    except (
            OSError,
            ValueError,
           ) as exc:
        parser.exit(2, f"fakegpu plan-training: {exc}\n")


def _detect_framework(config: Mapping[str, Any]) -> str:
    if "zero_optimization" in config or any(
        key in config for key in ("train_batch_size", "train_micro_batch_size_per_gpu")
    ):
        return "deepspeed"
    distributed_type = str(config.get("distributed_type") or "").upper()
    if distributed_type or "deepspeed_config" in config or "fsdp_config" in config:
        return "accelerate"
    strategy = str(
        config.get("sharding_strategy")
        or config.get("fsdp_sharding_strategy")
        or ""
    ).lower()
    if "fsdp2" in str(config.get("framework") or "").lower():
        return "fsdp2"
    if strategy or any(str(key).startswith("fsdp_") for key in config):
        return "fsdp"
    return ""


def _normalize_deepspeed(config: Mapping[str, Any]) -> dict[str, Any]:
    zero = config.get("zero_optimization")
    zero_config = dict(zero) if isinstance(zero, Mapping) else {}
    stage = int(zero_config.get("stage", 0) or 0)
    if stage not in {0, 1, 2, 3}:
        raise TrainingPlanError("DeepSpeed ZeRO stage must be 0, 1, 2, or 3")
    offload_param = zero_config.get("offload_param")
    offload_optimizer = zero_config.get("offload_optimizer")
    precision = _deepspeed_precision(config)
    activation_checkpointing = config.get("activation_checkpointing")
    checkpoint_config = (
        activation_checkpointing
        if isinstance(activation_checkpointing, Mapping)
        else {}
    )
    world_size = _positive_int(
        config.get("world_size", 1),
        "world_size",
    )
    return {
        "framework": "deepspeed",
        "world_size": world_size,
        "zero_stage": stage,
        "sharding_strategy": "full_shard" if stage == 3 else "replicated",
        "parameter_offload_device": _offload_device(offload_param),
        "optimizer_offload_device": _offload_device(offload_optimizer),
        "mixed_precision": precision,
        "gradient_accumulation_steps": _positive_int(
            config.get("gradient_accumulation_steps", 1),
            "gradient_accumulation_steps",
        ),
        "activation_checkpointing": _bool_value(
            checkpoint_config.get("partition_activations")
            or checkpoint_config.get("cpu_checkpointing")
            or config.get("gradient_checkpointing")
        ),
        "overlap_communication": _bool_value(
            zero_config.get("overlap_comm", False)
        ),
        "contiguous_gradients": _bool_value(
            zero_config.get("contiguous_gradients", False)
        ),
        "reduce_bucket_bytes": _int_or_default(
            zero_config.get("reduce_bucket_size"),
            0,
        ),
        "allgather_bucket_bytes": _int_or_default(
            zero_config.get("allgather_bucket_size"),
            0,
        ),
        "stage3_prefetch_bucket_bytes": _int_or_default(
            zero_config.get("stage3_prefetch_bucket_size"),
            0,
        ),
    }


def _normalize_accelerate(config: Mapping[str, Any]) -> dict[str, Any]:
    distributed_type = str(config.get("distributed_type") or "NO").upper()
    world_size = _positive_int(
        config.get("num_processes", config.get("world_size", 1)),
        "num_processes",
    )
    if distributed_type == "DEEPSPEED":
        nested = config.get("deepspeed_config")
        if not isinstance(nested, Mapping):
            nested = {}
        merged = dict(nested)
        if "zero_optimization" not in merged and "zero_stage" in merged:
            merged["zero_optimization"] = {
                "stage": _int_or_default(merged.get("zero_stage"), 0),
                "offload_optimizer": {
                    "device": str(
                        merged.get("offload_optimizer_device") or "none"
                    ).lower()
                },
                "offload_param": {
                    "device": str(
                        merged.get("offload_param_device") or "none"
                    ).lower()
                },
            }
        merged.setdefault("world_size", world_size)
        normalized = _normalize_deepspeed(merged)
    elif "FSDP" in distributed_type:
        nested = config.get("fsdp_config")
        if not isinstance(nested, Mapping):
            nested = {}
        merged = dict(nested)
        merged.setdefault("world_size", world_size)
        normalized = _normalize_fsdp(
            merged,
            framework="fsdp2" if "2" in distributed_type else "fsdp",
        )
    else:
        normalized = {
            "framework": "accelerate",
            "world_size": world_size,
            "zero_stage": 0,
            "sharding_strategy": "replicated",
            "parameter_offload_device": "none",
            "optimizer_offload_device": "none",
            "gradient_accumulation_steps": 1,
            "activation_checkpointing": False,
            "overlap_communication": False,
            "reduce_bucket_bytes": 0,
            "allgather_bucket_bytes": 0,
        }
    normalized["accelerate_distributed_type"] = distributed_type
    normalized["mixed_precision"] = _normalize_precision(
        config.get("mixed_precision", normalized.get("mixed_precision", "no"))
    )
    return normalized


def _normalize_fsdp(
    config: Mapping[str, Any],
    *,
    framework: str,
) -> dict[str, Any]:
    raw_strategy = str(
        config.get("sharding_strategy")
        or config.get("fsdp_sharding_strategy")
        or "FULL_SHARD"
    ).upper()
    strategy = {
        "1": "full_shard",
        "FULL_SHARD": "full_shard",
        "2": "shard_grad_op",
        "SHARD_GRAD_OP": "shard_grad_op",
        "3": "no_shard",
        "NO_SHARD": "no_shard",
        "4": "hybrid_shard",
        "HYBRID_SHARD": "hybrid_shard",
        "5": "hybrid_shard",
        "_HYBRID_SHARD_ZERO2": "hybrid_shard",
    }.get(raw_strategy)
    if strategy is None:
        raise TrainingPlanError(
            f"unsupported FSDP sharding strategy {raw_strategy!r}"
        )
    cpu_offload = config.get("cpu_offload")
    if isinstance(cpu_offload, Mapping):
        offload_params = _bool_value(
            cpu_offload.get("offload_params", False)
        )
    else:
        offload_params = _bool_value(
            config.get("fsdp_offload_params", cpu_offload)
        )
    checkpointing = _bool_value(
        config.get("activation_checkpointing")
        or config.get("fsdp_activation_checkpointing")
    )
    return {
        "framework": framework,
        "world_size": _positive_int(
            config.get("world_size", config.get("num_processes", 1)),
            "world_size",
        ),
        "zero_stage": 0,
        "sharding_strategy": strategy,
        "parameter_offload_device": "cpu" if offload_params else "none",
        "optimizer_offload_device": "none",
        "mixed_precision": _normalize_precision(
            config.get(
                "mixed_precision",
                config.get("fsdp_mixed_precision", "no"),
            )
        ),
        "gradient_accumulation_steps": _positive_int(
            config.get("gradient_accumulation_steps", 1),
            "gradient_accumulation_steps",
        ),
        "activation_checkpointing": checkpointing,
        "overlap_communication": _bool_value(
            config.get("backward_prefetch")
            or config.get("fsdp_backward_prefetch")
        ),
        "reduce_bucket_bytes": _int_or_default(
            config.get("reduce_bucket_bytes"),
            0,
        ),
        "allgather_bucket_bytes": _int_or_default(
            config.get("allgather_bucket_bytes"),
            0,
        ),
        "forward_prefetch": _bool_value(
            config.get("forward_prefetch")
            or config.get("fsdp_forward_prefetch")
        ),
        "use_orig_params": _bool_value(
            config.get("use_orig_params")
            or config.get("fsdp_use_orig_params")
            or framework == "fsdp2"
        ),
    }


def _deepspeed_precision(config: Mapping[str, Any]) -> str:
    for name in ("fp8", "bf16", "fp16"):
        value = config.get(name)
        if isinstance(value, Mapping) and _bool_value(
            value.get("enabled", False)
        ):
            return name
    return "no"


def _normalize_precision(value: Any) -> str:
    normalized = str(value or "no").lower()
    aliases = {
        "none": "no",
        "false": "no",
        "16": "fp16",
        "16-mixed": "fp16",
        "bf16-mixed": "bf16",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"no", "fp16", "bf16", "fp8"}:
        raise TrainingPlanError(
            f"unsupported mixed precision setting {value!r}"
        )
    return normalized


def _offload_device(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "none"
    device = str(value.get("device") or "none").lower()
    return device if device in {"none", "cpu", "nvme"} else "unknown"


def _positive_int(value: Any, name: str) -> int:
    if str(value).strip().lower() == "auto":
        return 1
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainingPlanError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise TrainingPlanError(f"{name} must be a positive integer")
    return parsed


def _int_or_default(value: Any, default: int) -> int:
    if value is None or str(value).strip().lower() == "auto":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainingPlanError(
            f"expected an integer or 'auto', got {value!r}"
        ) from exc
    if parsed < 0:
        raise TrainingPlanError("memory bucket size must be non-negative")
    return parsed


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "none", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on", "auto"}:
            return True
        raise TrainingPlanError(
            f"expected a boolean (or 'auto'), got {value!r}"
        )
    return bool(value)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _phase(
    name: str,
    peak_bytes: int,
    components: Mapping[str, int],
) -> dict[str, Any]:
    normalized = {
        key: int(value)
        for key, value in components.items()
        if int(value) > 0
    }
    return {
        "phase": name,
        "peak_bytes": int(peak_bytes),
        "components": normalized,
        "component_sum_bytes": sum(normalized.values()),
    }


def _print_report(report: Mapping[str, Any]) -> None:
    configuration = report["configuration"]
    timeline = report["memory_timeline"]
    print("FakeGPU training memory plan")
    print(f"  framework: {configuration['source_framework']}")
    print(f"  world size: {report['inputs']['world_size']}")
    print(f"  sharding: {report['sharding']['strategy']}")
    print(
        "  estimated rank peak: "
        f"{int(report['estimated_rank_peak_bytes']) / 2**30:.3f} GiB "
        f"({timeline['peak_phase']})"
    )
    print(
        "  CPU offload: "
        f"{int(report['cpu_offload']['total_bytes']) / 2**30:.3f} GiB"
    )


__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_FRAMEWORKS",
    "TrainingPlanError",
    "estimate_training_plan",
    "main",
    "normalize_training_config",
]
