"""Measure one serving sample on a real CUDA device through Transformers.

Split out of ``calibration`` unchanged. This is the only part of the
calibration layer that touches ``torch`` and ``transformers``, and the only
part that refuses to run under FakeGPU: a calibration sample is worth
nothing unless it came from real hardware, so it verifies the environment
before measuring, then reports the measured phase peaks through the
serving-sample protocol.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._calibration_protocol import (
    SERVING_SAMPLE_SCHEMA_VERSION,
    CalibrationError,
    _nonempty_string,
    _normalize_serving_phase_peaks,
    _positive_plan_integer,
    _serving_plan_observation_context,
    _serving_sample_run_index,
    _validate_collector_environment_contract,
    _validate_serving_runner_sample,
)
from .llm_estimator import _decoder_dimensions, inspect_safetensors_checkpoint


def _load_optional_module(name: str, *, purpose: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise CalibrationError(
            f"{purpose} requires the optional {name!r} package"
        ) from exc


def _reject_simulated_cuda_environment() -> None:
    fakegpu_mode = os.environ.get("FAKEGPU_MODE", "").strip().lower()
    if fakegpu_mode and fakegpu_mode != "passthrough":
        raise CalibrationError(
            "serving samples require real CUDA; FAKEGPU_MODE must be unset "
            "or 'passthrough'"
        )


def _real_cuda_environment(
    torch_module: Any,
    *,
    framework: str,
    framework_version: str,
    software: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise CalibrationError(
            "building a CUDA serving sample requires an available real "
            "CUDA device"
        )
    _reject_simulated_cuda_environment()
    cuda_version = str(getattr(torch_module.version, "cuda", "") or "")
    if not cuda_version:
        raise CalibrationError("PyTorch does not report a CUDA runtime version")
    device_index = int(cuda.current_device())
    cuda.set_device(device_index)
    properties = cuda.get_device_properties(device_index)
    total_memory = getattr(properties, "total_memory", None)
    if (
        not isinstance(total_memory, int)
        or isinstance(total_memory, bool)
        or total_memory <= 0
    ):
        raise CalibrationError("CUDA device does not report total memory")
    capability = cuda.get_device_capability(device_index)
    if (
        not isinstance(capability, Sequence)
        or len(capability) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in capability
        )
    ):
        raise CalibrationError("CUDA device does not report compute capability")
    gpu_name = str(
        getattr(properties, "name", None)
        or cuda.get_device_name(device_index)
        or ""
    ).strip()
    if not gpu_name:
        raise CalibrationError("CUDA device does not report a name")

    software_report: dict[str, Any] = {
        "framework": framework,
        "framework_version": framework_version,
        "cuda_version": cuda_version,
        "torch_version": str(torch_module.__version__),
    }
    if software is not None:
        if not isinstance(software, Mapping):
            raise CalibrationError("software must be a mapping")
        reserved_keys = set(software_report)
        collisions = sorted(reserved_keys.intersection(map(str, software)))
        if collisions:
            raise CalibrationError(
                "software metadata cannot replace: "
                + ", ".join(collisions)
            )
        for key, item in software.items():
            normalized_key = _nonempty_string(str(key), "software key")
            try:
                json.dumps(item, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise CalibrationError(
                    f"software.{normalized_key} must be JSON-compatible"
                ) from exc
            software_report[normalized_key] = item

    environment: dict[str, Any] = {
        "backend": "cuda",
        "simulated": False,
        "gpu_name": gpu_name,
        "compute_capability": [int(capability[0]), int(capability[1])],
        "total_memory_bytes": int(total_memory),
        "device_index": device_index,
        "software": software_report,
    }
    gpu_uuid = getattr(properties, "uuid", None)
    if gpu_uuid:
        if isinstance(gpu_uuid, bytes):
            gpu_uuid = gpu_uuid.decode("utf-8", errors="replace")
        environment["gpu_uuid"] = str(gpu_uuid)
    allocator_reader = getattr(cuda, "get_allocator_backend", None)
    if callable(allocator_reader):
        environment["allocator"] = str(allocator_reader())
    return environment


def _transformers_torch_dtype(torch_module: Any, dtype: str) -> Any:
    try:
        return getattr(torch_module, dtype)
    except AttributeError as exc:
        raise CalibrationError(
            f"installed PyTorch does not expose dtype {dtype!r}"
        ) from exc


def _synthetic_token_id(config: Any) -> int:
    vocab_size = getattr(config, "vocab_size", None)
    if (
        not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size <= 0
    ):
        raise CalibrationError("Transformers model has no positive vocab_size")
    for name in ("bos_token_id", "pad_token_id", "eos_token_id"):
        value = getattr(config, name, None)
        if isinstance(value, list):
            value = value[0] if value else None
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value < vocab_size
        ):
            return int(value)
    return 0


def _reset_cuda_phase_peak(cuda: Any, device_index: int) -> None:
    cuda.synchronize(device_index)
    cuda.reset_peak_memory_stats(device_index)


def _next_transformers_token(
    output: Any,
    *,
    phase: str,
) -> tuple[Any, Any]:
    past_key_values = getattr(output, "past_key_values", None)
    logits = getattr(output, "logits", None)
    if past_key_values is None:
        raise CalibrationError(
            f"Transformers {phase} output did not return past_key_values"
        )
    if logits is None:
        raise CalibrationError(
            f"Transformers {phase} output did not return logits"
        )
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return past_key_values, next_token


def _transformers_serving_workload(
    plan: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    context = _serving_plan_observation_context(plan)
    if context["schema_version"] != "fakegpu.llm_serving_plan.v1":
        raise CalibrationError(
            "sample-transformers supports homogeneous serving plans only"
        )
    if context["planned_phases"] != ["prefill", "decode"]:
        raise CalibrationError(
            "sample-transformers requires prefill and decode phases"
        )
    inputs = context["inputs"]
    if str(inputs.get("runtime") or "generic") != "generic":
        raise CalibrationError(
            "sample-transformers requires a generic serving runtime plan"
        )
    speculative = inputs.get("speculative_decoding")
    if isinstance(speculative, Mapping) and speculative.get("enabled"):
        raise CalibrationError(
            "sample-transformers does not yet execute speculative decoding"
        )
    if inputs.get("kv_cache_strategy") != "dynamic":
        raise CalibrationError(
            "sample-transformers requires --kv-cache-strategy dynamic"
        )
    if int(inputs.get("shared_prefix_tokens") or 0) != 0:
        raise CalibrationError(
            "sample-transformers does not emulate shared-prefix caching"
        )
    prompt_tokens = _positive_plan_integer(inputs, "prompt_tokens")
    prefill_chunk_tokens = inputs.get("prefill_chunk_tokens")
    if (
        prefill_chunk_tokens is not None
        and int(prefill_chunk_tokens) < prompt_tokens
    ):
        raise CalibrationError(
            "sample-transformers does not emulate chunked prefill"
        )
    for key in ("kv_cache_max_tokens", "kv_cache_window_tokens"):
        if inputs.get(key) is not None:
            raise CalibrationError(
                f"sample-transformers does not emulate {key}"
            )

    weight_storage = plan.get("weight_storage")
    if not isinstance(weight_storage, Mapping):
        raise CalibrationError("serving plan has no weight_storage")
    quantization = weight_storage.get("quantization")
    if isinstance(quantization, Mapping) and quantization.get("enabled"):
        raise CalibrationError(
            "sample-transformers does not load quantized checkpoints"
        )
    if weight_storage.get("adapters"):
        raise CalibrationError(
            "sample-transformers does not load adapter checkpoints"
        )

    model = plan.get("model")
    if not isinstance(model, Mapping):
        raise CalibrationError("serving plan has no model identity")
    selected_model_dir = model_dir or model.get("path")
    if selected_model_dir is None:
        raise CalibrationError(
            "sample-transformers requires a local model directory"
        )
    root = Path(selected_model_dir).expanduser().resolve()
    if not root.is_dir():
        raise CalibrationError(f"model directory does not exist: {root}")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise CalibrationError(f"model config does not exist: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibrationError(
            f"model config is not valid JSON: {config_path}"
        ) from exc
    if not isinstance(config, Mapping):
        raise CalibrationError(f"model config must be an object: {config_path}")

    observed_dimensions = _decoder_dimensions(config)
    mismatched_dimensions = [
        key
        for key, value in observed_dimensions.items()
        if model.get(key) != value
    ]
    if mismatched_dimensions:
        raise CalibrationError(
            "model config no longer matches the serving plan: "
            + ", ".join(mismatched_dimensions)
        )
    expected_checkpoint = plan.get("checkpoint")
    observed_checkpoint = inspect_safetensors_checkpoint(root)
    if not isinstance(expected_checkpoint, Mapping) or dict(
        expected_checkpoint
    ) != observed_checkpoint:
        raise CalibrationError(
            "model checkpoint no longer matches the serving plan"
        )

    attention_implementation = str(
        inputs.get("attention_implementation") or ""
    )
    if attention_implementation not in {"eager", "sdpa"}:
        raise CalibrationError(
            "sample-transformers supports eager or sdpa attention only"
        )
    dtype = str(inputs.get("dtype") or "")
    if dtype not in {"float16", "bfloat16", "float32", "float64"}:
        raise CalibrationError(
            f"sample-transformers does not support dtype {dtype!r}"
        )
    return {
        "model_dir": root,
        "active_sequences": _positive_plan_integer(
            inputs,
            "active_sequences",
        ),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": _positive_plan_integer(
            inputs,
            "generated_tokens",
        ),
        "attention_implementation": attention_implementation,
        "dtype": dtype,
    }


def build_cuda_serving_sample(
    plan: Mapping[str, Any],
    *,
    phase_peaks: Mapping[str, int],
    metric: str,
    framework: str,
    framework_version: str,
    run_index: int | None = None,
    software: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serving-runner protocol sample from a real CUDA runtime.

    This is the compatibility entry point for vLLM and custom runners that
    already have trustworthy phase measurements. CUDA device and software
    identity are collected here so each runner does not need to duplicate the
    versioned protocol.
    """

    context = _serving_plan_observation_context(plan)
    normalized_peaks = _normalize_serving_phase_peaks(
        phase_peaks,
        planned_phases=context["planned_phases"],
    )
    normalized_metric = _nonempty_string(metric, "metric")
    normalized_framework = _nonempty_string(framework, "framework")
    normalized_framework_version = _nonempty_string(
        framework_version,
        "framework_version",
    )
    normalized_run_index = _serving_sample_run_index(run_index)
    _validate_collector_environment_contract(
        context,
        run_index=normalized_run_index,
    )

    torch_module = _load_optional_module(
        "torch",
        purpose="building a real-CUDA serving sample",
    )
    environment = _real_cuda_environment(
        torch_module,
        framework=normalized_framework,
        framework_version=normalized_framework_version,
        software=software,
    )
    sample = {
        "schema_version": SERVING_SAMPLE_SCHEMA_VERSION,
        "workload_signature": context["workload_signature"],
        "run_index": normalized_run_index,
        "metric": normalized_metric,
        "phases": {
            phase: {"peak_bytes": peak}
            for phase, peak in normalized_peaks.items()
        },
        "environment": environment,
    }
    _validate_serving_runner_sample(
        sample,
        context=context,
        run_index=normalized_run_index,
    )
    return sample


def measure_transformers_serving_sample(
    plan: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike[str] | None = None,
    metric: str = "reserved",
    trust_remote_code: bool = False,
    run_index: int | None = None,
) -> dict[str, Any]:
    """Measure one homogeneous Transformers serving sample on real CUDA.

    The adapter executes a full prefill followed by token-at-a-time decoding
    with the returned dynamic KV cache. Optional framework dependencies are
    imported only when this function is called.
    """

    workload = _transformers_serving_workload(plan, model_dir=model_dir)
    metric_key = str(metric).strip().lower()
    metric_name = {
        "allocated": "torch.cuda.max_memory_allocated",
        "reserved": "torch.cuda.max_memory_reserved",
    }.get(metric_key)
    if metric_name is None:
        raise CalibrationError("metric must be 'allocated' or 'reserved'")

    torch_module = _load_optional_module(
        "torch",
        purpose="measuring Transformers serving memory",
    )
    transformers_module = _load_optional_module(
        "transformers",
        purpose="measuring Transformers serving memory",
    )
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise CalibrationError(
            "Transformers serving measurement requires an available real "
            "CUDA device"
        )
    _reject_simulated_cuda_environment()
    device_index = int(cuda.current_device())
    cuda.set_device(device_index)
    device = f"cuda:{device_index}"
    dtype = _transformers_torch_dtype(torch_module, workload["dtype"])

    load_kwargs: dict[str, Any] = {
        "attn_implementation": workload["attention_implementation"],
        "dtype": dtype,
        "local_files_only": True,
        "trust_remote_code": bool(trust_remote_code),
        "use_safetensors": True,
    }
    model_factory = transformers_module.AutoModelForCausalLM
    try:
        model = model_factory.from_pretrained(
            str(workload["model_dir"]),
            **load_kwargs,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        model = model_factory.from_pretrained(
            str(workload["model_dir"]),
            **load_kwargs,
        )
    model = model.to(device)
    model.eval()

    token_id = _synthetic_token_id(model.config)
    input_ids = torch_module.full(
        (
            workload["active_sequences"],
            workload["prompt_tokens"],
        ),
        token_id,
        dtype=torch_module.long,
        device=device,
    )
    peak_reader = getattr(cuda, f"max_memory_{metric_key}")
    with torch_module.inference_mode():
        _reset_cuda_phase_peak(cuda, device_index)
        prefill_output = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        cuda.synchronize(device_index)
        prefill_peak = int(peak_reader(device_index))
        past_key_values, next_token = _next_transformers_token(
            prefill_output,
            phase="prefill",
        )
        del prefill_output
        del input_ids

        cuda.synchronize(device_index)
        cuda.empty_cache()
        _reset_cuda_phase_peak(cuda, device_index)
        for _ in range(max(0, workload["generated_tokens"] - 1)):
            decode_output = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values, next_token = _next_transformers_token(
                decode_output,
                phase="decode",
            )
            del decode_output
        cuda.synchronize(device_index)
        decode_peak = int(peak_reader(device_index))

    return build_cuda_serving_sample(
        plan,
        phase_peaks={
            "prefill": prefill_peak,
            "decode": decode_peak,
        },
        metric=metric_name,
        framework="transformers",
        framework_version=str(transformers_module.__version__),
        run_index=run_index,
        software={
            "attention_implementation": workload[
                "attention_implementation"
            ],
            "cache_implementation": "dynamic",
            "dtype": workload["dtype"],
            "runner": "fakegpu.calibrate.sample-transformers",
        },
    )
