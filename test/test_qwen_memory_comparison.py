from __future__ import annotations

from verification.compare_qwen_memory import compare_reports, render_markdown


def _report(mode: str, scale: int = 0) -> dict:
    real = mode == "real"
    parameter_bytes = 16_000_000_000
    load = parameter_bytes + (2_000_000 if real else 0)
    inference_current = parameter_bytes + (12_000_000 if real else 2_000_000)
    inference_peak = parameter_bytes + (15_000_000 if real else 4_000_000)
    process_memory = inference_current + 400_000_000
    return {
        "status": "success",
        "mode": mode,
        "model_dir": "/models/qwen",
        "dtype": "bfloat16",
        "attention_implementation": "eager",
        "prompt_tokens": 9,
        "generated_tokens": [1, 2],
        "parameters": {
            "parameter_count": 8_000_000_000,
            "parameter_bytes": parameter_bytes,
        },
        "memory_phases": {
            "model_load_current_bytes": load,
            "inference_peak_bytes": inference_peak,
        },
        "measured_matmul_flops_by_step": [100_000_000_000, 10_000_000_000],
        "timeline": [
            {
                "label": "after_inference",
                "allocated_bytes": inference_current,
                "nvml": {"process_memory": process_memory} if real else None,
            }
        ],
        "static_estimate": {
            "checkpoint": {"parameter_count": 8_000_000_000},
            "memory": {
                "parameter_bytes": parameter_bytes,
                "estimated_tensor_peak_bytes": parameter_bytes + 5_000_000,
            },
            "compute": {"total_flops": 110_000_000_000 + scale},
        },
    }


def test_comparison_accepts_subpercent_memory_and_exact_flops() -> None:
    report = compare_reports(_report("real"), _report("fakecuda"))
    assert report["status"] == "success"
    assert report["calibration"]["nvml_runtime_overhead_bytes"] == 400_000_000
    assert report["checks"]["generated_tokens_match"] is True
    assert report["comparisons"]["static_flops"]["absolute_error_percent"] == 0
    markdown = render_markdown(report)
    assert "Qwen Inference Memory Validation" in markdown
    assert "virtual SMI vs real NVML" in markdown


def test_comparison_rejects_large_flop_error() -> None:
    fake = _report("fakecuda", scale=1_000_000_000)
    report = compare_reports(_report("real"), fake)
    assert report["status"] == "failed"
    assert report["checks"]["static_flop_error_below_0_01_percent"] is False
