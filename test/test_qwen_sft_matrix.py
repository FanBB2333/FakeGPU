from __future__ import annotations

from verification.summarize_qwen_sft_matrix import render_markdown, summarize_cases


def _report(mode: str, peak: int) -> dict:
    return {
        "status": "success",
        "mode": mode,
        "model_dir": "/models/Qwen3.5-0.8B",
        "gpu_name": "RTX PRO 5000" if mode == "real" else mode,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "optimizer": "adamw_single_tensor",
        "training_method": "lora",
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 1,
        "lora": {"rank": 8},
        "batch_size": 1,
        "sequence_length": 16,
        "data_seed": 7,
        "elapsed_seconds": 1.0,
        "parameters": {
            "parameter_count": 100,
            "trainable_parameter_count": 10,
        },
        "batch": {"fingerprint_sha256": "batch"},
        "memory_phases": {
            "forward_peak_bytes": peak - 30,
            "backward_peak_bytes": peak - 20,
            "optimizer_peak_bytes": peak - 10,
            "overall_peak_bytes": peak,
        },
        "static_analysis": {
            "checkpointing": "disabled",
            "gradient_accumulation": "single_microbatch_exact",
        },
    }


def test_matrix_summarizes_optional_fakecuda_report() -> None:
    report = summarize_cases(
        {
            "lora": {
                "real": _report("real", 1_000),
                "static": _report("static", 990),
                "fakecuda": _report("fakecuda", 980),
            }
        }
    )

    assert report["status"] == "success"
    assert report["case_count"] == 1
    assert report["cases"][0]["static_prediction_kind"] == "exact"
    assert report["cases"][0]["fakecuda_comparison"]["absolute_error_percent"] == 2.0
    assert "Qwen SFT Memory Matrix" in render_markdown(report)


def test_matrix_accepts_real_and_static_without_fakecuda() -> None:
    report = summarize_cases(
        {
            "static-only-validation": {
                "real": _report("real", 1_000),
                "static": _report("static", 995),
            }
        }
    )

    assert report["status"] == "success"
    assert report["cases"][0]["fakecuda_peak_bytes"] is None


def test_matrix_rejects_static_error_above_limit() -> None:
    report = summarize_cases(
        {
            "bad": {
                "real": _report("real", 1_000),
                "static": _report("static", 900),
            }
        }
    )

    assert report["status"] == "failed"
    assert report["cases"][0]["static_passed"] is False
