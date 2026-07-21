from __future__ import annotations

from verification.compare_qwen_sft_memory import compare_reports, render_markdown


def _execution_report(mode: str, *, peak_offset: int = 0) -> dict:
    real = mode == "real"
    parameter_bytes = 1_500_000_000
    base = {
        "status": "success",
        "mode": mode,
        "model_dir": "/models/Qwen3.5-0.8B",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "optimizer": "adamw_single_tensor",
        "batch_size": 1,
        "sequence_length": 16,
        "data_seed": 7,
        "gpu_name": "real GPU" if real else "fake GPU",
        "parameters": {
            "parameter_count": 750_000_000,
            "parameter_bytes": parameter_bytes,
        },
        "batch": {
            "masked_prefix_tokens": 8,
            "fingerprint_sha256": "same-random-batch",
        },
        "loss": 16.0 if real else 16.1,
        "phase_seconds": {
            "model_load": 1.0,
            "forward": 1.0,
            "backward": 1.0,
            "optimizer": 1.0,
        },
        "memory_phases": {
            "model_load_current_bytes": parameter_bytes + (2_000_000 if real else 0),
            "forward_peak_bytes": 1_800_000_000 + peak_offset,
            "backward_peak_bytes": 4_000_000_000 + peak_offset,
            "optimizer_peak_bytes": 7_000_000_000 + peak_offset,
            "overall_peak_bytes": 7_000_000_000 + peak_offset,
        },
    }
    return base


def _static_report(*, peak_offset: int = 0) -> dict:
    report = _execution_report("fakecuda")
    report.update(
        {
            "mode": "static",
            "gpu_name": "static FakeTensor CUDA target",
            "elapsed_seconds": 2.0,
            "memory_phases": {
                "overall_peak_bytes": 7_010_000_000 + peak_offset,
                "first_step_graph_phase_peak_bytes": 4_010_000_000 + peak_offset,
                "parameter_bytes": 1_500_000_000,
            },
            "static_estimate": {"tracking_confidence": "S2_aot_training_liveness"},
        }
    )
    report.pop("loss")
    report.pop("phase_seconds")
    return report


def test_comparison_accepts_matching_subpercent_sft_reports() -> None:
    report = compare_reports(
        _execution_report("real"),
        _execution_report("fakecuda"),
        _static_report(),
    )
    assert report["status"] == "success"
    assert report["checks"]["random_batch_exact"] is True
    assert report["checks"]["parameter_bytes_exact"] is True
    assert report["comparisons"]["static_overall_peak"]["absolute_error_percent"] < 1.0
    assert report["comparisons"]["static_first_step_graph_peak"]["absolute_error_percent"] < 1.0
    markdown = render_markdown(report)
    assert "Qwen SFT Memory Validation" in markdown
    assert "Overall peak: static graph vs real allocator" in markdown


def test_comparison_rejects_large_static_underestimate() -> None:
    report = compare_reports(
        _execution_report("real"),
        _execution_report("fakecuda"),
        _static_report(peak_offset=-1_000_000_000),
    )
    assert report["status"] == "failed"
    assert report["checks"]["static_overall_error_within_limit"] is False


def test_comparison_rejects_different_random_batch() -> None:
    fake = _execution_report("fakecuda")
    fake["batch"]["fingerprint_sha256"] = "different"
    try:
        compare_reports(_execution_report("real"), fake, _static_report())
    except ValueError as exc:
        assert "batches do not match" in str(exc)
    else:
        raise AssertionError("expected a mismatched random batch to be rejected")
