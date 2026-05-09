#!/usr/bin/env python3
"""Calibrate small RTX 3090 Ti real-CUDA workloads against fakecuda preflight."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = "calibration_rtx3090ti.json"
REPORT_MD = "calibration_rtx3090ti.md"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 3090 Ti"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/rtx3090ti_calibration")
    parser.add_argument("--worker", choices=["real", "fakecuda"], help=argparse.SUPPRESS)
    parser.add_argument("--workload", choices=list(_workloads()), help=argparse.SUPPRESS)
    parser.add_argument("--allow-other-gpu", action="store_true")
    ns = parser.parse_args(argv)

    if ns.worker:
        if not ns.workload:
            parser.error("--worker requires --workload")
        payload = _run_worker(ns.worker, ns.workload)
        print(json.dumps(payload, sort_keys=True))
        return 0

    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_calibration(out_dir=out_dir, allow_other_gpu=bool(ns.allow_other_gpu))
    (out_dir / REPORT_JSON).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / REPORT_MD).write_text(render_markdown(report), encoding="utf-8")
    print(f"calibration report: {out_dir / REPORT_JSON}")
    print(f"calibration markdown: {out_dir / REPORT_MD}")
    if str(report.get("status", "")).startswith("SKIP"):
        print(f"SKIP: {report.get('skip_reason')}")
    return 0


def run_calibration(*, out_dir: Path, allow_other_gpu: bool = False) -> dict[str, Any]:
    started = time.time()
    cuda = _cuda_probe()
    if cuda["status"] != "available":
        return _skip_report(cuda["reason"], started)
    if not allow_other_gpu and cuda["gpu"]["name"] != EXPECTED_GPU_NAME:
        return _skip_report(
            f"expected {EXPECTED_GPU_NAME}, got {cuda['gpu']['name']}",
            started,
            gpu=cuda["gpu"],
        )

    workloads: list[dict[str, Any]] = []
    for workload_name in _workloads():
        real = _run_real_worker(workload_name)
        fake = _run_fakecuda_preflight(workload_name, out_dir=out_dir)
        workloads.append(_compare_workload(workload_name, real, fake))

    return {
        "schema_version": "rtx3090ti_calibration.v1",
        "status": "PASS_CALIBRATED",
        "created_at_unix": int(started),
        "calibration_gpu": cuda["gpu"],
        "workloads": workloads,
        "notes": [
            "This calibration records memory peak error for small workloads that fit on a single RTX 3090 Ti.",
            "It does not claim A100/H100 numerical parity or cluster performance.",
            "passthrough and hybrid clamp calibration are not part of this first lightweight suite.",
        ],
    }


def _skip_report(reason: str, started: float, gpu: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": "rtx3090ti_calibration.v1",
        "status": "SKIP_NO_RTX3090TI",
        "created_at_unix": int(started),
        "skip_reason": reason,
        "calibration_gpu": gpu,
        "workloads": [],
        "notes": ["RTX 3090 Ti calibration requires a real CUDA-visible RTX 3090 Ti."],
    }


def _cuda_probe() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"status": "missing_torch", "reason": f"torch import failed: {exc}"}

    if not torch.cuda.is_available():
        return {"status": "no_cuda", "reason": "torch.cuda.is_available() is false"}

    try:
        props = torch.cuda.get_device_properties(0)
        return {
            "status": "available",
            "gpu": {
                "name": str(props.name),
                "total_memory": int(props.total_memory),
                "compute_capability": f"{int(props.major)}.{int(props.minor)}",
                "sm_count": int(props.multi_processor_count),
            },
        }
    except Exception as exc:
        return {"status": "probe_failed", "reason": str(exc)}


RecordFn = Callable[[str], None]


def _workloads() -> dict[str, Callable[[str, RecordFn | None], dict[str, Any]]]:
    workloads: dict[str, Callable[[str, RecordFn | None], dict[str, Any]]] = {
        "tensor_256mb": _workload_tensor_256mb,
        "mlp_train_step": _workload_mlp_train_step,
        "tiny_transformer_step": _workload_tiny_transformer_step,
    }
    if _module_available("transformers"):
        workloads["hf_tiny_gpt2_step"] = _workload_hf_tiny_gpt2_step
    if _module_available("transformers") and _module_available("peft"):
        workloads["peft_lora_tiny_step"] = _workload_peft_lora_tiny_step
    return workloads


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _run_worker(worker: str, workload_name: str) -> dict[str, Any]:
    if worker == "fakecuda":
        import fakegpu

        fakegpu.init(runtime="fakecuda", profile="rtx3090ti", device_count=1)
        stage_ctx = fakegpu.stage(workload_name)
    else:
        stage_ctx = contextlib.nullcontext()

    import torch

    if worker == "real" and not torch.cuda.is_available():
        raise RuntimeError("real worker requires torch.cuda.is_available()")

    device = "cuda"
    if worker == "real":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            torch.cuda.reset_accumulated_memory_stats()
        except Exception:
            pass

    timeline: list[dict[str, Any]] = []

    def record(label: str) -> None:
        _record_memory_point(label, timeline)

    with stage_ctx:
        result = _workloads()[workload_name](device, record)

    if worker == "real":
        torch.cuda.synchronize()
        result["peak_memory"] = int(torch.cuda.max_memory_allocated())
        result["peak_reserved_memory"] = int(torch.cuda.max_memory_reserved())
        stats = torch.cuda.memory_stats()
        requested_peak = stats.get("requested_bytes.all.peak")
        if requested_peak is not None:
            result["requested_peak_memory"] = int(requested_peak)
        torch.cuda.empty_cache()
    else:
        result["peak_memory"] = int(torch.cuda.max_memory_allocated())
        result["peak_reserved_memory"] = int(torch.cuda.max_memory_allocated())
        result["requested_peak_memory"] = int(torch.cuda.max_memory_allocated())

    result["timeline"] = timeline
    result["worker"] = worker
    result["workload"] = workload_name
    return result


def _record_memory_point(label: str, timeline: list[dict[str, Any]]) -> None:
    import torch

    if device_is_cuda := bool(torch.cuda.is_available()):
        try:
            torch.cuda.synchronize()
        except Exception:
            device_is_cuda = False
    stats = torch.cuda.memory_stats() if device_is_cuda else {}
    point = {
        "label": label,
        "current_memory": int(torch.cuda.memory_allocated()) if device_is_cuda else 0,
        "peak_memory": int(torch.cuda.max_memory_allocated()) if device_is_cuda else 0,
    }
    for src_key, dst_key in (
        ("requested_bytes.all.current", "requested_current_memory"),
        ("requested_bytes.all.peak", "requested_peak_memory"),
        ("reserved_bytes.all.current", "reserved_current_memory"),
        ("reserved_bytes.all.peak", "reserved_peak_memory"),
    ):
        value = stats.get(src_key)
        if value is not None:
            point[dst_key] = int(value)
    timeline.append(point)


def _workload_tensor_256mb(device: str, record: RecordFn | None = None) -> dict[str, Any]:
    import torch

    elements = 64 * 1024 * 1024
    x = torch.empty((elements,), device=device, dtype=torch.float32)
    x.fill_(1.0)
    if record is not None:
        record("after_tensor_fill")
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    requested_bytes = elements * 4
    del x
    return {"requested_bytes": requested_bytes}


def _workload_mlp_train_step(device: str, record: RecordFn | None = None) -> dict[str, Any]:
    import torch

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 4096),
        torch.nn.GELU(),
        torch.nn.Linear(4096, 1024),
    ).to(device)
    if record is not None:
        record("after_model_to")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn((16, 1024), device=device)
    if record is not None:
        record("after_input")
    loss = model(x).square().mean()
    if record is not None:
        record("after_forward_loss")
    loss.backward()
    if record is not None:
        record("after_backward")
    optimizer.step()
    if record is not None:
        record("after_optimizer_step")
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    del loss, x, optimizer, model
    return {"parameter_bytes": int(parameter_bytes)}


def _workload_tiny_transformer_step(device: str, record: RecordFn | None = None) -> dict[str, Any]:
    import torch

    torch.manual_seed(0)

    vocab_size = 512
    batch_size = 4
    seq_len = 64
    hidden_size = 128
    layers = 2

    class TinyTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token = torch.nn.Embedding(vocab_size, hidden_size)
            self.position = torch.nn.Parameter(torch.zeros(1, seq_len, hidden_size))
            self.blocks = torch.nn.ModuleList(
                [
                    torch.nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=4,
                        dim_feedforward=hidden_size * 4,
                        dropout=0.0,
                        activation="gelu",
                        batch_first=True,
                    )
                    for _ in range(layers)
                ]
            )
            self.head = torch.nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = self.token(input_ids) + self.position
            if record is not None:
                record("after_embed_add")
            for block_index, block in enumerate(self.blocks):
                x = block(x)
                if record is not None:
                    record(f"after_transformer_block_{block_index}")
            return self.head(x)

    model = TinyTransformer().to(device)
    if record is not None:
        record("after_model_to")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    if record is not None:
        record("after_input")
    logits = model(input_ids)
    if record is not None:
        record("after_forward")
    loss = logits.square().mean()
    if record is not None:
        record("after_loss")
    loss.backward()
    if record is not None:
        record("after_backward")
    optimizer.step()
    if record is not None:
        record("after_optimizer_step")
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    del loss, logits, input_ids, optimizer, model
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "layers": layers,
        "parameter_bytes": int(parameter_bytes),
    }


def _workload_hf_tiny_gpt2_step(device: str, record: RecordFn | None = None) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, GPT2Config

    torch.manual_seed(0)

    vocab_size = 256
    batch_size = 2
    seq_len = 32
    hidden_size = 128
    layers = 2

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_ctx=seq_len,
        n_embd=hidden_size,
        n_layer=layers,
        n_head=4,
        n_inner=hidden_size * 2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    model = AutoModelForCausalLM.from_config(config).to(device)
    if record is not None:
        record("after_model_to")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
    if record is not None:
        record("after_input")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    if record is not None:
        record("after_forward")
    loss = outputs.logits.square().mean()
    if record is not None:
        record("after_loss")
    loss.backward()
    if record is not None:
        record("after_backward")
    optimizer.step()
    if record is not None:
        record("after_optimizer_step")
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    del loss, outputs, input_ids, attention_mask, optimizer, model
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "layers": layers,
        "vocab_size": vocab_size,
        "parameter_bytes": int(parameter_bytes),
        "uses_attention_mask": True,
        "framework": "transformers",
        "model_family": "gpt2",
    }


def _workload_peft_lora_tiny_step(device: str, record: RecordFn | None = None) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, GPT2Config

    torch.manual_seed(0)

    vocab_size = 256
    batch_size = 2
    seq_len = 32
    hidden_size = 128
    layers = 2
    lora_rank = 4

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_ctx=seq_len,
        n_embd=hidden_size,
        n_layer=layers,
        n_head=4,
        n_inner=hidden_size * 2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    base_model = AutoModelForCausalLM.from_config(config)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn", "c_proj"],
        fan_in_fan_out=True,
    )
    model = get_peft_model(base_model, lora_config).to(device)
    if record is not None:
        record("after_model_to")

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=1e-3)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
    if record is not None:
        record("after_input")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    if record is not None:
        record("after_forward")
    loss = outputs.logits.square().mean()
    if record is not None:
        record("after_loss")
    loss.backward()
    if record is not None:
        record("after_backward")
    optimizer.step()
    if record is not None:
        record("after_optimizer_step")
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    trainable_parameter_bytes = sum(p.numel() * p.element_size() for p in trainable_parameters)
    del loss, outputs, input_ids, attention_mask, optimizer, model, base_model
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "layers": layers,
        "vocab_size": vocab_size,
        "parameter_bytes": int(parameter_bytes),
        "trainable_parameter_bytes": int(trainable_parameter_bytes),
        "lora_rank": lora_rank,
        "uses_attention_mask": True,
        "framework": "peft",
        "model_family": "gpt2",
    }


def _run_real_worker(workload_name: str) -> dict[str, Any]:
    return _run_json_subprocess(
        [sys.executable, str(Path(__file__).resolve()), "--worker", "real", "--workload", workload_name],
        env=_child_env(),
    )


def _run_fakecuda_preflight(workload_name: str, *, out_dir: Path) -> dict[str, Any]:
    report_dir = out_dir / f"fakecuda_{workload_name}"
    command = [
        sys.executable,
        "-m",
        "fakegpu",
        "preflight",
        "--runtime",
        "fakecuda",
        "--profile",
        "rtx3090ti",
        "--device-count",
        "1",
        "--stage",
        workload_name,
        "--report-dir",
        str(report_dir),
        "--strict",
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "fakecuda",
        "--workload",
        workload_name,
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=_child_env(),
        text=True,
        capture_output=True,
    )
    report_path = report_dir / "preflight_report.json"
    stdout_path = report_dir / "preflight_stdout.log"
    stderr_path = report_dir / "preflight_stderr.log"
    preflight_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    worker_payload = _load_last_json_line(stdout_path)
    peak = 0
    devices = preflight_report.get("devices")
    if isinstance(devices, list) and devices:
        peak = int(devices[0].get("peak_memory", 0) or 0)
    result = dict(worker_payload)
    result.update({
        "returncode": int(completed.returncode),
        "status": preflight_report.get("status", "MISSING_REPORT"),
        "peak_memory": peak,
        "report": str(report_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    })
    return result


def _load_last_json_line(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        return payload if isinstance(payload, dict) else {}
    return {}


def _run_json_subprocess(command: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"worker produced no JSON: {' '.join(command)}")
    return json.loads(lines[-1])


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("XONSH_HISTORY_BACKEND", "dummy")
    pythonpath = str(ROOT)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    return env


def _compare_workload(name: str, real: dict[str, Any], fake: dict[str, Any]) -> dict[str, Any]:
    real_peak = int(real.get("peak_memory", 0) or 0)
    fake_peak = int(fake.get("peak_memory", 0) or 0)
    abs_error = fake_peak - real_peak
    rel_error = (abs(abs_error) / real_peak * 100.0) if real_peak > 0 else None
    factor = (real_peak / fake_peak) if fake_peak > 0 else None
    gap_analysis = _timeline_gap_analysis(real.get("timeline"), fake.get("timeline"))
    return {
        "name": name,
        "real_cuda": real,
        "fakecuda_preflight": fake,
        "peak_error_bytes": abs_error,
        "peak_error_percent": round(rel_error, 3) if rel_error is not None else None,
        "calibration_factor": round(factor, 3) if factor is not None else None,
        "gap_analysis": gap_analysis,
        "likely_gap_reason": _likely_gap_reason(rel_error, gap_analysis),
    }


def _timeline_gap_analysis(real_timeline: Any, fake_timeline: Any) -> dict[str, Any]:
    if not isinstance(real_timeline, list) or not isinstance(fake_timeline, list):
        return {"available": False, "reason": "missing timeline"}

    fake_by_label = {
        str(item.get("label")): item
        for item in fake_timeline
        if isinstance(item, dict) and item.get("label")
    }
    rows: list[dict[str, Any]] = []
    for real_item in real_timeline:
        if not isinstance(real_item, dict) or not real_item.get("label"):
            continue
        label = str(real_item["label"])
        fake_item = fake_by_label.get(label)
        if not isinstance(fake_item, dict):
            continue
        real_current = int(real_item.get("current_memory", 0) or 0)
        fake_current = int(fake_item.get("current_memory", 0) or 0)
        real_peak = int(real_item.get("peak_memory", 0) or 0)
        fake_peak = int(fake_item.get("peak_memory", 0) or 0)
        rows.append(
            {
                "label": label,
                "real_current_memory": real_current,
                "fake_current_memory": fake_current,
                "current_gap_bytes": real_current - fake_current,
                "real_peak_memory": real_peak,
                "fake_peak_memory": fake_peak,
                "peak_gap_bytes": real_peak - fake_peak,
            }
        )

    if not rows:
        return {"available": False, "reason": "no aligned timeline labels"}
    largest_current = max(rows, key=lambda row: int(row.get("current_gap_bytes", 0)))
    largest_peak = max(rows, key=lambda row: int(row.get("peak_gap_bytes", 0)))
    return {
        "available": True,
        "largest_current_gap": largest_current,
        "largest_peak_gap": largest_peak,
        "points": rows,
    }


def _likely_gap_reason(error_percent: float | None, gap_analysis: dict[str, Any]) -> str:
    if error_percent is not None and error_percent <= 15.0:
        return "within_lightweight_calibration_tolerance"
    if not gap_analysis.get("available"):
        return "untracked_cuda_backend_allocation"
    largest = gap_analysis.get("largest_current_gap")
    label = str(largest.get("label", "")) if isinstance(largest, dict) else ""
    if "transformer_block" in label or "forward" in label or "loss" in label:
        return "cuda_backend_hidden_activation_or_workspace"
    if "backward" in label:
        return "cuda_backend_backward_saved_tensor_or_workspace"
    if "optimizer" in label:
        return "cuda_optimizer_backend_hidden_allocation"
    return "untracked_cuda_backend_allocation"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RTX 3090 Ti Calibration Report",
        "",
        f"**Status:** `{report.get('status')}`",
        "",
    ]
    if str(report.get("status", "")).startswith("SKIP"):
        lines.extend([f"**Skip reason:** {report.get('skip_reason')}", ""])
        return "\n".join(lines)

    gpu = report.get("calibration_gpu") or {}
    lines.extend(
        [
            "## Calibration GPU",
            "",
            f"- name: `{gpu.get('name')}`",
            f"- compute capability: `{gpu.get('compute_capability')}`",
            f"- total memory: `{_fmt_bytes(int(gpu.get('total_memory', 0) or 0))}`",
            f"- SM count: `{gpu.get('sm_count')}`",
            "",
            "## Workloads",
            "",
            "| workload | real peak | fakecuda peak | error | error % | factor | likely reason |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in report.get("workloads", []):
        real_peak = int(item.get("real_cuda", {}).get("peak_memory", 0) or 0)
        fake_peak = int(item.get("fakecuda_preflight", {}).get("peak_memory", 0) or 0)
        err = int(item.get("peak_error_bytes", 0) or 0)
        pct = item.get("peak_error_percent")
        pct_text = "" if pct is None else f"{pct:.3f}%"
        factor = item.get("calibration_factor")
        factor_text = "" if factor is None else f"{factor:.3f}x"
        lines.append(
            f"| `{item.get('name')}` | {_fmt_bytes(real_peak)} | {_fmt_bytes(fake_peak)} | {_fmt_bytes(err)} | {pct_text} | {factor_text} | `{item.get('likely_gap_reason', '')}` |"
        )
    gap_rows = []
    for item in report.get("workloads", []):
        gap = item.get("gap_analysis") or {}
        largest = gap.get("largest_current_gap") if isinstance(gap, dict) else None
        if not isinstance(largest, dict):
            continue
        gap_rows.append(
            "| `{name}` | `{label}` | {real} | {fake} | {gap_bytes} |".format(
                name=item.get("name"),
                label=largest.get("label"),
                real=_fmt_bytes(int(largest.get("real_current_memory", 0) or 0)),
                fake=_fmt_bytes(int(largest.get("fake_current_memory", 0) or 0)),
                gap_bytes=_fmt_bytes(int(largest.get("current_gap_bytes", 0) or 0)),
            )
        )
    if gap_rows:
        lines.extend(
            [
                "",
                "## Largest Timeline Gaps",
                "",
                "| workload | point | real current | fake current | gap |",
                "|---|---|---:|---:|---:|",
                *gap_rows,
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            *[f"- {note}" for note in report.get("notes", [])],
            "",
        ]
    )
    return "\n".join(lines)


def _fmt_bytes(value: int) -> str:
    sign = "-" if value < 0 else ""
    raw = abs(int(value))
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(raw)
    unit = units[0]
    for candidate in units:
        unit = candidate
        if amount < 1024.0 or candidate == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{sign}{raw} B"
    return f"{sign}{amount:.2f} {unit}"


if __name__ == "__main__":
    raise SystemExit(main())
