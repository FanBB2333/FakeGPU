#!/usr/bin/env python3
"""Calibrate small RTX 3090 Ti real-CUDA workloads against fakecuda preflight."""

from __future__ import annotations

import argparse
import contextlib
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


def _workloads() -> dict[str, Callable[[str], dict[str, Any]]]:
    return {
        "tensor_256mb": _workload_tensor_256mb,
        "mlp_train_step": _workload_mlp_train_step,
    }


def _run_worker(worker: str, workload_name: str) -> dict[str, Any]:
    import torch

    if worker == "real" and not torch.cuda.is_available():
        raise RuntimeError("real worker requires torch.cuda.is_available()")

    if worker == "fakecuda":
        import fakegpu

        stage_ctx = fakegpu.stage(workload_name)
    else:
        stage_ctx = contextlib.nullcontext()

    device = "cuda"
    if worker == "real":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with stage_ctx:
        result = _workloads()[workload_name](device)

    if worker == "real":
        torch.cuda.synchronize()
        result["peak_memory"] = int(torch.cuda.max_memory_allocated())
        torch.cuda.empty_cache()
    else:
        result["peak_memory"] = int(torch.cuda.max_memory_allocated())

    result["worker"] = worker
    result["workload"] = workload_name
    return result


def _workload_tensor_256mb(device: str) -> dict[str, Any]:
    import torch

    elements = 64 * 1024 * 1024
    x = torch.empty((elements,), device=device, dtype=torch.float32)
    x.fill_(1.0)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    requested_bytes = elements * 4
    del x
    return {"requested_bytes": requested_bytes}


def _workload_mlp_train_step(device: str) -> dict[str, Any]:
    import torch

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 4096),
        torch.nn.GELU(),
        torch.nn.Linear(4096, 1024),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn((16, 1024), device=device)
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    del loss, x, optimizer, model
    return {"parameter_bytes": int(parameter_bytes)}


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
    preflight_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    peak = 0
    devices = preflight_report.get("devices")
    if isinstance(devices, list) and devices:
        peak = int(devices[0].get("peak_memory", 0) or 0)
    return {
        "returncode": int(completed.returncode),
        "status": preflight_report.get("status", "MISSING_REPORT"),
        "peak_memory": peak,
        "report": str(report_path),
        "stdout": str(report_dir / "preflight_stdout.log"),
        "stderr": str(report_dir / "preflight_stderr.log"),
    }


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
    return {
        "name": name,
        "real_cuda": real,
        "fakecuda_preflight": fake,
        "peak_error_bytes": abs_error,
        "peak_error_percent": round(rel_error, 3) if rel_error is not None else None,
    }


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
            "| workload | real peak | fakecuda peak | error | error % |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in report.get("workloads", []):
        real_peak = int(item.get("real_cuda", {}).get("peak_memory", 0) or 0)
        fake_peak = int(item.get("fakecuda_preflight", {}).get("peak_memory", 0) or 0)
        err = int(item.get("peak_error_bytes", 0) or 0)
        pct = item.get("peak_error_percent")
        pct_text = "" if pct is None else f"{pct:.3f}%"
        lines.append(
            f"| `{item.get('name')}` | {_fmt_bytes(real_peak)} | {_fmt_bytes(fake_peak)} | {_fmt_bytes(err)} | {pct_text} |"
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
