#!/usr/bin/env python3
"""Run and compare a two-rank Hybrid FSDP Qwen SFT memory experiment."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WORKER = REPO_ROOT / "verification" / "qwen_fsdp_sft_memory_worker.py"
CLUSTER_CONFIG = REPO_ROOT / "verification" / "data" / "cluster_hybrid_2r.yaml"
SCHEMA_VERSION = "fakegpu.qwen_fsdp_sft_memory_comparison.v1"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_unix_socket(
    socket_path: Path,
    process: subprocess.Popen[str],
    *,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if socket_path.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"coordinator exited with code {process.returncode}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        time.sleep(0.05)
    raise TimeoutError(f"coordinator socket was not created: {socket_path}")


def _shutdown_unix_coordinator(socket_path: Path) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(5.0)
        sock.connect(str(socket_path))
        sock.sendall(b"SHUTDOWN\n")
        response = sock.recv(4096).decode("utf-8", errors="replace")
    if not response.startswith("OK "):
        raise RuntimeError(f"coordinator rejected shutdown: {response.strip()}")


def _close_coordinator(
    process: subprocess.Popen[str],
    socket_path: Path,
) -> None:
    if process.poll() is not None:
        return
    try:
        _shutdown_unix_coordinator(socket_path)
        process.wait(timeout=10)
    except Exception:
        process.kill()
        process.wait(timeout=5)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _comparison(predicted: int, observed: int) -> dict[str, int | float]:
    signed = predicted - observed
    return {
        "predicted": predicted,
        "observed": observed,
        "signed_error": signed,
        "absolute_error_bytes": abs(signed),
        "absolute_error_percent": (
            round(100.0 * abs(signed) / observed, 9) if observed else 0.0
        ),
    }


def compare_fsdp_reports(
    rank_reports: list[Mapping[str, Any]],
    static_report: Mapping[str, Any],
    cluster_report: Mapping[str, Any],
    *,
    max_error_percent: float,
) -> dict[str, Any]:
    from fakegpu.fsdp_memory import estimate_full_shard_sft_memory

    if len(rank_reports) != 2:
        raise ValueError("exactly two rank reports are required")
    if static_report.get("status") != "success" or static_report.get("mode") != "static":
        raise ValueError("a successful static Qwen SFT report is required")
    if static_report.get("training_method") != "full":
        raise ValueError("the FSDP experiment currently requires full training")

    matching_fields = (
        "model_dir",
        "dtype",
        "attention_implementation",
        "training_method",
        "gradient_checkpointing",
        "gradient_accumulation_steps",
        "batch_size",
        "sequence_length",
        "data_seed",
    )
    reference = rank_reports[0]
    for rank, report in enumerate(rank_reports):
        if report.get("status") != "success":
            raise ValueError(f"rank {rank} did not complete successfully")
        if int(report.get("rank", -1)) != rank or int(report.get("world_size", 0)) != 2:
            raise ValueError(f"rank metadata mismatch in report {rank}")
        for field in matching_fields:
            if report.get(field) != static_report.get(field):
                raise ValueError(f"rank {rank} has mismatched {field}")
        if report.get("batch", {}).get("fingerprint_sha256") != static_report.get(
            "batch", {}
        ).get("fingerprint_sha256"):
            raise ValueError(f"rank {rank} has a mismatched random batch")
        if report.get("fsdp", {}).get("sharding_plan") != reference.get(
            "fsdp", {}
        ).get("sharding_plan"):
            raise ValueError("rank reports contain different sharding plans")

    sharding_plan = reference["fsdp"]["sharding_plan"]
    projection = estimate_full_shard_sft_memory(static_report, sharding_plan)
    static_parameter_bytes = int(static_report["static_estimate"]["parameter_bytes"])
    for rank, report in enumerate(rank_reports):
        logical_parameter_bytes = int(
            report["parameters"]["logical"]["parameter_bytes"]
        )
        if logical_parameter_bytes != static_parameter_bytes:
            raise ValueError(
                f"rank {rank} logical parameters differ from static analysis"
            )

    phase_fields = {
        "graph": ("first_step_graph_peak_bytes", "graph_peak_bytes"),
        "optimizer": ("optimizer_peak_bytes", "optimizer_peak_bytes"),
        "overall": ("first_step_peak_bytes", "overall_peak_bytes"),
    }
    rank_results = []
    for rank, report in enumerate(rank_reports):
        comparisons = {
            phase: _comparison(
                int(projection[predicted_field]),
                int(report["memory_phases"][observed_field]),
            )
            for phase, (predicted_field, observed_field) in phase_fields.items()
        }
        passed = all(
            float(comparison["absolute_error_percent"]) <= max_error_percent
            for comparison in comparisons.values()
        )
        rank_results.append(
            {
                "rank": rank,
                "status": "success" if passed else "failed",
                "gpu_name": report["gpu_name"],
                "compute_capability": report["compute_capability"],
                "torch_version": report["torch_version"],
                "torch_cuda_version": report["torch_cuda_version"],
                "loss": float(report["loss"]),
                "local_parameter_bytes": int(
                    report["parameters"]["local_shard"]["parameter_bytes"]
                ),
                "local_gradient_bytes": int(
                    report["parameters"]["local_gradient_bytes"]
                ),
                "optimizer_state_tensor_bytes": int(
                    report["parameters"]["optimizer_state_tensor_bytes"]
                ),
                "comparisons": comparisons,
                "phase_seconds": report["phase_seconds"],
            }
        )

    collectives = cluster_report.get("collectives", {})
    for name in ("all_gather", "reduce_scatter"):
        if int(collectives.get(name, {}).get("calls", 0)) <= 0:
            raise ValueError(f"cluster report did not record {name}")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "success"
            if all(result["status"] == "success" for result in rank_results)
            else "failed"
        ),
        "max_error_percent": max_error_percent,
        "model_dir": reference["model_dir"],
        "dtype": reference["dtype"],
        "batch_size": int(reference["batch_size"]),
        "sequence_length": int(reference["sequence_length"]),
        "gradient_checkpointing": bool(reference["gradient_checkpointing"]),
        "gpu_name": reference["gpu_name"],
        "compute_capability": reference["compute_capability"],
        "torch_version": reference["torch_version"],
        "torch_cuda_version": reference["torch_cuda_version"],
        "projection": projection,
        "sharding_plan": sharding_plan,
        "ranks": rank_results,
        "communication": {
            "all_gather": collectives["all_gather"],
            "reduce_scatter": collectives["reduce_scatter"],
            "node_pairs": cluster_report.get("node_pairs", []),
            "operation_timeline": cluster_report.get("operation_timeline", {}),
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    projection = report["projection"]
    rows = []
    for rank in report["ranks"]:
        comparisons = rank["comparisons"]
        rows.append(
            f"| {rank['rank']} | {_gib(comparisons['graph']['observed']):.3f} | "
            f"{_gib(comparisons['graph']['predicted']):.3f} | "
            f"{comparisons['graph']['absolute_error_percent']:.3f}% | "
            f"{_gib(comparisons['optimizer']['observed']):.3f} | "
            f"{_gib(comparisons['optimizer']['predicted']):.3f} | "
            f"{comparisons['optimizer']['absolute_error_percent']:.3f}% | "
            f"{_gib(comparisons['overall']['observed']):.3f} | "
            f"{_gib(comparisons['overall']['predicted']):.3f} | "
            f"{comparisons['overall']['absolute_error_percent']:.3f}% | "
            f"{rank['status']} |"
        )

    collectives = report["communication"]
    return "\n".join(
        [
            "# Qwen Hybrid FSDP SFT Memory Validation",
            "",
            f"**Status:** `{report['status']}`",
            "",
            f"- GPU: {report['gpu_name']} (compute capability "
            f"{'.'.join(str(value) for value in report['compute_capability'])})",
            f"- PyTorch/CUDA: `{report['torch_version']}` / `{report['torch_cuda_version']}`",
            f"- Model: `{report['model_dir']}`",
            f"- Workload: batch {report['batch_size']}, sequence {report['sequence_length']}, "
            f"dtype `{report['dtype']}`",
            f"- FSDP units: {report['sharding_plan']['unit_count']}; local parameter shard "
            f"{_gib(projection['local_parameter_bytes']):.3f} GiB; largest all-gather unit "
            f"{_gib(projection['all_gather_workspace_bytes']):.3f} GiB",
            "",
            "| Rank | Graph observed GiB | Graph predicted GiB | Graph error | Optimizer observed GiB | Optimizer predicted GiB | Optimizer error | Overall observed GiB | Overall predicted GiB | Overall error | Status |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            *rows,
            "",
            "## Communication",
            "",
            f"- All-gather calls: {collectives['all_gather']['calls']}; bytes: "
            f"{collectives['all_gather']['bytes']:,}",
            f"- Reduce-scatter calls: {collectives['reduce_scatter']['calls']}; bytes: "
            f"{collectives['reduce_scatter']['bytes']:,}",
            "",
        ]
    )


def _gib(value: int) -> float:
    return int(value) / (1024**3)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    rank_dir = output_dir / "ranks"
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_dir.mkdir(parents=True, exist_ok=True)
    socket_path = output_dir / "coordinator.sock"
    cluster_report_path = output_dir / "cluster-report.json"
    comparison_path = output_dir / "comparison.json"
    markdown_path = output_dir / "comparison.md"
    for stale in (
        socket_path,
        cluster_report_path,
        comparison_path,
        markdown_path,
        rank_dir / "rank_0.json",
        rank_dir / "rank_1.json",
    ):
        stale.unlink(missing_ok=True)

    build_dir = args.build_dir.expanduser().resolve()
    coordinator_bin = build_dir / "fakegpu-coordinator"
    nccl_lib = build_dir / (
        "libnccl.dylib" if sys.platform == "darwin" else "libnccl.so.2"
    )
    static_report_path = args.static_report.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    for required in (
        coordinator_bin,
        nccl_lib,
        static_report_path,
        model_dir,
        WORKER,
        CLUSTER_CONFIG,
    ):
        if not required.exists():
            raise FileNotFoundError(f"missing required input: {required}")

    coordinator_env = dict(os.environ)
    coordinator_env.pop("LD_PRELOAD", None)
    coordinator_env.update(
        {
            "FAKEGPU_MODE": "simulate",
            "FAKEGPU_DIST_MODE": "simulate",
            "FAKEGPU_CLUSTER_CONFIG": str(CLUSTER_CONFIG),
            "FAKEGPU_COORDINATOR_TRANSPORT": "unix",
            "FAKEGPU_COORDINATOR_ADDR": str(socket_path),
            "FAKEGPU_CLUSTER_REPORT_PATH": str(cluster_report_path),
            "FAKEGPU_OPERATION_TIMELINE_LIMIT": "4096",
        }
    )
    coordinator = subprocess.Popen(
        [
            str(coordinator_bin),
            "--transport",
            "unix",
            "--address",
            str(socket_path),
        ],
        cwd=REPO_ROOT,
        env=coordinator_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_unix_socket(
            socket_path,
            coordinator,
            timeout=args.startup_timeout,
        )
        worker_env = dict(os.environ)
        worker_env["LD_PRELOAD"] = str(nccl_lib)
        worker_env.update(
            {
                "FAKEGPU_MODE": "hybrid",
                "FAKEGPU_DEVICE_COUNT": "1",
                "FAKEGPU_DIST_MODE": "simulate",
                "FAKEGPU_CLUSTER_CONFIG": str(CLUSTER_CONFIG),
                "FAKEGPU_COORDINATOR_TRANSPORT": "unix",
                "FAKEGPU_COORDINATOR_ADDR": str(socket_path),
                "FAKEGPU_COORDINATOR_TIMEOUT_MS": str(
                    max(1, round(args.timeout * 1000))
                ),
                "OMP_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--master-addr=127.0.0.1",
            f"--master-port={_find_free_port()}",
            str(WORKER),
            f"--model-dir={model_dir}",
            f"--report-dir={rank_dir}",
            f"--batch-size={args.batch_size}",
            f"--sequence-length={args.sequence_length}",
            f"--dtype={args.dtype}",
            f"--attention-implementation={args.attention_implementation}",
            f"--learning-rate={args.learning_rate}",
            f"--seed={args.seed}",
            f"--data-seed={args.data_seed}",
            f"--process-group-timeout={args.timeout}",
        ]
        if args.gradient_checkpointing:
            command.append("--gradient-checkpointing")
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=worker_env,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                f"FSDP workers exited with code {completed.returncode}"
            )

        _shutdown_unix_coordinator(socket_path)
        coordinator_stdout, coordinator_stderr = coordinator.communicate(timeout=10)
        if coordinator.returncode != 0:
            raise RuntimeError(
                f"coordinator exited with {coordinator.returncode}\n"
                f"stdout:\n{coordinator_stdout}\nstderr:\n{coordinator_stderr}"
            )

        rank_reports = [
            _load_json(rank_dir / f"rank_{rank}.json") for rank in range(2)
        ]
        static_report = _load_json(static_report_path)
        cluster_report = _load_json(cluster_report_path)
        report = compare_fsdp_reports(
            rank_reports,
            static_report,
            cluster_report,
            max_error_percent=args.max_error_percent,
        )
        report["artifacts"] = {
            "static_report": str(static_report_path),
            "cluster_report": str(cluster_report_path),
            "rank_reports": [
                str(rank_dir / f"rank_{rank}.json") for rank in range(2)
            ],
        }
        comparison_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "comparison": str(comparison_path),
                    "markdown": str(markdown_path),
                },
                indent=2,
            )
        )
        return report
    finally:
        _close_coordinator(coordinator, socket_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=REPO_ROOT / "build",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "build" / "qwen-fsdp-sft-memory",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa"),
        default="sdpa",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--data-seed", type=int, default=20260721)
    parser.add_argument("--max-error-percent", type=float, default=10.0)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least two")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero")
    if args.max_error_percent < 0:
        parser.error("--max-error-percent must be non-negative")
    if args.startup_timeout <= 0 or args.timeout <= 0:
        parser.error("timeouts must be greater than zero")

    try:
        report = _run(args)
    except Exception as exc:
        print(
            f"Qwen FSDP SFT memory validation failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
