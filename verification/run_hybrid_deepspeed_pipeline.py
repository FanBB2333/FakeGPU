#!/usr/bin/env python3
"""Run DeepSpeed Pipeline Parallel through Hybrid Fake NCCL."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verification.run_hybrid_deepspeed_numerics import (  # noqa: E402
    _close_coordinator,
    _find_free_port,
    _load_rank_reports,
    _temporary_coordinator_socket,
    _wait_for_unix_socket,
)


WORKER = ROOT / "verification" / "deepspeed_pipeline_worker.py"
CLUSTER_CONFIG = ROOT / "verification" / "data" / "cluster_hybrid_2r.yaml"
EXPECTED_FINAL = [
    [0.45, -0.35, -0.55, 0.65],
    [0.45, 0.65],
]


def _vector_close(
    actual: object,
    expected: list[float],
    tolerance: float,
) -> bool:
    return (
        isinstance(actual, list)
        and len(actual) == len(expected)
        and all(
            abs(float(actual_value) - expected_value) <= tolerance
            for actual_value, expected_value in zip(actual, expected)
        )
    )


def _validate_rank_reports(
    reports: list[dict[str, Any]],
    *,
    precision: str,
    activation_checkpoint_interval: int,
) -> None:
    tolerance = 1e-6 if precision == "fp32" else 2e-2
    if len(reports) != 2:
        raise AssertionError(f"expected two pipeline reports, got {len(reports)}")
    for rank, report in enumerate(reports):
        if report.get("status") != "success":
            raise AssertionError(f"pipeline rank {rank} failed: {report}")
        if report.get("engine_type") != "PipelineEngine":
            raise AssertionError(f"rank {rank} did not create PipelineEngine")
        if report.get("pipe_parallel_size") != 2:
            raise AssertionError(f"rank {rank} pipeline size mismatch")
        if report.get("pipe_stage_id") != rank:
            raise AssertionError(f"rank {rank} pipeline stage mismatch")
        if report.get("gradient_accumulation_steps") != 2:
            raise AssertionError(f"rank {rank} accumulation mismatch")
        if report.get("global_steps") != 1:
            raise AssertionError(f"rank {rank} optimizer step mismatch")
        if report.get("activation_checkpoint_interval") != (
            activation_checkpoint_interval
        ):
            raise AssertionError(f"rank {rank} checkpoint interval mismatch")
        if abs(float(report.get("loss", float("nan"))) - 4.25) > tolerance:
            raise AssertionError(f"rank {rank} pipeline loss mismatch")
        all_parameters = report.get("all_stage_parameters")
        if not isinstance(all_parameters, list) or len(all_parameters) != 2:
            raise AssertionError(f"rank {rank} parameter report is incomplete")
        for stage_id, expected in enumerate(EXPECTED_FINAL):
            if not _vector_close(
                all_parameters[stage_id],
                expected,
                tolerance,
            ):
                raise AssertionError(
                    f"rank {rank} stage {stage_id} parameter mismatch: "
                    f"{all_parameters[stage_id]} != {expected}"
                )


def _validate_communication(cluster_report: dict[str, Any]) -> dict[str, int]:
    p2p = cluster_report.get("point_to_point", {})
    summary = {
        "operations": int(p2p.get("operations", 0)),
        "sends": int(p2p.get("sends", 0)),
        "bytes": int(p2p.get("bytes", 0)),
    }
    if min(summary.values()) <= 0:
        raise AssertionError(f"pipeline P2P communication is empty: {p2p}")
    ranks = cluster_report.get("ranks", [])
    if len(ranks) != 2 or any(
        int(rank.get("point_to_point_calls", 0)) <= 0 for rank in ranks
    ):
        raise AssertionError(f"pipeline rank P2P accounting is empty: {ranks}")
    node_pairs = cluster_report.get("node_pairs", [])
    if len(node_pairs) != 1:
        raise AssertionError(f"expected one pipeline node pair: {node_pairs}")
    if int(node_pairs[0].get("point_to_point_operations", 0)) <= 0:
        raise AssertionError(f"node-pair P2P accounting is empty: {node_pairs}")
    return summary


def _run_case(
    *,
    build_dir: Path,
    report_root: Path,
    precision: str,
    activation_checkpoint_interval: int,
    timeout: float,
) -> dict[str, Any]:
    coordinator_bin = build_dir / "fakegpu-coordinator"
    nccl_lib = (
        build_dir / "libnccl.dylib"
        if sys.platform == "darwin"
        else build_dir / "libnccl.so.2"
    )
    case_root = report_root / (
        f"{precision}-checkpoint{activation_checkpoint_interval}"
    )
    case_root.mkdir(parents=True, exist_ok=True)
    rank_report_dir = case_root / "ranks"
    cluster_report_path = case_root / "cluster-report.json"
    socket_directory, socket_path = _temporary_coordinator_socket()

    coordinator_env = dict(os.environ)
    coordinator_env.pop("LD_PRELOAD", None)
    coordinator_env.pop("DYLD_INSERT_LIBRARIES", None)
    coordinator_env.update(
        {
            "FAKEGPU_MODE": "simulate",
            "FAKEGPU_DIST_MODE": "simulate",
            "FAKEGPU_CLUSTER_CONFIG": str(CLUSTER_CONFIG),
            "FAKEGPU_COORDINATOR_TRANSPORT": "unix",
            "FAKEGPU_COORDINATOR_ADDR": str(socket_path),
            "FAKEGPU_CLUSTER_REPORT_PATH": str(cluster_report_path),
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
        cwd=ROOT,
        env=coordinator_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_unix_socket(socket_path, coordinator)
        worker_env = dict(os.environ)
        worker_env.pop("DYLD_INSERT_LIBRARIES", None)
        worker_env["LD_PRELOAD"] = str(nccl_lib)
        worker_env.update(
            {
                "FAKEGPU_MODE": "hybrid",
                "FAKEGPU_DIST_MODE": "simulate",
                "FAKEGPU_CLUSTER_CONFIG": str(CLUSTER_CONFIG),
                "FAKEGPU_COORDINATOR_TRANSPORT": "unix",
                "FAKEGPU_COORDINATOR_ADDR": str(socket_path),
                "FAKEGPU_COORDINATOR_TIMEOUT_MS": "90000",
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
            "--report-dir",
            str(rank_report_dir),
            "--precision",
            precision,
            "--activation-checkpoint-interval",
            str(activation_checkpoint_interval),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=worker_env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
            raise RuntimeError(
                f"DeepSpeed pipeline worker exited with "
                f"code {completed.returncode}"
            )

        reports = _load_rank_reports(rank_report_dir, 2)
        _validate_rank_reports(
            reports,
            precision=precision,
            activation_checkpoint_interval=activation_checkpoint_interval,
        )

        from verification.run_hybrid_deepspeed_numerics import (  # noqa: PLC0415
            _shutdown_unix_coordinator,
        )

        _shutdown_unix_coordinator(socket_path)
        coordinator.wait(timeout=5)
        cluster_report = json.loads(
            cluster_report_path.read_text(encoding="utf-8")
        )
        p2p = _validate_communication(cluster_report)
        pair = cluster_report["node_pairs"][0]
        first = reports[0]
        return {
            "status": "success",
            "precision": precision,
            "activation_checkpoint_interval": activation_checkpoint_interval,
            "deepspeed_version": first["deepspeed_version"],
            "torch_version": first["torch_version"],
            "torch_cuda_version": first["torch_cuda_version"],
            "physical_device_name": first["physical_device_name"],
            "physical_compute_capability": first[
                "physical_compute_capability"
            ],
            "loss": first["loss"],
            "all_stage_parameters": first["all_stage_parameters"],
            "peak_allocated_bytes_by_rank": [
                int(report["peak_allocated_bytes"]) for report in reports
            ],
            "point_to_point": p2p,
            "node_pair_total_bytes": int(pair["total_bytes"]),
            "node_pair_peak_bytes_per_operation": int(
                pair["peak_combined_bytes_per_operation"]
            ),
            "cluster_markdown_report": cluster_report["cluster"][
                "markdown_report_path"
            ],
        }
    finally:
        _close_coordinator(coordinator, socket_path)
        coordinator.communicate(timeout=5)
        socket_directory.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "all"),
        default="fp32",
    )
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args(argv)

    if importlib.util.find_spec("deepspeed") is None:
        print(
            "DeepSpeed is not installed in the selected Python environment: "
            f"{sys.executable}",
            file=sys.stderr,
        )
        return 2
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    build_dir = args.build_dir.resolve()
    required = [
        build_dir / "fakegpu-coordinator",
        (
            build_dir / "libnccl.dylib"
            if sys.platform == "darwin"
            else build_dir / "libnccl.so.2"
        ),
        CLUSTER_CONFIG,
        WORKER,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("missing required artifacts: " + ", ".join(missing), file=sys.stderr)
        return 2

    report_root = (
        args.report_dir.resolve()
        if args.report_dir is not None
        else ROOT
        / "build"
        / "deepspeed_pipeline"
        / time.strftime("%Y%m%d-%H%M%S")
    )
    precisions = ("fp32", "bf16") if args.precision == "all" else (args.precision,)
    checkpoint_intervals = (0, 1) if args.activation_checkpointing else (0,)
    results = [
        _run_case(
            build_dir=build_dir,
            report_root=report_root,
            precision=precision,
            activation_checkpoint_interval=interval,
            timeout=args.timeout,
        )
        for precision in precisions
        for interval in checkpoint_intervals
    ]
    summary = {
        "schema_version": "fakegpu.deepspeed_pipeline.v1",
        "status": "success",
        "results": results,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    summary_path = report_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
