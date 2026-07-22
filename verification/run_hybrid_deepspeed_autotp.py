#!/usr/bin/env python3
"""Run DeepSpeed AutoTP training through Hybrid Fake NCCL."""

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

from verification.deepspeed_autotp_worker import (  # noqa: E402
    EXPECTED_COLUMN,
    EXPECTED_LOSS,
    EXPECTED_OUTPUT,
    EXPECTED_ROW,
    _tensor_close,
)
from verification.run_hybrid_deepspeed_numerics import (  # noqa: E402
    _close_coordinator,
    _find_free_port,
    _load_rank_reports,
    _temporary_coordinator_socket,
    _wait_for_unix_socket,
)


WORKER = ROOT / "verification" / "deepspeed_autotp_worker.py"
CLUSTER_CONFIG = ROOT / "verification" / "data" / "cluster_hybrid_2r.yaml"


def _validate_rank_reports(
    reports: list[dict[str, Any]],
    *,
    zero_stage: int,
    precision: str,
) -> None:
    if len(reports) != 2:
        raise AssertionError(f"expected two AutoTP reports, got {len(reports)}")
    tolerance = 1e-6 if precision == "fp32" else 2e-2
    expected_metadata = {
        "world_size": 2,
        "zero_stage": zero_stage,
        "effective_zero_stage": zero_stage,
        "precision": precision,
        "engine_type": "DeepSpeedEngine",
        "autotp_size": 2,
        "tensor_parallel_world_size": 2,
        "column_local_shape": [2, 4],
        "row_local_shape": [2, 2],
        "global_steps": 1,
    }
    for rank, report in enumerate(reports):
        if report.get("status") != "success":
            raise AssertionError(f"AutoTP rank {rank} failed: {report}")
        for field, expected in expected_metadata.items():
            if report.get(field) != expected:
                raise AssertionError(
                    f"AutoTP rank {rank} {field} mismatch: "
                    f"{report.get(field)} != {expected}"
                )
        if report.get("rank") != rank:
            raise AssertionError(f"AutoTP rank {rank} identity mismatch")
        if report.get("tensor_parallel_rank") != rank:
            raise AssertionError(f"AutoTP rank {rank} TP rank mismatch")
        if not str(report.get("column_layer_type", "")).startswith("Linear"):
            raise AssertionError(f"AutoTP rank {rank} column was not replaced")
        if not str(report.get("row_layer_type", "")).startswith("Linear"):
            raise AssertionError(f"AutoTP rank {rank} row was not replaced")
        if not _tensor_close(report.get("output"), EXPECTED_OUTPUT, tolerance):
            raise AssertionError(f"AutoTP rank {rank} output mismatch")
        if abs(float(report.get("loss", float("nan"))) - EXPECTED_LOSS) > tolerance:
            raise AssertionError(f"AutoTP rank {rank} loss mismatch")
        if not _tensor_close(
            report.get("full_column_weight_after_step"),
            EXPECTED_COLUMN,
            tolerance,
        ):
            raise AssertionError(f"AutoTP rank {rank} column update mismatch")
        if not _tensor_close(
            report.get("full_row_weight_after_step"),
            EXPECTED_ROW,
            tolerance,
        ):
            raise AssertionError(f"AutoTP rank {rank} row update mismatch")


def _validate_communication(cluster_report: dict[str, Any]) -> dict[str, int]:
    collectives = cluster_report.get("collectives", {})
    summary = {
        name: int(collectives.get(name, {}).get("calls", 0))
        for name in ("all_reduce", "all_gather")
    }
    if min(summary.values()) <= 0:
        raise AssertionError(f"AutoTP collective accounting is incomplete: {summary}")
    node_pairs = cluster_report.get("node_pairs", [])
    if len(node_pairs) != 1 or int(node_pairs[0].get("total_bytes", 0)) <= 0:
        raise AssertionError(f"AutoTP node-pair accounting is empty: {node_pairs}")
    return summary


def _run_case(
    *,
    build_dir: Path,
    report_root: Path,
    zero_stage: int,
    precision: str,
    timeout: float,
) -> dict[str, Any]:
    coordinator_bin = build_dir / "fakegpu-coordinator"
    nccl_lib = (
        build_dir / "libnccl.dylib"
        if sys.platform == "darwin"
        else build_dir / "libnccl.so.2"
    )
    case_root = report_root / f"zero{zero_stage}-{precision}"
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
            "--zero-stage",
            str(zero_stage),
            "--precision",
            precision,
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
                f"DeepSpeed AutoTP worker exited with code {completed.returncode}"
            )

        reports = _load_rank_reports(rank_report_dir, 2)
        _validate_rank_reports(
            reports,
            zero_stage=zero_stage,
            precision=precision,
        )

        from verification.run_hybrid_deepspeed_numerics import (  # noqa: PLC0415
            _shutdown_unix_coordinator,
        )

        _shutdown_unix_coordinator(socket_path)
        coordinator.wait(timeout=5)
        cluster_report = json.loads(
            cluster_report_path.read_text(encoding="utf-8")
        )
        communication = _validate_communication(cluster_report)
        first = reports[0]
        pair = cluster_report["node_pairs"][0]
        return {
            "status": "success",
            "zero_stage": zero_stage,
            "precision": precision,
            "deepspeed_version": first["deepspeed_version"],
            "torch_version": first["torch_version"],
            "torch_cuda_version": first["torch_cuda_version"],
            "physical_device_name": first["physical_device_name"],
            "physical_compute_capability": first[
                "physical_compute_capability"
            ],
            "output": first["output"],
            "loss": first["loss"],
            "full_column_weight_after_step": first[
                "full_column_weight_after_step"
            ],
            "full_row_weight_after_step": first[
                "full_row_weight_after_step"
            ],
            "peak_allocated_bytes_by_rank": [
                int(report["peak_allocated_bytes"]) for report in reports
            ],
            "collective_calls": communication,
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
        "--zero-stage",
        choices=("0", "1", "2", "all"),
        default="0",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "all"),
        default="fp32",
    )
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
        / "deepspeed_autotp"
        / time.strftime("%Y%m%d-%H%M%S")
    )
    zero_stages = (0, 1, 2) if args.zero_stage == "all" else (int(args.zero_stage),)
    precisions = ("fp32", "bf16") if args.precision == "all" else (args.precision,)
    results = [
        _run_case(
            build_dir=build_dir,
            report_root=report_root,
            zero_stage=zero_stage,
            precision=precision,
            timeout=args.timeout,
        )
        for zero_stage in zero_stages
        for precision in precisions
    ]
    summary = {
        "schema_version": "fakegpu.deepspeed_autotp.v1",
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
