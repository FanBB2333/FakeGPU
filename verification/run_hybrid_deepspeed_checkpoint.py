#!/usr/bin/env python3
"""Validate DeepSpeed ZeRO checkpoint save, resume, and consolidation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "test" / "test_deepspeed_checkpoint_worker.py"
CLUSTER_CONFIGS = {
    2: ROOT / "verification" / "data" / "cluster_hybrid_2r.yaml",
    4: ROOT / "verification" / "data" / "cluster_hybrid_4r.yaml",
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verification.run_hybrid_deepspeed_numerics import (  # noqa: E402
    _close_coordinator,
    _collective_calls,
    _find_free_port,
    _load_rank_reports,
    _shutdown_unix_coordinator,
    _validate_collective_calls,
    _wait_for_unix_socket,
)


SCHEMA_VERSION = "fakegpu.hybrid_deepspeed_checkpoint.v1"


def _matrix_close(
    actual: object,
    expected: object,
    tolerance: float,
) -> bool:
    if not isinstance(actual, list) or not isinstance(expected, list):
        return False
    if len(actual) != len(expected):
        return False
    for actual_row, expected_row in zip(actual, expected):
        if not isinstance(actual_row, list) or not isinstance(
            expected_row,
            list,
        ):
            return False
        if len(actual_row) != len(expected_row):
            return False
        if any(
            abs(float(actual_value) - float(expected_value)) > tolerance
            for actual_value, expected_value in zip(
                actual_row,
                expected_row,
            )
        ):
            return False
    return True


def validate_checkpoint_rank_reports(
    reports: list[dict[str, Any]],
    *,
    world_size: int,
    zero_stage: int,
    precision: str,
) -> None:
    if len(reports) != world_size:
        raise AssertionError(
            f"expected {world_size} rank reports, got {len(reports)}"
        )
    tolerance = 1e-6 if precision == "fp32" else 2e-2
    reference = reports[0]
    reference_parameter = reference.get("parameters", {}).get(
        "resumed_after_second"
    )
    reference_files = reference.get("checkpoint_files")
    for rank, report in enumerate(reports):
        if report.get("status") != "success":
            raise AssertionError(
                f"DeepSpeed checkpoint rank {rank} failed: {report}"
            )
        if int(report.get("rank", -1)) != rank:
            raise AssertionError(f"report rank mismatch: {report}")
        if int(report.get("zero_stage", -1)) != zero_stage:
            raise AssertionError(f"ZeRO stage mismatch: {report}")
        if int(report.get("effective_zero_stage", -1)) != zero_stage:
            raise AssertionError(f"effective ZeRO stage mismatch: {report}")
        if report.get("precision") != precision:
            raise AssertionError(f"precision mismatch: {report}")
        if int(report.get("global_steps_after_resume", -1)) != 2:
            raise AssertionError(f"global step was not restored: {report}")
        client_state = report.get("client_state", {})
        if client_state.get("validation_token") != (
            "fakegpu-deepspeed-checkpoint-v1"
        ):
            raise AssertionError(f"client state was not restored: {report}")
        if int(client_state.get("saved_global_steps", -1)) != 1:
            raise AssertionError(f"client step was not restored: {report}")
        learning_rates = report.get("learning_rates", {})
        if not (
            float(learning_rates.get("saved", -1))
            == float(learning_rates.get("restored", -2))
            and float(
                learning_rates.get("uninterrupted_after_second", -1)
            )
            == float(learning_rates.get("resumed_after_second", -2))
        ):
            raise AssertionError(f"scheduler state mismatch: {report}")
        parameters = report.get("parameters", {})
        if not _matrix_close(
            parameters.get("saved"),
            parameters.get("restored"),
            tolerance,
        ):
            raise AssertionError(f"model state mismatch: {report}")
        if not _matrix_close(
            parameters.get("uninterrupted_after_second"),
            parameters.get("resumed_after_second"),
            tolerance,
        ):
            raise AssertionError(f"optimizer state mismatch: {report}")
        if not _matrix_close(
            parameters.get("resumed_after_second"),
            reference_parameter,
            tolerance,
        ):
            raise AssertionError(
                f"rank {rank} resumed parameter differs: {report}"
            )
        if report.get("checkpoint_files") != reference_files:
            raise AssertionError(
                f"rank {rank} saw a different checkpoint inventory"
            )
    if not isinstance(reference_files, list) or "latest" not in reference_files:
        raise AssertionError(
            f"checkpoint inventory is incomplete: {reference_files}"
        )
    if not any("model_states.pt" in path for path in reference_files):
        raise AssertionError(
            f"checkpoint has no model state: {reference_files}"
        )
    if not any("optim_states.pt" in path for path in reference_files):
        raise AssertionError(
            f"checkpoint has no optimizer state: {reference_files}"
        )


def _consolidate_zero_checkpoint(
    checkpoint_dir: Path,
    reports: list[dict[str, Any]],
    *,
    precision: str,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415
    from deepspeed.runtime.fp16.loss_scaler import LossScaler  # noqa: PLC0415
    from deepspeed.runtime.zero.config import (  # noqa: PLC0415
        ZeroStageEnum,
    )
    from deepspeed.utils.zero_to_fp32 import (  # noqa: PLC0415
        get_fp32_state_dict_from_zero_checkpoint,
    )

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([ZeroStageEnum, LossScaler])
    state_dict = get_fp32_state_dict_from_zero_checkpoint(
        str(checkpoint_dir),
        tag="global_step1",
    )
    if "weight" not in state_dict:
        raise AssertionError(
            f"consolidated state dict does not contain weight: {state_dict.keys()}"
        )
    weight = state_dict["weight"].detach().float().cpu().tolist()
    expected = reports[0]["parameters"]["saved"]
    tolerance = 1e-6 if precision == "fp32" else 2e-2
    if not _matrix_close(weight, expected, tolerance):
        raise AssertionError(
            f"consolidated parameter mismatch: {weight} != {expected}"
        )
    return {
        "status": "success",
        "keys": sorted(state_dict),
        "weight": weight,
        "dtype": str(state_dict["weight"].dtype),
    }


def _run_stage(
    *,
    build_dir: Path,
    report_root: Path,
    world_size: int,
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
    cluster_config = CLUSTER_CONFIGS[world_size]
    stage_root = report_root / f"zero{zero_stage}-{precision}-{world_size}r"
    stage_root.mkdir(parents=True, exist_ok=True)
    socket_path = stage_root / "coordinator.sock"
    cluster_report_path = stage_root / "cluster-report.json"
    rank_report_dir = stage_root / "ranks"
    checkpoint_dir = stage_root / "checkpoint"

    coordinator_env = dict(os.environ)
    coordinator_env.pop("LD_PRELOAD", None)
    coordinator_env.pop("DYLD_INSERT_LIBRARIES", None)
    coordinator_env.update(
        {
            "FAKEGPU_MODE": "simulate",
            "FAKEGPU_DIST_MODE": "simulate",
            "FAKEGPU_CLUSTER_CONFIG": str(cluster_config),
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
                "FAKEGPU_CLUSTER_CONFIG": str(cluster_config),
                "FAKEGPU_COORDINATOR_TRANSPORT": "unix",
                "FAKEGPU_COORDINATOR_ADDR": str(socket_path),
                "FAKEGPU_COORDINATOR_TIMEOUT_MS": "180000",
            }
        )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            f"--nproc-per-node={world_size}",
            "--master-addr=127.0.0.1",
            f"--master-port={_find_free_port()}",
            str(WORKER),
            "--report-dir",
            str(rank_report_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
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
                f"DeepSpeed ZeRO-{zero_stage} checkpoint workers exited "
                f"with code {completed.returncode}"
            )

        rank_reports = _load_rank_reports(rank_report_dir, world_size)
        validate_checkpoint_rank_reports(
            rank_reports,
            world_size=world_size,
            zero_stage=zero_stage,
            precision=precision,
        )
        consolidation = None
        if zero_stage in (2, 3):
            consolidation = _consolidate_zero_checkpoint(
                checkpoint_dir,
                rank_reports,
                precision=precision,
            )

        _shutdown_unix_coordinator(socket_path)
        coordinator.wait(timeout=5)
        cluster_report = json.loads(
            cluster_report_path.read_text(encoding="utf-8")
        )
        calls = _collective_calls(cluster_report)
        _validate_collective_calls(calls, zero_stage)
        node_pairs = cluster_report.get("node_pairs")
        if not isinstance(node_pairs, list) or not node_pairs:
            raise AssertionError("checkpoint node-pair report is empty")
        total_bytes = sum(
            int(pair.get("total_bytes", 0)) for pair in node_pairs
        )
        peak_bytes = max(
            int(pair.get("peak_combined_bytes_per_operation", 0))
            for pair in node_pairs
        )
        if total_bytes <= 0 or peak_bytes <= 0:
            raise AssertionError(
                f"checkpoint communication was empty: {node_pairs}"
            )

        first = rank_reports[0]
        return {
            "status": "success",
            "zero_stage": zero_stage,
            "precision": precision,
            "world_size": world_size,
            "torch_version": first["torch_version"],
            "torch_cuda_version": first["torch_cuda_version"],
            "deepspeed_version": first["deepspeed_version"],
            "physical_device_name": first["physical_device_name"],
            "physical_compute_capability": first[
                "physical_compute_capability"
            ],
            "checkpoint_files": first["checkpoint_files"],
            "parameters": first["parameters"],
            "learning_rates": first["learning_rates"],
            "consolidation": consolidation,
            "collective_calls": calls,
            "node_pair_total_bytes": total_bytes,
            "node_pair_peak_bytes_per_operation": peak_bytes,
        }
    finally:
        _close_coordinator(coordinator, socket_path)
        coordinator.communicate(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--world-size", type=int, choices=(2, 4), default=2)
    parser.add_argument(
        "--zero-stage",
        choices=("0", "1", "2", "3", "all"),
        default="all",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    if importlib.util.find_spec("deepspeed") is None:
        print(
            "DeepSpeed is not installed in the selected Python environment: "
            f"{sys.executable}",
            file=sys.stderr,
        )
        return 2

    build_dir = args.build_dir.resolve()
    nccl_lib = (
        build_dir / "libnccl.dylib"
        if sys.platform == "darwin"
        else build_dir / "libnccl.so.2"
    )
    for artifact in (
        build_dir / "fakegpu-coordinator",
        nccl_lib,
        WORKER,
        CLUSTER_CONFIGS[args.world_size],
    ):
        if not artifact.exists():
            print(f"missing required artifact: {artifact}", file=sys.stderr)
            return 2

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.report_dir is None:
        temporary = tempfile.TemporaryDirectory(
            prefix="fakegpu-hybrid-deepspeed-checkpoint-"
        )
        report_root = Path(temporary.name)
    else:
        report_root = args.report_dir.resolve()
        report_root.mkdir(parents=True, exist_ok=True)

    stages = (
        (0, 1, 2, 3)
        if args.zero_stage == "all"
        else (int(args.zero_stage),)
    )
    try:
        results = [
            _run_stage(
                build_dir=build_dir,
                report_root=report_root,
                world_size=args.world_size,
                zero_stage=stage,
                precision=args.precision,
                timeout=args.timeout,
            )
            for stage in stages
        ]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "success",
            "world_size": args.world_size,
            "precision": args.precision,
            "zero_stages": list(stages),
            "results": results,
        }
        if args.report_dir is not None:
            (report_root / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
