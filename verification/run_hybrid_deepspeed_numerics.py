#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "test" / "test_deepspeed_hybrid_numerics.py"
CLUSTER_CONFIGS = {
    2: REPO_ROOT / "verification" / "data" / "cluster_hybrid_2r.yaml",
    4: REPO_ROOT / "verification" / "data" / "cluster_hybrid_4r.yaml",
}
COLLECTIVE_NAMES = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "reduce",
    "all_to_all",
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _shutdown_unix_coordinator(socket_path: Path) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        sock.connect(str(socket_path))
        sock.sendall(b"SHUTDOWN\n")
        sock.recv(4096)


def _wait_for_unix_socket(
    socket_path: Path,
    process: subprocess.Popen[str],
) -> None:
    deadline = time.monotonic() + 5.0
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


def _close_coordinator(
    coordinator: subprocess.Popen[str],
    socket_path: Path,
) -> None:
    if coordinator.poll() is not None:
        return
    try:
        _shutdown_unix_coordinator(socket_path)
        coordinator.wait(timeout=3)
    except Exception:
        coordinator.kill()
        coordinator.wait(timeout=5)


def _load_rank_reports(
    report_dir: Path,
    world_size: int,
) -> list[dict[str, Any]]:
    reports = []
    for rank in range(world_size):
        path = report_dir / f"rank_{rank}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing rank report: {path}")
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


def _matrix_close(
    actual: object,
    expected: list[list[float]],
    tolerance: float,
) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for actual_row, expected_row in zip(actual, expected):
        if not isinstance(actual_row, list) or len(actual_row) != len(
            expected_row
        ):
            return False
        if any(
            abs(float(actual_value) - expected_value) > tolerance
            for actual_value, expected_value in zip(actual_row, expected_row)
        ):
            return False
    return True


def _validate_rank_reports(
    reports: list[dict[str, Any]],
    *,
    world_size: int,
    zero_stage: int,
    precision: str,
    offload: str = "none",
) -> None:
    rank_average = (world_size + 1.0) / 2.0
    gradient_scale = rank_average * 1.5
    expected_final = [
        [1.0 - 0.1 * gradient_scale, -0.2 * gradient_scale]
    ]
    tolerance = 1e-6 if precision == "fp32" else 1e-2
    for rank, report in enumerate(reports):
        if report.get("status") != "success":
            raise AssertionError(
                f"DeepSpeed ZeRO-{zero_stage} rank {rank} failed: {report}"
            )
        if report.get("zero_stage") != zero_stage:
            raise AssertionError(
                f"rank {rank} reported ZeRO-{report.get('zero_stage')}"
            )
        if report.get("effective_zero_stage") != zero_stage:
            raise AssertionError(
                f"rank {rank} constructed the wrong ZeRO stage: {report}"
            )
        if report.get("gradient_accumulation_steps") != 2:
            raise AssertionError(
                f"rank {rank} did not configure gradient accumulation"
            )
        offload_report = report.get("offload", {})
        optimizer_requested = offload != "none"
        parameters_requested = offload == "optimizer-and-parameter"
        if bool(offload_report.get("optimizer_requested")) != (
            optimizer_requested
        ):
            raise AssertionError(f"rank {rank} optimizer offload mismatch")
        if bool(offload_report.get("parameters_requested")) != (
            parameters_requested
        ):
            raise AssertionError(f"rank {rank} parameter offload mismatch")
        if optimizer_requested:
            state_devices = offload_report.get("optimizer_state_devices")
            if state_devices != ["cpu"]:
                raise AssertionError(
                    f"rank {rank} optimizer state was not offloaded: "
                    f"{state_devices}"
                )
        if parameters_requested and offload_report.get(
            "local_parameter_partition_device"
        ) != "cpu":
            raise AssertionError(
                f"rank {rank} parameter partition was not offloaded: "
                f"{offload_report}"
            )
        if report.get("micro_step_1_global_steps") != 0:
            raise AssertionError(
                f"rank {rank} updated before the accumulation boundary"
            )
        if report.get("micro_step_2_global_steps") != 1:
            raise AssertionError(
                f"rank {rank} did not update at the accumulation boundary"
            )
        parameters = report.get("parameters_after_micro_steps")
        if not isinstance(parameters, list) or len(parameters) != 2:
            raise AssertionError(
                f"rank {rank} did not report two micro steps: {parameters}"
            )
        if not _matrix_close(parameters[0], [[1.0, 0.0]], tolerance):
            raise AssertionError(
                f"rank {rank} changed parameters too early: {parameters[0]}"
            )
        if not _matrix_close(parameters[1], expected_final, tolerance):
            raise AssertionError(
                f"rank {rank} parameter mismatch: "
                f"{parameters[1]} != {expected_final}"
            )
        gathered = report.get("gathered_parameters")
        if not isinstance(gathered, list) or len(gathered) != world_size:
            raise AssertionError(
                f"rank {rank} cross-rank result is incomplete: {gathered}"
            )
        if any(
            not (
                isinstance(value, list)
                and len(value) == 2
                and abs(float(value[0]) - expected_final[0][0]) <= tolerance
                and abs(float(value[1]) - expected_final[0][1]) <= tolerance
            )
            for value in gathered
        ):
            raise AssertionError(
                f"rank {rank} observed inconsistent parameters: {gathered}"
            )


def _collective_calls(cluster_report: dict[str, Any]) -> dict[str, int]:
    collectives = cluster_report.get("collectives", {})
    return {
        name: int(collectives.get(name, {}).get("calls", 0))
        for name in COLLECTIVE_NAMES
    }


def _validate_collective_calls(
    calls: dict[str, int],
    zero_stage: int,
) -> None:
    if calls["broadcast"] < 1:
        raise AssertionError(
            f"DeepSpeed ZeRO-{zero_stage} did not broadcast model parameters"
        )
    if calls["all_reduce"] + calls["reduce_scatter"] < 1:
        raise AssertionError(
            f"DeepSpeed ZeRO-{zero_stage} did not reduce gradients"
        )
    if zero_stage == 3 and calls["all_gather"] < 1:
        raise AssertionError("DeepSpeed ZeRO-3 did not gather parameters")


def _run_stage(
    *,
    build_dir: Path,
    report_root: Path,
    world_size: int,
    zero_stage: int,
    precision: str,
    offload: str,
    timeout: float,
) -> dict[str, Any]:
    coordinator_bin = build_dir / "fakegpu-coordinator"
    nccl_lib = (
        build_dir / "libnccl.dylib"
        if sys.platform == "darwin"
        else build_dir / "libnccl.so.2"
    )
    cluster_config = CLUSTER_CONFIGS[world_size]
    stage_root = report_root / (
        f"zero{zero_stage}-{precision}-{world_size}r-{offload}"
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    socket_path = stage_root / "coordinator.sock"
    cluster_report_path = stage_root / "cluster-report.json"
    rank_report_dir = stage_root / "ranks"

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
        cwd=REPO_ROOT,
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
                "FAKEGPU_COORDINATOR_TIMEOUT_MS": "90000",
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
            "--zero-stage",
            str(zero_stage),
            "--precision",
            precision,
        ]
        if offload != "none":
            command.append("--offload-optimizer")
        if offload == "optimizer-and-parameter":
            command.append("--offload-parameters")
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
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
            return_code = completed.returncode
            raise RuntimeError(
                f"DeepSpeed ZeRO-{zero_stage} worker exited with "
                f"code {return_code}"
            )

        rank_reports = _load_rank_reports(rank_report_dir, world_size)
        _validate_rank_reports(
            rank_reports,
            world_size=world_size,
            zero_stage=zero_stage,
            precision=precision,
            offload=offload,
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
            raise AssertionError(
                f"DeepSpeed ZeRO-{zero_stage} node-pair report is empty"
            )
        total_bytes = sum(
            int(pair.get("total_bytes", 0)) for pair in node_pairs
        )
        peak_bytes = max(
            int(pair.get("peak_combined_bytes_per_operation", 0))
            for pair in node_pairs
        )
        if total_bytes <= 0 or peak_bytes <= 0:
            raise AssertionError(
                f"DeepSpeed ZeRO-{zero_stage} communication was empty: "
                f"{node_pairs}"
            )
        markdown_report_path = Path(
            cluster_report["cluster"]["markdown_report_path"]
        )
        if not markdown_report_path.is_file():
            raise AssertionError(
                "cluster Markdown report was not generated: "
                f"{markdown_report_path}"
            )

        first_report = rank_reports[0]
        return {
            "status": "success",
            "zero_stage": zero_stage,
            "precision": precision,
            "world_size": world_size,
            "offload": offload,
            "deepspeed_version": first_report["deepspeed_version"],
            "torch_version": first_report["torch_version"],
            "torch_cuda_version": first_report["torch_cuda_version"],
            "physical_device_name": first_report["physical_device_name"],
            "physical_compute_capability": first_report[
                "physical_compute_capability"
            ],
            "optimizer_type": first_report["optimizer_type"],
            "parameter_dtype": first_report["parameter_dtype"],
            "offload_state": first_report["offload"],
            "parameter_after_step": first_report[
                "parameters_after_micro_steps"
            ][1],
            "collective_calls": calls,
            "node_pair_total_bytes": total_bytes,
            "node_pair_peak_bytes_per_operation": peak_bytes,
            "cluster_markdown_report": str(markdown_report_path),
        }
    finally:
        _close_coordinator(coordinator, socket_path)
        coordinator.communicate(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run DeepSpeed ZeRO numeric validation on one physical CUDA GPU "
            "while FakeGPU simulates NCCL communication for two or four ranks."
        )
    )
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
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
    parser.add_argument(
        "--offload",
        choices=("none", "optimizer", "optimizer-and-parameter"),
        default="none",
    )
    parser.add_argument("--timeout", type=float, default=240.0)
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
    cluster_config = CLUSTER_CONFIGS[args.world_size]
    for artifact in (
        build_dir / "fakegpu-coordinator",
        nccl_lib,
        WORKER,
        cluster_config,
    ):
        if not artifact.exists():
            print(f"missing required artifact: {artifact}", file=sys.stderr)
            return 2

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.report_dir is None:
        temporary = tempfile.TemporaryDirectory(
            prefix="fakegpu-hybrid-deepspeed-numerics-"
        )
        report_root = Path(temporary.name)
    else:
        report_root = args.report_dir.resolve()
        report_root.mkdir(parents=True, exist_ok=True)

    zero_stages = (
        (0, 1, 2, 3)
        if args.zero_stage == "all"
        else (int(args.zero_stage),)
    )
    if args.offload != "none" and any(
        stage not in (2, 3) for stage in zero_stages
    ):
        parser.error("optimizer offload requires ZeRO stage 2 or 3")
    if args.offload == "optimizer-and-parameter" and zero_stages != (3,):
        parser.error("parameter offload requires --zero-stage 3")
    try:
        stage_reports = [
            _run_stage(
                build_dir=build_dir,
                report_root=report_root,
                world_size=args.world_size,
                zero_stage=zero_stage,
                precision=args.precision,
                offload=args.offload,
                timeout=args.timeout,
            )
            for zero_stage in zero_stages
        ]
        summary = {
            "schema_version": "fakegpu.hybrid_deepspeed_numerics.v1",
            "status": "success",
            "world_size": args.world_size,
            "precision": args.precision,
            "offload": args.offload,
            "zero_stages": list(zero_stages),
            "results": stage_reports,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
