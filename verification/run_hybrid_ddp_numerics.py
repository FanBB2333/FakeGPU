#!/usr/bin/env python3

from __future__ import annotations

import argparse
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
WORKER = REPO_ROOT / "test" / "test_ddp_hybrid_numerics.py"
CLUSTER_CONFIG = REPO_ROOT / "verification" / "data" / "cluster_hybrid_2r.yaml"


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


def _load_rank_reports(report_dir: Path, world_size: int) -> list[dict[str, Any]]:
    reports = []
    for rank in range(world_size):
        path = report_dir / f"rank_{rank}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing rank report: {path}")
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


EXPECTED_RESULTS: dict[str, dict[str, list[list[float]]]] = {
    "basic": {
        "gradient": [[1.5, 3.0]],
        "parameters_after_step": [[0.85, -0.3]],
    },
    "no-sync": {
        "gradient": [[4.5, 9.0]],
        "parameters_after_step": [[0.55, -0.9]],
    },
    "find-unused": {
        "gradient": [[0.5, 1.0], [1.0, 2.0]],
        "parameters_after_step": [[0.95, -0.1], [-0.1, 0.8]],
    },
    "static-graph": {
        "gradient": [[1.5, 3.0]],
        "parameters_after_step": [[0.7, -0.6]],
    },
}
EXPECTED_ALL_REDUCE_CALLS = {
    "basic": 2,
    "no-sync": 2,
    "find-unused": 3,
    "static-graph": 4,
}


def _nested_allclose(
    actual: object,
    expected: list[list[float]],
    *,
    tolerance: float = 1e-6,
) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for actual_row, expected_row in zip(actual, expected):
        if not isinstance(actual_row, list) or len(actual_row) != len(expected_row):
            return False
        if any(
            abs(float(actual_value) - expected_value) > tolerance
            for actual_value, expected_value in zip(actual_row, expected_row)
        ):
            return False
    return True


def _validate_variant_reports(
    variant: str,
    reports: list[dict[str, Any]],
) -> None:
    expected = EXPECTED_RESULTS[variant]
    for rank, report in enumerate(reports):
        if report.get("status") != "success":
            raise AssertionError(f"{variant} rank {rank} failed: {report}")
        if report.get("variant") != variant:
            raise AssertionError(
                f"rank {rank} reported variant={report.get('variant')!r}"
            )
        for field, expected_value in expected.items():
            if not _nested_allclose(report.get(field), expected_value):
                raise AssertionError(
                    f"{variant} rank {rank} {field} mismatch: "
                    f"{report.get(field)} != {expected_value}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run two real-CUDA ranks on one physical GPU while fake NCCL "
            "validates DDP gradient averaging and common execution options."
        )
    )
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--variant",
        choices=[*EXPECTED_RESULTS, "all"],
        default="basic",
        help="DDP execution path to validate (default: basic).",
    )
    args = parser.parse_args()
    variants = list(EXPECTED_RESULTS) if args.variant == "all" else [args.variant]

    build_dir = args.build_dir.resolve()
    coordinator_bin = build_dir / "fakegpu-coordinator"
    nccl_lib = (
        build_dir / "libnccl.dylib"
        if sys.platform == "darwin"
        else build_dir / "libnccl.so.2"
    )
    for artifact in (coordinator_bin, nccl_lib, WORKER, CLUSTER_CONFIG):
        if not artifact.exists():
            print(f"missing required artifact: {artifact}", file=sys.stderr)
            return 2

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.report_dir is None:
        temporary = tempfile.TemporaryDirectory(
            prefix="fakegpu-hybrid-ddp-numerics-"
        )
        report_root = Path(temporary.name)
    else:
        report_root = args.report_dir.resolve()
        report_root.mkdir(parents=True, exist_ok=True)

    socket_path = report_root / "coordinator.sock"
    cluster_report_path = report_root / "cluster_report.json"
    rank_report_dir = report_root / "ranks"
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
                "FAKEGPU_CLUSTER_CONFIG": str(CLUSTER_CONFIG),
                "FAKEGPU_COORDINATOR_TRANSPORT": "unix",
                "FAKEGPU_COORDINATOR_ADDR": str(socket_path),
                "FAKEGPU_COORDINATOR_TIMEOUT_MS": "60000",
            }
        )
        reports_by_variant: dict[str, list[dict[str, Any]]] = {}
        for variant in variants:
            variant_report_dir = rank_report_dir / variant
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
                str(variant_report_dir),
                "--variant",
                variant,
            ]
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
                return completed.returncode

            rank_reports = _load_rank_reports(variant_report_dir, 2)
            _validate_variant_reports(variant, rank_reports)
            reports_by_variant[variant] = rank_reports

        _shutdown_unix_coordinator(socket_path)
        coordinator.wait(timeout=5)
        cluster_report = json.loads(
            cluster_report_path.read_text(encoding="utf-8")
        )
        expected_all_reduce_calls = sum(
            EXPECTED_ALL_REDUCE_CALLS[variant] for variant in variants
        )
        all_reduce_calls = cluster_report["collectives"]["all_reduce"]["calls"]
        if all_reduce_calls != expected_all_reduce_calls:
            raise AssertionError(
                "unexpected DDP all-reduce count: "
                f"{all_reduce_calls} != {expected_all_reduce_calls}"
            )
        all_gather_calls = cluster_report["collectives"]["all_gather"]["calls"]
        if all_gather_calls < len(variants):
            raise AssertionError(
                "parameter consistency did not issue one all-gather per variant"
            )
        node_pairs = cluster_report.get("node_pairs")
        if not isinstance(node_pairs, list) or len(node_pairs) != 1:
            raise AssertionError(
                f"expected one complete node-pair entry, got {node_pairs!r}"
            )
        node_pair = node_pairs[0]
        if int(node_pair.get("total_bytes", 0)) <= 0:
            raise AssertionError(f"DDP node-pair traffic was empty: {node_pair}")
        if int(node_pair.get("peak_combined_bytes_per_operation", 0)) <= 0:
            raise AssertionError(f"DDP node-pair peak was empty: {node_pair}")
        markdown_report_path = Path(
            cluster_report["cluster"]["markdown_report_path"]
        )
        if not markdown_report_path.is_file():
            raise AssertionError(
                f"cluster Markdown report was not generated: "
                f"{markdown_report_path}"
            )

        first_report = reports_by_variant[variants[0]][0]
        summary: dict[str, Any] = {
            "status": "success",
            "physical_device_name": first_report["physical_device_name"],
            "physical_compute_capability": first_report[
                "physical_compute_capability"
            ],
            "torch_version": first_report["torch_version"],
            "torch_cuda_version": first_report["torch_cuda_version"],
            "variants": {
                variant: {
                    "gradient": reports[0]["gradient"],
                    "parameters_after_step": reports[0][
                        "parameters_after_step"
                    ],
                    "training_steps": reports[0]["training_steps"],
                    "ddp_options": reports[0]["ddp_options"],
                }
                for variant, reports in reports_by_variant.items()
            },
            "all_reduce_calls": all_reduce_calls,
            "all_gather_calls": all_gather_calls,
            "node_pair_total_bytes": node_pair["total_bytes"],
            "node_pair_peak_bytes_per_operation": node_pair[
                "peak_combined_bytes_per_operation"
            ],
            "cluster_markdown_report": str(markdown_report_path),
        }
        if variants == ["basic"]:
            summary["gradient"] = first_report["gradient"]
            summary["parameters_after_step"] = first_report[
                "parameters_after_step"
            ]
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        if coordinator.poll() is None:
            try:
                _shutdown_unix_coordinator(socket_path)
                coordinator.wait(timeout=3)
            except Exception:
                coordinator.kill()
                coordinator.wait(timeout=5)
        coordinator.communicate(timeout=5)
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
