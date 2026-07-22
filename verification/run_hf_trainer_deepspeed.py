#!/usr/bin/env python3
"""Validate Hugging Face Trainer with Hybrid DeepSpeed and Fake NCCL."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "verification" / "hf_trainer_deepspeed_worker.py"
CLUSTER_CONFIGS = {
    2: ROOT / "verification" / "data" / "cluster_hybrid_2r.yaml",
    4: ROOT / "verification" / "data" / "cluster_hybrid_4r.yaml",
}
SCHEMA_VERSION = "fakegpu.hf_trainer_deepspeed.v1"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verification.run_hybrid_deepspeed_numerics import (  # noqa: E402
    _close_coordinator,
    _collective_calls,
    _find_free_port,
    _load_rank_reports,
    _shutdown_unix_coordinator,
    _temporary_coordinator_socket,
    _validate_collective_calls,
    _wait_for_unix_socket,
)


def validate_hf_trainer_reports(
    reports: list[dict[str, Any]],
    *,
    world_size: int,
    zero_stage: int,
    precision: str,
    max_steps: int,
) -> None:
    if len(reports) != world_size:
        raise AssertionError(
            f"expected {world_size} rank reports, got {len(reports)}"
        )
    fingerprints = {
        report.get("dataset", {}).get("fingerprint_sha256")
        for report in reports
    }
    if len(fingerprints) != 1:
        raise AssertionError(
            f"Trainer ranks used different datasets: {fingerprints}"
        )
    for rank, report in enumerate(reports):
        if report.get("status") != "success":
            raise AssertionError(
                f"Hugging Face Trainer rank {rank} failed: {report}"
            )
        if int(report.get("rank", -1)) != rank:
            raise AssertionError(f"rank report mismatch: {report}")
        if report.get("precision") != precision:
            raise AssertionError(f"precision mismatch: {report}")
        if int(report.get("zero_stage", -1)) != zero_stage:
            raise AssertionError(f"ZeRO stage mismatch: {report}")
        identity = report.get("trainer_identity", {})
        if int(identity.get("process_index", -1)) != rank:
            raise AssertionError(
                f"Accelerate process index mismatch: {report}"
            )
        if int(identity.get("local_process_index", -1)) != 0:
            raise AssertionError(
                f"logical rank {rank} did not map to physical cuda:0"
            )
        if int(identity.get("world_size", -1)) != world_size:
            raise AssertionError(f"Trainer world size mismatch: {report}")
        engine = report.get("engine", {})
        if engine.get("type") != "DeepSpeedEngine":
            raise AssertionError(f"Trainer did not use DeepSpeed: {report}")
        if int(engine.get("effective_zero_stage", -1)) != zero_stage:
            raise AssertionError(f"effective ZeRO stage mismatch: {report}")
        training = report.get("training", {})
        if int(training.get("global_step", -1)) != max_steps:
            raise AssertionError(f"Trainer step mismatch: {report}")
        if not math.isfinite(float(training.get("training_loss", math.nan))):
            raise AssertionError(f"Trainer loss is not finite: {report}")
        probe = report.get("parameter_update_probe", {})
        if float(probe.get("update_norm", 0)) <= 0:
            raise AssertionError(f"Trainer parameter did not update: {report}")
        if probe.get("before_sha256") == probe.get("after_sha256"):
            raise AssertionError(f"Trainer parameter hash did not change: {report}")
        gathered = probe.get("gathered_after_sha256")
        if not isinstance(gathered, list) or len(gathered) != world_size:
            raise AssertionError(f"cross-rank probe is incomplete: {report}")
        if len(set(gathered)) != 1:
            raise AssertionError(f"Trainer parameters differ by rank: {report}")
        resolved = report.get("resolved_deepspeed_config", {})
        if int(resolved.get("zero_optimization", {}).get("stage", -1)) != (
            zero_stage
        ):
            raise AssertionError(f"resolved DeepSpeed config mismatch: {report}")


def _run(
    *,
    build_dir: Path,
    report_dir: Path,
    workload: str,
    model_dir: Path | None,
    world_size: int,
    zero_stage: int,
    precision: str,
    batch_size: int,
    sequence_length: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    gradient_checkpointing: bool,
    timeout: float,
) -> dict[str, Any]:
    coordinator_bin = build_dir / "fakegpu-coordinator"
    nccl_lib = (
        build_dir / "libnccl.dylib"
        if sys.platform == "darwin"
        else build_dir / "libnccl.so.2"
    )
    cluster_config = CLUSTER_CONFIGS[world_size]
    report_dir.mkdir(parents=True, exist_ok=True)
    socket_directory, socket_path = _temporary_coordinator_socket()
    cluster_report_path = report_dir / "cluster-report.json"
    rank_report_dir = report_dir / "ranks"
    trainer_output_dir = report_dir / "trainer-output"

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
            "--workload",
            workload,
            "--report-dir",
            str(rank_report_dir),
            "--trainer-output-dir",
            str(trainer_output_dir),
            "--zero-stage",
            str(zero_stage),
            "--precision",
            precision,
            "--batch-size",
            str(batch_size),
            "--sequence-length",
            str(sequence_length),
            "--gradient-accumulation-steps",
            str(gradient_accumulation_steps),
            "--max-steps",
            str(max_steps),
        ]
        if model_dir is not None:
            command.extend(["--model-dir", str(model_dir)])
        if gradient_checkpointing:
            command.append("--gradient-checkpointing")
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
                "Hugging Face Trainer DeepSpeed workers exited with code "
                f"{completed.returncode}"
            )

        reports = _load_rank_reports(rank_report_dir, world_size)
        validate_hf_trainer_reports(
            reports,
            world_size=world_size,
            zero_stage=zero_stage,
            precision=precision,
            max_steps=max_steps,
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
            raise AssertionError("Trainer node-pair report is empty")
        total_bytes = sum(
            int(pair.get("total_bytes", 0)) for pair in node_pairs
        )
        peak_bytes = max(
            int(pair.get("peak_combined_bytes_per_operation", 0))
            for pair in node_pairs
        )
        first = reports[0]
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "success",
            "workload": workload,
            "world_size": world_size,
            "zero_stage": zero_stage,
            "precision": precision,
            "torch_version": first["torch_version"],
            "torch_cuda_version": first["torch_cuda_version"],
            "transformers_version": first["transformers_version"],
            "accelerate_version": first["accelerate_version"],
            "deepspeed_version": first["deepspeed_version"],
            "peft_version": first["peft_version"],
            "physical_device_name": first["physical_device_name"],
            "physical_compute_capability": first[
                "physical_compute_capability"
            ],
            "training": first["training"],
            "engine": first["engine"],
            "parameter_update_probe": first["parameter_update_probe"],
            "rank_peak_allocated_bytes": [
                int(report["memory"]["peak_allocated_bytes"])
                for report in reports
            ],
            "collective_calls": calls,
            "node_pair_total_bytes": total_bytes,
            "node_pair_peak_bytes_per_operation": peak_bytes,
        }
    finally:
        _close_coordinator(coordinator, socket_path)
        coordinator.communicate(timeout=5)
        socket_directory.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument(
        "--workload",
        choices=("tiny", "qwen-lora"),
        default="tiny",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--world-size", type=int, choices=(2, 4), default=2)
    parser.add_argument("--zero-stage", type=int, choices=(2, 3), default=3)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="bf16",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    if args.workload == "qwen-lora" and args.model_dir is None:
        parser.error("--model-dir is required for --workload qwen-lora")
    for dependency in ("deepspeed", "transformers", "accelerate"):
        if importlib.util.find_spec(dependency) is None:
            print(
                f"{dependency} is not installed in {sys.executable}",
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
            prefix="fakegpu-hf-trainer-deepspeed-"
        )
        report_dir = Path(temporary.name)
    else:
        report_dir = args.report_dir.resolve()
    try:
        summary = _run(
            build_dir=build_dir,
            report_dir=report_dir,
            workload=args.workload,
            model_dir=(
                args.model_dir.expanduser().resolve()
                if args.model_dir is not None
                else None
            ),
            world_size=args.world_size,
            zero_stage=args.zero_stage,
            precision=args.precision,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
            max_steps=args.max_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            timeout=args.timeout,
        )
        if args.report_dir is not None:
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "summary.json").write_text(
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
