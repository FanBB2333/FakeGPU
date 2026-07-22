#!/usr/bin/env python3
"""Run a multi-rank Hybrid DeepSpeed Qwen3.5 LoRA SFT experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verification.run_hybrid_deepspeed_numerics import (  # noqa: E402
    CLUSTER_CONFIGS,
    _close_coordinator,
    _collective_calls,
    _find_free_port,
    _temporary_coordinator_socket,
    _validate_collective_calls,
    _wait_for_unix_socket,
)


WORKER = ROOT / "verification" / "qwen_deepspeed_lora_sft_worker.py"
SCHEMA_VERSION = "fakegpu.qwen_deepspeed_lora_sft.v1"


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


def validate_qwen_deepspeed_reports(
    reports: list[dict[str, Any]],
    *,
    world_size: int,
    zero_stage: int,
    gradient_accumulation_steps: int,
) -> None:
    if len(reports) != world_size:
        raise AssertionError(
            f"expected {world_size} rank reports, got {len(reports)}"
        )
    reference_probe: float | None = None
    for rank, report in enumerate(reports):
        if report.get("status") != "success":
            raise AssertionError(
                f"Qwen DeepSpeed rank {rank} failed: {report}"
            )
        if report.get("rank") != rank:
            raise AssertionError(
                f"rank report order mismatch: expected {rank}, got "
                f"{report.get('rank')}"
            )
        if report.get("world_size") != world_size:
            raise AssertionError(f"rank {rank} world size mismatch")
        if report.get("zero_stage") != zero_stage:
            raise AssertionError(f"rank {rank} ZeRO stage mismatch")
        engine = report.get("engine", {})
        if engine.get("effective_zero_stage") != zero_stage:
            raise AssertionError(
                f"rank {rank} effective ZeRO stage mismatch: {engine}"
            )
        if (
            engine.get("gradient_accumulation_steps")
            != gradient_accumulation_steps
        ):
            raise AssertionError(
                f"rank {rank} gradient accumulation mismatch: {engine}"
            )
        global_steps = report.get("global_steps_after_microsteps")
        expected_steps = [
            0
            for _ in range(max(0, gradient_accumulation_steps - 1))
        ] + [1]
        if global_steps != expected_steps:
            raise AssertionError(
                f"rank {rank} optimizer boundary mismatch: "
                f"{global_steps} != {expected_steps}"
            )
        loss = float(report.get("loss", float("nan")))
        if not math.isfinite(loss) or loss <= 0:
            raise AssertionError(f"rank {rank} loss is invalid: {loss}")
        parameters = report.get("parameters", {})
        if int(parameters.get("trainable_parameter_count", 0)) <= 0:
            raise AssertionError(
                f"rank {rank} did not report trainable LoRA parameters"
            )
        probe = report.get("optimizer_update_probe", {})
        before = float(probe.get("before_squared_norm", float("nan")))
        after = float(probe.get("after_squared_norm", float("nan")))
        if not math.isfinite(before) or not math.isfinite(after):
            raise AssertionError(f"rank {rank} update probe is non-finite")
        if after <= before:
            raise AssertionError(
                f"rank {rank} LoRA probe was not updated: {before} -> {after}"
            )
        gathered = probe.get("gathered_after_squared_norms")
        if not isinstance(gathered, list) or len(gathered) != world_size:
            raise AssertionError(
                f"rank {rank} update probe gather is incomplete: {gathered}"
            )
        if any(
            not math.isclose(
                float(value),
                after,
                rel_tol=1e-5,
                abs_tol=1e-12,
            )
            for value in gathered
        ):
            raise AssertionError(
                f"rank {rank} observed inconsistent updates: {gathered}"
            )
        if reference_probe is None:
            reference_probe = after
        elif not math.isclose(
            after,
            reference_probe,
            rel_tol=1e-5,
            abs_tol=1e-12,
        ):
            raise AssertionError(
                f"rank {rank} update differs from rank 0: "
                f"{after} != {reference_probe}"
            )
        peak = int(
            report.get("memory_phases", {}).get("overall_peak_bytes", 0)
        )
        if peak <= 0:
            raise AssertionError(
                f"rank {rank} did not report a CUDA memory peak"
            )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen Hybrid DeepSpeed LoRA SFT",
        "",
        f"- Status: `{report['status']}`",
        f"- Model: `{report['model_dir']}`",
        f"- GPU: `{report['physical_device_name']}`",
        f"- DeepSpeed: `{report['deepspeed_version']}`",
        f"- ZeRO stage: `{report['zero_stage']}`",
        f"- World size: `{report['world_size']}`",
        f"- Precision: `{report['dtype']}`",
        f"- Sequence length: `{report['sequence_length']}`",
        "",
        "## Per-rank result",
        "",
        "| Rank | Loss | Peak CUDA memory | LoRA probe after step |",
        "|---:|---:|---:|---:|",
    ]
    for rank in report["ranks"]:
        peak_gib = rank["overall_peak_bytes"] / (1024**3)
        lines.append(
            f"| {rank['rank']} | {rank['loss']:.6f} | "
            f"{peak_gib:.3f} GiB | "
            f"{rank['update_probe_after']:.9g} |"
        )
    lines.extend(
        [
            "",
            "## Communication",
            "",
            "| Collective | Calls |",
            "|---|---:|",
        ]
    )
    for name, calls in report["collective_calls"].items():
        lines.append(f"| `{name}` | {calls} |")
    lines.extend(
        [
            "",
            f"- Node-pair total: `{report['node_pair_total_bytes']}` bytes",
            "- Node-pair peak per operation: "
            f"`{report['node_pair_peak_bytes_per_operation']}` bytes",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--world-size", type=int, choices=(2, 4), default=2)
    parser.add_argument("--zero-stage", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa"),
        default="sdpa",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--checkpoint-implementation",
        choices=("reentrant", "non-reentrant"),
        default="reentrant",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", default="all-linear")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--data-seed", type=int, default=20260722)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least two")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be greater than zero")

    if importlib.util.find_spec("deepspeed") is None:
        print(
            "DeepSpeed is not installed in the selected Python environment: "
            f"{sys.executable}",
            file=sys.stderr,
        )
        return 2

    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    build_dir = args.build_dir.expanduser().resolve()
    cluster_config = CLUSTER_CONFIGS[args.world_size]
    coordinator_bin = build_dir / "fakegpu-coordinator"
    nccl_lib = (
        build_dir / "libnccl.dylib"
        if sys.platform == "darwin"
        else build_dir / "libnccl.so.2"
    )
    for artifact in (
        model_dir,
        WORKER,
        coordinator_bin,
        nccl_lib,
        cluster_config,
    ):
        if not artifact.exists():
            print(f"missing required artifact: {artifact}", file=sys.stderr)
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    rank_report_dir = output_dir / "ranks"
    socket_directory, socket_path = _temporary_coordinator_socket()
    cluster_report_path = output_dir / "cluster-report.json"
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"

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
                "FAKEGPU_COORDINATOR_TIMEOUT_MS": "600000",
            }
        )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            f"--nproc-per-node={args.world_size}",
            "--master-addr=127.0.0.1",
            f"--master-port={_find_free_port()}",
            str(WORKER),
            "--model-dir",
            str(model_dir),
            "--report-dir",
            str(rank_report_dir),
            "--zero-stage",
            str(args.zero_stage),
            "--batch-size",
            str(args.batch_size),
            "--sequence-length",
            str(args.sequence_length),
            "--dtype",
            args.dtype,
            "--attention-implementation",
            args.attention_implementation,
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--checkpoint-implementation",
            args.checkpoint_implementation,
            "--lora-rank",
            str(args.lora_rank),
            "--lora-alpha",
            str(args.lora_alpha),
            "--lora-dropout",
            str(args.lora_dropout),
            "--lora-target-modules",
            args.lora_target_modules,
            "--learning-rate",
            str(args.learning_rate),
            "--seed",
            str(args.seed),
            "--data-seed",
            str(args.data_seed),
        ]
        if args.gradient_checkpointing:
            command.append("--gradient-checkpointing")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=worker_env,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )
        if completed.returncode != 0:
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
            raise RuntimeError(
                "Qwen DeepSpeed workers exited with code "
                f"{completed.returncode}"
            )

        rank_reports = _load_rank_reports(
            rank_report_dir,
            args.world_size,
        )
        validate_qwen_deepspeed_reports(
            rank_reports,
            world_size=args.world_size,
            zero_stage=args.zero_stage,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
        )

        from verification.run_hybrid_deepspeed_numerics import (  # noqa: PLC0415
            _shutdown_unix_coordinator,
        )

        _shutdown_unix_coordinator(socket_path)
        coordinator.wait(timeout=5)
        cluster_report = json.loads(
            cluster_report_path.read_text(encoding="utf-8")
        )
        calls = _collective_calls(cluster_report)
        _validate_collective_calls(calls, args.zero_stage)
        node_pairs = cluster_report.get("node_pairs")
        if not isinstance(node_pairs, list) or not node_pairs:
            raise AssertionError("DeepSpeed node-pair report is empty")
        total_bytes = sum(
            int(pair.get("total_bytes", 0)) for pair in node_pairs
        )
        peak_bytes = max(
            int(pair.get("peak_combined_bytes_per_operation", 0))
            for pair in node_pairs
        )
        if total_bytes <= 0 or peak_bytes <= 0:
            raise AssertionError(
                f"DeepSpeed communication was empty: {node_pairs}"
            )

        first = rank_reports[0]
        rank_summaries = [
            {
                "rank": int(report["rank"]),
                "loss": float(report["loss"]),
                "overall_peak_bytes": int(
                    report["memory_phases"]["overall_peak_bytes"]
                ),
                "forward_peak_bytes": int(
                    report["memory_phases"]["forward_peak_bytes"]
                ),
                "backward_peak_bytes": int(
                    report["memory_phases"]["backward_peak_bytes"]
                ),
                "optimizer_peak_bytes": int(
                    report["memory_phases"]["optimizer_peak_bytes"]
                ),
                "update_probe_after": float(
                    report["optimizer_update_probe"][
                        "after_squared_norm"
                    ]
                ),
                "rank_report": str(
                    rank_report_dir / f"rank_{report['rank']}.json"
                ),
            }
            for report in rank_reports
        ]
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "success",
            "model_dir": str(model_dir),
            "training_method": "lora",
            "world_size": args.world_size,
            "zero_stage": args.zero_stage,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "gradient_accumulation_steps": (
                args.gradient_accumulation_steps
            ),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "checkpoint_implementation": (
                args.checkpoint_implementation
                if args.gradient_checkpointing
                else None
            ),
            "lora": first["lora"],
            "physical_device_name": first["physical_device_name"],
            "physical_compute_capability": first[
                "physical_compute_capability"
            ],
            "torch_version": first["torch_version"],
            "torch_cuda_version": first["torch_cuda_version"],
            "transformers_version": first["transformers_version"],
            "peft_version": first["peft_version"],
            "deepspeed_version": first["deepspeed_version"],
            "optimizer_type": first["engine"]["optimizer_type"],
            "parameters": first["parameters"],
            "max_overall_peak_bytes": max(
                rank["overall_peak_bytes"] for rank in rank_summaries
            ),
            "ranks": rank_summaries,
            "collective_calls": calls,
            "node_pair_total_bytes": total_bytes,
            "node_pair_peak_bytes_per_operation": peak_bytes,
            "cluster_report": str(cluster_report_path),
            "cluster_markdown_report": cluster_report["cluster"][
                "markdown_report_path"
            ],
            "summary_markdown": str(markdown_path),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(
            render_markdown(summary),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "success",
                    "model_dir": str(model_dir),
                    "zero_stage": args.zero_stage,
                    "world_size": args.world_size,
                    "dtype": args.dtype,
                    "max_overall_peak_bytes": summary[
                        "max_overall_peak_bytes"
                    ],
                    "node_pair_total_bytes": total_bytes,
                    "summary": str(summary_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        _close_coordinator(coordinator, socket_path)
        coordinator.communicate(timeout=5)
        socket_directory.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
