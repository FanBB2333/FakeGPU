#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from elastic_ddp_training_state_validation import (
    validate_elastic_ddp_training_state_reports,
)
from test_elastic_ddp_checkpoint_resume import (
    _run_torchrun,
    _unused_local_port,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "verification" / "elastic_ddp_training_state_worker.py"


def _read_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"elastic training-state report is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate AdamW, scheduler, RNG, and partial gradient-accumulation "
            "recovery after a local torchrun worker-group restart."
        )
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    with tempfile.TemporaryDirectory(
        prefix="fakegpu-elastic-ddp-training-state-"
    ) as raw:
        report_dir = Path(raw) / "reports"
        checkpoint_dir = Path(raw) / "checkpoints"
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "1"
        if sys.platform == "darwin":
            env["GLOO_SOCKET_IFNAME"] = "lo0"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--master-addr=127.0.0.1",
            f"--master-port={_unused_local_port()}",
            "--max-restarts=1",
            str(WORKER),
            f"--report-dir={report_dir}",
            f"--checkpoint-dir={checkpoint_dir}",
            "--node-name=local",
            "--backend=gloo",
            "--trace-store",
            "--fail-rank=1",
            "--resume-rank-shift=1",
            "--survivor-wait-seconds=10",
        ]
        completed = _run_torchrun(command, env=env, timeout=args.timeout)
        if completed.returncode != 0:
            raise AssertionError(
                f"torchrun exited with {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        initial = [
            _read_report(report_dir / f"attempt_0_local_local_{rank}.json")
            for rank in range(2)
        ]
        restarted = [
            _read_report(report_dir / f"attempt_1_local_local_{rank}.json")
            for rank in range(2)
        ]
        summary = validate_elastic_ddp_training_state_reports(
            initial,
            restarted,
            backend="gloo",
            require_physical_gpu=False,
            check_checkpoint_files=True,
        )
        if summary["local_restart_counts"] != {"0": 1, "1": 1}:
            raise AssertionError(
                "single-agent torchrun did not restart both local workers"
            )
        if int(summary["failed_rank"]) != 1:
            raise AssertionError("local validation did not fail rank 1")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
