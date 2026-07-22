#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "verification" / "elastic_ddp_checkpoint_worker.py"
EXPECTED_GRADIENT = [[1.5, 3.0]]
EXPECTED_STEP_ONE_PARAMETERS = [[0.85, -0.3]]
EXPECTED_STEP_ONE_MOMENTUM = [[1.5, 3.0]]
EXPECTED_STEP_TWO_PARAMETERS = [[0.565, -0.87]]
EXPECTED_STEP_TWO_MOMENTUM = [[2.85, 5.7]]
EXPECTED_RESTORED_STATE = [[0.85, -0.3, 1.5, 3.0]] * 2
EXPECTED_FINAL_STATE = [[0.565, -0.87, 2.85, 5.7]] * 2


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _run_torchrun(
    command: list[str],
    *,
    env: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise AssertionError(
            f"torchrun timed out after {timeout:.1f} seconds\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        ) from None
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout,
        stderr,
    )


def _close(actual: object, expected: list[list[float]]) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for actual_row, expected_row in zip(actual, expected):
        if not isinstance(actual_row, list) or len(actual_row) != len(expected_row):
            return False
        if any(
            abs(float(actual_value) - expected_value) > 1e-6
            for actual_value, expected_value in zip(actual_row, expected_row)
        ):
            return False
    return True


def _read_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"elastic checkpoint report is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_reports(report_dir: Path) -> dict[str, Any]:
    initial = [
        _read_report(report_dir / f"attempt_0_local_local_{rank}.json")
        for rank in range(2)
    ]
    restarted = [
        _read_report(report_dir / f"attempt_1_local_local_{rank}.json")
        for rank in range(2)
    ]

    if [item.get("status") for item in initial] != [
        "waiting_for_restart",
        "expected_process_exit",
    ]:
        raise AssertionError(f"unexpected initial worker states: {initial}")
    if any(int(item.get("completed_steps", -1)) != 1 for item in initial):
        raise AssertionError(f"initial completed steps were invalid: {initial}")
    if any(int(item.get("restart_count", -1)) != 0 for item in initial):
        raise AssertionError(f"initial restart counters were invalid: {initial}")
    if any(int(item.get("restart_arrival_count", -1)) != 1 for item in initial):
        raise AssertionError(f"initial arrival counters were invalid: {initial}")
    if any(int(item.get("local_restart_count", -1)) != 0 for item in initial):
        raise AssertionError(f"initial local restart counters were invalid: {initial}")
    if any(item.get("observed_local_restart_counts") != [0, 0] for item in initial):
        raise AssertionError(f"initial restart generation was invalid: {initial}")
    if int(initial[1].get("expected_process_exit_code", -1)) != 86:
        raise AssertionError(f"initial failure marker was invalid: {initial[1]}")
    if any(
        item.get("initial_process_group_destroyed_before_exit") is not False
        for item in initial
    ):
        raise AssertionError(
            f"initial process groups were unexpectedly cleaned up: {initial}"
        )
    if any(not item.get("store_events") for item in initial + restarted):
        raise AssertionError("elastic checkpoint store tracing was empty")
    if any(
        not _close(item.get("gradient_after_step_one"), EXPECTED_GRADIENT)
        or not _close(
            item.get("parameters_after_step_one"),
            EXPECTED_STEP_ONE_PARAMETERS,
        )
        or not _close(
            item.get("momentum_after_step_one"),
            EXPECTED_STEP_ONE_MOMENTUM,
        )
        or not _close(
            item.get("gathered_state_after_step_one"),
            EXPECTED_RESTORED_STATE,
        )
        for item in initial
    ):
        raise AssertionError(f"initial checkpoint state was invalid: {initial}")

    if any(item.get("status") != "success" for item in restarted):
        raise AssertionError(f"restarted workers failed: {restarted}")
    if any(int(item.get("restart_count", -1)) != 1 for item in restarted):
        raise AssertionError(f"restart counters were invalid: {restarted}")
    if any(int(item.get("restart_arrival_count", -1)) != 2 for item in restarted):
        raise AssertionError(f"restart arrival counters were invalid: {restarted}")
    if any(int(item.get("local_restart_count", -1)) != 1 for item in restarted):
        raise AssertionError(f"local restart counters were invalid: {restarted}")
    if any(
        max(int(value) for value in item.get("observed_local_restart_counts", []))
        != 1
        for item in restarted
    ):
        raise AssertionError(f"restart generation was invalid: {restarted}")
    if sorted(int(item["rank"]) for item in restarted) != [0, 1]:
        raise AssertionError(f"restarted ranks were invalid: {restarted}")
    for before, after in zip(initial, restarted):
        if int(before["pid"]) == int(after["pid"]):
            raise AssertionError(
                "torchrun did not replace every initial worker process"
            )
        if after.get("checkpoint_loaded") is not True:
            raise AssertionError(f"checkpoint was not loaded: {after}")
        if int(after.get("restored_completed_steps", -1)) != 1:
            raise AssertionError(f"training step was not restored: {after}")
        saved = before.get("saved_checkpoint", {})
        loaded = after.get("loaded_checkpoint", {})
        if (
            not saved.get("sha256")
            or saved.get("sha256") != loaded.get("sha256")
            or int(saved.get("bytes", 0)) != int(loaded.get("bytes", -1))
        ):
            raise AssertionError(
                f"loaded checkpoint differs from the saved file: {before}, {after}"
            )
        if not Path(str(loaded.get("path", ""))).is_file():
            raise AssertionError(f"checkpoint file disappeared: {loaded}")
        if (
            not _close(after.get("restored_parameters"), EXPECTED_STEP_ONE_PARAMETERS)
            or not _close(after.get("restored_momentum"), EXPECTED_STEP_ONE_MOMENTUM)
            or not _close(
                after.get("gathered_restored_state"), EXPECTED_RESTORED_STATE
            )
            or not _close(after.get("gradient_after_step_two"), EXPECTED_GRADIENT)
            or not _close(
                after.get("parameters_after_step_two"),
                EXPECTED_STEP_TWO_PARAMETERS,
            )
            or not _close(
                after.get("momentum_after_step_two"),
                EXPECTED_STEP_TWO_MOMENTUM,
            )
            or not _close(
                after.get("gathered_state_after_step_two"), EXPECTED_FINAL_STATE
            )
        ):
            raise AssertionError(f"resumed training state was invalid: {after}")

    run_ids = {str(item.get("run_id", "")) for item in initial + restarted}
    if len(run_ids) != 1 or not next(iter(run_ids)):
        raise AssertionError(f"elastic run IDs were inconsistent: {run_ids}")

    return {
        "schema_version": "fakegpu.elastic_ddp_checkpoint_validation.v1",
        "status": "success",
        "backend": "gloo",
        "world_size": 2,
        "failed_rank": 1,
        "failure_exit_code": 86,
        "restart_count": 1,
        "run_id": next(iter(run_ids)),
        "completed_steps": 2,
        "initial_pids": [int(item["pid"]) for item in initial],
        "restarted_pids": [int(item["pid"]) for item in restarted],
        "restored_parameters": restarted[0]["restored_parameters"],
        "restored_momentum": restarted[0]["restored_momentum"],
        "parameters_after_step_two": restarted[0]["parameters_after_step_two"],
        "momentum_after_step_two": restarted[0]["momentum_after_step_two"],
        "initial_reports": initial,
        "restarted_reports": restarted,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate local two-worker DDP checkpoint recovery after a "
            "torchrun restart."
        )
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    with tempfile.TemporaryDirectory(
        prefix="fakegpu-elastic-ddp-checkpoint-"
    ) as raw:
        report_dir = Path(raw) / "reports"
        checkpoint_dir = Path(raw) / "checkpoints"
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "1"
        if sys.platform == "darwin":
            env["GLOO_SOCKET_IFNAME"] = "lo0"
        master_port = _unused_local_port()
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--master-addr=127.0.0.1",
            f"--master-port={master_port}",
            "--max-restarts=1",
            str(WORKER),
            f"--report-dir={report_dir}",
            f"--checkpoint-dir={checkpoint_dir}",
            "--node-name=local",
            "--backend=gloo",
            "--trace-store",
            "--fail-rank=1",
            "--survivor-wait-seconds=10",
        ]
        completed = _run_torchrun(command, env=env, timeout=args.timeout)
        if completed.returncode != 0:
            raise AssertionError(
                f"torchrun exited with {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        summary = _validate_reports(report_dir)

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
