#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = Path(os.environ.get("FAKEGPU_BUILD_DIR", REPO_ROOT / "build"))
COORDINATOR_BIN = BUILD_DIR / "fakegpu-coordinator"


def request(socket_path: str, payload: str) -> str:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(socket_path)
        sock.sendall((payload + "\n").encode("utf-8"))
        data = b""
        while not data.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
    return data.decode("utf-8").strip()


def main() -> int:
    if not COORDINATOR_BIN.exists():
        print(f"missing coordinator binary: {COORDINATOR_BIN}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(
        prefix="fgpu-coord-",
        dir="/tmp",
    ) as tmpdir:
        socket_path = os.path.join(tmpdir, "coordinator.sock")
        report_path = Path(tmpdir) / "cluster-report.json"
        markdown_path = Path(tmpdir) / "cluster-report.md"
        env = dict(os.environ)
        env["FAKEGPU_CLUSTER_REPORT_PATH"] = str(report_path)
        env["FAKEGPU_CLUSTER_REPORT_MARKDOWN_PATH"] = str(markdown_path)
        proc = subprocess.Popen(
            [str(COORDINATOR_BIN), "--transport", "unix", "--address", socket_path],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            deadline = time.time() + 1.0
            while time.time() < deadline:
                if os.path.exists(socket_path):
                    break
                time.sleep(0.05)
            if not os.path.exists(socket_path):
                raise AssertionError("socket was not created within 1 second")

            response = request(socket_path, "PING")
            if "OK" not in response or "status=ready" not in response or "version=1" not in response:
                raise AssertionError(f"unexpected ping response: {response}")

            response = request(socket_path, "SHUTDOWN")
            if "OK" not in response or "status=shutting_down" not in response:
                raise AssertionError(f"unexpected shutdown response: {response}")

            proc.wait(timeout=2.0)
            if os.path.exists(socket_path):
                raise AssertionError("socket path still exists after shutdown")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("schema_version") != "cluster_report.v1":
                raise AssertionError(f"unexpected empty report: {report}")
            if any(
                int(stats.get("calls", -1)) != 0
                for stats in report.get("collectives", {}).values()
            ):
                raise AssertionError(f"empty report recorded traffic: {report}")
            if (
                report.get("cluster", {}).get("communicators") != 0
                or report.get("node_pairs") != []
                or report.get("ranks") != []
                or report.get("operation_timeline", {}).get("retained_entries")
                != 0
                or not markdown_path.is_file()
            ):
                raise AssertionError(f"empty report is incomplete: {report}")

            print("coordinator smoke test passed")
            return 0
        finally:
            if proc.poll() is None:
                proc.kill()
            stdout, stderr = proc.communicate()
            if proc.returncode not in (None, 0):
                sys.stderr.write(stdout)
                sys.stderr.write(stderr)


if __name__ == "__main__":
    raise SystemExit(main())
