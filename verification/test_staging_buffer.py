#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_BIN = REPO_ROOT / "build" / "fakegpu_staging_buffer_probe"


def run_checked(args: list[str]) -> None:
    completed = subprocess.run(
        args,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise AssertionError(f"command failed with exit code {completed.returncode}: {' '.join(args)}")


def main() -> int:
    if not PROBE_BIN.exists():
        print(f"missing staging buffer probe: {PROBE_BIN}", file=sys.stderr)
        return 2

    payload = bytes(range(256))
    payload_hex = payload.hex()

    with tempfile.TemporaryDirectory(prefix="fakegpu-staging-buffer-") as tmpdir:
        ready_file = Path(tmpdir) / "ready"
        done_file = Path(tmpdir) / "done"
        shm_name = f"/fakegpu-staging-{os.getpid()}-{time.time_ns()}"

        reader = subprocess.Popen(
            [
                str(PROBE_BIN),
                "--read",
                "--name",
                shm_name,
                "--ready-file",
                str(ready_file),
                "--done-file",
                str(done_file),
                "--expected-hex",
                payload_hex,
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        writer = subprocess.Popen(
            [
                str(PROBE_BIN),
                "--write",
                "--name",
                shm_name,
                "--ready-file",
                str(ready_file),
                "--done-file",
                str(done_file),
                "--payload-hex",
                payload_hex,
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if writer.poll() is not None and reader.poll() is not None:
                break
            time.sleep(0.05)
        if writer.poll() is None:
            writer.kill()
        if reader.poll() is None:
            reader.kill()

        writer_stdout, writer_stderr = writer.communicate()
        reader_stdout, reader_stderr = reader.communicate()
        if writer.returncode is None or reader.returncode is None:
            raise AssertionError("writer or reader did not terminate before the timeout")
        if writer.returncode != 0:
            sys.stderr.write(writer_stdout)
            sys.stderr.write(writer_stderr)
            raise AssertionError(f"writer failed with exit code {writer.returncode}")
        if reader.returncode != 0:
            sys.stderr.write(reader_stdout)
            sys.stderr.write(reader_stderr)
            raise AssertionError(f"reader failed with exit code {reader.returncode}")

        run_checked([str(PROBE_BIN), "--probe-missing", "--name", shm_name])
        print("staging buffer test passed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
