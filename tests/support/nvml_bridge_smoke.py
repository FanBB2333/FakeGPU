"""Exercise real publishers, bundled NVML, unmodified pynvml, and pip nvitop.

Run with the installed package outside the checkout (python -I), or with
PYTHONPATH=. and --lib-dir pointing at a fresh native build. Output is redacted.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


WORKER = """
import gc, sys
import fakegpu
fakegpu.init(profile="l4", device_count=2)
import torch
tensors = [torch.empty(4 * 1024 * 1024, device="cuda:0"),
           torch.empty(2 * 1024 * 1024, device="cuda:1")]
print("ready", flush=True)
for command in sys.stdin:
    if command.strip() == "free":
        tensors.clear()
        gc.collect()
        torch.cuda.empty_cache()
        print("freed", flush=True)
    else:
        break
"""


def eventually(check, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.05)
    raise AssertionError("NVML state did not converge before timeout")


def check_process_abi(library, handle, expected_pids):
    class V1(ctypes.Structure):
        _fields_ = [("pid", ctypes.c_uint), ("used", ctypes.c_ulonglong)]

    class V2(ctypes.Structure):
        _fields_ = V1._fields_ + [("gpu", ctypes.c_uint), ("compute", ctypes.c_uint)]

    for suffix, row in (("", V1), ("_v2", V2), ("_v3", V2)):
        fn = getattr(library, "nvmlDeviceGetComputeRunningProcesses" + suffix)
        count = ctypes.c_uint(0)
        assert fn(handle, ctypes.byref(count), None) == 7  # insufficient size
        assert count.value == len(expected_pids)
        capacity = count.value
        size = capacity * ctypes.sizeof(row)
        buffer = (ctypes.c_ubyte * (size + 16))(*([0xA5] * (size + 16)))
        count.value = capacity - 1
        assert fn(handle, ctypes.byref(count), buffer) == 7
        assert list(buffer) == [0xA5] * (size + 16)  # no partial/overrun write
        assert fn(handle, ctypes.byref(count), buffer) == 0
        assert list(buffer)[size:] == [0xA5] * 16
        rows = ctypes.cast(buffer, ctypes.POINTER(row))
        assert {rows[i].pid for i in range(capacity)} == expected_pids
        assert all(rows[i].used == 16 * 1024**2 for i in range(capacity))
        if suffix:
            assert all(rows[i].gpu == rows[i].compute == 0xFFFFFFFF for i in range(capacity))


def check_tui(command, env, pids):
    import fcntl
    import pty
    import select
    import struct
    import termios

    master, slave = pty.openpty()
    viewer = None
    try:
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 160, 0, 0))
        viewer = subprocess.Popen(
            [*command, "--", "--monitor", "--readonly", "--interval", "0.25"],
            stdin=slave, stdout=slave, stderr=slave,
            env=dict(env, TERM="xterm-256color"), start_new_session=True,
        )
        output = bytearray()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and viewer.poll() is None:
            if select.select([master], [], [], 0.2)[0]:
                output.extend(os.read(master, 65536))
            if all(str(pid).encode() in output for pid in pids) and b"\x1b[?1049h" in output:
                break
        assert b"\x1b[?1049h" in output, "nvitop did not enter its interactive screen"
        assert all(str(pid).encode() in output for pid in pids), "TUI omitted a publisher"
        os.write(master, b"q")
        deadline = time.monotonic() + 10
        # Keep draining while curses paints/restores the screen. Waiting with
        # unread PTY output can block the child before it handles the key.
        while viewer.poll() is None and time.monotonic() < deadline:
            if select.select([master], [], [], 0.2)[0]:
                os.read(master, 65536)
        assert viewer.poll() == 0, "nvitop TUI did not exit cleanly with q"
    finally:
        if viewer is not None and viewer.poll() is None:
            viewer.kill()
            viewer.wait(timeout=10)
        os.close(master)
        os.close(slave)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-dir")
    parser.add_argument("--tui", action="store_true", help="Also exercise curses in a pseudo-terminal.")
    args = parser.parse_args()
    from fakegpu.nvml import NvmlStateBridge
    import pynvml

    python = [sys.executable, *( ["-I"] if sys.flags.isolated else [])]
    with tempfile.TemporaryDirectory(prefix="fakegpu-nvml-") as directory:
        env = dict(os.environ, FAKEGPU_SMI_STATE_DIR=directory, FAKEGPU_TERMINAL_REPORT="0")
        env.pop("FAKEGPU_SMI_STATE_PATH", None)
        workers = []
        try:
            for _ in range(2):
                worker = subprocess.Popen(
                    [*python, "-u", "-c", WORKER], env=env, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                workers.append(worker)
                assert worker.stdout.readline().strip() == "ready", "worker startup failed"
            with NvmlStateBridge(state_dir=directory, lib_dir=args.lib_dir, refresh_interval=0.05) as bridge:
                pynvml.nvmlInit()
                assert pynvml.nvmlDeviceGetCount() == 2
                handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(2)]
                pids = {worker.pid for worker in workers}
                eventually(lambda: [pynvml.nvmlDeviceGetMemoryInfo(h).used for h in handles]
                           == [32 * 1024**2, 40 * 1024**2])
                # The caching allocator reserves a 20 MiB segment for each 8 MiB tensor.
                assert pynvml.nvmlDeviceGetMemoryInfo(handles[1]).used == 40 * 1024**2
                for handle in handles:
                    assert {p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)} == pids
                    assert pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle) == []
                    uuid = pynvml.nvmlDeviceGetUUID(handle)
                    by_uuid = pynvml.nvmlDeviceGetHandleByUUID(uuid)
                    assert ctypes.cast(handle, ctypes.c_void_p).value == ctypes.cast(by_uuid, ctypes.c_void_p).value
                check_process_abi(bridge._library, handles[0], pids)

                from nvitop import Device, NA

                devices = Device.all()
                assert len(devices) == 2
                assert devices[0].memory_used() == 32 * 1024**2
                assert set(devices[0].processes()) == pids
                assert devices[0].gpu_utilization() == NA
                assert devices[0].temperature() == NA
                assert devices[0].power_usage() == NA

                command = [*python, "-m", "fakegpu", "nvitop", "--state-dir", directory]
                if args.lib_dir:
                    command += ["--lib-dir", args.lib_dir]
                result = subprocess.run([*command, "--", "--once"], env=env, text=True,
                                        capture_output=True, timeout=30)
                assert result.returncode == 0, "nvitop --once failed: " + result.stderr
                assert all(str(pid) in result.stdout for pid in pids), "nvitop omitted a publisher"
                assert "L4" in result.stdout and "FakeGPU" in result.stdout
                if args.tui:
                    check_tui(command, env, pids)

                workers[0].stdin.write("free\n")
                workers[0].stdin.flush()
                assert workers[0].stdout.readline().strip() == "freed"
                eventually(lambda: pynvml.nvmlDeviceGetMemoryInfo(handles[0]).used == 16 * 1024**2)
                workers[1].kill()  # Abrupt exit leaves a running=true state file.
                workers[1].wait(timeout=10)
                eventually(lambda: pynvml.nvmlDeviceGetMemoryInfo(handles[0]).used == 0)
                eventually(lambda: {p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handles[0])}
                           == {workers[0].pid})
                workers[0].stdin.write("exit\n")
                workers[0].stdin.flush()
                workers[0].wait(timeout=10)
                eventually(lambda: pynvml.nvmlDeviceGetComputeRunningProcesses(handles[0]) == [])
                assert pynvml.nvmlDeviceGetCount() == 2  # Cached handles remain valid.
                assert pynvml.nvmlDeviceGetMemoryInfo(handles[0]).used == 0
                assert bridge.last_error is None
                assert len(list(Path(directory).glob("*.json"))) == 2  # Viewer never publishes itself.
                pynvml.nvmlShutdown()
            empty = subprocess.run([*command, "--", "--once"], env=env, text=True,
                                   capture_output=True, timeout=30)
            assert empty.returncode == 1 and "No live local FakeGPU publishers" in empty.stderr
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                worker.wait(timeout=10)
    print("PASS: two publishers / two synthetic devices; NVML v1/v2/v3 ABI; nvitop API/CLI; "
          "live memory updates; graceful/abrupt exit cleanup; no hardware telemetry; empty-state guidance.")
    if args.tui:
        print("PASS: interactive nvitop displayed both publishers and exited with q.")


if __name__ == "__main__":
    main()
