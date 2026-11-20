"""Verify actual torch allocations on eight synthetic A100s in live nvitop.

Only redacted numeric summaries are printed. No terminal capture is saved.
"""

from __future__ import annotations

import argparse
import codecs
import fcntl
import json
import os
from pathlib import Path
import pty
import queue
import select
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time


def eventually(check, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.05)
    raise AssertionError("monitor did not converge")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-dir")
    args = parser.parse_args()
    from fakegpu.nvml import NvmlStateBridge
    import pynvml
    import pyte

    python = [sys.executable, *(["-I"] if sys.flags.isolated else [])]
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="fakegpu-a100-monitor-") as directory:
        env = dict(os.environ, FAKEGPU_TERMINAL_REPORT="0", TERM="xterm-256color")
        worker = subprocess.Popen([
            *python, str(root / "examples/monitoring/torch_eight_a100.py"),
            "--state-dir", directory, "--phase-seconds", "4", "--cycles", "1",
        ], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        events = queue.Queue()

        def read_events():
            for line in worker.stdout:
                if line.startswith("{"):
                    events.put(json.loads(line))

        threading.Thread(target=read_events, daemon=True).start()
        master, slave = pty.openpty()
        viewer = None
        stop = threading.Event()
        output = bytearray()
        screen = pyte.Screen(160, 60)
        terminal = pyte.Stream(screen)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        def drain():
            while not stop.is_set():
                if select.select([master], [], [], 0.1)[0]:
                    data = os.read(master, 65536)
                    output.extend(data)
                    terminal.feed(decoder.decode(data))

        reader = threading.Thread(target=drain, daemon=True)
        try:
            first = events.get(timeout=25)
            assert first["phase"] == "idle"
            with NvmlStateBridge(state_dir=directory, lib_dir=args.lib_dir, telemetry="reference", refresh_interval=0.05) as bridge:
                pynvml.nvmlInit()
                from nvitop import Device, NA
                from nvitop.api.utils import bytes2human

                devices = Device.all()
                assert len(devices) == 8
                assert all(d.memory_total() == 80 * 1024**3 for d in devices)
                assert all("A100" in d.name() for d in devices)
                assert all(d.temperature() != NA and d.power_usage() != NA for d in devices)
                assert all(d.power_limit() == 400000 for d in devices)
                assert all(d.fan_speed() == NA for d in devices)
                fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 60, 160, 0, 0))
                command = [*python, "-m", "fakegpu", "nvitop", "--state-dir", directory, "--telemetry", "reference"]
                if args.lib_dir:
                    command += ["--lib-dir", args.lib_dir]
                viewer = subprocess.Popen([*command, "--", "--monitor", "--readonly", "--interval", "0.25"],
                                          stdin=slave, stdout=slave, stderr=slave, env=env, start_new_session=True)
                reader.start()
                eventually(lambda: b"\x1b[?1049h" in output)
                # The header tells viewers that the sensor readings are modeled.
                eventually(lambda: "Driver Version: FakeGPU (modeled)" in "\n".join(screen.display))
                saw_activity = False
                observed = []
                for phase in ("allocated", "grown", "released_extra", "released_all"):
                    event = events.get(timeout=20)
                    assert event["phase"] == phase
                    base = [32 + 4 * i for i in range(8)]
                    expected = [value + 64 for value in base] if phase == "grown" else base
                    if phase == "released_all":
                        expected = [0] * 8
                    assert event["allocated_mib"] == expected
                    eventually(lambda: [d.memory_used() // 1024**2 for d in devices] == expected)
                    assert all(worker.pid in d.processes() for d in devices)
                    # Reconstruct curses' incremental updates and check the
                    # same screen through growth AND both release phases.
                    patterns = {bytes2human(value * 1024**2) for value in expected}
                    eventually(lambda: all(value in "\n".join(screen.display) for value in patterns))
                    if phase in {"allocated", "grown"}:
                        eventually(lambda: any(d.gpu_utilization() > 0 for d in devices))
                        saw_activity = True
                        assert all(0 <= d.gpu_utilization() <= 100 for d in devices)
                        assert all(56000 <= d.power_usage() <= 400000 for d in devices)
                        samples = pynvml.nvmlDeviceGetProcessUtilization(devices[0]._handle, 0)
                        assert samples and samples[0].pid == worker.pid
                        assert 0 <= samples[0].smUtil <= 100
                    observed.append({
                        "phase": phase, "per_device_mib": expected,
                        "modeled_gpu_percent": [d.gpu_utilization() for d in devices],
                        "modeled_power_w": [round(d.power_usage() / 1000, 2) for d in devices],
                        "modeled_temperature_c": [d.temperature() for d in devices],
                    })
                assert saw_activity and bridge.last_error is None
                eventually(lambda: all(d.gpu_utilization() == 0 for d in devices))
                os.write(master, b"q")
                assert viewer.wait(timeout=10) == 0
                assert worker.wait(timeout=10) == 0
                eventually(lambda: all(not d.processes() for d in devices))
                pynvml.nvmlShutdown()
                print(json.dumps({"passed": True, "synthetic_devices": 8, "observed": observed,
                                  "tui_live_changes": True, "reference_sensors": True}, indent=2))
        finally:
            for process in (viewer, worker):
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
            stop.set()
            if reader.is_alive():
                reader.join(timeout=2)
            os.close(master)
            os.close(slave)


if __name__ == "__main__":
    main()
