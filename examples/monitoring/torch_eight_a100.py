"""Real CPU-backed torch tensors on eight simulated A100s, with changing usage.

With the background machine (``fakegpu console`` or ``fakegpu daemon start``)
running, this script joins it; watch it in the console or ``fakegpu nvitop``.
Without a machine it publishes to /tmp/fakegpu-smi:

Terminal 1: python examples/monitoring/torch_eight_a100.py
Terminal 2: fakegpu nvitop --state-dir /tmp/fakegpu-smi --telemetry reference
Use --cycles 1 for a finite run. Ctrl+C stops a continuous demo.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", help="Default: the running machine, else /tmp/fakegpu-smi.")
    parser.add_argument("--phase-seconds", type=float, default=8)
    parser.add_argument("--cycles", type=int, default=0, help="0 repeats until Ctrl+C.")
    args = parser.parse_args()
    if args.phase_seconds <= 0 or args.cycles < 0:
        parser.error("phase seconds must be positive and cycles nonnegative")
    from fakegpu.machine.client import daemon_pid
    from fakegpu.machine.config import MachinePaths

    if args.state_dir or daemon_pid(MachinePaths.from_env()) is None:
        os.environ["FAKEGPU_SMI_STATE_DIR"] = args.state_dir or "/tmp/fakegpu-smi"
        os.environ.pop("FAKEGPU_SMI_STATE_PATH", None)
    os.environ["FAKEGPU_TELEMETRY"] = "reference"
    os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"

    import fakegpu

    fakegpu.init(profile="a100", device_count=8)
    import torch

    torch.set_num_threads(2)
    from fakegpu.torch_patch import _smi_publisher

    base, extra = [], []

    def report(phase):
        gc.collect()
        torch.cuda.empty_cache()
        if _smi_publisher is not None:
            _smi_publisher.publish_once(running=True)
        print(json.dumps({
            "phase": phase, "devices": 8,
            "allocated_mib": [torch.cuda.memory_allocated(i) // 1024**2 for i in range(8)],
            "reserved_mib": [torch.cuda.memory_reserved(i) // 1024**2 for i in range(8)],
        }), flush=True)

    def hold(active):
        deadline = time.monotonic() + args.phase_seconds
        while time.monotonic() < deadline:
            if active:
                # Actual torch operations, backed by host memory. No large
                # imaginary allocations or fabricated GPU process counters.
                for tensor in base:
                    tensor.add_(0.001)
            else:
                time.sleep(0.1)

    cycle = 0
    try:
        report("idle")
        hold(False)
        while not args.cycles or cycle < args.cycles:
            base = [torch.ones((32 + 4 * i) * 1024**2 // 4, device=f"cuda:{i}") for i in range(8)]
            report("allocated")
            hold(True)
            extra = [torch.ones(64 * 1024**2 // 4, device=f"cuda:{i}") for i in range(8)]
            report("grown")
            hold(True)
            extra.clear()
            report("released_extra")
            hold(False)
            base.clear()
            report("released_all")
            hold(False)
            cycle += 1
    except KeyboardInterrupt:
        pass
    finally:
        base.clear()
        extra.clear()
        report("finished")
        if _smi_publisher is not None:
            _smi_publisher.stop()


if __name__ == "__main__":
    main()
