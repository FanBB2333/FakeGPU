"""Run the installed nvitop against FakeGPU's cross-process NVML bridge."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from ._cli import command_prog
from .machine.client import daemon_pid
from .machine.config import MachineConfig, MachinePaths
from .machine.layout import is_machine_state_dir, layout_state
from .nvml import NvmlStateBridge


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__),
        description="View local simulated GPU publishers in the installed nvitop (macOS/Linux).",
        epilog="Pass nvitop options after --, e.g. -- --once or -- --monitor. Quit the TUI with q.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--state-dir",
        help="Publisher directory (default: FAKEGPU_SMI_STATE_DIR, else the running fakegpu daemon).",
    )
    parser.add_argument("--state", action="append", default=[], help="Explicit publisher JSON file; repeatable.")
    parser.add_argument("--refresh-interval", type=float, default=0.25, help="State refresh seconds (default: 0.25).")
    parser.add_argument("--stale-after-seconds", type=float, default=2.0, help="Ignore older heartbeats (default: 2).")
    parser.add_argument("--lib-dir", help="Override bundled FakeGPU native library directory.")
    parser.add_argument("--telemetry", choices=("none", "reference"),
                        help="Modeled sensors: A100 reference readings, a default curve for other "
                             "profiles (default: the daemon's setting, else none); "
                             "publishers need FAKEGPU_TELEMETRY=reference.")
    args, remaining = parser.parse_known_args(argv)
    paths = MachinePaths.from_env()
    state_dir = args.state_dir or os.environ.get("FAKEGPU_SMI_STATE_DIR")
    if not (state_dir or args.state or os.environ.get("FAKEGPU_SMI_STATE_PATH")):
        if daemon_pid(paths) is not None:
            state_dir = str(paths.state_dir)
    # The machine lists its GPUs even when no process uses them.
    uses_machine = is_machine_state_dir(state_dir, paths)
    layout = (lambda: layout_state(paths)) if uses_machine else None
    if args.telemetry is None:
        args.telemetry = MachineConfig.load(paths.config).telemetry if uses_machine else "none"
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    old_argv = sys.argv
    try:
        with NvmlStateBridge(
            state_dir=state_dir, state_paths=args.state,
            refresh_interval=args.refresh_interval, stale_after_seconds=args.stale_after_seconds,
            lib_dir=args.lib_dir, telemetry=args.telemetry, layout=layout,
        ) as bridge:
            if bridge.device_count == 0:
                print("No live local FakeGPU publishers and no running machine. Start the machine "
                      "with 'fakegpu daemon start', or a workload with the same FAKEGPU_SMI_STATE_DIR, "
                      "then restart this viewer.", file=sys.stderr)
                if bridge.errors:
                    print(f"Ignored {len(bridge.errors)} unreadable/invalid state file(s).", file=sys.stderr)
                return 1
            from nvitop.cli import main as nvitop_main

            label = ("modeled sensors, A100 reference or default curve "
                     "(CPU activity proxy, not hardware measurements)"
                     if args.telemetry == "reference" else "hardware telemetry is N/A")
            print(f"FakeGPU NVML: simulated memory/processes; {label}.", file=sys.stderr)
            sys.argv = ["nvitop", *remaining]
            return nvitop_main()
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"fakegpu nvitop: {exc}", file=sys.stderr)
        if isinstance(exc, ImportError):
            print("Install: pip install 'fakegpu[monitoring]'", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
