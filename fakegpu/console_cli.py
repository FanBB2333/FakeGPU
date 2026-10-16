"""Open the terminal console: ``fakegpu console``."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ._cli import EXIT_FAILURE, EXIT_SUCCESS, EXIT_USAGE, command_prog
from .daemon_cli import add_machine_options, machine_changes
from .machine.client import MachineClient, MachineError, daemon_pid
from .machine.config import MachinePaths
from .machine.daemon import start_daemon, stop_daemon


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__),
        description=(
            "Watch and control the simulated GPU machine: devices, processes and events. "
            "Starts the machine if it is not running. With background on (the default), "
            "the machine keeps running after you quit; stop it with Q or 'fakegpu daemon stop'."
        ),
    )
    add_machine_options(parser)
    parser.add_argument("--interval", type=float, default=0.5, help="Refresh seconds (default: 0.5).")
    args = parser.parse_args(argv)
    try:
        from .console.app import ConsoleApp
    except ImportError:
        print("fakegpu console needs Textual: pip install 'fakegpu[console]'", file=sys.stderr)
        return EXIT_USAGE

    paths = MachinePaths.from_env()
    client = MachineClient(paths)
    try:
        start_daemon(paths, changes=machine_changes(args))
        result = ConsoleApp(client, interval=args.interval).run()
        if daemon_pid(paths) is None:
            print("FakeGPU machine is not running.")
        elif result == "stop" or not client.call("config.get")["background"]:
            stop_daemon(paths)
            print("FakeGPU machine stopped.")
        else:
            print(
                "FakeGPU machine keeps running in the background. "
                "Reopen it with 'fakegpu console', stop it with 'fakegpu daemon stop'."
            )
    except MachineError as exc:
        print(f"fakegpu console: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except ValueError as exc:
        print(f"fakegpu console: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
