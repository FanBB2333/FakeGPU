"""Manage the background FakeGPU machine: ``fakegpu daemon <action>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Sequence

from ._cli import EXIT_FAILURE, EXIT_SUCCESS, EXIT_USAGE, command_prog
from .machine.client import MachineClient, MachineError, daemon_pid
from .machine.config import TELEMETRY_MODES, MachineConfig, MachinePaths


def _bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"on", "true", "yes", "1"}:
        return True
    if lowered in {"off", "false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError("use on or off")


def add_machine_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--devices", help="Device layout, e.g. a100:8 or l40s:4.")
    parser.add_argument(
        "--platform",
        help="Host platform: auto (by form factor) or an ID from 'fakegpu platform list', e.g. hgx-a100-8.",
    )
    parser.add_argument(
        "--pcie-gen",
        dest="pcie_generation",
        type=int,
        metavar="N",
        help="PCIe generation of the PCIe slots (0 keeps the platform's).",
    )
    parser.add_argument(
        "--nvlink-bridges",
        metavar="PAIRS",
        help="NVLink bridges between PCIe cards, e.g. 0-1,2-3 ('' removes them).",
    )
    parser.add_argument("--telemetry", choices=TELEMETRY_MODES, help="Sensor model for temperature/power/clocks.")
    parser.add_argument(
        "--background",
        type=_bool,
        metavar="on|off",
        help="Keep the daemon running after the console exits.",
    )


def machine_changes(args: argparse.Namespace) -> dict[str, Any]:
    names = ("devices", "platform", "pcie_generation", "nvlink_bridges", "telemetry", "background")
    changes = {name: getattr(args, name) for name in names}
    return {name: value for name, value in changes.items() if value is not None}


def _gib(value: int) -> str:
    return f"{value / 1024**3:.1f}"


def _uptime(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds // 60 % 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def slot_label(slot: dict) -> str:
    if not slot:
        return "-"
    return f"{slot['type']} {slot['slot']}"


def link_label(slot: dict) -> str:
    if not slot:
        return "-"
    return slot.get("pcie") or slot.get("host_link") or "-"


def format_event(event: dict) -> str:
    level = {"warning": "WARN", "error": "ERROR"}.get(event["level"], "INFO")
    return f"{event['time']}  {level:<5}  {event['message']}"


def _print_status(status: dict) -> None:
    daemon, config = status["daemon"], status["config"]
    print(
        f"FakeGPU machine: running (pid {daemon['pid']}, up {_uptime(daemon['uptime_seconds'])}), "
        f"layout {config['devices']}, telemetry {config['telemetry']}, "
        f"background {'on' if config['background'] else 'off'}"
    )
    platform = status.get("platform") or {}
    if platform:
        print(f"Platform: {platform['name']} ({platform['id']}), {platform['slot_layout']}")
    print(f"State directory: {status['paths']['state_dir']}")
    print()
    print(
        f"{'GPU':<4}{'Profile':<16}{'Slot':<11}{'Link':<13}{'Memory GiB':>18}{'Util':>7}{'Temp':>7}"
        f"{'Power':>9}{'Procs':>7}"
    )
    for device in status["devices"]:
        memory, telemetry = device["memory"], device["telemetry"] or {}
        used = f"{_gib(memory['used_bytes'])}/{_gib(memory['total_bytes'])}"
        util = f"{telemetry['gpu_percent']}%" if telemetry else "N/A"
        temp = f"{telemetry['temperature_c']}C" if telemetry else "N/A"
        power = f"{telemetry['power_w']:.0f}W" if telemetry else "N/A"
        marker = "" if device["in_layout"] else " *"
        slot = device.get("slot") or {}
        print(
            f"{device['index']:<4}{device['profile_id'] + marker:<16}{slot_label(slot):<11}"
            f"{link_label(slot):<13}{used:>18}{util:>7}{temp:>7}{power:>9}{device['process_count']:>7}"
        )
    print()
    if not status["processes"]:
        print("No FakeCUDA processes. Scripts that call fakegpu.init() join this machine.")
    for process in status["processes"]:
        devices = ",".join(str(row["index"]) for row in process["devices"] if row["used_bytes"])
        print(
            f"{process['pid']:>7}  {process['status']:<8}{_gib(process['memory_bytes']):>7} GiB  "
            f"GPU {devices or '-':<10} {process['command'] or process['name']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__),
        description=(
            "Run a simulated GPU machine in the background. FakeCUDA processes started while it "
            "runs publish their state to it and, unless they choose devices, use its layout."
        ),
    )
    actions = parser.add_subparsers(dest="action", required=True)
    start = actions.add_parser("start", help="Start the daemon (or update a running one).")
    add_machine_options(start)
    actions.add_parser("stop", help="Stop the daemon; running workloads are not affected.")
    status = actions.add_parser("status", help="Show devices and processes.")
    status.add_argument("--json", action="store_true", help="Print the full status document.")
    events = actions.add_parser("events", help="Show recent events.")
    events.add_argument("-n", "--limit", type=int, default=50, help="Number of events (default: 50).")
    events.add_argument("-f", "--follow", action="store_true", help="Keep printing new events.")
    events.add_argument("--json", action="store_true", help="One JSON object per line.")
    config = actions.add_parser("config", help="Show or change the machine settings.")
    add_machine_options(config)
    config.add_argument("--json", action="store_true", help="Print settings as JSON.")
    actions.add_parser("run", help="Run the daemon in the foreground.")
    args = parser.parse_args(argv)

    paths = MachinePaths.from_env()
    client = MachineClient(paths)
    try:
        if args.action == "run":
            from .machine.daemon import run_daemon

            return run_daemon(paths)
        if args.action == "start":
            from .machine.daemon import start_daemon

            pid = start_daemon(paths, changes=machine_changes(args))
            config_now = client.call("config.get")
            print(f"FakeGPU machine running (pid {pid}): {config_now['devices']}, telemetry {config_now['telemetry']}")
            print("Open the console with: fakegpu console")
            return EXIT_SUCCESS
        if args.action == "stop":
            from .machine.daemon import stop_daemon

            print("FakeGPU machine stopped." if stop_daemon(paths) else "FakeGPU machine is not running.")
            return EXIT_SUCCESS
        if args.action == "config":
            changes = machine_changes(args)
            if daemon_pid(paths) is not None:
                settings = client.call("config.update", changes=changes) if changes else client.call("config.get")
            else:
                updated = MachineConfig.load(paths.config).updated(changes)
                if changes:
                    paths.create()
                    updated.save(paths.config)
                settings = updated.to_dict()
            if args.json:
                print(json.dumps(settings, indent=2))
            else:
                for name, value in settings.items():
                    print(f"{name}: {value}")
            return EXIT_SUCCESS
        if args.action == "status":
            if daemon_pid(paths) is None:
                print("FakeGPU machine is not running. Start it with: fakegpu daemon start")
                return EXIT_FAILURE
            document = client.call("status")
            if args.json:
                print(json.dumps(document, indent=2))
            else:
                _print_status(document)
            return EXIT_SUCCESS
        last = 0
        for event in client.call("events", after=0, limit=args.limit):
            print(json.dumps(event) if args.json else format_event(event))
            last = event["seq"]
        while args.follow:
            time.sleep(0.5)
            for event in client.call("events", after=last, limit=1000):
                print(json.dumps(event) if args.json else format_event(event), flush=True)
                last = event["seq"]
        return EXIT_SUCCESS
    except MachineError as exc:
        print(f"fakegpu daemon: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except ValueError as exc:
        print(f"fakegpu daemon: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
