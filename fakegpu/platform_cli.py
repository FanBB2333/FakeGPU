"""Inspect host platforms and the slots a device layout lands in: ``fakegpu platform``."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from typing import Sequence

from ._cli import EXIT_SUCCESS, EXIT_USAGE, command_prog
from .machine.config import MachineConfig, MachinePaths, parse_devices
from .platforms import AUTO, PlatformError, ResolvedPlatform, load_platforms, resolve_platform


def _add_layout_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--devices", help="Device layout, e.g. a100:8 (default: the machine's layout).")
    parser.add_argument("--platform", help="auto or a platform ID (default: the machine's platform).")
    parser.add_argument("--pcie-gen", dest="pcie_generation", type=int, metavar="N",
                        help="PCIe generation of the PCIe slots (0 keeps the platform's).")
    parser.add_argument("--nvlink-bridges", metavar="PAIRS", help="NVLink bridges between PCIe cards, e.g. 0-1,2-3.")


def _resolve(args: argparse.Namespace) -> ResolvedPlatform:
    config = MachineConfig.load(MachinePaths.from_env().config)
    devices = args.devices or config.devices
    explicit_devices = args.devices is not None
    platform = args.platform or (AUTO if explicit_devices else config.platform)
    return resolve_platform(
        parse_devices(devices),
        platform,
        pcie_generation=args.pcie_generation if args.pcie_generation is not None else config.pcie_generation,
        nvlink_bridges=args.nvlink_bridges if args.nvlink_bridges is not None else (
            "" if explicit_devices else config.nvlink_bridges
        ),
    )


def _print_list() -> None:
    rows = [(p.id, p.kind, str(p.capacity), p.name) for p in load_platforms().values()]
    _print_rows([("ID", "Kind", "GPUs", "Name"), *rows])
    print()
    print("auto picks a platform by form factor: SXM modules go on the HGX/DGX baseboard built for them,")
    print("superchips on their Grace platform, and PCIe cards in desktop, workstation or server slots.")


def _print_resolved(resolved: ResolvedPlatform) -> None:
    document = resolved.to_dict()
    print(f"Platform: {document['name']} ({document['id']})")
    print(f"Slots: {document['slot_layout']}")
    print(f"Wiring: {document['summary']}")
    print()
    header = ("GPU", "Slot", "Profile", "PCIe", "GPU max", "Host link GB/s", "NUMA", "NVLinks")
    rows = [header]
    for device in document["devices"]:
        host = device["host_bandwidth_gbps"]
        rows.append(
            (
                str(device["index"]),
                f"{device['slot']} {device['slot_type']}",
                device["profile_id"],
                device["pcie"] or "-",
                device["gpu_max_pcie"] or "-",
                f"{device['host_link']} {host:g}" if host else device["host_link"],
                str(device["numa_node"]),
                f"{device['nvlink_links']} ({device['nvlink_bandwidth_gbps']:g} GB/s)" if device["nvlink_links"] else "-",
            )
        )
    _print_rows(rows)
    if len(document["matrix"]) > 1:
        print()
        count = len(document["matrix"])
        _print_rows(
            [("", *(f"GPU{index}" for index in range(count)))]
            + [(f"GPU{index}", *cells) for index, cells in enumerate(document["matrix"])]
        )
    for note in document["notes"]:
        print(f"note: {note}")


def _print_rows(rows: Sequence[Sequence[str]]) -> None:
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    for row in rows:
        print("  ".join(value.ljust(widths[column]) for column, value in enumerate(row)).rstrip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__),
        description=(
            "List host platforms (HGX/DGX baseboards, Grace superchips, PCIe servers, workstations, "
            "desktops) and show which slots, PCIe links and NVLinks a device layout gets."
        ),
    )
    actions = parser.add_subparsers(dest="action", required=True)
    listing = actions.add_parser("list", help="List the platform templates.")
    listing.add_argument("--json", action="store_true", help="Print the templates as JSON.")
    show = actions.add_parser("show", help="Place a device layout into a platform and show its links.")
    _add_layout_options(show)
    show.add_argument("--json", action="store_true", help="Print the placement as JSON.")
    env = actions.add_parser(
        "env",
        help="Print shell exports that give a FakeCUDA process this platform's topology.",
    )
    _add_layout_options(env)
    args = parser.parse_args(argv)

    if args.action == "list":
        if args.json:
            print(json.dumps(
                [
                    {
                        "id": p.id,
                        "name": p.name,
                        "kind": p.kind,
                        "slots": p.capacity,
                        "slot_layout": p.describe_slots(),
                        "accepts": list(p.accepts),
                        "summary": p.summary,
                        "sources": list(p.sources),
                    }
                    for p in load_platforms().values()
                ],
                indent=2,
            ))
        else:
            _print_list()
        return EXIT_SUCCESS
    try:
        resolved = _resolve(args)
    except (PlatformError, ValueError) as exc:
        print(f"fakegpu platform: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if args.action == "env":
        print(f"export FAKEGPU_PROFILES={shlex.quote(','.join(resolved.profiles))}")
        for name, value in resolved.environment().items():
            print(f"export {name}={shlex.quote(value)}")
        return EXIT_SUCCESS
    if args.json:
        print(json.dumps(resolved.to_dict(), indent=2))
    else:
        _print_resolved(resolved)
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
