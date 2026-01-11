from __future__ import annotations

import argparse
import os
import sys

from ._api import env as fakegpu_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu",
        description="Run a command with FakeGPU libraries preloaded (LD_PRELOAD/DYLD_INSERT_LIBRARIES).",
    )
    parser.add_argument(
        "--mode",
        choices=["simulate", "passthrough", "hybrid"],
        help="FakeGPU mode. Equivalent to setting $FAKEGPU_MODE.",
    )
    parser.add_argument(
        "--oom-policy",
        choices=["clamp", "managed", "mapped_host", "spill_cpu"],
        help="Hybrid OOM policy. Equivalent to setting $FAKEGPU_OOM_POLICY.",
    )
    parser.add_argument(
        "--build-dir",
        help="Path to a FakeGPU CMake build directory containing the shared libraries (default: $FAKEGPU_BUILD_DIR or ./build).",
    )
    parser.add_argument(
        "--lib-dir",
        help="Path to a directory containing FakeGPU shared libraries (takes precedence over --build-dir).",
    )
    parser.add_argument(
        "--profile",
        help="GPU preset ID for all devices (e.g. a100, h100). Equivalent to setting $FAKEGPU_PROFILE.",
    )
    parser.add_argument(
        "--device-count",
        type=int,
        help="Number of fake devices to expose (default: 8). Equivalent to setting $FAKEGPU_DEVICE_COUNT.",
    )
    parser.add_argument(
        "--devices",
        help="Device preset spec, e.g. 'a100:4,h100:4' or 't4,l40s'. Equivalent to setting $FAKEGPU_PROFILES.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")

    ns = parser.parse_args(argv)

    cmd = list(ns.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("missing command to run")

    child_env = fakegpu_env(
        build_dir=ns.build_dir,
        lib_dir=ns.lib_dir,
        mode=ns.mode,
        oom_policy=ns.oom_policy,
        profile=ns.profile,
        device_count=ns.device_count,
        devices=ns.devices,
    )
    os.execvpe(cmd[0], cmd, child_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
