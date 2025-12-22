from __future__ import annotations

import argparse
import os
import sys

from ._api import env as fakegpu_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu",
        description="Run a command with FakeGPU libraries preloaded (LD_PRELOAD).",
    )
    parser.add_argument(
        "--build-dir",
        help="Path to a FakeGPU CMake build directory containing the .so files (default: $FAKEGPU_BUILD_DIR or ./build).",
    )
    parser.add_argument(
        "--lib-dir",
        help="Path to a directory containing FakeGPU shared libraries (takes precedence over --build-dir).",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")

    ns = parser.parse_args(argv)

    cmd = list(ns.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("missing command to run")

    child_env = fakegpu_env(build_dir=ns.build_dir, lib_dir=ns.lib_dir)
    os.execvpe(cmd[0], cmd, child_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

