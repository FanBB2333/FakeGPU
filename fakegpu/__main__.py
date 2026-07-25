from __future__ import annotations

import argparse
import importlib
import os
import sys

from ._api import env as fakegpu_env
from ._api import _warn_if_macos_injection_may_be_blocked


_BUILTIN_HANDLERS = {
    "demo": (".demo", "main"),
    "doctor": (".doctor", "main"),
    "preflight": (".preflight", "main"),
    "coordinator": (".distributed_cli", "coordinator_main"),
    "bandwidth": (".distributed_cli", "bandwidth_main"),
    "estimate-llm": (".llm_cli", "main"),
    "estimate-roofline": (".performance_model", "main"),
    "analyze-repo": (".repository_analyzer", "main"),
    "analyze-kernel": (".kernel_analysis", "main"),
    "calibrate": (".calibration", "main"),
    "plan-training": (".training_plan", "main"),
    "simulate-topology": (".topology", "main"),
    "replay-trace": (".trace_replay", "main"),
    "nvidia-smi": (".smi", "main"),
    "workspace-profiles": (".workspace_cli", "main"),
    "validate": (".validation", "main"),
    "capabilities": (".capabilities", "main"),
}
BUILTIN_COMMANDS = tuple(_BUILTIN_HANDLERS)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    handler_reference = _BUILTIN_HANDLERS.get(argv[0]) if argv else None
    if handler_reference is not None:
        module_name, function_name = handler_reference
        module = importlib.import_module(module_name, package=__package__)
        handler = getattr(module, function_name)
        return handler(argv[1:])

    parser = argparse.ArgumentParser(
        prog="fakegpu",
        description="Run a command with FakeGPU libraries preloaded (LD_PRELOAD/DYLD_INSERT_LIBRARIES).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Built-in commands:\n"
            + "".join(f"  fakegpu {command}\n" for command in BUILTIN_COMMANDS)
            + "\nRun 'fakegpu <command> --help' for details."
        ),
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
        "--unsupported-api",
        choices=["allow", "warn", "error"],
        help=(
            "Behavior for recognized native APIs that FakeGPU does not execute. "
            "Equivalent to setting $FAKEGPU_UNSUPPORTED_API."
        ),
    )
    parser.add_argument(
        "--dist-mode",
        choices=["disabled", "simulate", "proxy", "passthrough"],
        help="Distributed communication mode. Equivalent to setting $FAKEGPU_DIST_MODE.",
    )
    parser.add_argument(
        "--cluster-config",
        help="Path to a cluster topology config. Equivalent to setting $FAKEGPU_CLUSTER_CONFIG.",
    )
    parser.add_argument(
        "--coordinator-addr",
        help="Coordinator endpoint, e.g. 127.0.0.1:29591. Equivalent to setting $FAKEGPU_COORDINATOR_ADDR.",
    )
    parser.add_argument(
        "--coordinator-transport",
        choices=["tcp", "unix"],
        help="Coordinator transport. Equivalent to setting $FAKEGPU_COORDINATOR_TRANSPORT.",
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

    _warn_if_macos_injection_may_be_blocked(cmd[0])

    child_env = fakegpu_env(
        build_dir=ns.build_dir,
        lib_dir=ns.lib_dir,
        mode=ns.mode,
        oom_policy=ns.oom_policy,
        unsupported_api=ns.unsupported_api,
        dist_mode=ns.dist_mode,
        cluster_config=ns.cluster_config,
        coordinator_addr=ns.coordinator_addr,
        coordinator_transport=ns.coordinator_transport,
        profile=ns.profile,
        device_count=ns.device_count,
        devices=ns.devices,
    )
    os.execvpe(cmd[0], cmd, child_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
