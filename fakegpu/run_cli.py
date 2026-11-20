"""``fakegpu run``: run an unmodified command on simulated GPUs.

Every Python process of the command, including torchrun or accelerate
workers, sees the simulated GPUs through FakeCUDA as soon as it imports
torch. The command's output streams as usual, ``nvidia-smi`` on its ``PATH``
shows the simulated GPUs, and after it exits a summary states whether the
run fit in GPU memory and where each number comes from.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Sequence

from ._cli import EXIT_USAGE, command_prog
from ._run_summary import build_report, load_records, render
from .machine.client import daemon_pid
from .machine.config import MachineConfig, MachinePaths
from .machine.layout import build_layout_state
from .profile_catalog import ProfileCatalogError, get_profile, load_profiles

DEFAULT_LAYOUT = ("a100", 8)
LAYOUT_REFRESH_SECONDS = 0.5
EXIT_NOT_FOUND = 127
EXIT_NOT_EXECUTABLE = 126


class GpuSpecError(ValueError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__),
        description=(
            "Run a command on simulated GPUs without changing it. Python processes of the "
            "command, including torchrun and accelerate workers, see the GPUs as soon as they "
            "import torch; nvidia-smi on the command's PATH lists them. Afterwards a summary "
            "states whether the run fit in GPU memory."
        ),
        epilog=(
            "Examples:\n"
            "  fakegpu run --gpu a100x8 -- python train.py\n"
            "  fakegpu run --gpu h100:4,a100:4 -- torchrun --nproc_per_node 8 train.py\n"
            "\n"
            "Without --gpu the run uses the background machine when it is running "
            "('fakegpu daemon start'), so 'fakegpu console' shows it; otherwise "
            "FAKEGPU_PROFILES or FAKEGPU_PROFILE, else 8 x a100.\n"
            "The exit status is the command's."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gpu",
        metavar="SPEC",
        help="GPUs to simulate: a100x8, 8xa100, a100:8, or a list such as h100:4,a100:4.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Keep the run's JSON report and per-process records in this directory.",
    )
    parser.add_argument("--no-summary", action="store_true", help="Do not print the summary.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="The command to run, after --.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    command = list(ns.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("missing command, e.g. fakegpu run --gpu a100x8 -- python train.py")
    try:
        profiles, machine = _resolve_layout(ns.gpu)
    except GpuSpecError as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if ns.report_dir is not None:
        run_dir = ns.report_dir.expanduser().resolve()
        # Only the run's own subdirectory is cleared of an earlier run's records.
        for stale in (run_dir / "records").glob("process-*.json"):
            stale.unlink()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = Path(tempfile.mkdtemp(prefix="fakegpu-run-"))
    try:
        env = _child_environment(run_dir, profiles, machine)
        publisher = None if machine else _LayoutPublisher(run_dir / "smi" / "layout.json", profiles, env)
        start = time.monotonic()
        try:
            child = subprocess.Popen(command, env=env)
            exit_code = _wait(child)
        except FileNotFoundError:
            print(f"{parser.prog}: command not found: {command[0]}", file=sys.stderr)
            return EXIT_NOT_FOUND
        except PermissionError:
            print(f"{parser.prog}: permission denied: {command[0]}", file=sys.stderr)
            return EXIT_NOT_EXECUTABLE
        finally:
            if publisher is not None:
                publisher.stop()
        report = build_report(
            command=command,
            exit_code=exit_code,
            duration_seconds=time.monotonic() - start,
            layout=_layout_rows(profiles),
            records=load_records(run_dir / "records"),
        )
        _hand_records_to_outer_run(run_dir / "records")
        report_path = None
        if ns.report_dir is not None:
            report_path = run_dir / "run_report.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        if not ns.no_summary:
            sys.stderr.write(render(report, report_path=report_path))
            sys.stderr.flush()
        return exit_code
    finally:
        if ns.report_dir is None:
            shutil.rmtree(run_dir, ignore_errors=True)


def parse_gpu_spec(spec: str) -> list[str]:
    """Expand ``a100x8``, ``8xa100``, ``a100:8`` or ``h100:4,a100:4`` to one profile ID per GPU."""

    catalog = load_profiles()
    profiles: list[str] = []
    for item in spec.split(","):
        item = item.strip().lower()
        if not item:
            continue
        profile_id, count = item, 1
        if item not in catalog:
            # Try both orders: 2xrtx4090 also reads as GPU '2xrt' x 4090.
            matches = [
                match for pattern in (r"(?P<id>.+?)\s*[:x×]\s*(?P<count>\d+)", r"(?P<count>\d+)\s*[x×]\s*(?P<id>.+)")
                if (match := re.fullmatch(pattern, item))
            ]
            match = next((match for match in matches if match["id"] in catalog), matches[0] if matches else None)
            if match:
                profile_id, count = match["id"], int(match["count"])
        if profile_id not in catalog:
            close = difflib.get_close_matches(profile_id, list(catalog), n=4, cutoff=0.6)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise GpuSpecError(
                f"unknown GPU {profile_id!r} in --gpu {spec!r}.{hint} "
                "List them with 'fakegpu doctor --list-profiles'."
            )
        if count < 1:
            raise GpuSpecError(f"GPU count must be at least 1 in --gpu {spec!r}")
        profiles.extend([profile_id] * count)
    if not profiles:
        raise GpuSpecError("--gpu needs at least one GPU, e.g. a100x8")
    return profiles


def _resolve_layout(spec: str | None) -> tuple[list[str], bool]:
    """The run's profile IDs, and whether the run joins the background machine."""

    if spec:
        return parse_gpu_spec(spec), False
    environ = os.environ
    machine_off = environ.get("FAKEGPU_MACHINE", "").strip().lower() in {"0", "off", "false", "no"}
    paths = MachinePaths.from_env()
    if not machine_off and daemon_pid(paths) is not None:
        try:
            return MachineConfig.load(paths.config).profiles, True
        except (OSError, ValueError) as exc:
            raise GpuSpecError(f"cannot read the background machine's layout: {exc}") from exc
    if environ.get("FAKEGPU_PROFILES"):
        return parse_gpu_spec(environ["FAKEGPU_PROFILES"]), False
    count = environ.get("FAKEGPU_DEVICE_COUNT", "")
    profile_id = environ.get("FAKEGPU_PROFILE") or DEFAULT_LAYOUT[0]
    try:
        total = int(count) if count else DEFAULT_LAYOUT[1]
    except ValueError as exc:
        raise GpuSpecError(f"FAKEGPU_DEVICE_COUNT must be a number, not {count!r}") from exc
    return parse_gpu_spec(f"{profile_id}:{total}"), False


def _profiles_spec(profiles: Sequence[str]) -> str:
    groups: list[list] = []
    for profile_id in profiles:
        if groups and groups[-1][0] == profile_id:
            groups[-1][1] += 1
        else:
            groups.append([profile_id, 1])
    return ",".join(f"{profile_id}:{count}" for profile_id, count in groups)


def _child_environment(run_dir: Path, profiles: Sequence[str], machine: bool) -> dict[str, str]:
    env = dict(os.environ)
    bootstrap = Path(__file__).resolve().parent / "_bootstrap"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(bootstrap), env.get("PYTHONPATH")]))
    env["FAKEGPU_RUN_DIR"] = str(run_dir / "records")
    # One profile ID per GPU, for processes limited by CUDA_VISIBLE_DEVICES.
    env["FAKEGPU_RUN_LAYOUT"] = ",".join(profiles)
    # The run prints one summary for all processes instead of one per process.
    env["FAKEGPU_TERMINAL_REPORT"] = "0"
    if not machine:
        # A run with its own GPUs stays out of the background machine, so
        # nvidia-smi inside the run shows exactly the GPUs the run sees.
        env["FAKEGPU_MACHINE"] = "off"
        env["FAKEGPU_PROFILES"] = _profiles_spec(profiles)
        env["FAKEGPU_DEVICE_COUNT"] = str(len(profiles))
        env.pop("FAKEGPU_PROFILE", None)
        env["FAKEGPU_SMI_STATE_DIR"] = str(run_dir / "smi")
        env.pop("FAKEGPU_SMI_STATE_PATH", None)
    bin_dir = run_dir / "bin"
    _write_tool_shims(bin_dir)
    env["PATH"] = os.pathsep.join(filter(None, [str(bin_dir), env.get("PATH")]))
    return env


def _write_tool_shims(bin_dir: Path) -> None:
    """``nvidia-smi`` (and ``nvitop`` when installed) that show the run's GPUs."""

    bin_dir.mkdir(parents=True, exist_ok=True)
    python = shlex.quote(sys.executable)
    tools = {"nvidia-smi": "nvidia-smi"}
    if importlib.util.find_spec("nvitop") is not None:
        tools["nvitop"] = "nvitop --"
    for name, subcommand in tools.items():
        path = bin_dir / name
        path.write_text(f'#!/bin/sh\nexec {python} -m fakegpu {subcommand} "$@"\n', encoding="utf-8")
        path.chmod(0o755)


class _LayoutPublisher:
    """Keep the run's GPUs listed in nvidia-smi while no process uses them."""

    def __init__(self, path: Path, profiles: Sequence[str], env: dict[str, str]):
        self.path = path
        self.state = build_layout_state(list(profiles), env, pid=os.getpid(), process_name="fakegpu run")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="fakegpu-run-layout", daemon=True)
        self._publish()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def _run(self) -> None:
        while not self._stop.wait(LAYOUT_REFRESH_SECONDS):
            self._publish()

    def _publish(self) -> None:
        self.state["timestamp_ns"] = time.time_ns()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.tmp")
            temporary.write_text(json.dumps(self.state), encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError:
            pass  # nvidia-smi then lists only GPUs in use; the run goes on.


def _wait(child: subprocess.Popen) -> int:
    """Wait for the command; its exit status, or 128 + signal number."""

    def forward(signum: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    def interrupt(signum: int, frame: object) -> None:
        # Ctrl-C at the terminal reaches the command too; the run only waits
        # for it to finish so the summary can still print.
        if not _in_terminal_foreground():
            forward(signum, frame)

    handlers = {signal.SIGINT: interrupt}
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            handlers[getattr(signal, name)] = forward
    previous = {signum: signal.signal(signum, handler) for signum, handler in handlers.items()}
    try:
        code = child.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 128 - code if code < 0 else code


def _in_terminal_foreground() -> bool:
    """Whether the terminal's Ctrl-C reaches this process group, the command's."""

    try:
        fd = os.open("/dev/tty", os.O_RDONLY | os.O_NOCTTY)
    except (OSError, AttributeError):
        return False
    try:
        return os.tcgetpgrp(fd) == os.getpgrp()
    except (OSError, AttributeError):
        return False
    finally:
        os.close(fd)


def _hand_records_to_outer_run(records: Path) -> None:
    """In a nested ``fakegpu run``, the outer run's summary counts this run too."""

    outer = os.environ.get("FAKEGPU_RUN_DIR")
    if not outer or Path(outer).resolve() == records.resolve():
        return
    for record in records.glob("process-*.json"):
        try:
            Path(outer).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record, Path(outer) / record.name)
        except OSError:
            pass


def _layout_rows(profiles: Sequence[str]) -> list[dict]:
    rows = []
    for index, profile_id in enumerate(profiles):
        try:
            profile = get_profile(profile_id)
        except ProfileCatalogError:
            rows.append({"index": index, "profile_id": profile_id, "name": profile_id, "total_memory": 0})
            continue
        rows.append({
            "index": index,
            "profile_id": profile.id,
            "name": profile.torch_name or profile.name,
            "total_memory": profile.memory_bytes,
        })
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
