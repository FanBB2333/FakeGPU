"""Run the machine daemon in the background and serve its control socket."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping

from .client import MachineClient, MachineError, daemon_pid
from .config import MachineConfig, MachinePaths
from .service import MachineService

POLL_INTERVAL_SECONDS = 0.5


def run_daemon(paths: MachinePaths | None = None) -> int:
    """Serve the machine in this process until stopped."""

    paths = paths or MachinePaths.from_env()
    paths.create()
    with paths.pidfile.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("fakegpu daemon: already running", file=sys.stderr)
            return 1
        lock.seek(0)
        lock.truncate()
        lock.write(f"{os.getpid()}\n")
        lock.flush()
        try:
            asyncio.run(_serve(MachineService(paths)))
        finally:
            paths.socket.unlink(missing_ok=True)
            paths.pidfile.unlink(missing_ok=True)
    return 0


async def _serve(service: MachineService) -> None:
    stop = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while line := await reader.readline():
                try:
                    request = json.loads(line)
                except json.JSONDecodeError:
                    request = None
                if isinstance(request, dict):
                    response = service.dispatch(request)
                else:
                    response = {
                        "v": 1,
                        "id": None,
                        "ok": False,
                        "error": {"code": "bad_request", "message": "send one JSON object per line"},
                    }
                writer.write(json.dumps(response).encode() + b"\n")
                await writer.drain()
                if service.stop_requested:
                    stop.set()
        except ConnectionError:
            pass
        finally:
            writer.close()

    service.paths.socket.unlink(missing_ok=True)
    server = await asyncio.start_unix_server(handle, path=str(service.paths.socket))
    service.paths.socket.chmod(0o600)
    loop = asyncio.get_running_loop()
    for number in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(number, stop.set)
    config = service.config
    service.emit(
        "daemon.started",
        f"machine daemon started (pid {os.getpid()}): {config.devices}, telemetry {config.telemetry}",
    )
    async with server:
        while not stop.is_set():
            service.poll()
            try:
                await asyncio.wait_for(stop.wait(), POLL_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass
    service.emit("daemon.stopped", "machine daemon stopped")


def start_daemon(
    paths: MachinePaths | None = None,
    *,
    changes: Mapping[str, Any] | None = None,
    timeout: float = 15.0,
) -> int:
    """Start the daemon unless it already runs, apply ``changes``, return its PID."""

    paths = paths or MachinePaths.from_env()
    paths.create()
    pid = daemon_pid(paths)
    if pid is not None:
        if changes:
            MachineClient(paths).call("config.update", changes=dict(changes))
        return pid
    MachineConfig.load(paths.config).updated(changes or {}).save(paths.config)
    environment = dict(os.environ, FAKEGPU_MACHINE="off")
    # Works from a source checkout as well as from an installed package.
    package_parent = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [package_parent, environment.get("PYTHONPATH")])
    )
    with paths.log.open("ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "fakegpu", "daemon", "run", "--machine-root", str(paths.root)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            cwd="/",
            env=environment,
            start_new_session=True,
        )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return int(MachineClient(paths, timeout=1).call("hello")["daemon"]["pid"])
        except MachineError:
            time.sleep(0.05)
    raise MachineError("start_failed", f"the daemon did not start; see {paths.log}")


def stop_daemon(paths: MachinePaths | None = None, *, timeout: float = 10.0) -> bool:
    """Stop the daemon. Returns False when it was not running."""

    paths = paths or MachinePaths.from_env()
    pid = daemon_pid(paths)
    if pid is None:
        return False
    try:
        MachineClient(paths).call("daemon.stop")
    except MachineError:
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while daemon_pid(paths) is not None:
        if time.monotonic() > deadline:
            raise MachineError("stop_failed", f"daemon pid {pid} did not stop")
        time.sleep(0.05)
    return True
