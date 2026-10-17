"""Talk to a running machine daemon, and let FakeCUDA processes join it."""

from __future__ import annotations

import fcntl
import itertools
import json
import os
import socket
from typing import Any, MutableMapping

from ..platforms import apply_platform_environment
from .config import MachineConfig, MachinePaths

_ids = itertools.count(1)


class MachineError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def daemon_pid(paths: MachinePaths) -> int | None:
    """PID of the running daemon. The daemon holds a lock on its pidfile."""

    try:
        stream = paths.pidfile.open("r")
    except FileNotFoundError:
        return None
    with stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            text = stream.read().strip()
            return int(text) if text.isdigit() else None
        fcntl.flock(stream, fcntl.LOCK_UN)
        return None


class MachineClient:
    def __init__(self, paths: MachinePaths | None = None, *, timeout: float = 3.0):
        self.paths = paths or MachinePaths.from_env()
        self.timeout = timeout

    def call(self, method: str, **params: Any) -> Any:
        request = {"v": 1, "id": next(_ids), "method": method, "params": params}
        data = bytearray()
        connected = False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect(str(self.paths.socket))
                connected = True
                sock.sendall(json.dumps(request).encode() + b"\n")
                while not data.endswith(b"\n"):
                    chunk = sock.recv(1 << 16)
                    if not chunk:
                        raise MachineError("disconnected", "the daemon closed the connection")
                    data += chunk
        except TimeoutError as exc:
            raise MachineError("timeout", "the FakeGPU machine daemon did not respond in time") from exc
        except OSError as exc:
            if connected:
                raise MachineError("disconnected", "the connection to the machine daemon was lost") from exc
            raise MachineError("not_running", "the FakeGPU machine daemon is not running") from exc
        response = json.loads(data)
        if not response.get("ok"):
            error = response.get("error") or {}
            raise MachineError(str(error.get("code")), str(error.get("message")))
        return response["result"]


def attach_environment(
    environ: MutableMapping[str, str] | None = None,
    *,
    explicit_layout: bool = False,
) -> MachineConfig | None:
    """Point a starting FakeCUDA process at the running machine, if any.

    The process publishes its state to the machine and, unless it chose its
    own devices, uses the machine layout and platform (slots, PCIe links and
    NVLinks). ``FAKEGPU_MACHINE=off`` opts out.
    """

    environ = os.environ if environ is None else environ
    if environ.get("FAKEGPU_MACHINE", "").strip().lower() in {"0", "off", "false", "no"}:
        return None
    paths = MachinePaths.from_env(environ)
    if daemon_pid(paths) is None:
        return None
    config = MachineConfig.load(paths.config)
    if "FAKEGPU_SMI_STATE_DIR" not in environ and "FAKEGPU_SMI_STATE_PATH" not in environ:
        environ["FAKEGPU_SMI_STATE_DIR"] = str(paths.state_dir)
    layout_variables = ("FAKEGPU_PROFILE", "FAKEGPU_PROFILES", "FAKEGPU_DEVICE_COUNT")
    if not explicit_layout and not any(name in environ for name in layout_variables):
        environ["FAKEGPU_PROFILES"] = config.devices
        if not environ.get("FAKEGPU_PLATFORM"):
            apply_platform_environment(
                environ,
                config.profiles,
                config.platform,
                pcie_generation=config.pcie_generation,
                nvlink_bridges=config.nvlink_bridges,
            )
    if config.telemetry == "reference":
        environ.setdefault("FAKEGPU_TELEMETRY", "reference")
    return config
