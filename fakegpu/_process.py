"""Identify local processes by PID plus birth time, on Linux and macOS.

A PID alone is not an identity: the kernel reuses it after the owner exits.
``process_marker`` returns the birth time and run state so callers can tell a
live owner from an unrelated process that later received the same PID.
"""

from __future__ import annotations

import ctypes
import errno
from functools import lru_cache
from pathlib import Path
import subprocess
import sys


class _DarwinBSDInfo(ctypes.Structure):
    # Public proc_bsdinfo ABI from the macOS SDK's <sys/proc_info.h>.
    _fields_ = [
        (name, ctypes.c_uint32)
        for name in (
            "pbi_flags",
            "pbi_status",
            "pbi_xstatus",
            "pbi_pid",
            "pbi_ppid",
            "pbi_uid",
            "pbi_gid",
            "pbi_ruid",
            "pbi_rgid",
            "pbi_svuid",
            "pbi_svgid",
            "rfu_1",
        )
    ] + [
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


@lru_cache(maxsize=1)
def _darwin_pidinfo():
    function = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True).proc_pidinfo
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    )
    function.restype = ctypes.c_int
    return function


def _darwin_process_marker(pid):
    info = _DarwinBSDInfo()
    ctypes.set_errno(0)
    size = _darwin_pidinfo()(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
    if size != ctypes.sizeof(info):
        error = ctypes.get_errno()
        if not size and error == errno.ESRCH:
            return None
        # Permission errors and incomplete reads cannot establish that an orphan
        # exited. Stop workspace reuse rather than silently overwriting its files.
        raise OSError(error or errno.EIO, "cannot inspect macOS process identity")
    if info.pbi_pid != pid or not info.pbi_start_tvsec:
        raise OSError(errno.EIO, "invalid macOS process identity")
    return {
        # Keep the existing record key, but namespace this microsecond birth time
        # so it cannot match a Linux clock-tick value from another workspace.
        "start_ticks": f"darwin:{info.pbi_start_tvsec}:{info.pbi_start_tvusec:06d}",
        "state": "Z" if info.pbi_status == 5 else "R",
    }


def process_marker(pid):
    """Return ``{"start_ticks", "state"}`` for a process, or None if it is gone."""

    if not isinstance(pid, int) or isinstance(pid, bool) or not 0 < pid < 2**31:
        return None
    if sys.platform == "darwin":
        return _darwin_process_marker(pid)
    try:
        # The command field can contain spaces and parentheses.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return {"start_ticks": fields[19], "state": fields[0]}


def is_alive(pid) -> bool:
    """True while ``pid`` runs and has not become a zombie."""

    try:
        marker = process_marker(pid)
    except OSError:
        # Another user's process: it exists, but we may not inspect it.
        return True
    return marker is not None and marker["state"] != "Z"


def describe(pid: int) -> dict[str, str]:
    """Owner and full command line of a live process, for local display only."""

    try:
        output = subprocess.run(
            ["ps", "-o", "user=", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        output = ""
    user, _, command = output.partition(" ")
    return {"user": user, "command": command.strip()}
