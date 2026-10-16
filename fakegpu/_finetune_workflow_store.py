"""Content identities, exclusive workspace writes and owned child processes."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import errno
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())

    def marker(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if marker(before) != marker(after) or marker(before) != marker(path.stat()):
        raise ValueError(f"file changed while hashing: {path}")
    return {"bytes": before.st_size, "sha256": digest.hexdigest()}


def _tree(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError(f"workspace artifacts cannot contain symlinks: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"workspace artifacts cannot contain symlinks: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = _file(path)
        elif not path.is_dir():
            raise ValueError(f"workspace artifacts must be ordinary files: {path}")
    return result


def _source_identity():
    root = Path(__file__).parent
    return {
        path.relative_to(root).as_posix(): _file(path)
        for path in sorted(root.rglob("*.py"))
    }


@contextmanager
def _locked(root, *, create=False):
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError(
            "native calibration workspaces require POSIX file locking"
        ) from exc
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError("calibration workspace does not exist")
    with (root / ".workflow.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another process owns this calibration workspace"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


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


def _process_marker(pid):
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


def _live(execution):
    if not execution.get("pid"):
        return False
    current = _process_marker(execution["pid"])
    return bool(
        current
        and current["state"] != "Z"
        and current["start_ticks"] == execution.get("process_start_ticks")
    )


def _stop(process):
    # Descendants may ignore SIGTERM even when their group leader exits.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _execute(
    command, directory, environment, timeout, *, cwd=None, purpose="native calibration"
):
    started = time.monotonic()
    execution = {"argv": command, "status": "starting", "started_at_utc": _now()}
    path = directory / "execution.json"
    _write(path, execution)
    with (directory / "run.log").open("x") as log:
        try:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                **({"cwd": cwd} if cwd is not None else {}),
            )
        except OSError as exc:
            execution.update(status="failed", error=str(exc), finished_at_utc=_now())
            _write(path, execution)
            raise
        execution.update(pid=process.pid, status="running")
        try:
            marker = _process_marker(process.pid)
            execution["process_start_ticks"] = marker["start_ticks"] if marker else None
            _write(path, execution)
            execution["returncode"] = process.wait(timeout=timeout)
            execution["status"] = "succeeded" if process.returncode == 0 else "failed"
        except BaseException as exc:
            _stop(process)
            execution.update(
                returncode=process.returncode,
                status="timed_out"
                if isinstance(exc, subprocess.TimeoutExpired)
                else "interrupted",
                error=str(exc),
            )
            raise
        finally:
            execution.update(
                finished_at_utc=_now(), process_wall_seconds=time.monotonic() - started
            )
            _write(path, execution)
    if process.returncode:
        raise RuntimeError(
            f"{purpose} exited {process.returncode}; preserved {directory / 'run.log'}"
        )
    return execution
