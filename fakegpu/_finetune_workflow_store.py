"""Content identities, exclusive workspace writes and owned child processes."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from uuid import uuid4

from ._local_checkpoint import _stat_marker
from ._process import process_marker


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

    marker = _stat_marker(before)
    if marker != _stat_marker(after) or marker != _stat_marker(path.stat()):
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


def _source_identity(root=Path(__file__).parent):
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


def _live(execution):
    if not execution.get("pid"):
        return False
    current = process_marker(execution["pid"])
    return bool(
        current
        and current["state"] != "Z"
        and current["start_ticks"] == execution.get("process_start_ticks")
    )


def _stop(process):
    # Descendants may ignore SIGTERM even when their group leader exits.
    # macOS answers EPERM instead of ESRCH when only zombies remain.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.wait()
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
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
                cwd=cwd,
            )
        except OSError as exc:
            execution.update(status="failed", error=str(exc), finished_at_utc=_now())
            _write(path, execution)
            raise
        execution.update(pid=process.pid, status="running")
        try:
            marker = process_marker(process.pid)
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
