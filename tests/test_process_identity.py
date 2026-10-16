"""Process ownership must survive resume without confusing reused PIDs."""

import ctypes
import errno
import json
import os
import subprocess
import sys
import time

import pytest

from fakegpu import _finetune_workflow_store as store
from fakegpu import _process as identity


@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="POSIX process identity"
)
def test_child_identity_tracks_live_and_reaped_process():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        marker = identity.process_marker(child.pid)
        execution = {"pid": child.pid, "process_start_ticks": marker["start_ticks"]}
        assert marker == identity.process_marker(child.pid)
        assert store._live(execution)
        assert not store._live({**execution, "process_start_ticks": "older-process"})
        child.terminate()
        child.wait(timeout=5)
        assert identity.process_marker(child.pid) is None
        assert not store._live(execution)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="POSIX zombie state"
)
def test_unreaped_child_does_not_block_workspace_resume():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            marker = identity.process_marker(child.pid)
            if marker is None or marker["state"] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("exited child was still identified as live")
        if marker:
            assert not store._live(
                {"pid": child.pid, "process_start_ticks": marker["start_ticks"]}
            )
    finally:
        child.wait(timeout=5)


@pytest.mark.parametrize("pid", [None, False, 0, -1, 2**32 + 1, "1"])
def test_invalid_pid_cannot_alias_an_existing_process(pid):
    assert identity.process_marker(pid) is None


def test_darwin_birth_time_preserves_microseconds_and_zombie_state(monkeypatch):
    def pidinfo(pid, flavor, argument, buffer, size):
        assert flavor == 3
        info = buffer._obj
        info.pbi_pid = pid
        info.pbi_start_tvsec = 123456
        info.pbi_start_tvusec = 7
        info.pbi_status = 5
        return size

    monkeypatch.setattr(identity, "_darwin_pidinfo", lambda: pidinfo)
    assert identity._darwin_process_marker(123) == {
        "start_ticks": "darwin:123456:000007",
        "state": "Z",
    }


@pytest.mark.parametrize("error", [errno.EPERM, errno.EACCES, errno.EIO])
def test_darwin_inspection_errors_do_not_assume_orphan_exited(monkeypatch, error):
    def pidinfo(*arguments):
        ctypes.set_errno(error)
        return 0

    monkeypatch.setattr(identity, "_darwin_pidinfo", lambda: pidinfo)
    with pytest.raises(OSError) as failure:
        identity._darwin_process_marker(123)
    assert failure.value.errno == error


def test_darwin_missing_process_has_no_marker(monkeypatch):
    def pidinfo(*arguments):
        ctypes.set_errno(errno.ESRCH)
        return 0

    monkeypatch.setattr(identity, "_darwin_pidinfo", lambda: pidinfo)
    assert identity._darwin_process_marker(123) is None


def test_linux_start_ticks_remain_compatible_with_existing_records(monkeypatch):
    monkeypatch.setattr(identity.sys, "platform", "linux")
    # A command may contain both spaces and parentheses; field 22 is the start.
    fields = ["S"] + ["0"] * 18 + ["123456"]
    monkeypatch.setattr(
        identity.Path,
        "read_text",
        lambda self: "123 (a (nested) command) " + " ".join(fields),
    )
    marker = identity.process_marker(123)
    assert marker == {"start_ticks": "123456", "state": "S"}
    assert store._live({"pid": 123, "process_start_ticks": "123456"})


@pytest.mark.skipif(os.name != "posix", reason="POSIX child process groups")
def test_identity_lookup_failure_stops_launched_child(tmp_path, monkeypatch):
    processes = []
    popen = subprocess.Popen

    def capture(*arguments, **options):
        child = popen(*arguments, **options)
        processes.append(child)
        return child

    def inaccessible(pid):
        raise PermissionError(errno.EPERM, "identity inspection denied")

    monkeypatch.setattr(store.subprocess, "Popen", capture)
    monkeypatch.setattr(store, "process_marker", inaccessible)
    with pytest.raises(PermissionError, match="inspection denied"):
        store._execute(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
            os.environ.copy(),
            timeout=5,
        )
    assert processes[0].poll() is not None
    execution = json.loads((tmp_path / "execution.json").read_text())
    assert execution["status"] == "interrupted"
    assert execution["returncode"] != 0
