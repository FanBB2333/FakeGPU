"""The background machine: events, control methods, attach and the daemon."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from fakegpu.machine.client import MachineClient, attach_environment, daemon_pid
from fakegpu.machine.config import MachineConfig, MachinePaths, parse_devices
from fakegpu.machine.daemon import start_daemon, stop_daemon
from fakegpu.machine.service import ENDED_RETENTION_SECONDS, MachineService

GIB = 1024**3


@pytest.fixture
def child():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield process
    process.kill()
    process.wait()


@pytest.fixture
def service(tmp_path):
    return MachineService(MachinePaths(tmp_path), MachineConfig(devices="a100:2", telemetry="none"))


def publish(service, pid, *, used=GIB, running=True, age=0.0, **extra):
    devices = [
        {"index": i, "profile_id": "a100", "total_memory": 80 * GIB, "reported_memory": used,
         "reported_peak_memory": used}
        for i in range(2)
    ]
    state = {
        "schema_version": "fakegpu.smi_state.v2", "hostname": socket.gethostname(), "pid": pid,
        "process_name": "train.py", "running": running, "timestamp_ns": time.time_ns() - int(age * 1e9),
        "devices": devices, **extra,
    }
    (service.paths.state_dir / f"{pid}.json").write_text(json.dumps(state))


def kinds(service):
    return [event["kind"] for event in service.events()]


def test_device_layout_expands_profile_counts():
    assert parse_devices("a100:2, h100") == ["a100", "a100", "h100"]
    for spec in ("", "a100:0", "a100:x", "no-such-gpu:1"):
        with pytest.raises(ValueError):
            parse_devices(spec)


def test_idle_machine_lists_its_layout(service):
    status = service.status()
    assert [device["profile_id"] for device in status["devices"]] == ["a100", "a100"]
    assert status["summary"]["memory_total_bytes"] == 2 * status["devices"][0]["memory"]["total_bytes"]
    assert status["processes"] == []


def test_process_lifecycle_events(service, child):
    publish(service, child.pid)
    service.poll()
    status = service.status()
    assert kinds(service) == ["process.attached"]
    assert status["processes"][0]["memory_bytes"] == 2 * GIB
    assert status["devices"][0]["memory"]["used_bytes"] == GIB

    publish(service, child.pid, age=5)
    service.poll()
    publish(service, child.pid)
    service.poll()
    assert kinds(service)[-2:] == ["process.stalled", "process.resumed"]

    publish(service, child.pid, running=False, exit_signal="SIGTERM")
    service.poll()
    event = service.events()[-1]
    assert (event["kind"], event["level"]) == ("process.terminated", "warning")
    assert "terminated by SIGTERM" in event["message"]
    assert service.status()["processes"] == []


def test_clean_exit_and_retention(service, child):
    publish(service, child.pid)
    service.poll()
    publish(service, child.pid, running=False)
    service.poll()
    assert kinds(service)[-1] == "process.exited"
    service.poll(time.time_ns() + int((ENDED_RETENTION_SECONDS + 1) * 1e9))
    assert not (service.paths.state_dir / f"{child.pid}.json").exists()
    assert service.status()["ended"][0]["pid"] == child.pid


def test_killed_process_without_final_heartbeat_vanishes(service, child):
    publish(service, child.pid)
    service.poll()
    child.kill()
    child.wait()
    service.poll()
    event = service.events()[-1]
    assert (event["kind"], event["level"]) == ("process.vanished", "error")
    assert "SIGKILL" in event["message"]


def test_memory_alerts(service, child):
    publish(service, child.pid, used=78 * GIB)
    service.poll()
    assert "memory.high" in kinds(service)
    assert service.status()["summary"]["alerts"] == 2
    publish(service, child.pid, used=GIB)
    service.poll()
    assert kinds(service).count("memory.normal") == 2


def test_dispatch_controls(service, child):
    assert service.dispatch({"id": 1, "method": "nope"})["error"]["code"] == "unknown_method"
    bad = service.dispatch({"method": "config.update", "params": {"changes": {"devices": "no-such-gpu"}}})
    assert bad["error"]["code"] == "invalid_params"
    ok = service.dispatch({"method": "config.update", "params": {"changes": {"devices": "h100:4"}}})
    assert ok["result"]["devices"] == "h100:4"
    assert MachineConfig.load(service.paths.config).devices == "h100:4"
    assert kinds(service)[-1] == "config.changed"

    missing = service.dispatch({"method": "process.signal", "params": {"pid": child.pid}})
    assert missing["error"]["code"] == "not_found"
    publish(service, child.pid)
    service.poll()
    sent = service.dispatch({"method": "process.signal", "params": {"pid": child.pid, "signal": "SIGTERM"}})
    assert sent["ok"]
    assert child.wait(timeout=5) == -signal.SIGTERM
    assert kinds(service)[-1] == "process.signal_sent"


def test_events_survive_a_restart(tmp_path):
    first = MachineService(MachinePaths(tmp_path))
    first.emit("daemon.started", "one")
    second = MachineService(MachinePaths(tmp_path))
    assert second.emit("daemon.started", "two")["seq"] == 2
    assert [event["message"] for event in second.events()] == ["one", "two"]


def test_attach_requires_a_running_daemon(tmp_path):
    paths = MachinePaths(tmp_path / "machine")
    paths.create()
    MachineConfig(devices="h100:2").save(paths.config)
    environ = {"FAKEGPU_HOME": str(tmp_path)}
    assert attach_environment(environ) is None and "FAKEGPU_PROFILES" not in environ
    with paths.pidfile.open("w") as lock:
        lock.write(f"{os.getpid()}\n")
        lock.flush()
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert daemon_pid(paths) == os.getpid()
        attach_environment(environ)
        assert environ["FAKEGPU_PROFILES"] == "h100:2"
        assert environ["FAKEGPU_SMI_STATE_DIR"] == str(paths.state_dir)
        assert environ["FAKEGPU_TELEMETRY"] == "reference"
        own = {"FAKEGPU_HOME": str(tmp_path), "FAKEGPU_PROFILES": "l4:1"}
        attach_environment(own)
        assert own["FAKEGPU_PROFILES"] == "l4:1"
        explicit = {"FAKEGPU_HOME": str(tmp_path)}
        attach_environment(explicit, explicit_layout=True)
        assert "FAKEGPU_PROFILES" not in explicit
        opted_out = {"FAKEGPU_HOME": str(tmp_path), "FAKEGPU_MACHINE": "off"}
        assert attach_environment(opted_out) is None


def test_publisher_records_sigterm(tmp_path):
    path = tmp_path / "state.json"
    script = textwrap.dedent(
        f"""
        import time
        from fakegpu.smi import SmiStatePublisher
        publisher = SmiStatePublisher({str(path)!r}, lambda: {{"devices": []}}, interval_seconds=0.05)
        publisher.start()
        print("ready", flush=True)
        time.sleep(60)
        """
    )
    process = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == "ready"
    process.terminate()
    assert process.wait(timeout=10) == -signal.SIGTERM
    state = json.loads(path.read_text())
    assert state["running"] is False
    assert state["exit_signal"] == "SIGTERM"


def test_daemon_watches_a_fakecuda_process(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("FAKEGPU_HOME", str(tmp_path))
    paths = MachinePaths.from_env()
    start_daemon(paths, changes={"devices": "a100:2", "telemetry": "reference"})
    try:
        client = MachineClient(paths)
        assert client.call("hello")["protocol"] == "fakegpu.machine.v1"
        worker = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(
                """
                import time, fakegpu
                fakegpu.init()
                import torch
                x = torch.ones(1024, 1024, device="cuda:1")
                print(torch.cuda.device_count(), flush=True)
                time.sleep(60)
                """
            )],
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]), "PYTHONWARNINGS": "ignore",
                 "FAKEGPU_TERMINAL_REPORT": "0", "FAKEGPU_MACHINE": "on"},
        )
        assert worker.stdout.readline().strip() == "2"
        deadline = time.monotonic() + 10
        while client.call("status")["devices"][1]["memory"]["used_bytes"] < 4 * 1024**2:
            assert time.monotonic() < deadline
            time.sleep(0.1)
        status = client.call("status")
        assert status["processes"][0]["pid"] == worker.pid
        assert status["devices"][0]["telemetry"]["source"] == "a100_reference_model"
        client.call("process.signal", pid=worker.pid, signal="SIGTERM")
        assert worker.wait(timeout=10) == -signal.SIGTERM
        deadline = time.monotonic() + 10
        while "process.terminated" not in [event["kind"] for event in client.call("events")]:
            assert time.monotonic() < deadline
            time.sleep(0.1)
    finally:
        assert stop_daemon(paths)
    assert daemon_pid(paths) is None
    assert not paths.socket.exists()
