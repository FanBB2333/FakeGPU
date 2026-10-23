from __future__ import annotations

import copy
import json
import os
import socket
import time

import pytest

from fakegpu.nvml import NvmlStateBridge, _Device, _Process


def _state():
    return {
        "schema_version": "fakegpu.smi_state.v2", "hostname": socket.gethostname(),
        "pid": os.getpid(), "running": True, "timestamp_ns": time.time_ns(),
        "devices": [{"index": 0, "profile_id": "l4", "total_memory": 1000,
                     "tracked_memory": 100, "reserved_memory": 200, "reported_memory": 200}],
    }


def test_reader_deduplicates_publisher_and_uses_allocator_memory(tmp_path):
    state = _state()
    (tmp_path / "old.json").write_text(json.dumps(state))
    state["timestamp_ns"] += 1
    state["devices"][0]["reported_memory"] = 300
    (tmp_path / "new.json").write_text(json.dumps(state))
    bridge = NvmlStateBridge(state_dir=tmp_path)
    inventory = bridge._inventory()
    devices, processes = bridge._arrays(inventory)
    assert len(devices) == len(processes) == 1
    assert devices[0].total_memory == 1000
    assert devices[0].used_memory == processes[0].used_memory == 300
    assert processes[0].pid == os.getpid()


def test_processes_appear_only_on_gpus_they_used(tmp_path):
    state = _state()
    unused = {"index": 1, "profile_id": "l4", "total_memory": 1000, "reported_memory": 0}
    released = {**unused, "index": 2, "reported_peak_memory": 50}
    state["devices"] += [unused, released]
    (tmp_path / "state.json").write_text(json.dumps(state))
    devices, processes = NvmlStateBridge(state_dir=tmp_path)._arrays(NvmlStateBridge(state_dir=tmp_path)._inventory())
    assert len(devices) == 3
    assert sorted(process.device_index for process in processes) == [0, 2]


@pytest.mark.parametrize("change", [
    {"running": False}, {"hostname": "remote-test-host"},
    {"timestamp_ns": 1}, {"timestamp_ns": time.time_ns() + 10**15},
    {"pid": 0}, {"pid": True}, {"pid": 2**31 - 1},
    {"timestamp_ns": "bad"}, {"schema_version": "unknown"},
    {"devices": [None]}, {"devices": "invalid"},
])
def test_reader_excludes_inactive_remote_and_invalid_states(tmp_path, change):
    state = _state()
    state.update(change)
    (tmp_path / "state.json").write_text(json.dumps(state))
    assert NvmlStateBridge(state_dir=tmp_path)._inventory()["process_count"] == 0


def test_bad_file_does_not_hide_live_publisher(tmp_path):
    (tmp_path / "broken.json").write_text("{")
    (tmp_path / "valid.json").write_text(json.dumps(_state()))
    bridge = NvmlStateBridge(state_dir=tmp_path)
    assert bridge._inventory()["process_count"] == 1
    assert len(bridge.errors) == 1


def test_deleted_file_clears_inventory(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(_state()))
    bridge = NvmlStateBridge(state_dir=tmp_path)
    assert bridge._inventory()["device_count"] == 1
    path.unlink()
    assert bridge._inventory()["device_count"] == 0


def test_reader_filters_zombies(tmp_path, monkeypatch):
    (tmp_path / "state.json").write_text(json.dumps(_state()))
    monkeypatch.setattr("fakegpu._smi_state.is_alive", lambda pid: False)
    assert NvmlStateBridge(state_dir=tmp_path)._inventory()["process_count"] == 0


@pytest.mark.parametrize("field,value", [("total_memory", 2**64), ("uuid", "bad\0id")])
def test_invalid_native_values_are_rejected_not_truncated(tmp_path, field, value):
    state = _state()
    state["devices"][0][field] = value
    (tmp_path / "state.json").write_text(json.dumps(state))
    bridge = NvmlStateBridge(state_dir=tmp_path)
    with pytest.raises(ValueError):
        bridge._arrays(bridge._inventory())


def test_snapshot_failure_clears_previous_usage(tmp_path, monkeypatch):
    bridge = NvmlStateBridge(state_dir=tmp_path)
    calls = []
    bridge._set_snapshot = lambda *args: calls.append(args) or 0

    def fail():
        raise OSError("reader failure")

    monkeypatch.setattr(bridge, "_inventory", fail)
    with pytest.raises(OSError):
        bridge.refresh()
    assert calls == [(1, 0, None, 0, None)]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_timing(value, tmp_path):
    with pytest.raises(ValueError):
        NvmlStateBridge(state_dir=tmp_path, refresh_interval=value)
    with pytest.raises(ValueError):
        NvmlStateBridge(state_dir=tmp_path, stale_after_seconds=value)


def test_snapshot_arrays_have_expected_layout_and_distinct_pids(tmp_path, monkeypatch):
    first = _state()
    second = copy.deepcopy(first)
    second["pid"] += 1
    monkeypatch.setattr("fakegpu._smi_state.is_alive", lambda pid: True)
    for i, state in enumerate((first, second)):
        (tmp_path / f"{i}.json").write_text(json.dumps(state))
    devices, processes = NvmlStateBridge._arrays(NvmlStateBridge(state_dir=tmp_path)._inventory())
    import ctypes

    assert ctypes.sizeof(_Device) == 48
    assert ctypes.sizeof(_Process) == 16
    assert len(devices) == 1 and len(processes) == 2
    assert devices[0].used_memory == 400
    assert {p.pid for p in processes} == {first["pid"], second["pid"]}


def test_nvlink_counters_survive_an_idle_layout_and_a_new_publisher(tmp_path):
    from fakegpu.link_traffic import empty_record

    bridge = NvmlStateBridge(state_dir=tmp_path)
    transfer = empty_record()
    transfer["links"]["nvlink"].update(tx_bytes=1000, rx_bytes=2000)
    device = {"uuid": "synthetic-links", "topology": {"slot": {"slot": 0}}, "transfer": transfer}

    def counters():
        links = bridge._links(device)
        return links.nvlink_tx_bytes, links.nvlink_rx_bytes

    assert counters() == (1000, 2000)
    device["transfer"] = None
    assert counters() == (1000, 2000)
    assert counters() == (1000, 2000)
    transfer["links"]["nvlink"].update(tx_bytes=100, rx_bytes=200)
    device["transfer"] = transfer
    assert counters() == (1100, 2200)


@pytest.mark.parametrize("selection,telemetry", [
    ("default", "reference"), ("argument", "reference"), ("environment", "reference"),
    ("single-file", "none"), ("disabled", "none"),
])
def test_nvitop_state_selection_and_machine_telemetry(tmp_path, monkeypatch, selection, telemetry):
    import sys
    from types import SimpleNamespace

    from fakegpu import nvitop_cli
    from fakegpu.machine.config import MachineConfig, MachinePaths

    monkeypatch.setenv("FAKEGPU_HOME", str(tmp_path))
    monkeypatch.delenv("FAKEGPU_SMI_STATE_DIR", raising=False)
    monkeypatch.delenv("FAKEGPU_SMI_STATE_PATH", raising=False)
    paths = MachinePaths.from_env()
    paths.create()
    MachineConfig(telemetry="reference").save(paths.config)
    monkeypatch.setattr(nvitop_cli, "daemon_pid", lambda _: 12345)
    monkeypatch.setattr(nvitop_cli, "is_machine_state_dir", lambda directory, _: (
        directory is not None and os.fspath(directory) == str(paths.state_dir)
    ))
    captured = {}

    class Bridge:
        device_count = 1

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(nvitop_cli, "NvmlStateBridge", Bridge)
    monkeypatch.setitem(sys.modules, "nvitop.cli", SimpleNamespace(main=lambda: 0))
    # Reference telemetry relabels nvitop's driver version on this stand-in.
    monkeypatch.setitem(sys.modules, "nvitop.api.device", SimpleNamespace(Device=type("Device", (), {})))
    arguments = []
    if selection in {"argument", "disabled"}:
        arguments += ["--state-dir", str(paths.state_dir)]
    if selection == "disabled":
        arguments += ["--telemetry", "none"]
    if selection == "environment":
        monkeypatch.setenv("FAKEGPU_SMI_STATE_DIR", str(paths.state_dir))
    if selection == "single-file":
        monkeypatch.setenv("FAKEGPU_SMI_STATE_PATH", str(tmp_path / "publisher.json"))
    assert nvitop_cli.main(arguments) == 0
    assert captured["telemetry"] == telemetry
    if selection == "single-file":
        assert captured["state_dir"] is None and captured["layout"] is None
    else:
        assert captured["state_dir"] == str(paths.state_dir)
