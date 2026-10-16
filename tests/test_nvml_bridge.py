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
    monkeypatch.setattr("fakegpu.nvml._process_marker", lambda pid: {"state": "Z"})
    assert NvmlStateBridge(state_dir=tmp_path)._inventory()["process_count"] == 0


@pytest.mark.parametrize("field,value", [("total_memory", 2**64), ("uuid", "bad\0id")])
def test_invalid_native_values_are_not_truncated(tmp_path, field, value):
    state = _state()
    state["devices"][0][field] = value
    (tmp_path / "state.json").write_text(json.dumps(state))
    bridge = NvmlStateBridge(state_dir=tmp_path)
    assert bridge._inventory()["device_count"] == 0
    assert bridge.errors


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
    monkeypatch.setattr("fakegpu.nvml._process_marker", lambda pid: {"state": "R"})
    for i, state in enumerate((first, second)):
        (tmp_path / f"{i}.json").write_text(json.dumps(state))
    devices, processes = NvmlStateBridge._arrays(NvmlStateBridge(state_dir=tmp_path)._inventory())
    import ctypes

    assert ctypes.sizeof(_Device) == 48
    assert ctypes.sizeof(_Process) == 16
    assert len(devices) == 1 and len(processes) == 2
    assert devices[0].used_memory == 400
    assert {p.pid for p in processes} == {first["pid"], second["pid"]}
