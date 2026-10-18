"""Portable machine settings imported through the daemon CLI."""

from __future__ import annotations

import json
import os
import time

import pytest

from fakegpu.daemon_cli import main
from fakegpu.machine.client import MachineClient, attach_environment
from fakegpu.machine.config import MachineConfig, MachinePaths
from fakegpu.machine.daemon import start_daemon, stop_daemon
from fakegpu.smi import SmiStatePublisher


@pytest.fixture
def machine_home(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKEGPU_HOME", str(tmp_path / "home"))
    return MachinePaths.from_env()


@pytest.mark.parametrize(
    "settings",
    [
        {"devices": "h100:4", "platform": "hgx-h100-4"},
        {
            "devices": "a100-pcie-80gb:2",
            "platform": "workstation",
            "pcie_generation": 3,
            "nvlink_bridges": "0-1",
            "telemetry": "none",
            "background": False,
            "stale_after_seconds": 5,
            "memory_warning_percent": 80,
        },
    ],
)
def test_export_import_preserves_settings_and_topology(
    machine_home, tmp_path, monkeypatch, capsys, settings
):
    original = MachineConfig(**settings)
    machine_home.create()
    original.save(machine_home.config)
    assert main(["config", "--json"]) == 0
    exported = capsys.readouterr().out
    assert json.loads(exported) == original.to_dict()
    source = tmp_path / "machine.json"
    source.write_text(exported, encoding="utf-8")

    monkeypatch.setenv("FAKEGPU_HOME", str(tmp_path / "other-home"))
    assert main(["config", "--from", str(source), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == original.to_dict()
    imported = MachineConfig.load(MachinePaths.from_env().config)
    assert imported == original
    assert imported.resolved_platform().to_dict() == original.resolved_platform().to_dict()
    assert imported.resolved_platform().environment() == original.resolved_platform().environment()


@pytest.mark.parametrize("existing", [False, True])
def test_partial_import_preserves_unspecified_settings(machine_home, tmp_path, capsys, existing):
    original = MachineConfig(devices="l4:2", telemetry="none") if existing else MachineConfig()
    if existing:
        machine_home.create()
        original.save(machine_home.config)
    source = tmp_path / "partial.json"
    source.write_text('{"background": false, "memory_warning_percent": 75}', encoding="utf-8")

    assert main(["config", "--from", str(source), "--json"]) == 0
    expected = {**original.to_dict(), "background": False, "memory_warning_percent": 75}
    assert json.loads(capsys.readouterr().out) == expected
    assert MachineConfig.load(machine_home.config).to_dict() == expected


def test_cli_overrides_file_before_validating_the_combined_layout(machine_home, tmp_path, capsys):
    source = tmp_path / "machine.json"
    source.write_text(
        json.dumps({
            "devices": "a100:8",
            "platform": "workstation",
            "pcie_generation": 3,
            "nvlink_bridges": "0-1",
            "background": True,
            "telemetry": "reference",
        }),
        encoding="utf-8",
    )

    assert main([
        "config", "--from", str(source), "--devices", "l4:2",
        "--pcie-gen", "0", "--nvlink-bridges", "", "--background", "off",
        "--telemetry", "none", "--json",
    ]) == 0
    settings = json.loads(capsys.readouterr().out)
    assert settings == MachineConfig(
        devices="l4:2", platform="workstation", telemetry="none", background=False
    ).to_dict()
    assert MachineConfig.load(machine_home.config).to_dict() == settings


@pytest.mark.parametrize(
    "document",
    [
        "", "{", "[]", "null", "true", "42", '"machine"',
        '{"unknown_setting": 1}',
        '{"devices": ""}',
        '{"devices": null}',
        '{"platform": "no-such-platform"}',
        '{"devices": "h100:2", "platform": "workstation"}',
        '{"pcie_generation": true}',
        '{"nvlink_bridges": []}',
        '{"telemetry": "invalid"}',
        '{"background": "off"}',
        '{"stale_after_seconds": "5"}',
        '{"stale_after_seconds": 0}',
        '{"memory_warning_percent": true}',
        '{"memory_warning_percent": 101}',
    ],
)
def test_invalid_import_leaves_saved_settings_unchanged(machine_home, tmp_path, capsys, document):
    machine_home.create()
    MachineConfig(devices="l4:1", telemetry="none").save(machine_home.config)
    before = machine_home.config.read_bytes()
    source = tmp_path / "invalid.json"
    source.write_text(document, encoding="utf-8")

    assert main(["config", "--from", str(source)]) != 0
    output = capsys.readouterr()
    assert output.err
    assert "Traceback" not in output.err
    assert machine_home.config.read_bytes() == before
    assert not machine_home.config.with_suffix(".tmp").exists()


def test_missing_import_does_not_create_machine_settings(machine_home, tmp_path, capsys):
    assert main(["config", "--from", str(tmp_path / "missing.json")]) != 0
    assert capsys.readouterr().err
    assert not machine_home.root.exists()


def test_empty_object_preserves_saved_settings(machine_home, tmp_path, capsys):
    machine_home.create()
    original = MachineConfig(devices="l4:2", background=False)
    original.save(machine_home.config)
    before = machine_home.config.read_bytes()
    source = tmp_path / "empty.json"
    source.write_text("{}", encoding="utf-8")

    assert main(["config", "--from", str(source), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == original.to_dict()
    assert machine_home.config.read_bytes() == before


def test_running_daemon_import_keeps_attached_workload_layout(machine_home, tmp_path, capsys):
    source = tmp_path / "machine.json"
    source.write_text('{"devices": "h100:2", "telemetry": "none"}', encoding="utf-8")
    start_daemon(machine_home, changes={"devices": "a100:2", "telemetry": "none"}, timeout=5)
    try:
        client = MachineClient(machine_home)
        old_environment = {"FAKEGPU_HOME": str(machine_home.root.parent)}
        original = attach_environment(old_environment)
        original_environment = dict(old_environment)
        publisher = SmiStatePublisher(
            machine_home.state_dir / f"{os.getpid()}.json",
            lambda: {"devices": [
                {"index": i, "profile_id": profile, "total_memory": 80 * 1024**3,
                 "current_memory": 1024}
                for i, profile in enumerate(original.profiles)
            ]},
        )
        publisher.publish_once(running=True)
        deadline = time.monotonic() + 5
        while not client.call("status")["processes"]:
            assert time.monotonic() < deadline
            time.sleep(0.05)

        assert main(["config", "--from", str(source), "--json"]) == 0
        imported = json.loads(capsys.readouterr().out)
        assert client.call("config.get") == imported
        assert MachineConfig.load(machine_home.config).to_dict() == imported
        assert old_environment == original_environment
        status = client.call("status")
        assert status["processes"][0]["layout"] == "a100:2"
        assert {device["profile_id"] for device in status["devices"] if device["in_layout"]} == {"h100"}
        assert {device["profile_id"] for device in status["devices"] if not device["in_layout"]} == {"a100"}
        new_environment = {"FAKEGPU_HOME": str(machine_home.root.parent)}
        assert attach_environment(new_environment).devices == "h100:2"
        assert new_environment["FAKEGPU_PLATFORM"] == "hgx-h100-4"

        before = machine_home.config.read_bytes()
        source.write_text('{"platform": "workstation"}', encoding="utf-8")
        assert main(["config", "--from", str(source)]) != 0
        assert capsys.readouterr().err
        assert client.call("config.get") == imported
        assert machine_home.config.read_bytes() == before
        assert client.call("status")["processes"][0]["layout"] == "a100:2"
    finally:
        stop_daemon(machine_home)
