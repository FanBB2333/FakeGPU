from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from fakegpu.link_topology import GpuSlot, LinkTopology, PcieLink
from fakegpu.link_traffic import LinkTraffic, explain_route
from fakegpu.machine.config import MachineConfig, MachinePaths, parse_devices
from fakegpu.platform_cli import main
from fakegpu.platforms import resolve_platform


@pytest.fixture(autouse=True)
def isolated_machine(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FAKEGPU_HOME", str(tmp_path / "fakegpu"))


def _json_route(capsys, *options: str) -> dict:
    assert main(["route", *options, "--json"]) == 0
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize(
    ("devices", "source", "target", "kind", "bandwidth", "middle"),
    [
        ("a100:8", 0, 1, "nvlink", 300.0, "nvswitch"),
        ("a100:4", 0, 1, "nvlink", 100.0, None),
        ("a100-pcie-80gb:8", 0, 1, "pcie_peer", 31.508, None),
        ("a100-pcie-80gb:8", 0, 7, "host_staged", 31.508, "host"),
        ("gb300-dgx-station,rtx-pro-6000-blackwell", 0, 1, "host_staged", 63.015, "host"),
        ("gb10", None, 0, "unified", 136.5, None),
        ("gb300-dgx-station", 0, None, "host", 450.0, None),
    ],
)
def test_route_json_explains_the_runtime_copy_model(capsys, devices, source, target, kind, bandwidth, middle):
    document = _json_route(
        capsys, "--devices", devices, "--from", "host" if source is None else str(source),
        "--to", "host" if target is None else str(target), "--bytes", "1GiB",
    )
    resolved = resolve_platform(parse_devices(devices))
    assert document["schema_version"] == "fakegpu.copy_route.v1"
    assert document["platform"] == {"id": resolved.platform.id, "name": resolved.platform.name}
    assert (document["source"], document["target"], document["bytes"]) == (source, target, 2**30)
    assert document["route"] == kind
    assert document["bandwidth_gbps"] == pytest.approx(bandwidth)
    assert document["modeled_seconds"] == pytest.approx(2**30 / (bandwidth * 1e9))
    assert document["modeled_seconds"] == LinkTraffic(lambda: resolved.topology).record(source, target, 2**30)
    assert document["relation"] == (
        resolved.topology.relation(source, target) if source is not None and target is not None else None
    )
    if source is not None:
        assert document["path"][0] == f"GPU{source}"
    if target is not None:
        assert document["path"][-1] == f"GPU{target}"
    if middle:
        assert any(middle in node.lower() for node in document["path"][1:-1])
    bottlenecks = [link for link in document["links"] if link["bottleneck"]]
    assert bottlenecks
    assert all(link["bandwidth_gbps"] == pytest.approx(bandwidth) for link in bottlenecks)
    assert document["notes"]


def test_host_staging_identifies_the_slower_link_in_both_directions(capsys):
    topology = resolve_platform(["gb300-dgx-station", "rtx-pro-6000-blackwell"]).topology
    for source, target in ((0, 1), (1, 0)):
        document = explain_route(topology, source, target, 10**9)
        links = {link["kind"]: link for link in document["links"]}
        assert links["c2c"]["bandwidth_gbps"] == 450.0
        assert not links["c2c"]["bottleneck"]
        assert links["pcie"]["bandwidth_gbps"] == pytest.approx(63.015)
        assert links["pcie"]["bottleneck"]
        assert document["modeled_seconds"] == pytest.approx(1 / 63.015)

    assert main([
        "route", "--devices", "gb300-dgx-station,rtx-pro-6000-blackwell",
        "--from", "0", "--to", "1", "--bytes", "1GB",
    ]) == 0
    text = capsys.readouterr().out
    for label in ("Path:", "Route:", "Bottleneck:", "Modeled time:", "63.015"):
        assert label in text
    assert "host" in text.lower() and "c2c" in text.lower() and "pcie" in text.lower()


def test_host_route_exposes_the_slower_cpu_uplink():
    topology = resolve_platform(["b300"] * 8).topology
    document = explain_route(topology, None, 0, 10**9)
    assert document["bandwidth_gbps"] == pytest.approx(63.015)
    link = document["links"][0]
    assert link["bottleneck"]
    assert "Gen6" in link["name"] and "Gen5" in link["name"]


def test_pcie_peer_route_uses_cpu_uplinks_only_when_the_path_reaches_them():
    slots = (
        GpuSlot(pcie=PcieLink(5, 16), host_pcie=PcieLink(3, 16)),
        GpuSlot(pcie=PcieLink(4, 16), host_pcie=PcieLink(3, 8)),
    )
    local = explain_route(LinkTopology(slots), 0, 1, 10**9)
    assert local["relation"] == "PIX"
    assert local["bandwidth_gbps"] == pytest.approx(31.508)
    assert [link["bandwidth_gbps"] for link in local["links"]] == pytest.approx([63.015, 31.508])

    through_host = explain_route(LinkTopology((slots[0], replace(slots[1], switch="1"))), 0, 1, 10**9)
    assert through_host["relation"] == "PHB"
    assert through_host["route"] == "pcie_peer"
    assert through_host["bandwidth_gbps"] == pytest.approx(7.877)
    assert [link["bandwidth_gbps"] for link in through_host["links"]] == pytest.approx([15.754, 7.877])


def test_unknown_bandwidth_is_not_reported_as_zero_cost():
    topology = LinkTopology((GpuSlot(pcie=None, host_link="unified"),))
    document = explain_route(topology, None, 0, 1024)
    assert document["bandwidth_gbps"] is None
    assert document["modeled_seconds"] is None
    assert not any(link["bottleneck"] for link in document["links"])


def test_partially_known_staging_bandwidth_does_not_claim_a_known_bottleneck():
    topology = LinkTopology((GpuSlot(pcie=None, host_link="c2c"), GpuSlot(p2p=False)))
    document = explain_route(topology, 0, 1, 10**9)
    assert document["modeled_seconds"] == LinkTraffic(lambda: topology).record(0, 1, 10**9)
    assert document["bandwidth_gbps"] == pytest.approx(31.508)
    assert not any(link["bottleneck"] for link in document["links"])
    assert any("bottleneck is unknown" in note for note in document["notes"])


@pytest.mark.parametrize(("size", "nbytes"), [("1", 1), ("3", 3), ("64KiB", 65536), ("1GB", 10**9), ("1GiB", 2**30)])
def test_byte_sizes_preserve_binary_decimal_and_unaligned_copies(capsys, size, nbytes):
    document = _json_route(capsys, "--from", "0", "--to", "1", "--bytes", size)
    assert document["bytes"] == nbytes


def test_route_inherits_machine_settings_and_keeps_config_unchanged(capsys):
    paths = MachinePaths.from_env()
    paths.root.mkdir(parents=True)
    MachineConfig(devices="rtx3090:2", platform="workstation", pcie_generation=3, nvlink_bridges="0-1").save(paths.config)
    original = paths.config.read_bytes()
    options = ("--from", "0", "--to", "1", "--bytes", "1GiB")

    inherited = _json_route(capsys, *options)
    assert inherited["platform"]["id"] == "workstation"
    assert inherited["route"] == "nvlink" and inherited["bandwidth_gbps"] == 56.25

    unbridged = _json_route(capsys, *options, "--nvlink-bridges", "")
    assert unbridged["route"] == "pcie_peer" and unbridged["bandwidth_gbps"] == pytest.approx(15.754)

    explicit = _json_route(capsys, *options, "--devices", "rtx4090:2", "--pcie-gen", "4")
    assert explicit["platform"]["id"] == "desktop"
    assert explicit["route"] == "host_staged" and explicit["bandwidth_gbps"] == pytest.approx(15.754)
    assert paths.config.read_bytes() == original
    assert list(paths.root.iterdir()) == [paths.config]


@pytest.mark.parametrize(
    ("source", "target", "size"),
    [("gpu", "1", "1"), ("-1", "1", "1"), ("8", "1", "1"), ("0", "0", "1"),
     ("host", "host", "1"), ("0", "1", "0"), ("0", "1", "-1"), ("0", "1", "1.5MiB"), ("0", "1", "1XB")],
)
def test_invalid_transfer_arguments_are_usage_errors(source, target, size):
    result = subprocess.run(
        [sys.executable, "-m", "fakegpu", "platform", "route", "--from", source, "--to", target, "--bytes", size],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2, result.stderr
    assert result.stderr and "Traceback" not in result.stderr


def test_route_command_does_not_import_torch_or_create_daemon_files():
    result = subprocess.run(
        [sys.executable, "-c", "from fakegpu.platform_cli import main; import sys; "
         "assert main(['route', '--from', '0', '--to', '1', '--bytes', '1']) == 0; "
         "assert 'torch' not in sys.modules; assert 'fakegpu.machine.service' not in sys.modules"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not Path(os.environ["FAKEGPU_HOME"]).exists()
