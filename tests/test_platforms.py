"""Host platforms, slot placement and the link topology FakeCUDA processes model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fakegpu.link_topology import GpuSlot, LinkTopology, NvlinkGroup, PcieLink
from fakegpu.machine.config import MachineConfig, parse_devices
from fakegpu.platform_cli import main as platform_main
from fakegpu.platforms import (
    PlatformError,
    export_platform,
    load_platforms,
    resolve_platform,
    topology_from_environment,
)
from fakegpu.profile_catalog import get_profile, load_profiles
from fakegpu.smi import SmiStatePublisher, main as smi_main


def resolve(spec: str, platform: str = "auto", **options):
    return resolve_platform(parse_devices(spec), platform, **options)


@pytest.mark.parametrize(
    ("devices", "platform"),
    [
        ("a100:8", "hgx-a100-8"),
        ("a100:2", "hgx-a100-4"),
        ("h100:8", "hgx-h100-8"),
        ("h100:4", "hgx-h100-4"),
        # No 4-GPU H200 wiring is published, so the 8-GPU board is used.
        ("h200:4", "hgx-h100-8"),
        ("b200:8", "hgx-b200-8"),
        ("b300:8", "hgx-b300-8"),
        ("v100:8", "dgx-1-v100"),
        ("p100-sxm2-16gb:8", "dgx-1-p100"),
        ("v100-sxm3-32gb:16", "dgx-2"),
        ("gh200-144gb", "gh200"),
        ("gb200:4", "gb200-nvl72-tray"),
        ("gb300:4", "gb300-nvl72-tray"),
        ("gb300-dgx-station,rtx-pro-6000-blackwell", "dgx-station-gb300"),
        ("gb10", "dgx-spark"),
        ("jetson-agx-orin-64gb", "jetson"),
        ("rtx4090-laptop", "laptop"),
        ("rtx4090:2", "desktop"),
        ("rtx3090:3", "workstation"),
        ("rtx-a6000:2", "workstation"),
        ("a100-pcie-80gb:8", "pcie-server"),
        ("l4", "pcie-server"),
    ],
)
def test_auto_picks_the_platform_for_the_form_factor(devices, platform):
    assert resolve(devices).platform.id == platform


@pytest.mark.parametrize(
    ("devices", "platform", "options", "message"),
    [
        ("a100:4,h100:4", "auto", {}, "holds one GPU model"),
        ("a100,a100-pcie-80gb", "auto", {}, "need different slots"),
        ("a100:9", "auto", {}, "exceed the HGX A100 8-GPU"),
        ("rtx4090:9", "auto", {}, "exceed"),
        ("a100:2", "hgx-h100-8", {}, "it fits hgx-a100-8, hgx-a100-4"),
        ("a100-pcie-80gb:2", "hgx-a100-8", {}, "has 8x SXM"),
        ("h100:2", "desktop", {}, "is one of the SXM modules"),
        ("rtx4090:3", "desktop", {}, "exceed the 2 slots"),
        ("a100:2", "auto", {"nvlink_bridges": "0-1"}, "wires its GPUs itself"),
        ("rtx4090:2", "auto", {"nvlink_bridges": "0-1"}, "no NVLink bridge connector"),
        ("a100-pcie-80gb:4", "auto", {"nvlink_bridges": "0-1-2-3"}, "join 2 GPUs, not 4"),
        ("a100-pcie-80gb:4", "auto", {"nvlink_bridges": "0-1,1-2"}, "already has a bridge"),
        ("a100-pcie-80gb,a100-pcie-40gb", "auto", {"nvlink_bridges": "0-1"}, "different models"),
        ("rtx4090", "auto", {"pcie_generation": 7}, "pcie_generation must be"),
        ("a100", "no-such-board", {}, "unknown platform"),
    ],
)
def test_layouts_that_do_not_fit_explain_why(devices, platform, options, message):
    with pytest.raises(PlatformError, match=message):
        resolve(devices, platform, **options)


def test_dgx1_v100_reproduces_the_published_hybrid_cube_mesh():
    matrix = resolve("v100:8").to_dict()["matrix"]
    assert matrix[0] == ["X", "NV1", "NV1", "NV2", "NV2", "SYS", "SYS", "SYS"]
    assert matrix[5] == ["SYS", "NV2", "SYS", "SYS", "NV1", "X", "NV2", "NV1"]
    topology = resolve("v100:8").topology
    assert all(len(topology.ports(index)) == 6 for index in range(8))


def test_switch_fabrics_bond_every_link_of_each_gpu():
    a100 = resolve("a100:8").topology
    assert {a100.relation(0, peer) for peer in range(1, 8)} == {"NV12"}
    assert a100.peer_bandwidth_gbps(0, 7) == 300
    assert [port.remote for port in a100.ports(3)] == ["nvswitch"] * 12
    b200 = resolve("b200:8").topology
    assert b200.relation(1, 6) == "NV18" and b200.peer_bandwidth_gbps(1, 6) == 900


def test_dgx2_pcie_tree_matches_the_published_pci_only_matrix():
    topology = resolve("v100-sxm3-32gb:16").topology
    assert [topology.pcie_relation(0, peer) for peer in range(8)] == [
        "X", "PIX", "PXB", "PXB", "NODE", "NODE", "NODE", "NODE",
    ]
    assert topology.pcie_relation(0, 8) == "SYS"


def test_pcie_links_train_at_the_slower_of_card_and_slot():
    desktop = resolve("rtx3090:2").topology
    # Two cards split the CPU's 16 lanes; the Gen4 card sets the generation.
    assert [str(slot.pcie) for slot in desktop.slots] == ["4x8", "4x8"]
    assert str(desktop.slots[0].gpu_max_pcie) == "4x16"
    assert str(resolve("rtx3090").topology.slots[0].pcie) == "4x16"
    assert str(resolve("rtx-pro-6000-blackwell:2", "workstation", pcie_generation=4).topology.slots[1].pcie) == "4x16"
    b300 = resolve("b300:8").topology.slots[0]
    assert str(b300.pcie) == "6x16" and b300.host_bandwidth_gbps == PcieLink(5, 16).bandwidth_gbps


def test_cpu_links_use_c2c_and_unified_memory_where_there_is_no_pcie_path():
    assert resolve("gh200-96gb").topology.host_bandwidth_gbps(0) == 450
    spark = resolve("gb10").topology.slots[0]
    assert spark.pcie is None and spark.host_link == "unified"
    assert spark.host_bandwidth_gbps == get_profile("gb10").memory_bandwidth_gbps / 2


def test_peer_access_follows_links_sockets_and_documented_policy():
    server = resolve("a100-pcie-80gb:8").topology
    assert server.relation(0, 1) == "PIX" and server.can_access_peer(0, 1)
    assert server.relation(0, 2) == "NODE" and server.can_access_peer(0, 2)
    assert server.relation(0, 4) == "SYS" and not server.can_access_peer(0, 4)
    assert not server.can_access_peer(3, 3)
    assert not resolve("rtx4090:2").topology.can_access_peer(0, 1)
    bridged = resolve("rtx3090:2", nvlink_bridges="0-1").topology
    assert bridged.relation(0, 1) == "NV4" and bridged.peer_bandwidth_gbps(0, 1) == 56.25
    assert not resolve("gb300-dgx-station,rtx-pro-6000-blackwell").topology.can_access_peer(0, 1)
    nvl = resolve("h200-nvl:4", nvlink_bridges="0-1-2-3").topology
    assert nvl.relation(0, 3) == "NV6" and len(nvl.ports(0)) == 18


def test_unknown_module_nvlink_falls_back_to_the_board_with_a_note():
    resolved = resolve("a800-sxm4-80gb:8")
    assert resolved.topology.relation(0, 1) == "NV12"
    assert any("a800-sxm4-80gb does not record its NVLink ports" in note for note in resolved.notes)


def test_platform_templates_cover_the_catalog_modules():
    platforms = load_platforms()
    profiles = load_profiles()
    accepted = {profile_id for platform in platforms.values() for profile_id in platform.accepts}
    assert accepted <= set(profiles)
    for profile in profiles.values():
        if profile.form_factor in ("sxm", "superchip"):
            assert profile.id in accepted, profile.id
    for platform in platforms.values():
        if platform.nvlink and platform.nvlink["fabric"] == "direct":
            for left, right, _ in platform.nvlink["pairs"]:
                assert left < right < platform.capacity, platform.id


def test_topology_round_trips_through_the_environment():
    for spec in ("a100:8", "v100:8", "rtx3090:2", "gb200:4", "gb10", "b300:8"):
        resolved = resolve(spec, nvlink_bridges="0-1" if spec == "rtx3090:2" else "")
        environ = resolved.environment()
        parsed = LinkTopology.from_environment([get_profile(p) for p in resolved.profiles], environ)
        assert parsed.valid, parsed.error
        assert parsed.environment() == environ
        count = len(resolved.profiles)
        assert [[parsed.relation(a, b) for b in range(count)] for a in range(count)] == resolved.to_dict()["matrix"]


def test_environment_parsing_rejects_bad_specs_without_links():
    profiles = [get_profile("a100")] * 2
    for environ, message in (
        ({"FAKEGPU_SLOTS": "type=sxm,pcie=4x16"}, "describes 1 GPUs but the process has 2"),
        ({"FAKEGPU_SLOTS": "type=sxm,pcie=9x16;type=sxm"}, "PCIe generation 9"),
        ({"FAKEGPU_SLOTS": "type=sxm,bogus=1;type=sxm"}, "unknown slot field"),
        ({"FAKEGPU_NVLINK_FABRIC": "mesh:0,2:1:25"}, "beyond the 2 GPUs"),
        ({"FAKEGPU_NVLINK_FABRIC": "ring:0,1:1:25"}, "kind 'ring'"),
        ({"FAKEGPU_NVLINK_FABRIC": "mesh:0,1:19:25"}, "1 to 18 links"),
        ({"FAKEGPU_NVLINK_FABRIC": "switch:0-1:12:25;switch:0-1:12:25"}, "more than 18 NVLinks"),
    ):
        topology = LinkTopology.from_environment(profiles, environ)
        assert topology.source == "modeled_environment_invalid"
        assert message in topology.error
        assert topology.nvlink == () and topology.can_access_peer(0, 1)


def test_legacy_nvlink_groups_keep_one_link_per_peer():
    topology = LinkTopology.from_environment(
        [get_profile("a100")] * 4,
        {"FAKEGPU_NVLINK_GROUPS": "0,1,2;1,2", "FAKEGPU_NVLINK_BANDWIDTH_GBPS": "800"},
    )
    assert topology.source == "modeled_environment"
    assert [(port.peer, port.bandwidth_gbps) for port in topology.ports(1)] == [(0, 800.0), (2, 800.0)]
    assert topology.relation(0, 2) == "NV1" and topology.relation(0, 3) == "PIX"


def test_unconfigured_processes_keep_each_gpu_at_its_own_link():
    topology = topology_from_environment(["h100", "rtx3090"], {})
    assert not topology.configured
    assert [str(slot.pcie) for slot in topology.slots] == ["5x16", "4x16"]
    assert topology.relation(0, 1) == "PIX" and topology.can_access_peer(0, 1)
    assert topology == LinkTopology((GpuSlot(pcie=PcieLink(5, 16)), GpuSlot(pcie=PcieLink(4, 16))))


def test_platform_name_alone_is_resolved_for_the_process_devices():
    environ = {"FAKEGPU_PROFILES": "a100-pcie-80gb:2", "FAKEGPU_PLATFORM": "workstation", "FAKEGPU_NVLINK_BRIDGES": "0-1"}
    topology = topology_from_environment(["a100-pcie-80gb"] * 2, environ)
    assert topology.platform == "workstation" and topology.relation(0, 1) == "NV12"
    export_platform(environ)
    assert environ["FAKEGPU_NVLINK_FABRIC"] == "mesh:0,1:12:25"
    bad = topology_from_environment(["a100"], {"FAKEGPU_PLATFORM": "desktop"})
    assert bad.source == "modeled_environment_invalid" and "FAKEGPU_PLATFORM" in bad.error
    with pytest.raises(PlatformError):
        export_platform({"FAKEGPU_PROFILES": "a100", "FAKEGPU_PLATFORM": "desktop"})


def test_machine_config_validates_the_platform():
    assert MachineConfig().resolved_platform().platform.id == "hgx-a100-8"
    with pytest.raises(ValueError, match="holds one GPU model"):
        MachineConfig(devices="a100:4,h100:4")
    assert MachineConfig(devices="a100:4,h100:4", platform="generic").resolved_platform().topology.relation(0, 4) == "PIX"
    with pytest.raises(ValueError, match="pcie_generation"):
        MachineConfig(pcie_generation="5")
    with pytest.raises(ValueError, match="unknown machine setting"):
        MachineConfig().updated({"slots": 4})


def test_smi_shows_the_platform_matrix_slots_and_links(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("FAKEGPU_PLATFORM", "auto")
    monkeypatch.setenv("FAKEGPU_PROFILES", "a100-pcie-80gb:8")
    path = tmp_path / "state.json"
    state = SmiStatePublisher(
        path,
        lambda: {
            "devices": [
                {"index": index, "profile_id": "a100-pcie-80gb", "total_memory": 80 * 2**30}
                for index in range(8)
            ]
        },
    ).publish_once(running=True)
    topology = state["topology"]
    assert (topology["schema_version"], topology["platform"], topology["link_count"]) == (
        "fakegpu.device_topology.v2", "pcie-server", 0,
    )
    device = state["devices"][2]["topology"]
    assert device["slot"] == {"type": "pcie", "numa_node": 0, "host_bridge": 1, "pcie_switch": "1"}
    assert device["pcie"]["generation"] == 4 and device["peer_access"] == [0, 1, 3]

    assert smi_main(["topo", "-m", "--state", str(path)]) == 0
    matrix = capsys.readouterr().out
    assert "Platform: PCIe server, 8 slots (pcie-server)" in matrix
    assert "GPU0  X     PIX   NODE  NODE  SYS" in matrix
    assert "NUMA Affinity" in matrix and "PXB  = Connection traversing multiple PCIe bridges" in matrix

    assert smi_main([
        "--state", str(path), "--query-gpu",
        "index,platform,slot.type,numa.node,pcie.link.gen.current,pcie.link.width.max,host_link.bandwidth",
        "--format", "json", "-i", "5",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["records"] == [
        {
            "index": 5,
            "platform": "pcie-server",
            "slot.type": "pcie",
            "numa.node": 1,
            "pcie.link.gen.current": 4,
            "pcie.link.width.max": 16,
            "host_link.bandwidth": 31.5,
        }
    ]


def test_platform_command_lists_shows_and_exports(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAKEGPU_HOME", str(tmp_path))
    assert platform_main(["list"]) == 0
    assert "hgx-b300-8" in capsys.readouterr().out
    assert platform_main(["show"]) == 0
    shown = capsys.readouterr().out
    assert "HGX A100 8-GPU (hgx-a100-8)" in shown and "NV12" in shown
    assert platform_main(["show", "--devices", "rtx3090:2", "--nvlink-bridges", "0-1", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["id"] == "desktop" and document["matrix"][0] == ["X", "NV4"]
    assert platform_main(["env", "--devices", "h100:2"]) == 0
    assert "export FAKEGPU_NVLINK_FABRIC=mesh:0,1:6:25" in capsys.readouterr().out
    assert platform_main(["show", "--devices", "a100,h100"]) == 2
    assert "holds one GPU model" in capsys.readouterr().err


def test_nvlink_group_encoding_accepts_ranges():
    assert NvlinkGroup.decode("switch:0-3,6:12:25").members == (0, 1, 2, 3, 6)
