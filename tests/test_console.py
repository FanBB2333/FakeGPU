"""Drive the terminal console against a real machine service."""

from __future__ import annotations

import asyncio
import io
import signal

import pytest

pytest.importorskip("textual")

from rich.console import Console  # noqa: E402
from textual.widgets import DataTable, Input, Select, Static  # noqa: E402

from fakegpu.console.app import Confirm, ConsoleApp, ProfilePicker, machine_charts, plan_layout  # noqa: E402
from fakegpu.console.widgets import Chart, selected_key  # noqa: E402
from fakegpu.machine.client import MachineError  # noqa: E402
from fakegpu.machine.config import MachineConfig, MachinePaths  # noqa: E402
from fakegpu.machine.service import MachineService  # noqa: E402
from test_machine import child, publish  # noqa: E402,F401


class LocalClient:
    """Call the service in-process, polling before each status like the daemon."""

    def __init__(self, service: MachineService):
        self.service = service
        self.paths = service.paths

    def call(self, method, **params):
        if method == "status":
            self.service.poll()
        response = self.service.dispatch({"method": method, "params": params})
        if not response["ok"]:
            raise MachineError(response["error"]["code"], response["error"]["message"])
        return response["result"]


@pytest.fixture
def service(tmp_path):
    return MachineService(MachinePaths(tmp_path), MachineConfig(devices="a100:2", telemetry="reference"))


def test_console_shows_and_controls_the_machine(service, child):  # noqa: F811
    publish(service, child.pid)
    app = ConsoleApp(LocalClient(service), interval=0.1)

    async def scenario():
        async with app.run_test(size=(180, 60)) as pilot:
            await pilot.pause()
            assert len(app.cards) == 2
            assert app.query_one("#overview-processes", DataTable).row_count == 1
            assert [event["kind"] for event in app.events] == ["process.attached"]

            await pilot.press("3")
            await pilot.pause()
            app.show_device_detail()
            recorder = Console(width=160, record=True, file=io.StringIO())
            recorder.print(app.query_one("#device-detail", Static).content)
            detail = recorder.export_text()
            assert "Copies" in detail and "none recorded" in detail
            assert "NVLink" in detail  # trend rows follow the HGX board's links
            await pilot.press("1")
            await pilot.pause()

            await pilot.press("k")
            await pilot.pause()
            assert isinstance(app.screen, Confirm)
            await pilot.press("y")
            await pilot.pause()
            assert child.wait(timeout=5) == -signal.SIGTERM

            await pilot.press("6")
            app.query_one("#profile-filter", Input).value = "h100"
            await pilot.pause()
            await pilot.press("a")
            await pilot.pause()
            assert app.query_one("#cfg-devices", Input).value.endswith(",h100:1")
            # A100 and H100 modules need different baseboards.
            assert "one GPU model" in str(app.query_one("#cfg-devices-check", Static).render())
            await pilot.click("#cfg-apply")
            await pilot.pause()
            assert service.config.devices == "a100:2"
            app.query_one("#cfg-platform", Select).value = "generic"
            await pilot.pause()
            await pilot.click("#cfg-apply")
            await pilot.pause()
            assert (service.config.devices, service.config.platform) == ("a100:2,h100:1", "generic")
            assert len(app.cards) == 3

            await pilot.press("b")
            await pilot.pause()
            assert service.config.background is False
            await pilot.press("q")

    asyncio.run(scenario())
    assert app.return_value == "detach"
    kinds = [event["kind"] for event in service.events()]
    assert {"process.attached", "process.signal_sent", "config.changed"} <= set(kinds)


def test_console_survives_a_stopped_machine(service):
    class Offline(LocalClient):
        def call(self, method, **params):
            raise MachineError("not_running", "the FakeGPU machine daemon is not running")

    app = ConsoleApp(Offline(service), interval=0.1)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert "not running" in app.sub_title
            await pilot.press("k")
            await pilot.press("Q")
            await pilot.pause()
            await pilot.press("y")

    asyncio.run(scenario())
    assert app.return_value == "stop"


def text(renderable, width=160):
    recorder = Console(width=width, record=True, color_system=None, file=io.StringIO())
    recorder.print(renderable)
    return recorder.export_text()


def test_layout_plan_moves_to_a_platform_that_holds_the_devices():
    config = MachineConfig(devices="a100:4").to_dict()
    smaller = plan_layout(config, ["a100"] * 2)
    assert (smaller.changes, smaller.platform, smaller.notes) == ({"devices": "a100:2"}, "HGX A100 4-GPU", ())

    mixed = plan_layout(config, ["a100", "h100"])
    assert mixed.changes == {"devices": "a100:1,h100:1", "platform": "generic"}
    assert "one GPU model" in mixed.notes[0]

    bridged = MachineConfig(devices="rtx3090:2", nvlink_bridges="0-1").to_dict()
    single = plan_layout(bridged, ["rtx3090"])
    assert single.changes == {"devices": "rtx3090:1", "nvlink_bridges": ""}
    assert single.notes == ("NVLink bridges 0-1 removed",)

    assert "at least one device" in plan_layout(config, []).error
    assert "exceed" in plan_layout(config, ["a100"] * 65).error


def test_chart_scales_samples_to_its_rows():
    chart = Chart("Power", [None, 0, 100, 400], maximum=400, unit="W", color="red", height=2)
    lines = text(chart, width=30).splitlines()
    assert lines[0].startswith("● Power  400W  peak 400W")
    assert lines[1] == "400W┤" + " " * 24 + "█"
    assert lines[2] == "   0┤" + " " * 23 + "▄█"


def test_machine_charts_combine_the_layout():
    def device(percent, total, gpu, power, temperature, in_layout=True):
        return {
            "in_layout": in_layout, "power_limit_w": 400.0,
            "memory": {"used_bytes": int(total * percent / 100), "total_bytes": total},
            "telemetry": {"gpu_percent": gpu, "power_w": power, "temperature_c": temperature},
            "history": {
                "time": [1.0, 1.5], "memory_percent": [0.0, percent], "gpu": [None, gpu],
                "power_w": [None, power], "temperature": [None, temperature],
                "pcie_gbps": [None, 1.0], "c2c_gbps": [None, None], "nvlink_gbps": [None, 2.0],
            },
        }

    gib = 1024**3
    charts = machine_charts([
        device(50, 80 * gib, 20, 100.0, 40), device(25, 40 * gib, 60, 300.0, 50),
        device(90, 80 * gib, 100, 400.0, 90, in_layout=False),
    ])
    memory, utilization, power, temperature, links = (chart.values[-1] for chart in charts)
    assert round(memory, 2) == round((40 + 10) / 120 * 100, 2)
    assert (utilization, power, temperature, links) == (40, 400.0, 50, 6.0)
    assert charts[2].maximum == 800.0
    assert [chart.values[0] for chart in charts] == [0.0, None, None, None, None]


def test_console_edits_the_layout_per_gpu(service):
    app = ConsoleApp(LocalClient(service), interval=0.1)

    async def scenario():
        async with app.run_test(size=(180, 60)) as pilot:
            await pilot.pause()
            await pilot.press("e")  # only on the Devices tab
            await pilot.pause()
            assert not isinstance(app.screen, ProfilePicker)

            await pilot.click(app.cards[1])
            await pilot.pause()
            devices = app.query_one("#devices-table", DataTable)
            assert app.query_one("TabbedContent").active == "devices"
            assert selected_key(devices) == "gpu-1"

            await pilot.press("e")
            await pilot.pause()
            picker = app.screen
            assert isinstance(picker, ProfilePicker)
            picker.query_one("#picker-filter", Input).value = "h100"
            await pilot.pause()
            assert "generic" in text(picker.query_one("#picker-preview", Static).content)
            await pilot.press("enter")
            await pilot.pause()
            assert (service.config.devices, service.config.platform) == ("a100:1,h100:1", "generic")

            await pilot.press("a")  # starts from the selected GPU's profile
            await pilot.pause()
            app.screen.query_one("#picker-count", Input).value = "2"
            await pilot.pause()
            await pilot.click("#picker-apply")
            await pilot.pause()
            assert service.config.devices == "a100:1,h100:3"
            assert selected_key(devices) == "gpu-2"

            await pilot.press("x")
            await pilot.pause()
            assert isinstance(app.screen, Confirm)
            await pilot.press("y")
            await pilot.pause()
            assert service.config.devices == "a100:1,h100:2"

            await pilot.press("E")
            await pilot.pause()
            app.screen.query_one("#picker-filter", Input).value = "a100-pcie-80gb"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert service.config.devices == "a100-pcie-80gb:3"
            assert app.query_one("#cfg-devices", Input).value == "a100-pcie-80gb:3"

            await pilot.press("1", "v")
            await pilot.pause()
            gpu_list = app.query_one("#gpu-list", DataTable)
            assert gpu_list.display and not app.query_one("#gpus").display
            assert gpu_list.row_count == 3
            await pilot.press("down", "enter")  # opens that GPU on the Devices tab
            await pilot.pause()
            assert selected_key(devices) == "gpu-1"
            await pilot.press("2")
            await pilot.pause()
            assert app.query_one("TabbedContent").active == "processes"
            await pilot.press("q")

    asyncio.run(scenario())
