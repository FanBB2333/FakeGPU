"""Drive the terminal console against a real machine service."""

from __future__ import annotations

import asyncio
import signal

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable, Input  # noqa: E402

from fakegpu.console.app import Confirm, ConsoleApp  # noqa: E402
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
            await pilot.click("#cfg-apply")
            await pilot.pause()
            assert service.config.devices == "a100:2,h100:1"
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
