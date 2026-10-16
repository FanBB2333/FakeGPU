"""Terminal console for the background FakeGPU machine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from ..machine.client import MachineError
from ..machine.config import parse_devices
from ..machine.service import EVENT_BUFFER
from ..profile_catalog import load_profiles
from .widgets import (
    GpuCard,
    StatTile,
    bar,
    duration,
    event_line,
    key_values,
    selected_key,
    size,
    sparkline,
    sync_rows,
)

TABS = ("overview", "processes", "devices", "events", "machine", "profiles")
TELEMETRY_LABELS = {"reference": "A100 reference (modeled)", "none": "none (sensors N/A)"}


class Confirm(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "dismiss(False)", "Cancel"), Binding("y", "dismiss(True)", "Yes")]

    def __init__(self, question: str):
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self.question)
            with Horizontal():
                yield Button("Yes", variant="error", id="yes")
                yield Button("Cancel", id="no")

    @on(Button.Pressed)
    def answer(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


class ConsoleApp(App[str]):
    """Watch and control the machine through any client with ``call(method, **params)``.

    ``run()`` returns ``"detach"`` or ``"stop"``; the caller decides what to do
    with the daemon, so the console itself never outlives or kills it.
    """

    TITLE = "FakeGPU Console"
    CSS = """
    #stats { height: 4; margin: 0 1; }
    TabbedContent, TabbedContent > ContentSwitcher, TabPane { height: 1fr; }
    StatTile {
        width: 1fr; height: 4; border: round $primary-darken-2; text-align: center;
        text-wrap: nowrap; text-overflow: ellipsis;
    }
    #gpus { grid-size: 4; grid-gutter: 0 1; height: auto; max-height: 60%; overflow-y: auto; margin: 0 1; }
    GpuCard { height: 9; border: round $panel-lighten-2; padding: 0 1; }
    GpuCard.busy { border: round $success; }
    GpuCard.warn { border: round $warning; }
    GpuCard.alert { border: round $error; }
    #lower { height: 1fr; margin: 0 1; }
    #overview-processes { width: 3fr; }
    #overview-events { width: 2fr; border-left: tall $panel; }
    #processes-table, #devices-table { height: 1fr; min-height: 5; }
    #ended-table { height: auto; max-height: 7; }
    .detail { height: auto; max-height: 50%; border-top: tall $panel; padding: 0 1; }
    .section { text-style: bold; color: $accent; margin-top: 1; }
    #event-filters { height: 3; }
    #event-filter { width: 1fr; }
    #event-level { width: 30; }
    #machine-body { layout: horizontal; }
    #machine-form { width: 92; height: auto; padding: 0 2; }
    #machine-form Input, #machine-form Select { width: 1fr; max-width: 60; }
    #machine-form .row { height: 3; align: left middle; }
    #machine-form .row Label { width: 26; }
    #machine-info { width: 1fr; height: auto; padding: 1 2; border-left: tall $panel; }
    #machine-body.stacked { layout: vertical; }
    #machine-body.stacked #machine-form { width: 1fr; }
    #machine-body.stacked #machine-info { border-left: none; border-top: tall $panel; }
    #machine-form .buttons { height: 3; margin-top: 1; }
    #machine-form .buttons Button { margin-right: 2; }
    #cfg-devices-check { margin-bottom: 1; }
    #profile-filter { margin: 0 1; }
    Confirm { align: center middle; }
    #dialog { width: 64; height: auto; border: thick $warning; background: $surface; padding: 1 2; }
    #dialog Horizontal { height: auto; align: center middle; margin-top: 1; }
    #dialog Button { margin: 0 1; }
    """
    BINDINGS = [
        Binding("q", "detach", "Quit"),
        Binding("Q", "stop_machine", "Stop machine"),
        Binding("b", "toggle_background", "Background"),
        Binding("k", "signal('SIGTERM')", "Terminate"),
        Binding("K", "signal('SIGKILL')", "Kill"),
        Binding("i", "signal('SIGINT')", "Interrupt", show=False),
        Binding("a", "add_profile", "Add to layout", show=False),
        Binding("s", "start_machine", "Start machine", show=False),
        *[Binding(str(number), f"tab('{tab}')", tab.title(), show=False) for number, tab in enumerate(TABS, 1)],
    ]

    def __init__(self, client: Any, *, interval: float = 0.5):
        super().__init__()
        self.client = client
        self.interval = interval
        self.machine: dict[str, Any] | None = None
        self.last_seq = 0
        self.events: list[dict[str, Any]] = []
        self.backlog_loaded = False
        self.form_loaded = False
        self.cards: list[GpuCard] = []
        self.profiles = sorted(load_profiles().values(), key=lambda p: (p.architecture, p.id))

    # ------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="stats"):
            for name in ("devices", "memory", "power", "processes", "alerts"):
                yield StatTile(id=f"stat-{name}")
        with TabbedContent(initial="overview"):
            with TabPane("1 Overview", id="overview"):
                yield Grid(id="gpus")
                with Horizontal(id="lower"):
                    yield DataTable(id="overview-processes", cursor_type="row")
                    yield RichLog(id="overview-events", max_lines=300, wrap=True, min_width=20)
            with TabPane("2 Processes", id="processes"):
                yield DataTable(id="processes-table", cursor_type="row")
                with VerticalScroll(classes="detail"):
                    yield Static(id="process-detail")
                yield Static("Recently ended", classes="section")
                yield DataTable(id="ended-table", cursor_type="row")
            with TabPane("3 Devices", id="devices"):
                yield DataTable(id="devices-table", cursor_type="row")
                with VerticalScroll(classes="detail"):
                    yield Static(id="device-detail")
            with TabPane("4 Events", id="events"):
                with Horizontal(id="event-filters"):
                    yield Input(placeholder="Filter events by text", id="event-filter")
                    yield Select(
                        [("All levels", "info"), ("Warnings and errors", "warning"), ("Errors", "error")],
                        allow_blank=False,
                        value="info",
                        id="event-level",
                    )
                yield RichLog(id="events-log", max_lines=EVENT_BUFFER, wrap=True, min_width=20)
            with TabPane("5 Machine", id="machine"):
                with VerticalScroll(id="machine-body"):
                    with Vertical(id="machine-form"):
                        yield Static("Device layout", classes="section")
                        yield Input(placeholder="a100:8 or a100:4,h100:4", id="cfg-devices")
                        yield Static(id="cfg-devices-check")
                        with Horizontal(classes="row"):
                            yield Label("Sensor model")
                            yield Select(
                                [(label, mode) for mode, label in TELEMETRY_LABELS.items()],
                                allow_blank=False,
                                id="cfg-telemetry",
                            )
                        with Horizontal(classes="row"):
                            yield Label("Keep running after quit")
                            yield Switch(id="cfg-background")
                        with Horizontal(classes="row"):
                            yield Label("Memory warning at (%)")
                            yield Input(id="cfg-warning", type="number")
                        with Horizontal(classes="row"):
                            yield Label("Heartbeat timeout (s)")
                            yield Input(id="cfg-stale", type="number")
                        with Horizontal(classes="buttons"):
                            yield Button("Apply", variant="primary", id="cfg-apply")
                            yield Button("Reload", id="cfg-reload")
                            yield Button("Stop machine", variant="error", id="stop-machine")
                    yield Static(id="machine-info")
            with TabPane("6 Profiles", id="profiles"):
                yield Input(placeholder="Filter by id, name, architecture or segment; press a to add", id="profile-filter")
                yield DataTable(id="profiles-table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        columns = {
            "overview-processes": ("PID", "Name", "Status", "GPUs", "Memory", "GPU %", "Uptime"),
            "processes-table": (
                "PID", "User", "Name", "Status", "GPUs", "Memory", "Peak", "GPU %", "Stage", "Backend",
                "Uptime", "Heartbeat", "Command",
            ),
            "ended-table": ("PID", "Name", "Result", "Runtime", "Peak", "Ended (UTC)"),
            "devices-table": (
                "GPU", "Profile", "Name", "Arch", "CC", "Used", "Total", "Mem %", "Procs", "Util",
                "Temp", "Power", "SM clock",
            ),
            "profiles-table": ("ID", "Name", "Architecture", "Segment", "Memory", "CC", "SMs", "Power"),
        }
        for table_id, labels in columns.items():
            self.query_one(f"#{table_id}", DataTable).add_columns(*labels)
        self.show_profiles()
        self.refresh_machine()
        self.set_interval(self.interval, self.refresh_machine)

    def on_resize(self) -> None:
        self.query_one("#gpus", Grid).styles.grid_size_columns = max(1, (self.size.width - 2) // 42)
        # Settings and machine details side by side only when both fit.
        self.query_one("#machine-body").set_class(self.size.width < 140, "stacked")

    # ----------------------------------------------------------- refresh

    def refresh_machine(self) -> None:
        try:
            status = self.client.call("status")
            events = self.client.call("events", after=self.last_seq, limit=EVENT_BUFFER)
        except MachineError as exc:
            self.show_offline(str(exc))
            return
        self.machine = status
        self.show_status(status)
        for event in events:
            self.add_event(event)
        self.backlog_loaded = True

    def show_offline(self, reason: str) -> None:
        if self.machine is not None:
            self.notify(f"Lost the machine: {reason}", severity="error")
        self.machine = None
        self.sub_title = "machine is not running; press s to start it"
        for name in ("devices", "memory", "power", "processes", "alerts"):
            self.query_one(f"#stat-{name}", StatTile).show("-", name)
        self.show_cards([], 100)
        for table_id in ("#overview-processes", "#processes-table", "#devices-table"):
            self.query_one(table_id, DataTable).clear()

    def show_status(self, status: dict[str, Any]) -> None:
        daemon, config, summary = status["daemon"], status["config"], status["summary"]
        self.sub_title = (
            f"pid {daemon['pid']}, up {duration(daemon['uptime_seconds'])}, layout {config['devices']}, "
            f"telemetry {config['telemetry']}, background {'on' if config['background'] else 'off'}"
        )
        total, used = summary["memory_total_bytes"], summary["memory_used_bytes"]
        percent = used / total * 100 if total else 0
        self.query_one("#stat-devices", StatTile).show(str(summary["device_count"]), f"devices ({config['devices']})")
        self.query_one("#stat-memory", StatTile).show(f"{size(used)} / {size(total)}", f"memory, {percent:.1f}% used")
        self.query_one("#stat-processes", StatTile).show(
            str(summary["process_count"]), f"processes, {summary['stalled_count']} stalled"
        )
        self.query_one("#stat-alerts", StatTile).show(
            str(summary["alerts"]), "memory alerts", "bold red" if summary["alerts"] else "bold green"
        )
        sensed = [d for d in status["devices"] if d["telemetry"]]
        if sensed:
            draw = sum(d["telemetry"]["power_w"] for d in sensed)
            limit = sum(d["power_limit_w"] or 0 for d in sensed)
            self.query_one("#stat-power", StatTile).show(
                f"{draw:.0f} / {limit:.0f} W", f"power, modeled on {len(sensed)} GPU(s)"
            )
        else:
            self.query_one("#stat-power", StatTile).show("N/A", "power (no sensor model)", "dim")
        self.show_cards(status["devices"], config["memory_warning_percent"])
        self.show_processes(status["processes"], status["ended"])
        self.show_devices(status["devices"])
        self.show_machine(status)

    def show_cards(self, devices: list[dict], warning: float) -> None:
        while len(self.cards) > len(devices):
            self.cards.pop().remove()
        if len(self.cards) < len(devices):
            new = [GpuCard() for _ in range(len(devices) - len(self.cards))]
            self.query_one("#gpus", Grid).mount_all(new)
            self.cards.extend(new)
        for card, device in zip(self.cards, devices):
            card.show(device, warning)

    def show_processes(self, processes: list[dict], ended: list[dict]) -> None:
        def gpus(process: dict) -> str:
            return ",".join(str(row["index"]) for row in process["devices"] if row["used_bytes"]) or "-"

        def gpu_percent(process: dict) -> str:
            return "N/A" if process["gpu_percent"] is None else f"{process['gpu_percent']}%"

        status_style = {"running": "green", "stalled": "yellow"}
        sync_rows(self.query_one("#overview-processes", DataTable), [
            (str(p["pid"]), (
                str(p["pid"]), p["name"], Text(p["status"], style=status_style.get(p["status"], "")), gpus(p),
                size(p["memory_bytes"]), gpu_percent(p), duration(p["runtime_seconds"]),
            ))
            for p in processes
        ])
        sync_rows(self.query_one("#processes-table", DataTable), [
            (str(p["pid"]), (
                str(p["pid"]), p["user"], p["name"], Text(p["status"], style=status_style.get(p["status"], "")),
                gpus(p), size(p["memory_bytes"]), size(p["peak_bytes"]), gpu_percent(p), p["stage"], p["backend"],
                duration(p["runtime_seconds"]), f"{p['heartbeat_age_seconds']:.1f}s", p["command"] or p["name"],
            ))
            for p in processes
        ])
        result_style = {"exited": "green", "terminated": "yellow", "vanished": "red"}
        sync_rows(self.query_one("#ended-table", DataTable), [
            (f"{p['pid']}-{p['ended_at']}", (
                str(p["pid"]), p["name"], Text(p["reason"], style=result_style.get(p["status"], "")),
                duration(p["runtime_seconds"]), size(p["peak_bytes"]), (p["ended_at"] or "")[11:19],
            ))
            for p in ended
        ])
        self.show_process_detail()

    def show_devices(self, devices: list[dict]) -> None:
        rows = []
        for d in devices:
            memory, sensors = d["memory"], d["telemetry"] or {}
            rows.append((d["uuid"], (
                str(d["index"]), d["profile_id"], d["name"], d["architecture"], d["compute_capability"],
                size(memory["used_bytes"]), size(memory["total_bytes"]), f"{memory['percent']:.1f}%",
                str(d["process_count"]),
                f"{sensors['gpu_percent']}%" if sensors else "N/A",
                f"{sensors['temperature_c']}°C" if sensors else "N/A",
                f"{sensors['power_w']:.0f} W" if sensors else "N/A",
                f"{sensors['sm_clock_mhz']} MHz" if sensors else "N/A",
            )))
        sync_rows(self.query_one("#devices-table", DataTable), rows)
        self.show_device_detail()

    @on(DataTable.RowHighlighted, "#processes-table")
    def show_process_detail(self) -> None:
        detail = self.query_one("#process-detail", Static)
        pid = selected_key(self.query_one("#processes-table", DataTable))
        process = next((p for p in (self.machine or {}).get("processes", []) if str(p["pid"]) == pid), None)
        if process is None:
            detail.update(Text("No FakeCUDA processes. Scripts that call fakegpu.init() join this machine.", style="dim"))
            return
        totals = {d["index"]: d["memory"]["total_bytes"] for d in (self.machine or {}).get("devices", [])}
        devices = Table.grid(padding=(0, 2))
        for row in process["devices"]:
            if row["used_bytes"]:
                share = row["used_bytes"] / totals[row["index"]] * 100 if totals.get(row["index"]) else 0
                devices.add_row(f"GPU {row['index']} {row['profile_id']}", bar(share, 30), size(row["used_bytes"]))
        if not devices.row_count:
            devices.add_row(Text("No device memory in use yet.", style="dim"))
        software = ", ".join(f"{k} {v}" for k, v in process["software"].items() if isinstance(v, str) and v)
        detail.update(Group(
            key_values([
                ("Command", process["command"] or process["name"]),
                ("User", process["user"] or "-"),
                ("Attached", f"{process['attached_at']} ({duration(process['runtime_seconds'])} ago)"),
                ("Layout", process["layout"]),
                ("Stage / backend", f"{process['stage']} / {process['backend']}"),
                ("Tracking", process["tracking_confidence"]),
                ("Software", software or "-"),
                ("Keys", "k terminate (SIGTERM), K kill (SIGKILL), i interrupt (SIGINT)"),
            ]),
            devices,
        ))

    @on(DataTable.RowHighlighted, "#devices-table")
    def show_device_detail(self) -> None:
        detail = self.query_one("#device-detail", Static)
        uuid = selected_key(self.query_one("#devices-table", DataTable))
        device = next((d for d in (self.machine or {}).get("devices", []) if d["uuid"] == uuid), None)
        if device is None:
            detail.update("")
            return
        memory, allocator, history = device["memory"], device["allocator"], device.get("history", {})
        trends = Table.grid(padding=(0, 2))
        for label, key, maximum, unit in (
            ("Memory", "memory_percent", 100.0, "%"),
            ("Utilization", "gpu", 100.0, "%"),
            ("Temperature", "temperature", 100.0, "°C"),
            ("Power", "power_w", float(device["power_limit_w"] or 1), "W"),
        ):
            values = history.get(key, [])
            last = next((value for value in reversed(values) if value is not None), None)
            trends.add_row(label, sparkline(values, 60, maximum), "N/A" if last is None else f"{last:g}{unit}")
        categories = Table.grid(padding=(0, 2))
        for name, value in sorted(allocator["categories"].items(), key=lambda item: -item[1])[:8]:
            share = value / memory["used_bytes"] * 100 if memory["used_bytes"] else 0
            categories.add_row(name, bar(share, 20), size(value))
        largest = Table.grid(padding=(0, 2))
        for allocation in allocator["largest_allocations"][:5]:
            largest.add_row(size(allocation.get("bytes")), str(allocation.get("category", "")),
                            f"pid {allocation.get('pid', '-')}")
        detail.update(Group(
            key_values([
                ("Device", f"GPU {device['index']} {device['name']} ({device['profile_id']})"),
                ("UUID", device["uuid"]),
                ("Memory", f"{size(memory['used_bytes'])} used, peak {size(memory['peak_bytes'])}, "
                           f"tracked {size(memory['tracked_bytes'])}, reserved {size(memory['reserved_bytes'])}"),
                ("Allocator", f"{allocator['allocation_count']} allocations in {allocator['segment_count']} segments"),
                ("Sensors", TELEMETRY_LABELS["reference"] if device["telemetry"] else "N/A for this profile"),
            ]),
            Text("Last 60 seconds", style="bold"), trends,
            Text("Memory by category", style="bold"), categories,
            Text("Largest allocations", style="bold"), largest,
        ))

    def show_machine(self, status: dict[str, Any]) -> None:
        config, daemon, paths = status["config"], status["daemon"], status["paths"]
        if not self.form_loaded:
            self.load_form(config)
        home = str(Path.home())

        def path(value: str) -> str:
            return "~" + value[len(home):] if value.startswith(home + "/") else value

        self.query_one("#machine-info", Static).update(Group(
            Text("Machine", style="bold"),
            key_values([
                ("Daemon", f"pid {daemon['pid']} on {daemon['host']}"),
                ("Version", f"fakegpu {daemon['version']}"),
                ("Started", f"{daemon['started_at']} (up {duration(daemon['uptime_seconds'])})"),
                ("Sensors", TELEMETRY_LABELS[config["telemetry"]]),
                ("Data", Text(path(paths["root"]), overflow="fold")),
                ("Joining", "fakegpu.init() joins; FAKEGPU_MACHINE=off opts out"),
            ]),
            Text(),
            Text("Other views", style="bold"),
            key_values([
                ("nvitop", "fakegpu nvitop"),
                ("nvidia-smi", Text(f"fakegpu nvidia-smi --state-dir {path(paths['state_dir'])}", overflow="fold")),
                ("Events", "fakegpu daemon events -f"),
            ]),
        ))

    def load_form(self, config: dict[str, Any]) -> None:
        self.query_one("#cfg-devices", Input).value = config["devices"]
        self.query_one("#cfg-telemetry", Select).value = config["telemetry"]
        self.query_one("#cfg-background", Switch).value = config["background"]
        self.query_one("#cfg-warning", Input).value = f"{config['memory_warning_percent']:g}"
        self.query_one("#cfg-stale", Input).value = f"{config['stale_after_seconds']:g}"
        self.form_loaded = True

    def show_profiles(self, text: str = "") -> None:
        needle = text.strip().lower()
        sync_rows(self.query_one("#profiles-table", DataTable), [
            (p.id, (
                p.id, p.name, p.architecture, p.segment, size(p.memory_bytes), p.compute_capability_text,
                str(p.sm_count or "-"), f"{p.max_power_limit_mw / 1000:.0f} W" if p.max_power_limit_mw else "-",
            ))
            for p in self.profiles
            if not needle or needle in f"{p.id} {p.name} {p.architecture} {p.segment}".lower()
        ])

    # ------------------------------------------------------------ events

    def add_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        del self.events[:-EVENT_BUFFER]
        self.last_seq = event["seq"]
        line = event_line(event)
        self.query_one("#overview-events", RichLog).write(line)
        if self.event_visible(event):
            self.query_one("#events-log", RichLog).write(line)
        if self.backlog_loaded and (event["level"] != "info" or event["kind"].startswith("process.")):
            severity = {"warning": "warning", "error": "error"}.get(event["level"], "information")
            self.notify(event["message"], severity=severity, timeout=6)

    def event_visible(self, event: dict[str, Any]) -> bool:
        order = {"info": 0, "warning": 1, "error": 2}
        level = self.query_one("#event-level", Select).value
        needle = self.query_one("#event-filter", Input).value.strip().lower()
        return order.get(event["level"], 0) >= order[level] and needle in event["message"].lower()

    @on(Input.Changed, "#event-filter")
    @on(Select.Changed, "#event-level")
    def filter_events(self) -> None:
        log = self.query_one("#events-log", RichLog)
        log.clear()
        for event in self.events:
            if self.event_visible(event):
                log.write(event_line(event))

    # ------------------------------------------------------- machine form

    @on(Input.Changed, "#cfg-devices")
    def check_layout(self, event: Input.Changed) -> None:
        check = self.query_one("#cfg-devices-check", Static)
        try:
            profiles = parse_devices(event.value)
        except ValueError as exc:
            check.update(Text(str(exc), style="red"))
            return
        catalog = {p.id: p for p in self.profiles}
        total = sum(catalog[p].memory_bytes for p in profiles)
        check.update(Text(f"{len(profiles)} device(s), {size(total)} in total", style="green"))

    @on(Button.Pressed, "#cfg-apply")
    def apply_config(self) -> None:
        try:
            changes = {
                "devices": self.query_one("#cfg-devices", Input).value.strip(),
                "telemetry": self.query_one("#cfg-telemetry", Select).value,
                "background": self.query_one("#cfg-background", Switch).value,
                "memory_warning_percent": float(self.query_one("#cfg-warning", Input).value),
                "stale_after_seconds": float(self.query_one("#cfg-stale", Input).value),
            }
            config = self.client.call("config.update", changes=changes)
        except (MachineError, ValueError) as exc:
            self.notify(f"Not saved: {exc}", severity="error")
            return
        self.load_form(config)
        self.notify(f"Saved. New processes use {config['devices']}; running ones keep their devices.")
        self.refresh_machine()

    @on(Button.Pressed, "#cfg-reload")
    def reload_config(self) -> None:
        if self.machine:
            self.load_form(self.machine["config"])

    @on(Button.Pressed, "#stop-machine")
    def stop_button(self) -> None:
        self.action_stop_machine()

    @on(Input.Changed, "#profile-filter")
    def filter_profiles(self, event: Input.Changed) -> None:
        self.show_profiles(event.value)

    @on(DataTable.RowSelected, "#profiles-table")
    def select_profile(self) -> None:
        self.action_add_profile()

    # ----------------------------------------------------------- actions

    def action_tab(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab

    def action_detach(self) -> None:
        self.exit("detach")

    def action_stop_machine(self) -> None:
        def answer(confirmed: bool | None) -> None:
            if confirmed:
                self.exit("stop")

        self.push_screen(Confirm("Stop the machine and quit? Running workloads keep running."), answer)

    def action_toggle_background(self) -> None:
        if self.machine is None:
            return
        enabled = not self.machine["config"]["background"]
        try:
            self.client.call("config.update", changes={"background": enabled})
        except MachineError as exc:
            self.notify(str(exc), severity="error")
            return
        self.form_loaded = False
        self.notify(
            "Background on: the machine keeps running after you quit."
            if enabled
            else "Background off: quitting the console also stops the machine."
        )
        self.refresh_machine()

    def action_signal(self, name: str) -> None:
        tab = self.query_one(TabbedContent).active
        table_id = {"overview": "#overview-processes", "processes": "#processes-table"}.get(tab)
        pid = selected_key(self.query_one(table_id, DataTable)) if table_id else None
        if pid is None or self.machine is None:
            self.notify("Select a process on the Overview or Processes tab first.", severity="warning")
            return
        process = next(p for p in self.machine["processes"] if str(p["pid"]) == pid)

        def answer(confirmed: bool | None) -> None:
            if not confirmed:
                return
            try:
                self.client.call("process.signal", pid=int(pid), signal=name, source="console")
            except MachineError as exc:
                self.notify(str(exc), severity="error")

        self.push_screen(Confirm(f"Send {name} to {process['name']} (pid {pid})?"), answer)

    def action_add_profile(self) -> None:
        if self.query_one(TabbedContent).active != "profiles":
            return
        profile = selected_key(self.query_one("#profiles-table", DataTable))
        if profile is None:
            return
        field = self.query_one("#cfg-devices", Input)
        groups = [item.strip() for item in field.value.split(",") if item.strip()]
        name, _, count = groups[-1].partition(":") if groups else ("", "", "")
        if name == profile:
            groups[-1] = f"{profile}:{int(count or 1) + 1}"
        else:
            groups.append(f"{profile}:1")
        field.value = ",".join(groups)
        self.action_tab("machine")
        self.notify(f"Layout is now {field.value}. Press Apply to save it.")

    def action_start_machine(self) -> None:
        if self.machine is not None:
            return
        from ..machine.daemon import start_daemon

        try:
            start_daemon(self.client.paths)
        except MachineError as exc:
            self.notify(str(exc), severity="error")
            return
        self.notify("Machine started.")
        self.refresh_machine()
