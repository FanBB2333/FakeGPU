"""Terminal console for the background FakeGPU machine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
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
from ..machine.config import layout_text, parse_devices
from ..link_traffic import copies_summary, format_rate, throughput_summary
from ..platforms import AUTO, _pcie_generations, load_platforms, resolve_platform
from ..machine.service import EVENT_BUFFER
from ..profile_catalog import get_profile, load_profiles
from .widgets import (
    ALERT,
    COLORS,
    NAME,
    OK,
    THEME,
    WARNING,
    Chart,
    GpuCard,
    StatTile,
    bar,
    chart_grid,
    device_key,
    duration,
    event_line,
    key_values,
    latest,
    memory_pair,
    nice_ceiling,
    number,
    selected_key,
    size,
    sparkline,
    sync_rows,
)

TABS = ("overview", "processes", "devices", "events", "machine", "profiles")
TELEMETRY_LABELS = {"reference": "modeled (A100 reference, default curve)", "none": "none (sensors N/A)"}
SENSOR_SOURCES = {
    "a100_reference_model": "A100 reference readings, follow activity (modeled)",
    "default_model": "default curve: idle 30°C and 1 W, follow activity (modeled)",
}
# Actions that only apply on some tabs; elsewhere their keys are inactive and
# left out of the footer.
TAB_ACTIONS = {
    "signal": ("overview", "processes"),
    "toggle_gpu_view": ("overview",),
    "change_gpu": ("devices",),
    "replace_layout": ("devices",),
    "add_gpu": ("devices",),
    "remove_gpu": ("devices",),
    "add_profile": ("profiles",),
}


def slot_text(slot: dict[str, Any] | None) -> str:
    if not slot:
        return "not in the machine layout"
    parts = [f"{slot['slot_type']} slot {slot['slot']}", slot["pcie"] or "no PCIe link"]
    if slot["host_bandwidth_gbps"]:
        parts.append(f"host {slot['host_link']} {slot['host_bandwidth_gbps']:g} GB/s per direction")
    if slot["nvlink_links"]:
        parts.append(f"{slot['nvlink_links']} NVLinks, {slot['nvlink_bandwidth_gbps']:g} GB/s")
    parts.append(f"NUMA {slot['numa_node']}")
    return ", ".join(parts)


def device_links(slot: dict[str, Any] | None) -> list[tuple[str, str, float | None]]:
    """Links of a GPU as (kind, label, GB/s per direction) for its trend rows."""
    if not slot:
        return [("pcie", "PCIe", None)]
    links: list[tuple[str, str, float | None]] = []
    if slot["pcie"]:
        links.append(("pcie", "PCIe", slot["host_bandwidth_gbps"] if slot["host_link"] == "pcie" else None))
    if slot["host_link"] == "c2c":
        links.append(("c2c", "C2C", slot["host_bandwidth_gbps"]))
    if slot["nvlink_links"]:
        links.append(("nvlink", "NVLink", slot["nvlink_bandwidth_gbps"]))
    return links


def link_io_text(transfer: dict[str, Any] | None) -> str:
    if not transfer:
        return "N/A"
    total = sum(
        entry["tx_bytes_per_second"] + entry["rx_bytes_per_second"] for entry in transfer["links"].values()
    )
    return format_rate(total)


def short(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def sources_text(devices: Sequence[dict[str, Any]]) -> str:
    """Where the charted readings come from: memory is tracked, sensors are modeled."""
    sensors = "sensors modeled" if any(device["telemetry"] for device in devices) else "sensors N/A"
    return f"memory tracked, {sensors}"


def window_text(times: Sequence[float]) -> str:
    return f"last {times[-1] - times[0]:.0f} s" if len(times) > 1 else "just started"


def device_charts(device: dict[str, Any], height: int = 4) -> list[Chart]:
    """Recent memory, sensor and link readings of one GPU."""
    history, memory = device["history"], device["memory"]
    power = history["power_w"]
    charts = [
        Chart(
            "Memory", history["memory_percent"], maximum=100, unit="%", color=COLORS["memory"],
            current=f"{size(memory['used_bytes'])} / {size(memory['total_bytes'])} ({memory['percent']:.1f}%)",
            height=height,
        ),
        Chart("Utilization", history["gpu"], maximum=100, unit="%", color=COLORS["gpu"], height=height),
        Chart(
            "Temperature", history["temperature"], maximum=100, unit="°C",
            color=COLORS["temperature"], height=height,
        ),
        Chart(
            "Power", power, unit="W", color=COLORS["power"], height=height,
            maximum=device["power_limit_w"] or nice_ceiling(max([v for v in power if v is not None], default=1)),
        ),
    ]
    for link, label, bandwidth in device_links(device["slot"]):
        values = history[f"{link}_gbps"]
        peak = max([v for v in values if v is not None], default=1.0)
        charts.append(Chart(
            f"{label} TX+RX", values, unit=" GB/s", color=COLORS[link], height=height,
            # TX plus RX against both directions of the link
            maximum=2 * bandwidth if bandwidth else nice_ceiling(peak),
        ))
    return charts


def machine_charts(devices: list[dict[str, Any]], height: int = 3) -> list[Chart]:
    """Readings of every GPU in the layout combined into one chart each."""
    layout = [device for device in devices if device["in_layout"]]
    length = max((len(device["history"]["time"]) for device in layout), default=0)

    def aligned(key: str) -> list[list[float | None]]:
        # GPUs added later have shorter histories; line them up on the newest sample.
        return [[None] * (length - len(values)) + values for values in (d["history"][key] for d in layout)]

    def samples(key: str) -> list[list[float]]:
        return [[value for value in column if value is not None] for column in zip(*aligned(key))]

    totals = [device["memory"]["total_bytes"] for device in layout]
    total = sum(totals)
    memory = [
        sum((value or 0) * capacity for value, capacity in zip(column, totals)) / total if total else None
        for column in zip(*aligned("memory_percent"))
    ]
    utilization = [sum(column) / len(column) if column else None for column in samples("gpu")]
    power = [sum(column) if column else None for column in samples("power_w")]
    temperature = [max(column) if column else None for column in samples("temperature")]
    links = [samples(f"{link}_gbps") for link in ("pcie", "c2c", "nvlink")]
    traffic = [sum(sum(column) for column in columns) if any(columns) else None for columns in zip(*links)]

    used = sum(device["memory"]["used_bytes"] for device in layout)
    sensed = [device for device in layout if device["telemetry"]]
    no_sensors = None if sensed else "N/A"
    readings = [device["telemetry"] for device in sensed]
    limit = sum(device["power_limit_w"] or 0 for device in sensed)
    peak_power = max([value for value in power if value is not None], default=0.0)
    return [
        Chart(
            "Memory", memory, maximum=100, unit="%", color=COLORS["memory"], height=height,
            current=f"{memory_pair(used, total)} ({used / total * 100 if total else 0:.0f}%)",
        ),
        Chart(
            "Utilization", utilization, maximum=100, unit="%", color=COLORS["gpu"], height=height,
            current=no_sensors or (
                f"{sum(r['gpu_percent'] for r in readings) / len(readings):.0f}% avg, "
                f"{max(r['gpu_percent'] for r in readings)}% max"
            ),
        ),
        Chart(
            "Power", power, unit="W", color=COLORS["power"], height=height,
            # Profiles without a catalog power cap still draw power.
            maximum=limit if peak_power <= limit else nice_ceiling(peak_power),
            current=no_sensors or f"{sum(r['power_w'] for r in readings):.0f} / {limit:.0f} W",
        ),
        Chart(
            "Temperature", temperature, maximum=100, unit="°C", color=COLORS["temperature"], height=height,
            current=no_sensors or f"{max(r['temperature_c'] for r in readings)}°C max",
        ),
        Chart(
            "Link I/O", traffic, unit=" GB/s", color=COLORS["link"], height=height,
            maximum=nice_ceiling(max([v for v in traffic if v is not None], default=1.0)),
            current=f"{number(latest(traffic) or 0)} GB/s",
        ),
    ]


@dataclass(frozen=True)
class LayoutPlan:
    """A device list and the machine settings that can hold it."""

    profiles: tuple[str, ...]
    changes: dict[str, Any]
    platform: str | None = None
    notes: tuple[str, ...] = ()
    error: str | None = None


def plan_layout(config: Mapping[str, Any], profiles: Sequence[str]) -> LayoutPlan:
    """Settings for a new device list, keeping the platform when it still fits.

    Tries the configured platform and NVLink bridges first, then ``auto``,
    then ``auto`` without bridges, then the generic host; ``notes`` says what
    had to change and why.
    """
    if not profiles:
        return LayoutPlan((), {}, error="the machine needs at least one device")
    platform, bridges = config["platform"], config["nvlink_bridges"]
    reason = None
    for candidate, links in dict.fromkeys([(platform, bridges), (AUTO, bridges), (AUTO, ""), ("generic", "")]):
        try:
            resolved = resolve_platform(
                profiles, candidate, pcie_generation=int(config["pcie_generation"]), nvlink_bridges=links
            )
        except ValueError as exc:
            reason = reason or str(exc)
            continue
        changes: dict[str, Any] = {"devices": layout_text(profiles)}
        notes = []
        if candidate != platform:
            changes["platform"] = candidate
            notes.append(f"platform {platform} → {candidate}, because {reason}")
        if links != bridges:
            changes["nvlink_bridges"] = links
            notes.append(f"NVLink bridges {bridges} removed")
        return LayoutPlan(tuple(profiles), changes, resolved.platform.name, tuple(notes))
    return LayoutPlan(tuple(profiles), {}, error=reason)


def plan_view(plan: LayoutPlan, profile_id: str) -> Group:
    profile = get_profile(profile_id)
    power = f" · {profile.max_power_limit_mw / 1000:.0f} W" if profile.max_power_limit_mw else ""
    heading = Text.assemble(
        (profile.id, NAME), "  ",
        f"{profile.torch_name or profile.name} · {profile.architecture.capitalize()} · "
        f"{size(profile.memory_bytes)} · CC {profile.compute_capability_text}{power}",
    )
    if plan.error:
        return Group(heading, Text(f"Cannot build this machine: {plan.error}", style=ALERT))
    total = sum(get_profile(item).memory_bytes for item in plan.profiles)
    rows: list[tuple[str, Any]] = [
        ("Layout", f"{plan.changes['devices']} ({len(plan.profiles)} GPUs, {size(total)})"),
        ("Platform", plan.platform),
    ]
    rows += [("Also", Text(note, style=WARNING, overflow="fold")) for note in plan.notes]
    return Group(heading, key_values(rows))


def platform_view(platform: dict[str, Any]) -> list[Any]:
    """Slot map and GPU link matrix of the machine's platform."""

    slots = Table(box=None, padding=(0, 1), header_style="dim")
    for label in ("GPU", "Slot", "Profile", "PCIe", "Host link", "NUMA", "NVLinks"):
        slots.add_column(label, no_wrap=True)
    for device in platform["devices"]:
        host = device["host_bandwidth_gbps"]
        nvlinks = device["nvlink_links"]
        slots.add_row(
            str(device["index"]),
            f"{device['slot']} {device['slot_type']}",
            device["profile_id"],
            device["pcie"] or "-",
            f"{device['host_link']} {host:g} GB/s" if host else device["host_link"],
            str(device["numa_node"]),
            f"{nvlinks} ({device['nvlink_bandwidth_gbps']:g} GB/s)" if nvlinks else "-",
        )
    matrix = Table(box=None, padding=(0, 1), header_style="dim")
    count = len(platform["matrix"])
    matrix.add_column("")
    for column in range(count):
        matrix.add_column(f"GPU{column}", no_wrap=True)
    for row, cells in enumerate(platform["matrix"]):
        matrix.add_row(f"GPU{row}", *[Text(cell, style=OK if cell.startswith("NV") else "") for cell in cells])
    parts: list[Any] = [
        Text("Platform", style="bold"),
        key_values([
            ("Platform", f"{platform['name']} ({platform['id']})"),
            ("Slots", platform["slot_layout"]),
            ("Wiring", Text(platform["summary"], overflow="fold")),
        ]),
        slots,
    ]
    if count > 1:
        parts.extend([Text("GPU links (as nvidia-smi topo -m)", style="dim"), matrix])
    parts.extend(Text(note, style=WARNING, overflow="fold") for note in platform["notes"])
    return parts


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


class ProfilePicker(ModalScreen["tuple[str, int] | None"]):
    """Choose a catalog profile, and a count when ``count`` is given.

    ``preview(profile, count)`` plans the machine the choice would make; the
    picker shows that plan and only accepts a choice that can be built.
    """

    BINDINGS = [
        Binding("escape", "dismiss(None)", "Cancel"),
        Binding("down", "move(1)", show=False),
        Binding("up", "move(-1)", show=False),
    ]

    def __init__(
        self,
        heading: str,
        profiles: Sequence[Any],
        current: str | None,
        preview: Callable[[str, int], LayoutPlan],
        count: int | None = None,
    ):
        super().__init__()
        self.heading = heading
        self.profiles = profiles
        self.current = current
        self.preview = preview
        self.count = count

    def compose(self) -> ComposeResult:
        with Vertical(id="picker"):
            yield Label(self.heading, id="picker-title")
            with Horizontal(id="picker-search"):
                yield Input(placeholder="Filter by id, name, architecture or segment", id="picker-filter")
                if self.count is not None:
                    yield Label("Count", id="picker-count-label")
                    yield Input(str(self.count), type="integer", id="picker-count")
            yield DataTable(id="picker-table", cursor_type="row")
            yield Static(id="picker-preview")
            with Horizontal(id="picker-buttons"):
                yield Label("↑↓ choose · Enter apply · Esc cancel", id="picker-help")
                yield Button("Apply", variant="primary", id="picker-apply")
                yield Button("Cancel", id="picker-cancel")

    def on_mount(self) -> None:
        table = self.query_one("#picker-table", DataTable)
        table.add_columns("ID", "Name", "Architecture", "Form", "Segment", "Memory", "CC", "Power")
        self.show_profiles("")
        keys = [row.value for row in table.rows]
        if self.current in keys:
            table.move_cursor(row=keys.index(self.current))
        self.query_one("#picker-filter", Input).focus()

    def show_profiles(self, text: str) -> None:
        needle = text.strip().lower()
        sync_rows(self.query_one("#picker-table", DataTable), [
            (p.id, (
                Text(p.id, style=NAME if p.id == self.current else ""), short(p.torch_name or p.name, 34),
                p.architecture, p.form_factor or "-", p.segment, size(p.memory_bytes), p.compute_capability_text,
                f"{p.max_power_limit_mw / 1000:.0f} W" if p.max_power_limit_mw else "-",
            ))
            for p in self.profiles
            if not needle or needle in f"{p.id} {p.name} {p.architecture} {p.segment}".lower()
        ])
        self.show_preview()

    def choice(self) -> tuple[str, int] | None:
        profile = selected_key(self.query_one("#picker-table", DataTable))
        if profile is None:
            return None
        if self.count is None:
            return profile, 1
        try:
            return profile, int(self.query_one("#picker-count", Input).value)
        except ValueError:
            return profile, 0

    @on(DataTable.RowHighlighted)
    @on(Input.Changed, "#picker-count")
    def show_preview(self) -> None:
        preview = self.query_one("#picker-preview", Static)
        apply = self.query_one("#picker-apply", Button)
        choice = self.choice()
        apply.disabled = True
        if choice is None:
            preview.update(Text("No profile matches the filter.", style="dim"))
        elif choice[1] < 1:
            preview.update(Text("Count must be at least 1.", style=ALERT))
        else:
            plan = self.preview(*choice)
            preview.update(plan_view(plan, choice[0]))
            apply.disabled = plan.error is not None

    @on(Input.Changed, "#picker-filter")
    def filter_profiles(self, event: Input.Changed) -> None:
        self.show_profiles(event.value)

    @on(DataTable.RowSelected)
    @on(Input.Submitted)
    @on(Button.Pressed, "#picker-apply")
    def submit(self) -> None:
        choice = self.choice()
        if choice is not None and choice[1] >= 1 and self.preview(*choice).error is None:
            self.dismiss(choice)

    @on(Button.Pressed, "#picker-cancel")
    def cancel(self) -> None:
        self.dismiss(None)

    def action_move(self, step: int) -> None:
        table = self.query_one("#picker-table", DataTable)
        table.action_cursor_down() if step > 0 else table.action_cursor_up()


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
    #machine-trends {
        height: auto; margin: 0 1; padding: 0 1; border: round $panel-lighten-2; border-title-color: $accent;
    }
    #gpus { grid-size: 4; grid-gutter: 0 1; height: auto; max-height: 60%; overflow-y: auto; margin: 0 1 1 1; }
    #gpu-list { height: auto; max-height: 60%; margin: 0 1 1 1; display: none; }
    GpuCard { height: 9; border: round $panel-lighten-2; padding: 0 1; }
    GpuCard:hover { background: $boost; }
    GpuCard.busy { border: round $success; }
    GpuCard.warn { border: round $warning; }
    GpuCard.alert { border: round $error; }
    #lower { height: 1fr; margin: 0 1; }
    #overview-processes { width: 3fr; }
    #overview-events { width: 2fr; border-left: tall $panel; }
    #processes-table { height: 1fr; min-height: 5; }
    #ended-table { height: auto; max-height: 7; }
    .detail { height: auto; max-height: 50%; border-top: tall $panel; padding: 0 1; }
    #devices-toolbar { height: 1; margin: 1 1 0 1; }
    #devices-toolbar Button { margin-right: 2; }
    #devices-layout { width: 1fr; color: $text-muted; text-wrap: nowrap; text-overflow: ellipsis; }
    #devices-table { height: auto; max-height: 40%; min-height: 3; margin-top: 1; }
    #device-detail-scroll { height: 1fr; border-top: tall $panel; padding: 0 1; }
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
    #dialog { width: 72; height: auto; border: thick $warning; background: $surface; padding: 1 2; }
    #dialog Horizontal { height: auto; align: center middle; margin-top: 1; }
    #dialog Button { margin: 0 1; }
    ProfilePicker { align: center middle; }
    #picker { width: 140; max-width: 96%; height: 90%; border: thick $primary; background: $surface; padding: 0 1; }
    #picker-title { text-style: bold; color: $accent; margin-top: 1; }
    #picker-search { height: 3; }
    #picker-filter { width: 1fr; }
    #picker-count-label { padding: 1 1 0 2; }
    #picker-count { width: 12; }
    #picker-table { height: 1fr; }
    #picker-preview { height: auto; max-height: 9; padding-top: 1; }
    #picker-buttons { height: 3; align: right middle; }
    #picker-help { width: 1fr; color: $text-muted; }
    #picker-buttons Button { margin-left: 2; }
    """
    BINDINGS = [
        Binding("q", "detach", "Quit"),
        Binding("Q", "stop_machine", "Stop machine"),
        Binding("b", "toggle_background", "Background"),
        Binding("k", "signal('SIGTERM')", "Terminate"),
        Binding("K", "signal('SIGKILL')", "Kill"),
        Binding("i", "signal('SIGINT')", "Interrupt", show=False),
        Binding("v", "toggle_gpu_view", "Cards/list"),
        Binding("e", "change_gpu", "Change GPU"),
        Binding("E", "replace_layout", "Replace all"),
        Binding("a", "add_gpu", "Add GPUs"),
        Binding("x,delete", "remove_gpu", "Remove GPU"),
        Binding("a", "add_profile", "Add to layout"),
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
        self.gpu_view = "cards"
        self.select_device: str | None = None
        self.profiles = sorted(load_profiles().values(), key=lambda p: (p.architecture, p.id))

    # ------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="stats"):
            for name in ("devices", "memory", "power", "processes", "alerts"):
                yield StatTile(id=f"stat-{name}")
        with TabbedContent(initial="overview"):
            with TabPane("1 Overview", id="overview"):
                yield Static(id="machine-trends")
                yield Grid(id="gpus")
                yield DataTable(id="gpu-list", cursor_type="row")
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
                with Horizontal(id="devices-toolbar"):
                    yield Button("e Change GPU", id="gpu-change", compact=True, action="app.change_gpu")
                    yield Button("E Replace all", id="gpu-replace", compact=True, action="app.replace_layout")
                    yield Button("a Add GPUs", id="gpu-add", compact=True, action="app.add_gpu")
                    yield Button("x Remove GPU", id="gpu-remove", compact=True, action="app.remove_gpu")
                    yield Static(id="devices-layout")
                yield DataTable(id="devices-table", cursor_type="row")
                with VerticalScroll(id="device-detail-scroll"):
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
                        yield Input(placeholder="a100:8, h100:4 or l40s:2,l4:2", id="cfg-devices")
                        yield Static(id="cfg-devices-check")
                        with Horizontal(classes="row"):
                            yield Label("Platform")
                            yield Select(
                                [("Auto (by form factor)", AUTO)]
                                + [(f"{platform.name} ({platform.id})", platform.id)
                                   for platform in load_platforms().values()],
                                allow_blank=False,
                                id="cfg-platform",
                            )
                        with Horizontal(classes="row"):
                            yield Label("PCIe slot generation")
                            yield Select(
                                [("Platform default", 0)] + [(f"Gen{gen}", gen) for gen in _pcie_generations()],
                                allow_blank=False,
                                id="cfg-pcie",
                            )
                        with Horizontal(classes="row"):
                            yield Label("NVLink bridges")
                            yield Input(placeholder="0-1,2-3 (PCIe cards)", id="cfg-bridges")
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
                            yield Button("Stop machine", variant="error", id="stop-machine", action="app.stop_machine")
                    yield Static(id="machine-info")
            with TabPane("6 Profiles", id="profiles"):
                yield Input(placeholder="Filter by id, name, architecture or segment; press a to add", id="profile-filter")
                yield DataTable(id="profiles-table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.register_theme(THEME)
        self.theme = THEME.name
        columns = {
            "gpu-list": ("GPU", "Model", "Memory", "Utilization", "Power", "Temp", "Procs", "Utilization trend"),
            "overview-processes": ("PID", "Name", "Status", "GPUs", "Memory", "GPU %", "Uptime"),
            "processes-table": (
                "PID", "User", "Name", "Status", "GPUs", "Memory", "Peak", "GPU %", "Stage", "Backend",
                "Uptime", "Heartbeat", "Command",
            ),
            "ended-table": ("PID", "Name", "Result", "Runtime", "Peak", "Ended (UTC)"),
            "devices-table": (
                "GPU", "Profile", "Name", "Arch", "CC", "Used", "Total", "Mem %", "Procs", "Util",
                "Temp", "Power", "SM clock", "Link I/O",
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

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        tabs = TAB_ACTIONS.get(action)
        if tabs is None:
            return True
        try:
            return self.query_one(TabbedContent).active in tabs
        except NoMatches:
            return False

    @on(TabbedContent.TabActivated)
    def tab_changed(self) -> None:
        self.refresh_bindings()

    # ----------------------------------------------------------- refresh

    def refresh_machine(self) -> None:
        # Shutdown removes the widgets before it stops the refresh timer.
        if not self.is_running:
            return
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
        trends = self.query_one("#machine-trends", Static)
        trends.border_title = None
        trends.update("")
        self.query_one("#devices-layout", Static).update("")
        for table_id in ("#gpu-list", "#overview-processes", "#processes-table", "#devices-table"):
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
        self.query_one("#stat-memory", StatTile).show(
            f"{size(used)} / {size(total)}", f"memory, {percent:.1f}% used", f"bold {COLORS['memory']}"
        )
        self.query_one("#stat-processes", StatTile).show(
            str(summary["process_count"]), f"processes, {summary['stalled_count']} stalled"
        )
        self.query_one("#stat-alerts", StatTile).show(
            str(summary["alerts"]), "memory alerts", f"bold {ALERT}" if summary["alerts"] else f"bold {OK}"
        )
        sensed = [d for d in status["devices"] if d["telemetry"]]
        if sensed:
            draw = sum(d["telemetry"]["power_w"] for d in sensed)
            limit = sum(d["power_limit_w"] or 0 for d in sensed)
            self.query_one("#stat-power", StatTile).show(
                f"{draw:.0f} / {limit:.0f} W", f"power, modeled on {len(sensed)} GPU(s)", f"bold {COLORS['power']}"
            )
        else:
            self.query_one("#stat-power", StatTile).show("N/A", "power (no sensor model)", "dim")
        self.show_trends(status["devices"])
        self.show_cards(status["devices"], config["memory_warning_percent"])
        self.show_gpu_list(status["devices"])
        self.show_processes(status["processes"], status["ended"])
        self.show_devices(status["devices"])
        self.show_machine(status)
        platform = status["platform"]["name"]
        held = "" if all(d["in_layout"] for d in status["devices"]) else " · * held by a running process, not in the layout"
        self.query_one("#devices-layout", Static).update(
            f"Layout {config['devices']} on {platform} · Enter or click a selected row to change that GPU{held}"
        )

    def show_trends(self, devices: list[dict]) -> None:
        panel = self.query_one("#machine-trends", Static)
        layout = [d for d in devices if d["in_layout"]]
        times = next((d["history"]["time"] for d in layout), [])
        panel.border_title = f"All {len(layout)} GPUs · {window_text(times)} · {sources_text(layout)}"
        panel.update(chart_grid(machine_charts(devices), max(1, (self.size.width - 6) // 32)))

    def show_cards(self, devices: list[dict], warning: float) -> None:
        while len(self.cards) > len(devices):
            self.cards.pop().remove()
        if len(self.cards) < len(devices):
            new = [GpuCard() for _ in range(len(devices) - len(self.cards))]
            self.query_one("#gpus", Grid).mount_all(new)
            self.cards.extend(new)
        for card, device in zip(self.cards, devices):
            card.show(device, warning)

    def show_gpu_list(self, devices: list[dict]) -> None:
        rows = []
        for d in devices:
            memory, sensors = d["memory"], d["telemetry"]
            limit = d["power_limit_w"]
            if sensors:
                power_percent = sensors["power_w"] / limit * 100 if limit else None
                utilization = Text.assemble(
                    bar(sensors["gpu_percent"], 10, COLORS["gpu"]), f" {sensors['gpu_percent']:>3}%"
                )
                power = Text.assemble(bar(power_percent, 10, COLORS["power"]), f" {sensors['power_w']:>4.0f} W")
                temperature = Text(f"{sensors['temperature_c']}°C", style=COLORS["temperature"])
            else:
                utilization = power = temperature = Text("N/A", style="dim")
            rows.append((device_key(d), (
                str(d["index"]), d["name"],
                Text.assemble(
                    bar(memory["percent"], 10, COLORS["memory"]),
                    f" {memory_pair(memory['used_bytes'], memory['total_bytes'])}",
                ),
                utilization, power, temperature, str(d["process_count"]),
                sparkline(d["history"]["gpu"], 30, color=COLORS["gpu"]),
            )))
        sync_rows(self.query_one("#gpu-list", DataTable), rows)

    def show_processes(self, processes: list[dict], ended: list[dict]) -> None:
        def gpus(process: dict) -> str:
            return ",".join(str(row["index"]) for row in process["devices"] if row["used_bytes"]) or "-"

        def gpu_percent(process: dict) -> str:
            return "N/A" if process["gpu_percent"] is None else f"{process['gpu_percent']}%"

        status_style = {"running": OK, "stalled": WARNING}
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
        result_style = {"exited": OK, "terminated": WARNING, "vanished": ALERT}
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
            # A GPU outside the layout is one a running process still holds
            # from the layout it started with.
            index = str(d["index"]) if d["in_layout"] else Text(f"{d['index']}*", style=WARNING)
            rows.append((device_key(d), (
                index, d["profile_id"], d["name"], d["architecture"], d["compute_capability"],
                size(memory["used_bytes"]), size(memory["total_bytes"]),
                Text(f"{memory['percent']:.1f}%", style=COLORS["memory"]),
                str(d["process_count"]),
                Text(f"{sensors['gpu_percent']}%", style=COLORS["gpu"]) if sensors else "N/A",
                Text(f"{sensors['temperature_c']}°C", style=COLORS["temperature"]) if sensors else "N/A",
                Text(f"{sensors['power_w']:.0f} W", style=COLORS["power"]) if sensors else "N/A",
                f"{sensors['sm_clock_mhz']} MHz" if sensors else "N/A",
                link_io_text(d["transfer"]),
            )))
        table = self.query_one("#devices-table", DataTable)
        sync_rows(table, rows)
        keys = [key for key, _ in rows]
        if self.select_device in keys:
            table.move_cursor(row=keys.index(self.select_device))
            self.select_device = None
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

    def selected_device(self) -> dict[str, Any] | None:
        key = selected_key(self.query_one("#devices-table", DataTable))
        return next((d for d in (self.machine or {}).get("devices", []) if device_key(d) == key), None)

    @on(DataTable.RowHighlighted, "#devices-table")
    def show_device_detail(self) -> None:
        detail = self.query_one("#device-detail", Static)
        device = self.selected_device()
        if device is None:
            detail.update("")
            return
        memory, allocator = device["memory"], device["allocator"]
        transfer = device["transfer"]
        links = [link for link, _label, _bandwidth in device_links(device["slot"])]
        categories = Table.grid(padding=(0, 2))
        for name, value in sorted(allocator["categories"].items(), key=lambda item: -item[1])[:8]:
            share = value / memory["used_bytes"] * 100 if memory["used_bytes"] else 0
            categories.add_row(name, bar(share, 20, COLORS["memory"]), size(value))
        if not categories.row_count:
            categories.add_row(Text("No allocations.", style="dim"))
        largest = Table.grid(padding=(0, 2))
        for allocation in allocator["largest_allocations"][:5]:
            largest.add_row(size(allocation.get("bytes")), str(allocation.get("category", "")),
                            f"pid {allocation.get('pid', '-')}")
        if not largest.row_count:
            largest.add_row(Text("No allocations.", style="dim"))
        times = device["history"]["time"]
        detail.update(Group(
            key_values([
                ("Device", f"GPU {device['index']} {device['name']} ({device['profile_id']})"),
                ("UUID", device["uuid"]),
                ("Slot", slot_text(device["slot"])),
                ("Copies", copies_summary(transfer, size) + " (modeled)" if transfer else "none recorded"),
                ("Links", f"{throughput_summary(transfer, links)} (last {transfer['window_seconds']:g} s)"
                          if transfer else "N/A"),
                ("Memory", f"{size(memory['used_bytes'])} used, peak {size(memory['peak_bytes'])}, "
                           f"tracked {size(memory['tracked_bytes'])}, reserved {size(memory['reserved_bytes'])}"),
                ("Allocator", f"{allocator['allocation_count']} allocations in {allocator['segment_count']} segments"),
                ("Sensors", SENSOR_SOURCES.get(device["telemetry"]["source"], "modeled")
                            if device["telemetry"] else "N/A (sensor model off, or no activity published)"),
            ]),
            Text(),
            Text(f"Trends, {window_text(times)} · {sources_text([device])}", style="bold"),
            chart_grid(device_charts(device), 2 if self.size.width >= 100 else 1, gap=1),
            Text(),
            Text("Memory by category", style="bold"), categories,
            Text("Largest allocations", style="bold"), largest,
        ))

    def show_machine(self, status: dict[str, Any]) -> None:
        config, daemon, paths = status["config"], status["daemon"], status["paths"]
        platform = status["platform"]
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
            *platform_view(platform),
            Text(),
            Text("Other views", style="bold"),
            key_values([
                ("nvitop", "fakegpu nvitop"),
                ("nvidia-smi", "fakegpu nvidia-smi"),
                ("Events", "fakegpu daemon events -f"),
            ]),
        ))

    def load_form(self, config: dict[str, Any]) -> None:
        self.query_one("#cfg-devices", Input).value = config["devices"]
        self.query_one("#cfg-platform", Select).value = config["platform"].strip().lower()
        self.query_one("#cfg-pcie", Select).value = config["pcie_generation"]
        self.query_one("#cfg-bridges", Input).value = config["nvlink_bridges"]
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
    @on(Input.Changed, "#cfg-bridges")
    @on(Select.Changed, "#cfg-platform")
    @on(Select.Changed, "#cfg-pcie")
    def check_layout(self) -> None:
        check = self.query_one("#cfg-devices-check", Static)
        try:
            profiles = parse_devices(self.query_one("#cfg-devices", Input).value)
            resolved = resolve_platform(
                profiles,
                str(self.query_one("#cfg-platform", Select).value),
                pcie_generation=int(self.query_one("#cfg-pcie", Select).value),
                nvlink_bridges=self.query_one("#cfg-bridges", Input).value,
            )
        except ValueError as exc:
            check.update(Text(str(exc), style=ALERT))
            return
        catalog = {p.id: p for p in self.profiles}
        total = sum(catalog[p].memory_bytes for p in profiles)
        check.update(Text(
            f"{len(profiles)} device(s), {size(total)} in total, on {resolved.platform.name}",
            style=OK,
        ))

    @on(Button.Pressed, "#cfg-apply")
    def apply_config(self) -> None:
        try:
            changes = {
                "devices": self.query_one("#cfg-devices", Input).value.strip(),
                "platform": self.query_one("#cfg-platform", Select).value,
                "pcie_generation": self.query_one("#cfg-pcie", Select).value,
                "nvlink_bridges": self.query_one("#cfg-bridges", Input).value.strip(),
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

    @on(Input.Changed, "#profile-filter")
    def filter_profiles(self, event: Input.Changed) -> None:
        self.show_profiles(event.value)

    @on(DataTable.RowSelected, "#profiles-table")
    def select_profile(self) -> None:
        self.action_add_profile()

    # ---------------------------------------------------- device layout

    @on(GpuCard.Selected)
    def card_selected(self, event: GpuCard.Selected) -> None:
        self.focus_device(event.key)

    @on(DataTable.RowSelected, "#gpu-list")
    def list_selected(self, event: DataTable.RowSelected) -> None:
        self.focus_device(event.row_key.value)

    def focus_device(self, key: str) -> None:
        """Open a GPU on the Devices tab."""
        self.action_tab("devices")
        table = self.query_one("#devices-table", DataTable)
        keys = [row.value for row in table.rows]
        if key in keys:
            table.move_cursor(row=keys.index(key))
        table.focus()

    @on(DataTable.RowSelected, "#devices-table")
    def change_selected(self) -> None:
        self.action_change_gpu()

    def layout(self) -> list[str] | None:
        return parse_devices(self.machine["config"]["devices"]) if self.machine else None

    def layout_index(self) -> int | None:
        """Layout position of the selected Devices row."""
        key = selected_key(self.query_one("#devices-table", DataTable))
        return int(key[4:]) if key and key.startswith("gpu-") else None

    def pick_profile(
        self,
        heading: str,
        current: str,
        build: Callable[[str, int], list[str]],
        describe: Callable[[str, int], str],
        *,
        count: int | None = None,
        select: Callable[[str, int], str] | None = None,
    ) -> None:
        config = dict(self.machine["config"])

        def done(choice: tuple[str, int] | None) -> None:
            if choice is not None:
                self.apply_layout(
                    plan_layout(config, build(*choice)), describe(*choice), select(*choice) if select else None
                )

        self.push_screen(
            ProfilePicker(heading, self.profiles, current, lambda p, n: plan_layout(config, build(p, n)), count),
            done,
        )

    def apply_layout(self, plan: LayoutPlan, message: str, select: str | None = None) -> None:
        try:
            config = self.client.call("config.update", changes=plan.changes)
        except MachineError as exc:
            self.notify(f"Not saved: {exc}", severity="error")
            return
        self.form_loaded = False
        self.select_device = select
        also = f" Also: {'; '.join(plan.notes)}." if plan.notes else ""
        self.notify(
            f"{message}. Layout {config['devices']} on {plan.platform}.{also} "
            "New processes use it; running ones keep their devices.",
            timeout=8,
        )
        self.refresh_machine()

    def action_change_gpu(self) -> None:
        layout, index = self.layout(), self.layout_index()
        if layout is None:
            return
        if index is None:
            self.notify("Select a GPU of the machine layout first.", severity="warning")
            return
        self.pick_profile(
            f"GPU {index}: replace {layout[index]} with",
            layout[index],
            lambda profile, _count: layout[:index] + [profile] + layout[index + 1:],
            lambda profile, _count: f"GPU {index} is now {profile}",
        )

    def action_replace_layout(self) -> None:
        layout, index = self.layout(), self.layout_index()
        if layout is None:
            return
        self.pick_profile(
            f"Replace all {len(layout)} GPUs with",
            layout[index if index is not None else 0],
            lambda profile, count: [profile] * count,
            lambda profile, count: f"The machine now has {count}x {profile}",
            count=len(layout),
        )

    def action_add_gpu(self) -> None:
        layout, index = self.layout(), self.layout_index()
        if layout is None:
            return
        self.pick_profile(
            f"Add GPUs after GPU {len(layout) - 1}",
            layout[index if index is not None else -1],
            lambda profile, count: layout + [profile] * count,
            lambda profile, count: f"Added {count}x {profile} as GPU {len(layout)}"
                                   + (f"-{len(layout) + count - 1}" if count > 1 else ""),
            count=1,
            select=lambda _profile, _count: f"gpu-{len(layout)}",
        )

    def action_remove_gpu(self) -> None:
        layout, index = self.layout(), self.layout_index()
        if layout is None:
            return
        if index is None:
            self.notify("Select a GPU of the machine layout first.", severity="warning")
            return
        plan = plan_layout(self.machine["config"], layout[:index] + layout[index + 1:])
        if plan.error:
            self.notify(f"Cannot remove GPU {index}: {plan.error}", severity="error")
            return
        also = f" Also: {'; '.join(plan.notes)}." if plan.notes else ""

        def answer(confirmed: bool | None) -> None:
            if confirmed:
                self.apply_layout(plan, f"Removed GPU {index}", f"gpu-{min(index, len(plan.profiles) - 1)}")

        self.push_screen(Confirm(
            f"Remove GPU {index} ({layout[index]})? New processes will see {len(plan.profiles)} GPUs "
            f"on {plan.platform}.{also}"
        ), answer)

    # ----------------------------------------------------------- actions

    def action_tab(self, tab: str) -> None:
        # When the focused widget's pane is hidden, Textual hands the focus to
        # a sibling on that pane, which switches the tab straight back. Move
        # the focus to the new tab's table first (none on tabs with inputs, so
        # the keys keep working there).
        target = {
            "overview": "#gpu-list" if self.gpu_view == "list" else "#overview-processes",
            "processes": "#processes-table",
            "devices": "#devices-table",
            "profiles": "#profiles-table",
        }.get(tab)
        if target:
            self.query_one(target).focus()
        else:
            self.set_focus(None)
        self.query_one(TabbedContent).active = tab

    def action_toggle_gpu_view(self) -> None:
        self.gpu_view = "list" if self.gpu_view == "cards" else "cards"
        self.query_one("#gpus", Grid).display = self.gpu_view == "cards"
        gpu_list = self.query_one("#gpu-list", DataTable)
        gpu_list.display = self.gpu_view == "list"
        (gpu_list if gpu_list.display else self.query_one("#overview-processes")).focus()

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
        table_id = {"overview": "#overview-processes", "processes": "#processes-table"}[tab]
        pid = selected_key(self.query_one(table_id, DataTable))
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
