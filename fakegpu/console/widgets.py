"""Small display widgets and text helpers for the console."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.widgets import DataTable, Static

SPARK = "▁▂▃▄▅▆▇█"
LEVEL_STYLE = {"info": "cyan", "warning": "yellow", "error": "bold red"}
LEVEL_LABEL = {"info": "INFO", "warning": "WARN", "error": "ERROR"}


def size(value: int | None) -> str:
    if not value:
        return "0"
    if value >= 1024**3:
        return f"{value / 1024**3:.1f} GiB"
    return f"{value / 1024**2:.0f} MiB"


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds // 60 % 60:02d}m"


def level_style(percent: float) -> str:
    return "red" if percent >= 90 else "yellow" if percent >= 70 else "green"


def bar(percent: float | None, width: int) -> Text:
    if percent is None:
        return Text("░" * width, style="dim")
    filled = round(max(0.0, min(100.0, percent)) / 100 * width)
    return Text.assemble(("█" * filled, level_style(percent)), ("░" * (width - filled), "dim"))


def sparkline(values: Sequence[float | None], width: int, maximum: float = 100.0) -> Text:
    text = Text()
    for value in list(values)[-width:]:
        if value is None:
            text.append(" ")
        else:
            index = min(len(SPARK) - 1, int(max(0.0, value) / maximum * (len(SPARK) - 1) + 0.5))
            text.append(SPARK[index], style=level_style(value / maximum * 100))
    return text


def event_line(event: dict[str, Any]) -> Text:
    level = event.get("level", "info")
    return Text.assemble(
        (event["time"][11:19], "dim"),
        " ",
        (f"{LEVEL_LABEL.get(level, level.upper()):<5}", LEVEL_STYLE.get(level, "")),
        " ",
        event["message"],
    )


def sync_rows(table: DataTable, rows: Iterable[tuple[str, tuple]]) -> None:
    """Update rows in place when the keys are unchanged, keep the selection."""

    rows = list(rows)
    keys = [key for key, _ in rows]
    current = [row.value for row in table.rows]
    if keys == current:
        columns = list(table.columns)
        for key, cells in rows:
            for column, value in zip(columns, cells):
                table.update_cell(key, column, value)
        return
    selected = current[table.cursor_row] if 0 <= table.cursor_row < len(current) else None
    table.clear()
    for key, cells in rows:
        table.add_row(*cells, key=key)
    if selected in keys:
        table.move_cursor(row=keys.index(selected))


def selected_key(table: DataTable) -> str | None:
    if not table.row_count:
        return None
    return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value


class StatTile(Static):
    def show(self, value: str, label: str, style: str = "bold") -> None:
        self.update(Text.assemble((value, style), "\n", (label, "dim")))


class GpuCard(Static):
    """One device: identity, memory, sensors and recent memory history."""

    BAR = 16

    def show(self, device: dict[str, Any], warning_percent: float) -> None:
        memory, telemetry = device["memory"], device["telemetry"]
        percent = memory["percent"]
        self.set_class(device["process_count"] > 0, "busy")
        self.set_class(percent >= warning_percent, "warn")
        self.set_class(memory["used_bytes"] > memory["total_bytes"], "alert")
        title = Text.assemble((f"GPU {device['index']} ", "bold"), (device["name"], "bold cyan"))
        if not device["in_layout"]:
            title.append("  (not in layout)", style="yellow")
        lines = [
            title,
            Text(
                f"{device['architecture']}  CC {device['compute_capability']}  "
                f"{device['process_count']} process(es)",
                style="dim",
            ),
            Text.assemble(
                "MEM  ",
                bar(percent, self.BAR),
                f" {size(memory['used_bytes'])} / {size(memory['total_bytes'])}",
            ),
        ]
        if telemetry:
            power_limit = device["power_limit_w"]
            lines += [
                Text.assemble("UTIL ", bar(telemetry["gpu_percent"], self.BAR), f" {telemetry['gpu_percent']}%"),
                Text(
                    f"TEMP {telemetry['temperature_c']}°C  PWR {telemetry['power_w']:.0f}"
                    f"{f'/{power_limit:.0f}' if power_limit else ''} W  {telemetry['pstate']}"
                ),
                Text(f"SM {telemetry['sm_clock_mhz']} MHz  MEM {telemetry['memory_clock_mhz']} MHz", style="dim"),
            ]
        else:
            lines += [Text("sensors N/A for this profile", style="dim"), Text(""), Text("")]
        history = device.get("history", {}).get("memory_percent", [])
        lines.append(Text.assemble(("mem  ", "dim"), sparkline(history, self.BAR + 14)))
        self.update(Group(*lines))


def key_values(rows: Iterable[tuple[str, Any]]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    for key, value in rows:
        table.add_row(key, value if isinstance(value, Text) else str(value))
    return table
