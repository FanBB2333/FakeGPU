"""Small display widgets and text helpers for the console."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.measure import Measurement
from rich.table import Table
from rich.text import Text
from textual.message import Message
from textual.widgets import DataTable, Static

SPARK = "▁▂▃▄▅▆▇█"
LEVEL_STYLE = {"info": "cyan", "warning": "yellow", "error": "bold red"}
LEVEL_LABEL = {"info": "INFO", "warning": "WARN", "error": "ERROR"}
# One color per kind of reading, the same in bars, charts and labels. Fixed
# RGB values keep them apart from the red/yellow/green of warning levels
# whatever the terminal's ANSI palette is.
COLORS = {
    "memory": "#c678dd",
    "gpu": "#98c379",
    "temperature": "#e5c07b",
    "power": "#ff9e64",
    "pcie": "#56b6c2",
    "nvlink": "#61afef",
    "c2c": "#73daca",
    "link": "#56b6c2",
}


def size(value: int | None) -> str:
    if not value:
        return "0"
    if value >= 1024**3:
        return f"{value / 1024**3:.1f} GiB"
    return f"{value / 1024**2:.0f} MiB"


def memory_pair(used: int, total: int) -> str:
    """``12.0/80.0 GiB``: used and total memory in little space."""
    if used >= 1024**3 or not used:
        return f"{used / 1024**3:.1f}/{total / 1024**3:.1f} GiB"
    return f"{size(used)}/{size(total)}"


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


def number(value: float) -> str:
    if value >= 100 or value == int(value):
        return f"{value:.0f}"
    return f"{value:.1f}" if value >= 10 else f"{value:.2g}"


def nice_ceiling(value: float) -> float:
    """Smallest 1, 2 or 5 times a power of ten that is at least ``value``."""
    if value <= 0:
        return 1.0
    scale = 10 ** math.floor(math.log10(value))
    return next(step * scale for step in (1, 2, 5, 10) if step * scale >= value)


def latest(values: Sequence[float | None]) -> float | None:
    return next((value for value in reversed(values) if value is not None), None)


def bar(percent: float | None, width: int, color: str | None = None) -> Text:
    """Fill to ``percent``; ``color`` names the reading, otherwise the level picks one."""
    if percent is None:
        return Text("░" * width, style="dim")
    filled = round(max(0.0, min(100.0, percent)) / 100 * width)
    return Text.assemble(("█" * filled, color or level_style(percent)), ("░" * (width - filled), "dim"))


def sparkline(
    values: Sequence[float | None], width: int, maximum: float = 100.0, color: str | None = None
) -> Text:
    text = Text()
    for value in list(values)[-width:]:
        if value is None:
            text.append(" ")
        else:
            index = min(len(SPARK) - 1, int(max(0.0, value) / maximum * (len(SPARK) - 1) + 0.5))
            text.append(SPARK[index], style=color or level_style(value / maximum * 100))
    return text


class Chart:
    """Recent samples as a filled area chart that spans the width it is given.

    The newest sample is at the right edge; a ``None`` sample (no reading)
    leaves its column empty.
    """

    def __init__(
        self,
        title: str,
        values: Sequence[float | None],
        *,
        maximum: float,
        color: str,
        unit: str = "",
        current: str | None = None,
        height: int = 4,
    ):
        self.title, self.values, self.color, self.unit = title, list(values), color, unit
        self.maximum = maximum if maximum > 0 else 1.0
        self.current = current
        self.height = height

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        return Measurement(16, options.max_width)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = options.max_width
        last = latest(self.values)
        peak = max((value for value in self.values if value is not None), default=None)
        current = self.current or ("N/A" if last is None else f"{number(last)}{self.unit}")
        header = Text.assemble(
            ("● ", self.color), (self.title, f"bold {self.color}"), "  ", (current, "bold"),
            no_wrap=True, overflow="ellipsis",
        )
        if peak is not None and self.current is None:
            header.append(f"  peak {number(peak)}{self.unit}", style="dim")
        header.truncate(width, overflow="ellipsis")
        yield header
        top = f"{number(self.maximum)}{self.unit}"
        axis = len(top)
        area = max(1, width - axis - 1)
        samples = self.values[-area:]
        samples = [None] * (area - len(samples)) + samples
        eighths = self.height * 8
        levels = []
        for value in samples:
            if value is None:
                levels.append(None)
                continue
            level = round(max(0.0, min(1.0, value / self.maximum)) * eighths)
            # Keep any activity visible, however small against the scale.
            levels.append(max(1, level) if value > 0 else 0)
        for row in range(self.height - 1, -1, -1):
            label = top if row == self.height - 1 else "0" if row == 0 else ""
            cells = []
            for level in levels:
                fill = 0 if level is None else level - row * 8
                cells.append(" " if fill <= 0 else "█" if fill >= 8 else SPARK[fill - 1])
            yield Text.assemble(
                (f"{label:>{axis}}{'┤' if label else '│'}", "dim"),
                ("".join(cells), self.color),
                no_wrap=True,
                overflow="crop",
            )


def chart_grid(charts: Sequence[Chart], columns: int, gap: int = 0) -> Table:
    """Charts side by side, ``columns`` per row, sharing the width evenly."""
    grid = Table.grid(expand=True, padding=(0, 2))
    columns = max(1, min(columns, len(charts)))
    for _ in range(columns):
        grid.add_column(ratio=1)
    for start in range(0, len(charts), columns):
        if start and gap:
            # Grids drop vertical padding, so blank lines separate the rows.
            grid.add_row("\n" * (gap - 1))
        row = list(charts[start:start + columns])
        grid.add_row(*row, *[""] * (columns - len(row)))
    return grid


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


def device_key(device: dict[str, Any]) -> str:
    """Row key of a device: its layout position, or its UUID when outside the layout."""
    return f"gpu-{device['index']}" if device["in_layout"] else device["uuid"]


def reading(label: str, color: str, percent: float | None, width: int, text: str) -> Text:
    """``LABEL ████░░░░ text`` with the label and bar in the reading's color."""
    style = "bold red" if percent is not None and percent >= 90 else ""
    return Text.assemble((f"{label:<5}", color), bar(percent, width, color), " ", (text, style))


class GpuCard(Static):
    """One device: identity, memory, sensors and recent history; click to open it."""

    BAR = 16

    class Selected(Message):
        def __init__(self, key: str):
            super().__init__()
            self.key = key

    key: str | None = None

    def show(self, device: dict[str, Any], warning_percent: float) -> None:
        memory, telemetry = device["memory"], device["telemetry"]
        history = device.get("history", {})
        percent = memory["percent"]
        self.key = device_key(device)
        self.set_class(device["process_count"] > 0, "busy")
        self.set_class(percent >= warning_percent, "warn")
        self.set_class(memory["used_bytes"] > memory["total_bytes"], "alert")
        title = Text.assemble((f"GPU {device['index']} ", "bold"), (device["name"], "bold cyan"))
        if not device["in_layout"]:
            title.append("  (not in layout)", style="yellow")
        processes = device["process_count"]
        lines = [
            title,
            Text.assemble(
                (f"{device['profile_id']}  {device['architecture']}  CC {device['compute_capability']}  ", "dim"),
                (f"{processes} process{'es' if processes != 1 else ''}", "bold green" if processes else "dim"),
            ),
            reading(
                "MEM", COLORS["memory"], percent, self.BAR, memory_pair(memory["used_bytes"], memory["total_bytes"])
            ),
        ]
        if telemetry:
            limit = device["power_limit_w"]
            power = telemetry["power_w"]
            lines += [
                reading("UTIL", COLORS["gpu"], telemetry["gpu_percent"], self.BAR, f"{telemetry['gpu_percent']}%"),
                reading(
                    "PWR", COLORS["power"], power / limit * 100 if limit else None, self.BAR,
                    f"{power:.0f}{f'/{limit:.0f}' if limit else ''} W {telemetry['pstate']}",
                ),
                Text.assemble(
                    ("TEMP ", COLORS["temperature"]), (f"{telemetry['temperature_c']}°C", "bold"),
                    (f"  SM {telemetry['sm_clock_mhz']}  MEM {telemetry['memory_clock_mhz']} MHz", "dim"),
                ),
            ]
        else:
            lines += [Text("sensors N/A for this profile", style="dim"), Text(""), Text("")]
        half = (self.BAR + 10) // 2
        lines.append(Text.assemble(
            ("mem ", COLORS["memory"]), sparkline(history.get("memory_percent", []), half, color=COLORS["memory"]),
            (" util ", COLORS["gpu"]), sparkline(history.get("gpu", []), half, color=COLORS["gpu"]),
        ))
        for line in lines:
            line.no_wrap, line.overflow = True, "ellipsis"
        self.update(Group(*lines))

    def on_click(self) -> None:
        if self.key is not None:
            self.post_message(self.Selected(self.key))


def key_values(rows: Iterable[tuple[str, Any]]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    for key, value in rows:
        table.add_row(key, value if isinstance(value, Text) else str(value))
    return table
