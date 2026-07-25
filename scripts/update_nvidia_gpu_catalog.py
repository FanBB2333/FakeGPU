#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "fakegpu" / "data" / "nvidia_compute_capabilities.json"
SOURCES = (
    "https://developer.nvidia.com/cuda/gpus",
    "https://developer.nvidia.com/cuda/gpus/legacy",
)
_CC_RE = re.compile(r"^\d+\.\d+$")


class _ComputeCapabilityTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_tbody = False
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tbody":
            self._in_tbody = True
        elif self._in_tbody and tag == "tr":
            self._row = []
        elif self._row is not None and tag == "td":
            self._cell = []
        elif self._cell is not None and tag == "br":
            self._cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "tbody":
            self._in_tbody = False
        elif tag == "td" and self._row is not None and self._cell is not None:
            self._row.append("".join(self._cell))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _fetch(url: str) -> str:
    request = Request(
        url,
        headers={"User-Agent": "FakeGPU NVIDIA profile catalog updater/1.0"},
    )
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _normalize_lines(cell: str) -> list[str]:
    result = []
    for raw in cell.replace("\xa0", " ").splitlines():
        value = re.sub(r"\s+", " ", raw).strip()
        if value:
            result.append(value)
    return result


def _normalize_models(cell: str) -> list[str]:
    models: list[str] = []
    for value in _normalize_lines(cell):
        # NVIDIA's current table renders these server-edition names on two
        # visual lines, unlike the line breaks that separate distinct models.
        if value == "Blackwell Server Edition" and models:
            models[-1] = f"{models[-1]} {value}"
        else:
            models.append(value)
    return models


def _parse_models(html: str) -> dict[str, str]:
    parser = _ComputeCapabilityTableParser()
    parser.feed(html)
    models: dict[str, str] = {}
    for row in parser.rows:
        first_cell = _normalize_lines(row[0]) if row else []
        if not first_cell or not _CC_RE.fullmatch(first_cell[0]):
            continue
        compute_capability = first_cell[0]
        for cell in row[1:]:
            for model in _normalize_models(cell):
                models[model] = compute_capability
    return models


def build_snapshot() -> dict[str, object]:
    models: dict[str, str] = {}
    for url in SOURCES:
        models.update(_parse_models(_fetch(url)))
    return {
        "schema_version": "nvidia-compute-capabilities.v1",
        "retrieved_at": datetime.now(UTC).date().isoformat(),
        "sources": list(SOURCES),
        "models": dict(sorted(models.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh the checked-in NVIDIA model-to-compute-capability snapshot "
            "from NVIDIA's current and legacy CUDA GPU tables."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the checked-in snapshot differs from NVIDIA's tables.",
    )
    args = parser.parse_args(argv)

    payload = build_snapshot()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()

    if args.check:
        try:
            current = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
        comparable_keys = ("schema_version", "sources", "models")
        if any(current.get(key) != payload.get(key) for key in comparable_keys):
            print(f"NVIDIA compute-capability snapshot is stale: {output}")
            return 1
        print(f"NVIDIA compute-capability snapshot is current: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(
        f"Wrote {len(payload['models'])} NVIDIA GPU mappings to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
