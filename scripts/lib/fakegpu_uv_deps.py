#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_with_tomllib(pyproject_path: Path) -> dict | None:
    try:
        import tomllib  # Python 3.11+
    except Exception:
        return None
    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))


def _strip_toml_comment(line: str) -> str:
    in_string = False
    escaped = False
    out: list[str] = []
    for ch in line:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            out.append(ch)
            in_string = not in_string
            continue
        if ch == "#" and not in_string:
            break
        out.append(ch)
    return "".join(out)


def _parse_fakegpu_uv_section(pyproject_text: str) -> dict[str, list[str]]:
    deps: dict[str, list[str]] = {}

    in_section = False
    current_key: str | None = None
    collecting = False

    def add_items_from_line(key: str, line: str) -> None:
        i = 0
        while True:
            start = line.find('"', i)
            if start == -1:
                break
            end = line.find('"', start + 1)
            if end == -1:
                break
            deps.setdefault(key, []).append(line[start + 1 : end])
            i = end + 1

    for raw in pyproject_text.splitlines():
        line = _strip_toml_comment(raw).strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            header = line[1:-1].strip()
            if header == "tool.fakegpu.uv":
                in_section = True
                continue
            if in_section:
                break
            continue

        if not in_section:
            continue

        if collecting and current_key is not None:
            add_items_from_line(current_key, line)
            if "]" in line:
                collecting = False
                current_key = None
            continue

        if "=" not in line:
            continue

        key, rhs = (part.strip() for part in line.split("=", 1))
        if not key:
            continue

        if "[" in rhs:
            current_key = key
            collecting = True
            add_items_from_line(current_key, rhs)
            if "]" in rhs:
                collecting = False
                current_key = None

    return deps


def load_deps(pyproject_path: Path, group: str) -> list[str]:
    parsed = _load_with_tomllib(pyproject_path)
    if parsed is not None:
        try:
            deps = parsed["tool"]["fakegpu"]["uv"][group]
        except KeyError as exc:  # pragma: no cover
            raise KeyError(f"Missing `[tool.fakegpu.uv].{group}` in {pyproject_path}") from exc
        if not isinstance(deps, list) or not all(isinstance(x, str) for x in deps):
            raise TypeError(f"`[tool.fakegpu.uv].{group}` must be a list of strings")
        return deps

    text = pyproject_path.read_text(encoding="utf-8")
    deps_map = _parse_fakegpu_uv_section(text)
    if group not in deps_map:
        raise KeyError(f"Missing `[tool.fakegpu.uv].{group}` in {pyproject_path}")
    return deps_map[group]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print dependency lists from `[tool.fakegpu.uv]` in pyproject.toml.")
    parser.add_argument("group", help="Key under `[tool.fakegpu.uv]`, e.g. 'linux_vm_gpu_mgmt'")
    parser.add_argument("--pyproject", default="pyproject.toml", help="Path to pyproject.toml (default: ./pyproject.toml)")
    parser.add_argument("--json", action="store_true", help="Emit JSON array instead of newline-separated output")
    ns = parser.parse_args(argv)

    try:
        deps = load_deps(Path(ns.pyproject), ns.group)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if ns.json:
        print(json.dumps(deps))
    else:
        for dep in deps:
            print(dep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
