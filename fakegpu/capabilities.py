from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "fakegpu.native_api_capabilities.v1"


class CapabilityCatalogError(ValueError):
    pass


def default_capability_catalog_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "data"
        / "native_api_capabilities.json"
    )


@lru_cache(maxsize=4)
def load_native_capabilities(
    path: str | Path | None = None,
) -> dict[str, Any]:
    resolved = (
        Path(path).expanduser().resolve()
        if path is not None
        else default_capability_catalog_path()
    )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapabilityCatalogError(
            f"cannot read native capability catalog {resolved}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CapabilityCatalogError("capability catalog root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CapabilityCatalogError(
            f"expected schema_version {SCHEMA_VERSION!r}"
        )

    groups = payload.get("groups")
    apis = payload.get("apis")
    if not isinstance(groups, list) or not isinstance(apis, list):
        raise CapabilityCatalogError("groups and apis must be arrays")

    validated_groups: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    for index, raw in enumerate(groups):
        if not isinstance(raw, Mapping):
            raise CapabilityCatalogError(f"groups[{index}] must be an object")
        item = dict(raw)
        for field in (
            "id",
            "library",
            "symbol_regex",
            "classification",
            "simulate_behavior",
        ):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise CapabilityCatalogError(
                    f"groups[{index}].{field} must be a non-empty string"
                )
        if item["id"] in seen_group_ids:
            raise CapabilityCatalogError(
                f"duplicate capability group {item['id']!r}"
            )
        seen_group_ids.add(item["id"])
        try:
            re.compile(item["symbol_regex"])
        except re.error as exc:
            raise CapabilityCatalogError(
                f"groups[{index}].symbol_regex is invalid: {exc}"
            ) from exc
        validated_groups.append(item)

    validated_apis: list[dict[str, Any]] = []
    seen_apis: set[str] = set()
    for index, raw in enumerate(apis):
        if not isinstance(raw, Mapping):
            raise CapabilityCatalogError(f"apis[{index}] must be an object")
        item = dict(raw)
        for field in (
            "api",
            "library",
            "classification",
            "simulate_behavior",
        ):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise CapabilityCatalogError(
                    f"apis[{index}].{field} must be a non-empty string"
                )
        if item["api"] in seen_apis:
            raise CapabilityCatalogError(
                f"duplicate native API capability {item['api']!r}"
            )
        seen_apis.add(item["api"])
        if not isinstance(item.get("policy_enforced"), bool):
            raise CapabilityCatalogError(
                f"apis[{index}].policy_enforced must be boolean"
            )
        validated_apis.append(item)

    return {
        "schema_version": SCHEMA_VERSION,
        "source": str(resolved),
        "groups": validated_groups,
        "apis": validated_apis,
    }


def native_capability_report(
    *,
    catalog_path: str | Path | None = None,
    source_root: str | Path | None = None,
    build_dir: str | Path | None = None,
) -> dict[str, Any]:
    catalog = load_native_capabilities(catalog_path)
    apis = list(catalog["apis"])
    groups = list(catalog["groups"])
    classifications: dict[str, int] = {}
    libraries: dict[str, int] = {}
    for api in apis:
        classification = str(api["classification"])
        library = str(api["library"])
        classifications[classification] = (
            classifications.get(classification, 0) + 1
        )
        libraries[library] = libraries.get(library, 0) + 1

    source_audit = (
        audit_native_capability_sources(source_root, catalog=catalog)
        if source_root is not None
        else {"status": "not_requested"}
    )
    export_audit = (
        audit_native_exports(build_dir, catalog=catalog)
        if build_dir is not None
        else {"status": "not_requested"}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_source": catalog["source"],
        "summary": {
            "group_count": len(groups),
            "explicit_api_count": len(apis),
            "policy_enforced_api_count": sum(
                bool(api["policy_enforced"]) for api in apis
            ),
            "classifications": dict(sorted(classifications.items())),
            "libraries": dict(sorted(libraries.items())),
        },
        "groups": groups,
        "apis": apis,
        "source_audit": source_audit,
        "export_audit": export_audit,
    }


def audit_native_capability_sources(
    source_root: str | Path,
    *,
    catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(source_root).expanduser().resolve()
    if root.name != "src" and (root / "src").is_dir():
        root = root / "src"
    if not root.is_dir():
        raise FileNotFoundError(f"native source directory not found: {root}")

    selected = dict(catalog or load_native_capabilities())
    api_entries = {
        str(item["api"]): dict(item)
        for item in selected.get("apis", [])
        if isinstance(item, Mapping)
    }
    policy_apis: set[str] = set()
    stub_apis: set[str] = set()
    scanned_files: list[str] = []
    for path in sorted((*root.rglob("*.cpp"), *root.rglob("*.hpp"))):
        text = path.read_text(encoding="utf-8", errors="replace")
        scanned_files.append(str(path))
        policy_apis.update(_policy_api_names(text))
        stub_apis.update(_stub_api_names(text))

    declared_policy_apis = {
        name
        for name, entry in api_entries.items()
        if bool(entry.get("policy_enforced"))
    }
    unclassified_policy = sorted(policy_apis - set(api_entries))
    unclassified_stubs = sorted(stub_apis - set(api_entries))
    declared_but_not_enforced = sorted(declared_policy_apis - policy_apis)
    passed = not (
        unclassified_policy
        or unclassified_stubs
        or declared_but_not_enforced
    )
    return {
        "status": "passed" if passed else "failed",
        "source_root": str(root),
        "scanned_file_count": len(scanned_files),
        "policy_api_count": len(policy_apis),
        "stub_api_count": len(stub_apis),
        "policy_apis": sorted(policy_apis),
        "stub_apis": sorted(stub_apis),
        "unclassified_policy_apis": unclassified_policy,
        "unclassified_stub_apis": unclassified_stubs,
        "declared_policy_apis_not_found": declared_but_not_enforced,
    }


def audit_native_exports(
    build_dir: str | Path,
    *,
    catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(build_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"native build directory not found: {root}")
    nm = shutil.which("nm")
    if nm is None:
        raise RuntimeError("nm is required to audit native exports")

    selected = dict(catalog or load_native_capabilities())
    groups = [
        (dict(group), re.compile(str(group["symbol_regex"])))
        for group in selected.get("groups", [])
        if isinstance(group, Mapping)
    ]
    explicit = {
        str(item["api"]): dict(item)
        for item in selected.get("apis", [])
        if isinstance(item, Mapping)
    }
    library_paths = _native_library_paths(root)
    expected_libraries = {
        str(group["library"]) for group, _pattern in groups
    }
    found_libraries = {_library_id(path.name) for path in library_paths}
    missing_libraries = sorted(expected_libraries - found_libraries)
    exports: list[dict[str, Any]] = []
    unclassified: list[str] = []
    library_mismatches: list[str] = []
    for library_path in library_paths:
        library_id = _library_id(library_path.name)
        command = (
            [nm, "-gU", str(library_path)]
            if library_path.suffix == ".dylib"
            else [nm, "-D", "--defined-only", str(library_path)]
        )
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"nm failed for {library_path}: {completed.stderr.strip()}"
            )
        for raw_symbol in _nm_symbols(completed.stdout):
            symbol = _normalize_export_symbol(raw_symbol, groups)
            if not _looks_like_vendor_api(symbol):
                continue
            entry = explicit.get(symbol)
            group = next(
                (
                    group
                    for group, pattern in groups
                    if pattern.search(symbol)
                    and library_id
                    in _allowed_export_libraries(str(group["library"]))
                ),
                None,
            )
            if (
                entry is not None
                and library_id
                not in _allowed_export_libraries(str(entry["library"]))
            ):
                library_mismatches.append(
                    f"{library_path.name}:{symbol}: expected "
                    f"{entry['library']}"
                )
                continue
            if entry is None and group is None:
                unclassified.append(f"{library_path.name}:{symbol}")
                continue
            exports.append(
                {
                    "library_file": library_path.name,
                    "symbol": symbol,
                    "classification": str(
                        (entry or group or {}).get("classification", "")
                    ),
                    "classification_source": (
                        "explicit_api" if entry is not None else "group"
                    ),
                }
            )

    return {
        "status": "passed" if (
            library_paths
            and not missing_libraries
            and not unclassified
            and not library_mismatches
        ) else (
            "no_libraries" if not library_paths else "failed"
        ),
        "build_dir": str(root),
        "library_count": len(library_paths),
        "classified_export_count": len(exports),
        "missing_libraries": missing_libraries,
        "unclassified_exports": sorted(unclassified),
        "library_mismatches": sorted(library_mismatches),
        "exports": exports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu capabilities",
        description=(
            "List native API behavior classifications and audit source/build "
            "coverage."
        ),
    )
    parser.add_argument("--catalog")
    parser.add_argument("--source-root")
    parser.add_argument("--build-dir")
    parser.add_argument("--library")
    parser.add_argument("--classification")
    parser.add_argument("--api")
    parser.add_argument(
        "--json",
        dest="json_path",
        nargs="?",
        const="-",
        help="Write JSON to PATH, or stdout when PATH is omitted.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 when a requested source/export audit fails.",
    )
    args = parser.parse_args(argv)

    try:
        report = native_capability_report(
            catalog_path=args.catalog,
            source_root=args.source_root,
            build_dir=args.build_dir,
        )
    except (
        CapabilityCatalogError,
        FileNotFoundError,
        OSError,
        RuntimeError,
    ) as exc:
        parser.exit(2, f"fakegpu capabilities: {exc}\n")

    report["apis"] = [
        api
        for api in report["apis"]
        if (
            args.library is None
            or str(api["library"]) == str(args.library)
        )
        and (
            args.classification is None
            or str(api["classification"]) == str(args.classification)
        )
        and (
            args.api is None
            or str(args.api).lower() in str(api["api"]).lower()
        )
    ]
    if args.json_path:
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.json_path == "-":
            print(payload, end="")
        else:
            output = Path(args.json_path).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
            print(f"Native capability report: {output}")
    else:
        _print_capability_report(report)

    audit_failed = any(
        str(report[name].get("status")) == "failed"
        for name in ("source_audit", "export_audit")
    )
    return 2 if args.strict and audit_failed else 0


def _policy_api_names(text: str) -> set[str]:
    pattern = re.compile(
        r"(?:rejectUnsupportedApi|record_unsupported_api)\s*\("
        r".{0,600}?[\"']((?:__)?(?:cuda|cu|nvml|cublas|nccl)"
        r"[A-Za-z0-9_]*)[\"']",
        re.DOTALL,
    )
    return {match.group(1) for match in pattern.finditer(text)}


def _stub_api_names(text: str) -> set[str]:
    lines = text.splitlines()
    results: set[str] = set()
    for index, line in enumerate(lines):
        if not (
            re.search(r"\(\s*stub\s*\)", line, re.IGNORECASE)
            or "Just a stub" in line
            or "without actually" in line
        ):
            continue
        stripped = line.strip()
        if (
            stripped.startswith("//")
            and re.search(r"\(\s*stub\s*\)", line, re.IGNORECASE)
            and "Just a stub" not in line
        ):
            name = _next_api_function(lines, index)
        else:
            name = _nearest_api_function(lines, index)
        if name:
            results.add(name)
    return results


def _nearest_api_function(lines: Sequence[str], index: int) -> str | None:
    function_pattern = re.compile(
        r"\b((?:__)?(?:cuda|cu|nvml|cublas|nccl)[A-Za-z0-9_]*)\s*\("
    )
    for line_index in range(index - 1, max(-1, index - 30), -1):
        line = lines[line_index]
        if "FGPU_LOG" in line or "record_unsupported_api" in line:
            continue
        match = function_pattern.search(line)
        if match:
            return match.group(1)
    return None


def _next_api_function(lines: Sequence[str], index: int) -> str | None:
    function_pattern = re.compile(
        r"\b((?:__)?(?:cuda|cu|nvml|cublas|nccl)[A-Za-z0-9_]*)\s*\("
    )
    for line_index in range(index + 1, min(len(lines), index + 30)):
        match = function_pattern.search(lines[line_index])
        if match:
            return match.group(1)
    return None


def _nm_symbols(output: str) -> list[str]:
    symbols = []
    for line in output.splitlines():
        fields = line.split()
        if fields:
            symbols.append(fields[-1])
    return symbols


def _normalize_export_symbol(
    symbol: str,
    groups: Sequence[tuple[Mapping[str, Any], re.Pattern[str]]],
) -> str:
    if any(pattern.search(symbol) for _group, pattern in groups):
        return symbol
    if symbol.startswith("_") and any(
        pattern.search(symbol[1:]) for _group, pattern in groups
    ):
        return symbol[1:]
    return symbol


def _looks_like_vendor_api(symbol: str) -> bool:
    return bool(
        re.match(r"^(?:__cuda|cuda|cublas|nvml|nccl|cu[A-Z])", symbol)
    )


def _library_id(filename: str) -> str:
    if filename.startswith("libcudart"):
        return "cudart"
    if filename.startswith("libcublas"):
        return "cublas"
    if filename.startswith("libnvidia-ml"):
        return "nvml"
    if filename.startswith("libnccl"):
        return "nccl"
    if filename.startswith("libcuda"):
        return "cuda_driver"
    return "unknown"


def _allowed_export_libraries(declared_library: str) -> set[str]:
    if declared_library == "cuda_driver":
        # The standalone Runtime compatibility library embeds Driver stubs so
        # it can service internal Runtime calls without another dependency.
        return {"cuda_driver", "cudart"}
    return {declared_library}


def _native_library_paths(root: Path) -> list[Path]:
    patterns = {
        "cuda_driver": ("libcuda.so*", "libcuda*.dylib"),
        "cudart": ("libcudart.so*", "libcudart*.dylib"),
        "cublas": ("libcublas.so*", "libcublas*.dylib"),
        "nvml": ("libnvidia-ml.so*", "libnvidia-ml*.dylib"),
        "nccl": ("libnccl.so*", "libnccl*.dylib"),
    }
    canonical_names = {
        "cuda_driver": ("libcuda.so.1", "libcuda.dylib"),
        "cudart": ("libcudart.so.12", "libcudart.dylib"),
        "cublas": ("libcublas.so.12", "libcublas.dylib"),
        "nvml": ("libnvidia-ml.so.1", "libnvidia-ml.dylib"),
        "nccl": ("libnccl.so.2", "libnccl.dylib"),
    }
    selected: list[Path] = []
    for library_id in ("cuda_driver", "cudart", "cublas", "nvml", "nccl"):
        canonical = next(
            (
                root / name
                for name in canonical_names[library_id]
                if (root / name).is_file()
            ),
            None,
        )
        if canonical is not None:
            selected.append(canonical)
            continue
        candidates = sorted(
            {
                candidate
                for pattern in patterns[library_id]
                for candidate in root.glob(pattern)
                if candidate.is_file()
            },
            key=lambda path: (len(path.name), path.name),
        )
        if candidates:
            selected.append(candidates[0])
    return selected


def _print_capability_report(report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    print("FakeGPU native API capabilities")
    print(
        f"  groups: {summary['group_count']}  "
        f"explicit APIs: {summary['explicit_api_count']}  "
        f"policy-enforced: {summary['policy_enforced_api_count']}"
    )
    print("")
    print(f"{'API':44} {'Library':13} {'Classification'}")
    print(f"{'-' * 44} {'-' * 13} {'-' * 28}")
    for item in report["apis"]:
        print(
            f"{str(item['api']):44} "
            f"{str(item['library']):13} "
            f"{str(item['classification'])}"
        )
    for name, label in (
        ("source_audit", "source audit"),
        ("export_audit", "export audit"),
    ):
        audit = report[name]
        if audit.get("status") != "not_requested":
            print(f"\n  {label}: {audit.get('status')}")


__all__ = [
    "CapabilityCatalogError",
    "SCHEMA_VERSION",
    "audit_native_capability_sources",
    "audit_native_exports",
    "load_native_capabilities",
    "native_capability_report",
]
