from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .profile_catalog import get_profile
from .structured_io import write_json


SCHEMA_VERSION = "fakegpu.kernel_static_analysis.v1"


class KernelAnalysisError(ValueError):
    pass


def analyze_kernel_file(
    path: str | Path,
    *,
    profile_id: str | None = None,
    threads_per_block: int = 256,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"kernel file not found: {resolved}")
    if threads_per_block <= 0:
        raise KernelAnalysisError("threads_per_block must be greater than zero")
    suffix = resolved.suffix.lower()
    try:
        text = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise KernelAnalysisError(
            f"kernel static analysis requires text input: {resolved}"
        ) from exc
    if suffix == ".ptx":
        analysis = analyze_ptx(text)
        kind = "ptx"
    elif suffix in {".sass", ".disasm"}:
        analysis = analyze_sass(text)
        kind = "sass"
    elif suffix in {".cu", ".cuh"}:
        analysis = analyze_cuda_source(text)
        kind = "cuda_source"
    else:
        raise KernelAnalysisError(
            "supported kernel text suffixes are .ptx, .sass, .disasm, .cu, and .cuh"
        )

    occupancy = None
    if profile_id and kind == "ptx":
        occupancy = estimate_occupancy(
            analysis,
            profile_id=profile_id,
            threads_per_block=threads_per_block,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(resolved),
        "kind": kind,
        "analysis": analysis,
        "occupancy": occupancy,
        "limitations": [
            "Instruction counts are static text counts and do not include dynamic branch or loop trip counts.",
            "FLOP counts cover recognized arithmetic opcodes only.",
            "Occupancy is a resource ceiling, not achieved utilization or latency.",
        ],
    }


def analyze_ptx(text: str) -> dict[str, Any]:
    cleaned = _strip_comments(text)
    targets = sorted(
        set(re.findall(r"\.target\s+([A-Za-z0-9_]+)", cleaned))
    )
    entries = re.findall(
        r"(?:\.visible\s+)?\.entry\s+([A-Za-z_$][\w$]*)",
        cleaned,
    )
    registers_by_type: Counter[str] = Counter()
    for declaration in re.finditer(
        r"\.reg\s+\.(\w+)\s+([^;]+);",
        cleaned,
    ):
        type_name = declaration.group(1)
        body = declaration.group(2)
        vector_counts = [
            int(value) for value in re.findall(r"<(\d+)>", body)
        ]
        if vector_counts:
            count = sum(vector_counts)
        else:
            count = len([value for value in body.split(",") if value.strip()])
        registers_by_type[type_name] += max(1, count)

    shared_bytes = 0
    shared_objects = []
    for match in re.finditer(
        r"\.shared(?:\s+\.align\s+\d+)?\s+\.(\w+)\s+"
        r"([A-Za-z_$][\w$]*)(?:\[(\d+)\])?\s*;",
        cleaned,
    ):
        type_name, name, count_raw = match.groups()
        count = int(count_raw or 1)
        element_bytes = _ptx_type_bytes(type_name)
        size = count * element_bytes
        shared_bytes += size
        shared_objects.append(
            {
                "name": name,
                "type": type_name,
                "elements": count,
                "bytes": size,
            }
        )

    instruction_counts: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    estimated_flops = 0
    tensor_core_flops_per_issue = 0
    pending_instruction = ""
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not pending_instruction and (
            line.startswith(".")
            or line.endswith(":")
            or line in {"{", "}"}
        ):
            continue
        pending_instruction = (
            f"{pending_instruction} {line}".strip()
            if pending_instruction
            else line
        )
        if not line.endswith(";"):
            continue
        statement = re.sub(
            r"^@!?%[\w$]+\s+",
            "",
            pending_instruction,
        )
        pending_instruction = ""
        opcode = statement.split(None, 1)[0].rstrip(";")
        if re.match(r"[A-Za-z]", opcode):
            instruction_counts[opcode] += 1
            instruction_class = _ptx_instruction_class(opcode)
            classes[instruction_class] += 1
            flops = _ptx_flops(opcode)
            estimated_flops += flops
            if opcode.startswith(("mma.", "wgmma.")):
                tensor_core_flops_per_issue += flops

    register_count = sum(registers_by_type.values())
    return {
        "targets": targets,
        "entry_points": sorted(set(entries)),
        "entry_point_count": len(set(entries)),
        "instruction_count": sum(instruction_counts.values()),
        "instruction_counts": dict(sorted(instruction_counts.items())),
        "instruction_classes": dict(sorted(classes.items())),
        "register_declarations": dict(sorted(registers_by_type.items())),
        "declared_register_count": register_count,
        "static_shared_memory_bytes": shared_bytes,
        "shared_objects": shared_objects,
        "recognized_flops_per_static_issue": estimated_flops,
        "recognized_tensor_core_flops_per_static_issue": (
            tensor_core_flops_per_issue
        ),
    }


def analyze_sass(text: str) -> dict[str, Any]:
    cleaned = _strip_comments(text, preserve_sass_addresses=True)
    opcodes: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    estimated_flops = 0
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        match = re.search(
            r"(?:/\*[0-9A-Fa-f]+\*/\s*)?([A-Z][A-Z0-9_.]+)"
            r"(?:\s|;|$)",
            line,
        )
        if not match:
            continue
        opcode = match.group(1)
        opcodes[opcode] += 1
        instruction_class = _sass_instruction_class(opcode)
        classes[instruction_class] += 1
        estimated_flops += _sass_flops(opcode)
    register_hints = [
        int(value)
        for value in re.findall(
            r"(?:REG(?:ISTERS)?|registers?)\s*[:=]\s*(\d+)",
            text,
            flags=re.IGNORECASE,
        )
    ]
    return {
        "instruction_count": sum(opcodes.values()),
        "instruction_counts": dict(sorted(opcodes.items())),
        "instruction_classes": dict(sorted(classes.items())),
        "register_count_hint": max(register_hints) if register_hints else None,
        "recognized_flops_per_static_issue": estimated_flops,
    }


def analyze_cuda_source(text: str) -> dict[str, Any]:
    cleaned = _strip_comments(text)
    kernels = re.findall(
        r"__global__\s+(?:[\w:<>]+\s+)+([A-Za-z_]\w*)\s*\(",
        cleaned,
    )
    launches = re.findall(r"([A-Za-z_]\w*)\s*<<<", cleaned)
    shared_static = re.findall(
        r"__shared__\s+[\w:<>]+\s+([A-Za-z_]\w*)\s*(?:\[(\d+)\])?",
        cleaned,
    )
    runtime_generation = [
        marker
        for marker in (
            "nvrtcCreateProgram",
            "nvrtcCompileProgram",
            "cuModuleLoadData",
            "cuModuleLoadDataEx",
        )
        if marker in cleaned
    ]
    return {
        "kernel_definitions": sorted(set(kernels)),
        "kernel_definition_count": len(kernels),
        "kernel_launches": dict(sorted(Counter(launches).items())),
        "kernel_launch_count": len(launches),
        "static_shared_declarations": [
            {"name": name, "elements": int(count or 1)}
            for name, count in shared_static
        ],
        "dynamic_shared_declaration_count": len(
            re.findall(r"extern\s+__shared__", cleaned)
        ),
        "runtime_generation_markers": runtime_generation,
    }


def estimate_occupancy(
    ptx_analysis: Mapping[str, Any],
    *,
    profile_id: str,
    threads_per_block: int,
) -> dict[str, Any]:
    profile = get_profile(profile_id)
    registers_per_thread = int(
        ptx_analysis.get("declared_register_count", 0) or 0
    )
    shared_per_block = int(
        ptx_analysis.get("static_shared_memory_bytes", 0) or 0
    )
    blocks_by_threads = (
        profile.max_threads_per_multiprocessor // threads_per_block
    )
    blocks_by_registers = (
        profile.regs_per_multiprocessor
        // (registers_per_thread * threads_per_block)
        if registers_per_thread > 0
        else profile.max_blocks_per_multiprocessor
    )
    blocks_by_shared = (
        profile.shared_mem_per_sm // shared_per_block
        if shared_per_block > 0
        else profile.max_blocks_per_multiprocessor
    )
    resident_blocks = min(
        profile.max_blocks_per_multiprocessor,
        blocks_by_threads,
        blocks_by_registers,
        blocks_by_shared,
    )
    active_threads = resident_blocks * threads_per_block
    constraints = {
        "architectural_block_limit": profile.max_blocks_per_multiprocessor,
        "threads": blocks_by_threads,
        "registers": blocks_by_registers,
        "shared_memory": blocks_by_shared,
    }
    minimum = min(constraints.values())
    limiting = sorted(
        name for name, value in constraints.items() if value == minimum
    )
    return {
        "profile_id": profile.id,
        "compute_capability": profile.compute_capability_text,
        "threads_per_block": threads_per_block,
        "registers_per_thread_upper_bound": registers_per_thread,
        "static_shared_memory_per_block_bytes": shared_per_block,
        "resident_blocks_per_sm_upper_bound": max(0, resident_blocks),
        "active_threads_per_sm_upper_bound": max(0, active_threads),
        "occupancy_upper_bound": (
            min(
                1.0,
                active_threads / profile.max_threads_per_multiprocessor,
            )
            if profile.max_threads_per_multiprocessor > 0
            else 0.0
        ),
        "block_limits": constraints,
        "limiting_resources": limiting,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu analyze-kernel",
        description=(
            "Statically inspect PTX, SASS text, or CUDA source for instructions, "
            "FLOPs, registers, shared memory, and occupancy ceilings."
        ),
    )
    parser.add_argument("path")
    parser.add_argument("--profile")
    parser.add_argument("--threads-per-block", type=int, default=256)
    parser.add_argument(
        "--json",
        dest="json_path",
        nargs="?",
        const="-",
        help="Write JSON to PATH, or stdout when PATH is omitted.",
    )
    args = parser.parse_args(argv)
    try:
        report = analyze_kernel_file(
            args.path,
            profile_id=args.profile,
            threads_per_block=args.threads_per_block,
        )
        if args.json_path:
            payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
            if args.json_path == "-":
                print(payload, end="")
            else:
                output = write_json(args.json_path, report)
                print(f"Kernel analysis: {output}")
        else:
            _print_report(report)
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"fakegpu analyze-kernel: {exc}\n")


def _strip_comments(
    text: str,
    *,
    preserve_sass_addresses: bool = False,
) -> str:
    if not preserve_sass_addresses:
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    return text


def _ptx_type_bytes(type_name: str) -> int:
    match = re.search(r"(\d+)$", type_name)
    return max(1, int(match.group(1)) // 8) if match else 1


def _ptx_instruction_class(opcode: str) -> str:
    base = opcode.split(".", 1)[0]
    if base in {"ld", "ldu", "prefetch"}:
        return "load"
    if base in {"st"}:
        return "store"
    if base in {"bra", "brx", "call", "ret", "exit"}:
        return "control"
    if base in {"bar", "membar", "atom", "red"}:
        return "synchronization_or_atomic"
    if base in {"mma", "wgmma"}:
        return "tensor_core"
    if base in {
        "add",
        "sub",
        "mul",
        "mad",
        "fma",
        "div",
        "sqrt",
        "rsqrt",
        "rcp",
        "sin",
        "cos",
        "ex2",
        "lg2",
    }:
        return "arithmetic"
    if base in {"cvt", "mov", "set", "setp", "selp", "shl", "shr"}:
        return "conversion_or_data_movement"
    return "other"


def _ptx_flops(opcode: str) -> int:
    if opcode.startswith(("mma.", "wgmma.")):
        match = re.search(r"m(\d+)n(\d+)k(\d+)", opcode)
        if match:
            m, n, k = (int(value) for value in match.groups())
            return 2 * m * n * k
        return 0
    base = opcode.split(".", 1)[0]
    vector_width = 2 if ".v2." in opcode else 4 if ".v4." in opcode else 1
    if base in {"fma", "mad"}:
        return 2 * vector_width
    if base in {"add", "sub", "mul", "div", "sqrt", "rsqrt", "rcp"}:
        return vector_width
    return 0


def _sass_instruction_class(opcode: str) -> str:
    if opcode.startswith(("LD", "LDC")):
        return "load"
    if opcode.startswith("ST"):
        return "store"
    if opcode.startswith(("BRA", "BRX", "CALL", "RET", "EXIT")):
        return "control"
    if opcode.startswith(("HMMA", "IMMA", "DMMA", "MMA")):
        return "tensor_core"
    if opcode.startswith(("F", "D", "H")):
        return "arithmetic"
    return "other"


def _sass_flops(opcode: str) -> int:
    if opcode.startswith(("HMMA", "IMMA", "DMMA", "MMA")):
        return 0
    if "FMA" in opcode or "MAD" in opcode:
        return 2
    if opcode.startswith(("FADD", "FMUL", "DADD", "DMUL", "HADD", "HMUL")):
        return 1
    return 0


def _print_report(report: Mapping[str, Any]) -> None:
    analysis = report["analysis"]
    print("FakeGPU kernel static analysis")
    print(f"  kind: {report['kind']}")
    print(f"  path: {report['path']}")
    if "instruction_count" in analysis:
        print(f"  static instructions: {analysis['instruction_count']}")
        print(
            "  recognized FLOPs/static issue: "
            f"{analysis['recognized_flops_per_static_issue']}"
        )
    if report["occupancy"] is not None:
        print(
            "  occupancy upper bound: "
            f"{report['occupancy']['occupancy_upper_bound'] * 100:.1f}%"
        )


__all__ = [
    "KernelAnalysisError",
    "SCHEMA_VERSION",
    "analyze_cuda_source",
    "analyze_kernel_file",
    "analyze_ptx",
    "analyze_sass",
    "estimate_occupancy",
    "main",
]
