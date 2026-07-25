from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .kernel_analysis import KernelAnalysisError, analyze_kernel_file


SCHEMA_VERSION = "fakegpu.repository_analysis.v1"

_IGNORED_DIRECTORIES = {
    ".claude",
    ".eggs",
    ".git",
    ".hg",
    ".idea",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
_TEXT_FILE_LIMIT = 2 * 1024 * 1024
_NATIVE_CUDA_SUFFIXES = {
    ".cu",
    ".cuh",
    ".ptx",
    ".sass",
    ".disasm",
    ".cubin",
    ".fatbin",
}
_BINARY_EXTENSION_SUFFIXES = {".so", ".dylib", ".dll", ".pyd"}
_FRAMEWORK_IMPORTS = {
    "torch": "pytorch",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "deepspeed": "deepspeed",
    "peft": "peft",
    "trl": "trl",
    "bitsandbytes": "bitsandbytes",
    "triton": "triton",
    "flash_attn": "flash_attention",
    "xformers": "xformers",
    "apex": "apex",
    "lightning": "lightning",
    "pytorch_lightning": "lightning",
    "torchtune": "torchtune",
}
_ENTRYPOINT_FILENAMES = {
    "train.py",
    "training.py",
    "finetune.py",
    "fine_tune.py",
    "sft.py",
    "main.py",
    "run.py",
}


class RepositoryAnalysisError(ValueError):
    pass


def analyze_repository(
    path: str | os.PathLike[str],
    *,
    entrypoints: Sequence[str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise RepositoryAnalysisError(f"repository directory not found: {root}")

    files = list(_iter_repository_files(root))
    suffix_counts = Counter(
        (file.suffix.lower() or "<none>") for file in files
    )
    python_files = [file for file in files if file.suffix.lower() == ".py"]
    native_cuda_files = [
        file for file in files if file.suffix.lower() in _NATIVE_CUDA_SUFFIXES
    ]
    binary_extensions = [
        file for file in files if file.suffix.lower() in _BINARY_EXTENSION_SUFFIXES
    ]

    imports: Counter[str] = Counter()
    call_markers: Counter[str] = Counter()
    syntax_errors: list[dict[str, Any]] = []
    discovered_entrypoints: set[str] = set()
    for file in python_files:
        relative = str(file.relative_to(root))
        try:
            text = _read_text(file)
        except OSError:
            continue
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=str(file))
        except SyntaxError as exc:
            syntax_errors.append(
                {
                    "path": relative,
                    "line": int(exc.lineno or 0),
                    "message": str(exc.msg),
                }
            )
            continue
        imports.update(_python_imports(tree))
        call_markers.update(_python_call_markers(tree))
        if file.name.lower() in _ENTRYPOINT_FILENAMES or _has_main_guard(tree):
            discovered_entrypoints.add(relative)

    dependency_names = _dependency_names(root, files)
    imported_frameworks = {
        framework
        for import_name, framework in _FRAMEWORK_IMPORTS.items()
        if import_name in imports
    }
    dependency_frameworks = {
        framework
        for dependency in dependency_names
        for import_name, framework in _FRAMEWORK_IMPORTS.items()
        if _normalize_dependency_name(dependency)
        == _normalize_dependency_name(import_name)
    }
    frameworks = sorted(imported_frameworks | dependency_frameworks)

    configured_entrypoints = _validate_entrypoints(root, entrypoints or ())
    discovered_entrypoints.update(_pyproject_entrypoints(root))
    selected_entrypoints = (
        configured_entrypoints
        if configured_entrypoints
        else sorted(discovered_entrypoints)
    )

    config_markers = _configuration_markers(files)
    build_system = _build_system_markers(files, root=root)
    kernel_analyses = _kernel_analyses(
        native_cuda_files,
        root=root,
    )
    findings = _build_findings(
        native_cuda_files=native_cuda_files,
        binary_extensions=binary_extensions,
        frameworks=frameworks,
        call_markers=call_markers,
        config_markers=config_markers,
        build_system_markers=build_system["markers"],
        syntax_errors=syntax_errors,
        entrypoints=selected_entrypoints,
        root=root,
    )
    blockers = [
        finding
        for finding in findings
        if finding["severity"] == "requires_real_gpu_or_hybrid"
    ]
    validation_requirements = [
        finding
        for finding in findings
        if finding["severity"] == "requires_targeted_validation"
    ]
    analysis_complete = not syntax_errors and bool(selected_entrypoints)
    if blockers:
        verdict = "requires_real_gpu_or_hybrid"
    elif validation_requirements:
        verdict = "requires_targeted_validation"
    elif not analysis_complete:
        verdict = "analysis_incomplete"
    else:
        verdict = "preflight_candidate"

    git = _git_metadata(root)
    report = {
        "schema_version": SCHEMA_VERSION,
        "repository": {
            "path": str(root),
            "name": root.name,
            **git,
        },
        "inventory": {
            "file_count": len(files),
            "python_file_count": len(python_files),
            "native_cuda_file_count": len(native_cuda_files),
            "binary_extension_count": len(binary_extensions),
            "suffix_counts": dict(sorted(suffix_counts.items())),
            "native_cuda_files": [
                str(file.relative_to(root)) for file in native_cuda_files
            ],
            "binary_extensions": [
                str(file.relative_to(root)) for file in binary_extensions
            ],
        },
        "python": {
            "imports": dict(sorted(imports.items())),
            "call_markers": dict(sorted(call_markers.items())),
            "syntax_errors": syntax_errors,
        },
        "kernel_static_analysis": kernel_analyses,
        "build_system": build_system,
        "dependencies": sorted(dependency_names),
        "frameworks": frameworks,
        "configuration_markers": sorted(config_markers),
        "entrypoints": selected_entrypoints,
        "findings": findings,
        "readiness": {
            "verdict": verdict,
            "fakecuda_candidate": not blockers and analysis_complete,
            "requires_real_gpu_or_hybrid": bool(blockers),
            "requires_targeted_validation": bool(validation_requirements),
            "blocker_count": len(blockers),
            "validation_requirement_count": len(validation_requirements),
            "static_analysis_complete": analysis_complete,
            "confidence": (
                "R2_static_repository_scan"
                if not syntax_errors
                else "R1_partial_static_repository_scan"
            ),
        },
        "recommended_experiments": _recommended_experiments(
            root=root,
            entrypoints=selected_entrypoints,
            blockers=blockers,
            frameworks=frameworks,
        ),
        "limitations": [
            "Static scanning does not execute repository code or infer runtime tensor shapes.",
            "Encrypted, downloaded, or data-dependent generated kernels may not be visible.",
            "Use fakegpu preflight and the ATen graph estimator for the selected entrypoint and shapes.",
        ],
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu analyze-repo",
        description=(
            "Inspect a repository for FakeGPU readiness, acceleration "
            "dependencies, entrypoints, and required validation."
        ),
    )
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        help="Repository-relative Python entrypoint; may be repeated.",
    )
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
        help=(
            "Return exit code 2 for repositories that require real GPU/hybrid "
            "execution or have incomplete static analysis."
        ),
    )
    args = parser.parse_args(argv)

    try:
        report = analyze_repository(args.path, entrypoints=args.entry)
    except (OSError, RepositoryAnalysisError, ValueError) as exc:
        parser.exit(2, f"fakegpu analyze-repo: {exc}\n")

    if args.json_path:
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.json_path == "-":
            print(payload, end="")
        else:
            output = Path(args.json_path).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
            print(f"Repository analysis: {output}")
    else:
        _print_repository_analysis(report)

    readiness = report["readiness"]
    strict_failure = (
        readiness["requires_real_gpu_or_hybrid"]
        or not readiness["static_analysis_complete"]
    )
    return 2 if args.strict and strict_failure else 0


def _iter_repository_files(root: Path) -> Iterable[Path]:
    git_files = _git_visible_files(root)
    if git_files is not None:
        yield from git_files
        return

    for current_root, directory_names, file_names in os.walk(root):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not _ignore_directory_name(name)
        )
        current = Path(current_root)
        for name in sorted(file_names):
            path = current / name
            if _is_scannable_file(path):
                yield path


def _git_visible_files(root: Path) -> list[Path] | None:
    if not (root / ".git").exists():
        return None
    try:
        import subprocess

        completed = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            cwd=root,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    files: list[Path] = []
    for encoded in completed.stdout.split(b"\0"):
        if not encoded:
            continue
        relative = Path(os.fsdecode(encoded))
        if relative.is_absolute() or ".." in relative.parts:
            continue
        path = root / relative
        if _is_scannable_file(path):
            files.append(path)
    return sorted(files, key=lambda path: os.fspath(path.relative_to(root)))


def _ignore_directory_name(name: str) -> bool:
    return (
        name in _IGNORED_DIRECTORIES
        or name.startswith(".cache")
        or name.startswith("build-")
        or name.startswith("cmake-build-")
    )


def _is_scannable_file(path: Path) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        return not (
            path.stat().st_size > _TEXT_FILE_LIMIT
            and path.suffix.lower()
            not in _BINARY_EXTENSION_SUFFIXES | _NATIVE_CUDA_SUFFIXES
        )
    except OSError:
        return False


def _read_text(path: Path) -> str | None:
    if path.stat().st_size > _TEXT_FILE_LIMIT:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _python_imports(tree: ast.AST) -> Counter[str]:
    imports: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.name.split(".", 1)[0]] += 1
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports[node.module.split(".", 1)[0]] += 1
    return imports


def _python_call_markers(tree: ast.AST) -> Counter[str]:
    aliases = _python_aliases(tree)
    markers: Counter[str] = Counter()
    known = {
        "torch.compile",
        "torch.jit.script",
        "torch.jit.trace",
        "torch.utils.cpp_extension.load",
        "torch.utils.cpp_extension.load_inline",
        "torch.utils.cpp_extension.CUDAExtension",
        "triton.jit",
        "triton.autotune",
        "cupy.RawKernel",
        "cupy.RawModule",
        "numba.cuda.jit",
        "torch.library.custom_op",
        "torch.library.Library",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _resolve_python_name(_qualified_ast_name(node.func), aliases)
        if not name:
            continue
        if name in known:
            markers[name] += 1
        elif name.endswith((".impl", ".define")) and name.startswith(
            "torch.library"
        ):
            markers["torch.library.custom_operator_registration"] += 1
        elif "nvrtc" in name.lower() or name in {
            "cuda.core.experimental.Program",
            "cuda.bindings.nvrtc.nvrtcCompileProgram",
        }:
            markers["runtime_generated_cuda"] += 1
        elif name.endswith(".cuda"):
            markers["tensor_or_module.cuda"] += 1
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            name = _resolve_python_name(_qualified_ast_name(target), aliases)
            if name in known:
                markers[name] += 1
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value
        if "__global__" in value or (
            ".version" in value and ".target" in value and ".entry" in value
        ):
            markers["embedded_cuda_or_ptx_source"] += 1
    return markers


def _python_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                local = item.asname or item.name.split(".", 1)[0]
                aliases[local] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                if item.name == "*":
                    continue
                local = item.asname or item.name
                aliases[local] = f"{node.module}.{item.name}"
    # Resolve simple function aliases such as ``compile_fn = torch.compile``.
    for _ in range(2):
        for node in ast.walk(tree):
            if (
                not isinstance(node, (ast.Assign, ast.AnnAssign))
                or isinstance(node, ast.Assign)
                and len(node.targets) != 1
            ):
                continue
            target = (
                node.targets[0]
                if isinstance(node, ast.Assign)
                else node.target
            )
            value = node.value
            if not isinstance(target, ast.Name) or value is None:
                continue
            qualified = _qualified_ast_name(value)
            if qualified:
                aliases[target.id] = _resolve_python_name(qualified, aliases)
    return aliases


def _resolve_python_name(name: str, aliases: Mapping[str, str]) -> str:
    if not name:
        return ""
    head, separator, tail = name.partition(".")
    resolved = aliases.get(head, head)
    return f"{resolved}.{tail}" if separator else resolved


def _qualified_ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_ast_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _has_main_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        try:
            rendered = ast.unparse(node.test)
        except Exception:
            continue
        if "__name__" in rendered and "__main__" in rendered:
            return True
    return False


def _dependency_names(root: Path, files: Sequence[Path]) -> set[str]:
    names: set[str] = set()
    for file in files:
        relative = file.relative_to(root)
        lower_name = file.name.lower()
        if lower_name.startswith("requirements") and file.suffix.lower() in {
            ".txt",
            ".in",
        }:
            text = _read_text(file)
            if text:
                for line in text.splitlines():
                    dependency = _requirement_name(line)
                    if dependency:
                        names.add(dependency)
        elif relative == Path("pyproject.toml"):
            names.update(_pyproject_dependencies(file))
        elif lower_name in {"environment.yml", "environment.yaml"}:
            text = _read_text(file)
            if text:
                names.update(_environment_dependencies(text))
    return names


def _requirement_name(line: str) -> str | None:
    value = line.strip()
    if not value or value.startswith(("#", "-", "git+", "http:", "https:")):
        return None
    value = value.split(";", 1)[0].strip()
    match = re.match(r"([A-Za-z0-9_.-]+)", value)
    return match.group(1) if match else None


def _pyproject_dependencies(path: Path) -> set[str]:
    try:
        import tomllib

        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (ImportError, OSError, ValueError):
        return set()
    project = payload.get("project")
    if not isinstance(project, Mapping):
        return set()
    values: list[Any] = list(project.get("dependencies") or [])
    optional = project.get("optional-dependencies")
    if isinstance(optional, Mapping):
        for dependencies in optional.values():
            if isinstance(dependencies, list):
                values.extend(dependencies)
    return {
        name
        for value in values
        if isinstance(value, str)
        for name in [_requirement_name(value)]
        if name
    }


def _environment_dependencies(text: str) -> set[str]:
    names = set()
    for line in text.splitlines():
        match = re.match(r"\s*-\s*([A-Za-z0-9_.-]+)", line)
        if match and match.group(1) not in {"pip", "python"}:
            names.add(match.group(1))
    return names


def _normalize_dependency_name(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _pyproject_entrypoints(root: Path) -> set[str]:
    path = root / "pyproject.toml"
    if not path.is_file():
        return set()
    try:
        import tomllib

        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (ImportError, OSError, ValueError):
        return set()
    scripts = (payload.get("project") or {}).get("scripts")
    if not isinstance(scripts, Mapping):
        return set()
    return {f"module:{value}" for value in scripts.values() if isinstance(value, str)}


def _validate_entrypoints(
    root: Path,
    entrypoints: Sequence[str | os.PathLike[str]],
) -> list[str]:
    selected = []
    for raw in entrypoints:
        path = (root / Path(raw)).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise RepositoryAnalysisError(
                f"entrypoint is outside repository: {raw}"
            ) from exc
        if not path.is_file():
            raise RepositoryAnalysisError(f"entrypoint not found: {relative}")
        selected.append(str(relative))
    return sorted(set(selected))


def _configuration_markers(files: Sequence[Path]) -> set[str]:
    markers: set[str] = set()
    for file in files:
        lower = file.name.lower()
        if "deepspeed" in lower:
            markers.add("deepspeed_config")
        if "fsdp" in lower:
            markers.add("fsdp_config")
        if lower in {"accelerate.yaml", "default_config.yaml"}:
            markers.add("accelerate_config")
        if lower.endswith((".json", ".yaml", ".yml")):
            try:
                text = _read_text(file)
            except OSError:
                text = None
            if text and "zero_optimization" in text:
                markers.add("deepspeed_zero")
            if text and any(
                token in text
                for token in (
                    "tensor_parallel",
                    "pipeline_parallel",
                    "expert_parallel",
                )
            ):
                markers.add("model_parallel")
    return markers


def _build_system_markers(
    files: Sequence[Path],
    *,
    root: Path,
) -> dict[str, Any]:
    patterns = {
        "cmake_cuda_language": re.compile(
            r"(?:enable_language\s*\(\s*CUDA|project\s*\([^)]*\bCUDA\b)",
            re.IGNORECASE | re.DOTALL,
        ),
        "cmake_cuda_toolkit": re.compile(
            r"(?:find_package\s*\(\s*CUDAToolkit|CUDA::)",
            re.IGNORECASE,
        ),
        "nvcc_compilation": re.compile(
            r"(?:\bnvcc\b|CMAKE_CUDA_COMPILER)",
            re.IGNORECASE,
        ),
        "setuptools_cuda_extension": re.compile(
            r"(?:CUDAExtension|BuildExtension\.with_options)",
        ),
        "runtime_cpp_extension": re.compile(
            r"(?:load_inline|cpp_extension\.load\s*\()",
        ),
        "runtime_nvrtc_compilation": re.compile(
            r"(?:nvrtcCreateProgram|nvrtcCompileProgram|cuModuleLoadDataEx?)",
        ),
        "custom_operator_registration": re.compile(
            r"(?:TORCH_LIBRARY|torch\.library\.(?:custom_op|Library))",
        ),
        "triton_kernel_decorator": re.compile(
            r"@(?:triton\.)?(?:jit|autotune)\b",
        ),
    }
    markers: Counter[str] = Counter()
    evidence: dict[str, list[str]] = defaultdict(list)
    candidate_names = {
        "cmakelists.txt",
        "setup.py",
        "pyproject.toml",
        "build",
        "build.bazel",
        "workspace",
        "meson.build",
    }
    for file in files:
        if (
            file.name.lower() not in candidate_names
            and file.suffix.lower()
            not in {".py", ".cpp", ".cc", ".cxx", ".cu", ".cuh"}
        ):
            continue
        try:
            text = _read_text(file)
        except OSError:
            continue
        if not text:
            continue
        relative = str(file.relative_to(root))
        for name, pattern in patterns.items():
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            markers[name] += len(matches)
            evidence[name].append(relative)
    return {
        "markers": dict(sorted(markers.items())),
        "evidence": {
            name: sorted(set(paths))
            for name, paths in sorted(evidence.items())
        },
    }


def _kernel_analyses(
    files: Sequence[Path],
    *,
    root: Path,
) -> list[dict[str, Any]]:
    analyses = []
    for file in files:
        if file.suffix.lower() not in {".ptx", ".sass", ".disasm", ".cu", ".cuh"}:
            continue
        try:
            report = analyze_kernel_file(file)
        except (FileNotFoundError, KernelAnalysisError, OSError, ValueError) as exc:
            analyses.append(
                {
                    "path": str(file.relative_to(root)),
                    "status": "analysis_failed",
                    "error": str(exc),
                }
            )
            continue
        analyses.append(
            {
                "path": str(file.relative_to(root)),
                "status": "analyzed",
                "kind": report["kind"],
                "analysis": report["analysis"],
            }
        )
    return analyses


def _build_findings(
    *,
    native_cuda_files: Sequence[Path],
    binary_extensions: Sequence[Path],
    frameworks: Sequence[str],
    call_markers: Mapping[str, int],
    config_markers: set[str],
    build_system_markers: Mapping[str, int],
    syntax_errors: Sequence[Mapping[str, Any]],
    entrypoints: Sequence[str],
    root: Path,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    def add(
        code: str,
        severity: str,
        detail: str,
        evidence: Sequence[str] = (),
    ) -> None:
        findings.append(
            {
                "code": code,
                "severity": severity,
                "detail": detail,
                "evidence": list(evidence),
            }
        )

    if native_cuda_files:
        add(
            "native_cuda_sources",
            "requires_real_gpu_or_hybrid",
            "The repository contains CUDA source or binary kernel files; FakeCUDA cannot execute arbitrary kernels.",
            [str(path.relative_to(root)) for path in native_cuda_files[:20]],
        )
    if binary_extensions:
        add(
            "binary_extensions",
            "requires_real_gpu_or_hybrid",
            "Prebuilt native extensions require ABI and CUDA compatibility checks.",
            [str(path.relative_to(root)) for path in binary_extensions[:20]],
        )
    for framework, detail in (
        ("triton", "Triton kernels require real GPU or Hybrid validation."),
        (
            "bitsandbytes",
            "bitsandbytes uses native CUDA kernels and allocator behavior.",
        ),
        (
            "flash_attention",
            "Flash Attention extensions require a matching CUDA build.",
        ),
        ("xformers", "xFormers may dispatch custom CUDA kernels."),
        ("apex", "Apex commonly depends on compiled CUDA extensions."),
    ):
        if framework in frameworks:
            add(
                f"framework_{framework}",
                "requires_real_gpu_or_hybrid",
                detail,
                [framework],
            )
    if any(
        marker.startswith("torch.utils.cpp_extension")
        for marker in call_markers
    ) or any(
        marker in build_system_markers
        for marker in (
            "setuptools_cuda_extension",
            "runtime_cpp_extension",
            "cmake_cuda_language",
            "nvcc_compilation",
        )
    ):
        add(
            "runtime_cuda_extension_build",
            "requires_real_gpu_or_hybrid",
            "The repository builds or configures a CUDA extension.",
            sorted(
                set(call_markers)
                | {
                    marker
                    for marker in build_system_markers
                    if marker
                    in {
                        "setuptools_cuda_extension",
                        "runtime_cpp_extension",
                        "cmake_cuda_language",
                        "nvcc_compilation",
                    }
                }
            ),
        )
    generated_markers = sorted(
        marker
        for marker in set(call_markers) | set(build_system_markers)
        if marker
        in {
            "triton.jit",
            "triton.autotune",
            "cupy.RawKernel",
            "cupy.RawModule",
            "numba.cuda.jit",
            "runtime_generated_cuda",
            "embedded_cuda_or_ptx_source",
            "runtime_nvrtc_compilation",
            "triton_kernel_decorator",
        }
    )
    if generated_markers:
        add(
            "generated_kernel_path",
            "requires_real_gpu_or_hybrid",
            "The repository contains decorated, embedded, or runtime-generated GPU kernels.",
            generated_markers,
        )
    custom_operator_markers = sorted(
        marker
        for marker in set(call_markers) | set(build_system_markers)
        if marker
        in {
            "torch.library.custom_op",
            "torch.library.Library",
            "torch.library.custom_operator_registration",
            "custom_operator_registration",
        }
    )
    if custom_operator_markers:
        add(
            "custom_operator_registration",
            "requires_targeted_validation",
            "Custom operators require an operator profile or a matching real-stack measurement.",
            custom_operator_markers,
        )
    if "torch.compile" in call_markers:
        add(
            "torch_compile",
            "requires_targeted_validation",
            "torch.compile may select Inductor or Triton code generation outside FakeCUDA coverage.",
            ["torch.compile"],
        )
    if "deepspeed" in frameworks or "deepspeed_config" in config_markers:
        add(
            "deepspeed",
            "requires_targeted_validation",
            "DeepSpeed support depends on ZeRO stage, offload, JIT operators, and version.",
            sorted(
                marker
                for marker in config_markers
                if marker.startswith("deepspeed")
            ),
        )
    if any(
        marker in config_markers
        for marker in ("fsdp_config", "model_parallel")
    ):
        add(
            "distributed_sharding",
            "requires_targeted_validation",
            "Sharding and model-parallel configuration needs a rank/precision matrix.",
            sorted(config_markers),
        )
    if syntax_errors:
        add(
            "python_parse_errors",
            "analysis_incomplete",
            "Some Python files could not be parsed.",
            [str(item["path"]) for item in syntax_errors[:20]],
        )
    if not entrypoints:
        add(
            "entrypoint_not_found",
            "analysis_incomplete",
            "No likely Python entrypoint was found; pass --entry for a runnable preflight target.",
        )
    return sorted(findings, key=lambda item: (item["severity"], item["code"]))


def _recommended_experiments(
    *,
    root: Path,
    entrypoints: Sequence[str],
    blockers: Sequence[Mapping[str, Any]],
    frameworks: Sequence[str],
) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = [
        {
            "name": "native_capability_strict",
            "purpose": "Reject recognized native calls that FakeGPU does not execute.",
            "environment": {"FAKEGPU_UNSUPPORTED_API": "error"},
        }
    ]
    if entrypoints:
        entrypoint = entrypoints[0]
        if entrypoint.startswith("module:"):
            module = entrypoint.removeprefix("module:").split(":", 1)[0]
            target = ["{python}", "-m", module]
        else:
            target = ["{python}", str(root / entrypoint)]
        experiments.append(
            {
                "name": "fakecuda_preflight",
                "purpose": "Exercise the selected repository path and capture stage memory.",
                "command": [
                    "{python}",
                    "-m",
                    "fakegpu",
                    "preflight",
                    "--runtime",
                    "fakecuda",
                    "--stage",
                    "forward",
                    "--strict",
                    "--",
                    *target,
                ],
            }
        )
    if blockers:
        experiments.append(
            {
                "name": "hybrid_or_passthrough_baseline",
                "purpose": "Validate compiled kernels and collect a real CUDA memory baseline.",
                "required": "real_gpu_and_matching_cuda_stack",
            }
        )
    if "deepspeed" in frameworks:
        experiments.append(
            {
                "name": "deepspeed_matrix",
                "purpose": "Test each selected ZeRO stage, precision, offload mode, and world size separately.",
            }
        )
    return experiments


def _git_metadata(root: Path) -> dict[str, Any]:
    git_dir = root / ".git"
    if not git_dir.exists():
        return {"git_commit": None, "git_dirty": None}
    try:
        import subprocess

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}


def _print_repository_analysis(report: Mapping[str, Any]) -> None:
    repository = report["repository"]
    readiness = report["readiness"]
    inventory = report["inventory"]
    print("FakeGPU repository analysis")
    print(f"  repository: {repository['path']}")
    print(f"  verdict: {readiness['verdict']}")
    print(
        f"  files: {inventory['file_count']}  "
        f"python: {inventory['python_file_count']}  "
        f"CUDA sources: {inventory['native_cuda_file_count']}"
    )
    print(f"  frameworks: {', '.join(report['frameworks']) or 'none detected'}")
    print(f"  entrypoints: {', '.join(report['entrypoints']) or 'none detected'}")
    if report["findings"]:
        print("\nFindings:")
        for finding in report["findings"]:
            print(
                f"  [{finding['severity']}] {finding['code']}: "
                f"{finding['detail']}"
            )


__all__ = [
    "RepositoryAnalysisError",
    "SCHEMA_VERSION",
    "analyze_repository",
]
