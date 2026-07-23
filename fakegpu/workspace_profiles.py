from __future__ import annotations

import fnmatch
import json
import os
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from .profile_catalog import GpuProfile, get_profile


SCHEMA_VERSION = "fakegpu.workspace_profiles.v1"
SUPPORTED_LIFETIMES = {"operator_local", "graph_phase_persistent"}
SUPPORTED_FORMULAS = {
    "fixed",
    "linear_io",
    "tiled_tensor",
}


class WorkspaceProfileError(ValueError):
    pass


def default_workspace_profile_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "workspace_profiles.json"


def load_workspace_profiles(
    paths: Sequence[str | os.PathLike[str]] | None = None,
) -> list[dict[str, Any]]:
    """Load and validate built-in plus optional workspace profile catalogs."""

    resolved = [default_workspace_profile_path()]
    env_paths = os.environ.get("FAKEGPU_WORKSPACE_PROFILE_PATHS", "")
    if env_paths:
        resolved.extend(
            Path(item).expanduser()
            for item in env_paths.split(os.pathsep)
            if item.strip()
        )
    if paths:
        resolved.extend(Path(item).expanduser() for item in paths)

    profiles: list[dict[str, Any]] = []
    seen_ids: dict[str, Path] = {}
    for path in resolved:
        path = path.resolve()
        try:
            stat = path.stat()
        except OSError as exc:
            raise WorkspaceProfileError(
                f"workspace profile catalog not found: {path}"
            ) from exc
        catalog = _load_workspace_catalog_cached(
            str(path),
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )
        for raw in catalog:
            profile = dict(raw)
            profile_id = str(profile["id"])
            previous = seen_ids.get(profile_id)
            if previous is not None:
                raise WorkspaceProfileError(
                    f"duplicate workspace profile id {profile_id!r}: "
                    f"{previous} and {path}"
                )
            seen_ids[profile_id] = path
            profiles.append(profile)
    return profiles


def match_workspace_profile(
    node: Any,
    target: str,
    *,
    target_device: Any,
    target_profile: str | GpuProfile | None = None,
    profile_paths: Sequence[str | os.PathLike[str]] | None = None,
) -> dict[str, Any] | None:
    """Return the highest-priority matching empirical workspace profile."""

    device_type = str(
        getattr(target_device, "type", str(target_device).split(":", 1)[0])
    )
    gpu_profile = _resolve_gpu_profile(target_profile)
    stack = _software_stack(gpu_profile)
    inputs = _node_input_tensors(node)
    outputs = list(_iter_tensor_leaves(getattr(node, "meta", {}).get("val")))
    signature = {
        **stack,
        "device_type": device_type,
        "operator": target,
        "input_shapes": [_shape_list(tensor) for tensor in inputs],
        "input_dtypes": [str(getattr(tensor, "dtype", "")) for tensor in inputs],
        "output_shapes": [_shape_list(tensor) for tensor in outputs],
        "output_dtypes": [str(getattr(tensor, "dtype", "")) for tensor in outputs],
    }

    matches: list[dict[str, Any]] = []
    for profile in load_workspace_profiles(profile_paths):
        if not _profile_matches(profile, signature):
            continue
        workspace_bytes, calculation = _workspace_bytes(
            profile,
            inputs=inputs,
            outputs=outputs,
        )
        matches.append(
            {
                "operator": target,
                "node": str(getattr(node, "name", "")),
                "profile": str(profile["id"]),
                "kind": str(profile.get("kind", "operator_workspace_storage")),
                "lifetime": str(profile["lifetime"]),
                "bytes": workspace_bytes,
                "confidence": str(profile.get("confidence", "external_profile")),
                "priority": int(profile.get("priority", 0)),
                "source": str(profile.get("_source", "")),
                "signature": signature,
                "calculation": calculation,
                "validated_envelope": dict(profile.get("validated_envelope") or {}),
            }
        )
    if not matches:
        return None
    matches.sort(
        key=lambda item: (
            int(item.get("priority", 0)),
            str(item.get("profile", "")),
        ),
        reverse=True,
    )
    selected = matches[0]
    selected["matched_profile_count"] = len(matches)
    selected["shadowed_profiles"] = [str(item["profile"]) for item in matches[1:]]
    return selected


def workspace_profile_summary(
    paths: Sequence[str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    profiles = load_workspace_profiles(paths)
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_count": len(profiles),
        "profile_ids": [str(profile["id"]) for profile in profiles],
        "sources": sorted({str(profile.get("_source", "")) for profile in profiles}),
    }


@lru_cache(maxsize=32)
def _load_workspace_catalog_cached(
    path_text: str,
    _mtime_ns: int,
    _size: int,
) -> tuple[dict[str, Any], ...]:
    path = Path(path_text)
    if not path.is_file():
        raise WorkspaceProfileError(f"workspace profile catalog not found: {path}")
    payload = _read_structured_file(path)
    if not isinstance(payload, Mapping):
        raise WorkspaceProfileError(f"{path}: catalog root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise WorkspaceProfileError(
            f"{path}: expected schema_version {SCHEMA_VERSION!r}"
        )
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list):
        raise WorkspaceProfileError(f"{path}: profiles must be a list")
    validated = []
    for index, raw in enumerate(raw_profiles):
        if not isinstance(raw, Mapping):
            raise WorkspaceProfileError(f"{path}: profiles[{index}] must be an object")
        profile = _validate_profile(dict(raw), path=path, index=index)
        profile["_source"] = str(path)
        validated.append(profile)
    return tuple(validated)


def _read_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise WorkspaceProfileError(f"{path}: invalid JSON: {exc}") from exc
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise WorkspaceProfileError(
                "YAML workspace profiles require PyYAML; install "
                "'fakegpu[validation]' or use JSON"
            ) from exc
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise WorkspaceProfileError(f"{path}: invalid YAML: {exc}") from exc
    raise WorkspaceProfileError(
        f"{path}: expected a .json, .yaml, or .yml workspace catalog"
    )


def _validate_profile(
    profile: dict[str, Any],
    *,
    path: Path,
    index: int,
) -> dict[str, Any]:
    prefix = f"{path}: profiles[{index}]"
    profile_id = profile.get("id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise WorkspaceProfileError(f"{prefix}: id must be a non-empty string")
    profile["id"] = profile_id.strip()

    operator = profile.get("operator")
    operator_regex = profile.get("operator_regex")
    if bool(operator) == bool(operator_regex):
        raise WorkspaceProfileError(
            f"{prefix}: set exactly one of operator or operator_regex"
        )
    if operator is not None and not isinstance(operator, str):
        raise WorkspaceProfileError(f"{prefix}: operator must be a string")
    if operator_regex is not None:
        if not isinstance(operator_regex, str):
            raise WorkspaceProfileError(f"{prefix}: operator_regex must be a string")
        try:
            re.compile(operator_regex)
        except re.error as exc:
            raise WorkspaceProfileError(
                f"{prefix}: invalid operator_regex: {exc}"
            ) from exc

    lifetime = profile.get("lifetime")
    if lifetime not in SUPPORTED_LIFETIMES:
        raise WorkspaceProfileError(
            f"{prefix}: lifetime must be one of "
            f"{', '.join(sorted(SUPPORTED_LIFETIMES))}"
        )
    match = profile.get("match", {})
    if not isinstance(match, Mapping):
        raise WorkspaceProfileError(f"{prefix}: match must be an object")
    profile["match"] = dict(match)

    formula = profile.get("formula")
    direct_bytes = profile.get("bytes")
    if (formula is None) == (direct_bytes is None):
        raise WorkspaceProfileError(f"{prefix}: set exactly one of bytes or formula")
    if direct_bytes is not None:
        if not isinstance(direct_bytes, int) or direct_bytes < 0:
            raise WorkspaceProfileError(
                f"{prefix}: bytes must be a non-negative integer"
            )
    if formula is not None:
        if not isinstance(formula, Mapping):
            raise WorkspaceProfileError(f"{prefix}: formula must be an object")
        kind = formula.get("kind")
        if kind not in SUPPORTED_FORMULAS:
            raise WorkspaceProfileError(
                f"{prefix}: formula.kind must be one of "
                f"{', '.join(sorted(SUPPORTED_FORMULAS))}"
            )
        profile["formula"] = dict(formula)
    priority = profile.get("priority", 0)
    if not isinstance(priority, int):
        raise WorkspaceProfileError(f"{prefix}: priority must be an integer")
    return profile


def _resolve_gpu_profile(
    target_profile: str | GpuProfile | None,
) -> GpuProfile | None:
    if isinstance(target_profile, GpuProfile):
        return target_profile
    profile_id = str(target_profile).strip() if target_profile else ""
    if not profile_id:
        profiles_env = os.environ.get("FAKEGPU_PROFILES", "")
        if profiles_env:
            profile_id = profiles_env.split(",", 1)[0].split(":", 1)[0].strip()
        else:
            profile_id = os.environ.get("FAKEGPU_PROFILE", "").strip()
    if not profile_id:
        return None
    try:
        return get_profile(profile_id)
    except Exception as exc:
        raise WorkspaceProfileError(
            f"unknown target GPU profile {profile_id!r}"
        ) from exc


def _software_stack(profile: GpuProfile | None) -> dict[str, Any]:
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda or "")
    except Exception:
        torch_version = ""
        cuda_version = ""
    return {
        "profile_id": profile.id if profile is not None else None,
        "architecture": profile.architecture if profile is not None else None,
        "compute_capability": (
            profile.compute_capability_text if profile is not None else None
        ),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
    }


def _profile_matches(
    profile: Mapping[str, Any],
    signature: Mapping[str, Any],
) -> bool:
    operator = str(signature["operator"])
    expected_operator = profile.get("operator")
    if expected_operator is not None and operator != str(expected_operator):
        return False
    operator_regex = profile.get("operator_regex")
    if operator_regex is not None and re.search(str(operator_regex), operator) is None:
        return False

    match = dict(profile.get("match") or {})
    scalar_lists = {
        "device_types": "device_type",
        "profile_ids": "profile_id",
        "architectures": "architecture",
        "compute_capabilities": "compute_capability",
    }
    for match_key, signature_key in scalar_lists.items():
        allowed = match.get(match_key)
        if allowed is None:
            continue
        if not isinstance(allowed, list) or signature.get(signature_key) not in allowed:
            return False

    for match_key, signature_key in (
        ("torch_versions", "torch_version"),
        ("cuda_versions", "cuda_version"),
    ):
        patterns = match.get(match_key)
        if patterns is None:
            continue
        if not isinstance(patterns, list) or not any(
            fnmatch.fnmatch(str(signature.get(signature_key) or ""), str(pattern))
            for pattern in patterns
        ):
            return False

    expected_dtypes = match.get("input_dtypes")
    if expected_dtypes is not None and not _sequence_matches(
        expected_dtypes,
        signature.get("input_dtypes") or [],
        _scalar_pattern_matches,
    ):
        return False
    expected_shapes = match.get("input_shapes")
    if expected_shapes is not None and not _sequence_matches(
        expected_shapes,
        signature.get("input_shapes") or [],
        _shape_matches,
    ):
        return False
    return True


def _sequence_matches(
    expected: Any,
    observed: Sequence[Any],
    matcher: Any,
) -> bool:
    if not isinstance(expected, list) or len(expected) != len(observed):
        return False
    return all(matcher(want, got) for want, got in zip(expected, observed))


def _scalar_pattern_matches(expected: Any, observed: Any) -> bool:
    return expected == "*" or fnmatch.fnmatch(str(observed), str(expected))


def _shape_matches(expected: Any, observed: Any) -> bool:
    if expected == "*":
        return True
    if not isinstance(expected, list) or not isinstance(observed, list):
        return False
    if len(expected) != len(observed):
        return False
    for constraint, value in zip(expected, observed):
        value = int(value)
        if constraint == "*":
            continue
        if isinstance(constraint, int):
            if value != constraint:
                return False
            continue
        if not isinstance(constraint, Mapping):
            return False
        if "min" in constraint and value < int(constraint["min"]):
            return False
        if "max" in constraint and value > int(constraint["max"]):
            return False
        if "multiple_of" in constraint and value % int(constraint["multiple_of"]):
            return False
    return True


def _workspace_bytes(
    profile: Mapping[str, Any],
    *,
    inputs: Sequence[Any],
    outputs: Sequence[Any],
) -> tuple[int, dict[str, Any]]:
    if "bytes" in profile:
        value = int(profile["bytes"])
        return value, {"kind": "fixed", "fixed_bytes": value}

    formula = dict(profile["formula"])
    kind = str(formula["kind"])
    fixed = int(formula.get("fixed_bytes", 0) or 0)
    alignment = int(formula.get("alignment_bytes", 1) or 1)
    if fixed < 0 or alignment <= 0:
        raise WorkspaceProfileError(
            f"workspace profile {profile['id']!r} has invalid formula values"
        )

    if kind == "fixed":
        raw_bytes = fixed
        calculation = {"kind": kind, "fixed_bytes": fixed}
    elif kind == "linear_io":
        input_multiplier = float(formula.get("input_bytes_multiplier", 0))
        output_multiplier = float(formula.get("output_bytes_multiplier", 0))
        input_bytes = sum(_tensor_bytes(tensor) for tensor in inputs)
        output_bytes = sum(_tensor_bytes(tensor) for tensor in outputs)
        raw_bytes = int(
            fixed + input_multiplier * input_bytes + output_multiplier * output_bytes
        )
        calculation = {
            "kind": kind,
            "fixed_bytes": fixed,
            "input_bytes": input_bytes,
            "input_bytes_multiplier": input_multiplier,
            "output_bytes": output_bytes,
            "output_bytes_multiplier": output_multiplier,
        }
    elif kind == "tiled_tensor":
        tensor_index = int(formula.get("tensor_index", 0))
        if tensor_index < 0 or tensor_index >= len(inputs):
            raise WorkspaceProfileError(
                f"workspace profile {profile['id']!r} tensor_index "
                f"{tensor_index} is outside {len(inputs)} inputs"
            )
        tensor = inputs[tensor_index]
        dimension = int(formula.get("tile_dimension", 0))
        shape = _shape_list(tensor)
        if dimension < 0:
            dimension += len(shape)
        if dimension < 0 or dimension >= len(shape):
            raise WorkspaceProfileError(
                f"workspace profile {profile['id']!r} tile_dimension "
                f"{dimension} is outside rank {len(shape)}"
            )
        tile_size = int(formula.get("tile_size", 1))
        multiplier = float(formula.get("tensor_bytes_multiplier", 1))
        if tile_size <= 0:
            raise WorkspaceProfileError(
                f"workspace profile {profile['id']!r} tile_size must be positive"
            )
        tile_count = max(1, (shape[dimension] + tile_size - 1) // tile_size)
        tensor_bytes = _tensor_bytes(tensor)
        raw_bytes = int(fixed + tile_count * multiplier * tensor_bytes)
        calculation = {
            "kind": kind,
            "fixed_bytes": fixed,
            "tensor_index": tensor_index,
            "tensor_bytes": tensor_bytes,
            "tensor_bytes_multiplier": multiplier,
            "tile_dimension": dimension,
            "tile_size": tile_size,
            "tile_count": tile_count,
        }
    else:  # pragma: no cover - validation rejects this path
        raise WorkspaceProfileError(f"unsupported formula kind {kind!r}")

    aligned = (raw_bytes + alignment - 1) // alignment * alignment
    calculation["raw_bytes"] = raw_bytes
    calculation["alignment_bytes"] = alignment
    calculation["aligned_bytes"] = aligned
    return aligned, calculation


def _node_input_tensors(node: Any) -> list[Any]:
    tensors: list[Any] = []
    for value in (*getattr(node, "args", ()), getattr(node, "kwargs", {})):
        tensors.extend(_node_value_tensors(value))
    return tensors


def _node_value_tensors(value: Any) -> list[Any]:
    if hasattr(value, "meta"):
        return list(_iter_tensor_leaves(value.meta.get("val")))
    if isinstance(value, Mapping):
        tensors = []
        for item in value.values():
            tensors.extend(_node_value_tensors(item))
        return tensors
    if isinstance(value, (tuple, list)):
        tensors = []
        for item in value:
            tensors.extend(_node_value_tensors(item))
        return tensors
    return []


def _iter_tensor_leaves(value: Any):
    try:
        import torch
    except Exception:
        return
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensor_leaves(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensor_leaves(item)


def _shape_list(tensor: Any) -> list[int]:
    return [int(value) for value in tuple(getattr(tensor, "shape", ()))]


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())
