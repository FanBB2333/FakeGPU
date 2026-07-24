from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .structured_io import load_mapping


SCHEMA_VERSION = "fakegpu.operator_profiles.v1"


class OperatorProfileError(ValueError):
    pass


def load_operator_profiles(
    paths: Sequence[str | Path],
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    ids: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        catalog = load_mapping(path)
        if catalog.get("schema_version") != SCHEMA_VERSION:
            raise OperatorProfileError(
                f"{path}: unsupported operator profile schema "
                f"{catalog.get('schema_version')!r}"
            )
        entries = catalog.get("profiles")
        if not isinstance(entries, list):
            raise OperatorProfileError(f"{path}: profiles must be a list")
        for index, raw in enumerate(entries):
            if not isinstance(raw, Mapping):
                raise OperatorProfileError(
                    f"{path}: profile {index} must be an object"
                )
            profile = _validate_profile(raw, source=path, index=index)
            profile_id = str(profile["id"])
            if profile_id in ids:
                raise OperatorProfileError(
                    f"duplicate operator profile id {profile_id!r}"
                )
            ids.add(profile_id)
            profiles.append(profile)
    return profiles


def match_operator_profile(
    profiles: Sequence[Mapping[str, Any]],
    *,
    operator: str,
    dtype: str | None = None,
    shape: Sequence[int] | None = None,
) -> dict[str, Any] | None:
    matches = []
    normalized_dtype = str(dtype).lower() if dtype is not None else None
    normalized_shape = tuple(int(value) for value in shape) if shape is not None else None
    for raw in profiles:
        match = raw.get("match")
        if not isinstance(match, Mapping):
            continue
        exact = match.get("operator")
        regex = match.get("operator_regex")
        if exact is not None and str(exact) != operator:
            continue
        if regex is not None and not re.search(str(regex), operator):
            continue
        configured_dtype = match.get("dtype")
        if (
            configured_dtype is not None
            and str(configured_dtype).lower() != normalized_dtype
        ):
            continue
        configured_shape = match.get("shape")
        if configured_shape is not None:
            if normalized_shape is None or tuple(configured_shape) != normalized_shape:
                continue
        specificity = sum(
            value is not None
            for value in (exact, regex, configured_dtype, configured_shape)
        )
        matches.append((int(raw.get("priority", 0)), specificity, str(raw["id"]), raw))
    if not matches:
        return None
    matches.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return dict(matches[0][3])


def evaluate_operator_profile(
    profile: Mapping[str, Any],
    *,
    numel: int | None = None,
) -> dict[str, Any]:
    model = profile.get("model")
    if not isinstance(model, Mapping):
        raise OperatorProfileError(
            f"profile {profile.get('id')!r} has no model"
        )
    elements = int(numel or 0)
    if elements < 0:
        raise OperatorProfileError("numel must be non-negative")
    output = {"profile_id": str(profile["id"])}
    for field in (
        "workspace_bytes",
        "flops",
        "memory_bytes",
        "latency_us",
        "launch_count",
    ):
        fixed = model.get(field)
        per_element = model.get(f"{field}_per_element")
        if fixed is None and per_element is None:
            continue
        value = float(fixed or 0) + float(per_element or 0) * elements
        if value < 0:
            raise OperatorProfileError(
                f"profile {profile['id']!r} produces negative {field}"
            )
        output[field] = (
            int(round(value))
            if field != "latency_us"
            else float(value)
        )
    output["fused_operators"] = list(profile.get("fused_operators") or [])
    output["confidence"] = str(profile.get("confidence") or "user_profile")
    return output


def _validate_profile(
    raw: Mapping[str, Any],
    *,
    source: Path,
    index: int,
) -> dict[str, Any]:
    profile_id = raw.get("id")
    if not isinstance(profile_id, str) or not profile_id:
        raise OperatorProfileError(
            f"{source}: profile {index} requires a non-empty id"
        )
    match = raw.get("match")
    if not isinstance(match, Mapping):
        raise OperatorProfileError(
            f"{source}: profile {profile_id!r} requires match"
        )
    exact = match.get("operator")
    regex = match.get("operator_regex")
    if exact is None and regex is None:
        raise OperatorProfileError(
            f"{source}: profile {profile_id!r} requires operator or operator_regex"
        )
    if regex is not None:
        try:
            re.compile(str(regex))
        except re.error as exc:
            raise OperatorProfileError(
                f"{source}: invalid regex in {profile_id!r}: {exc}"
            ) from exc
    shape = match.get("shape")
    if shape is not None and (
        not isinstance(shape, list)
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in shape
        )
    ):
        raise OperatorProfileError(
            f"{source}: profile {profile_id!r} has an invalid shape"
        )
    model = raw.get("model")
    if not isinstance(model, Mapping) or not model:
        raise OperatorProfileError(
            f"{source}: profile {profile_id!r} requires a non-empty model"
        )
    allowed_fields = {
        "workspace_bytes",
        "workspace_bytes_per_element",
        "flops",
        "flops_per_element",
        "memory_bytes",
        "memory_bytes_per_element",
        "latency_us",
        "latency_us_per_element",
        "launch_count",
        "launch_count_per_element",
    }
    unknown = sorted(set(model) - allowed_fields)
    if unknown:
        raise OperatorProfileError(
            f"{source}: profile {profile_id!r} has unknown model fields: "
            + ", ".join(unknown)
        )
    for key, value in model.items():
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise OperatorProfileError(
                f"{source}: profile {profile_id!r} field {key!r} must be numeric"
            ) from exc
        if parsed < 0:
            raise OperatorProfileError(
                f"{source}: profile {profile_id!r} field {key!r} must be non-negative"
            )
    profile = dict(raw)
    profile["match"] = dict(match)
    profile["model"] = dict(model)
    profile["source"] = str(source)
    return profile


__all__ = [
    "OperatorProfileError",
    "SCHEMA_VERSION",
    "evaluate_operator_profile",
    "load_operator_profiles",
    "match_operator_profile",
]
