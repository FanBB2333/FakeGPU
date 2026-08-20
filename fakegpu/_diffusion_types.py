"""The diffusion-estimate error type and the value checks that raise it."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class DiffusionEstimateError(ValueError):
    pass


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DiffusionEstimateError("expected an object")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise DiffusionEstimateError(
            f"{name} must be a positive integer"
        )
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise DiffusionEstimateError(
            f"{name} must be a non-negative integer"
        )
    return value


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise DiffusionEstimateError(
            f"{name} must be a finite non-negative number"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DiffusionEstimateError(
            f"{name} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(number) or number < 0:
        raise DiffusionEstimateError(
            f"{name} must be a finite non-negative number"
        )
    return number


def _positive_integer_list(
    value: Any,
    name: str,
) -> list[int]:
    if not isinstance(value, list) or not value:
        raise DiffusionEstimateError(
            f"{name} must be a non-empty integer array"
        )
    return [
        _positive_integer(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    ]


def _nonnegative_integer_list(
    value: Any,
    name: str,
) -> list[int]:
    if not isinstance(value, list) or not value:
        raise DiffusionEstimateError(
            f"{name} must be a non-empty integer array"
        )
    return [
        _nonnegative_integer(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    ]


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor
