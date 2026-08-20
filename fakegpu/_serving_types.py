"""The serving-plan error type and the integer checks that raise it."""

from __future__ import annotations

from typing import Any


class ServingPlanError(ValueError):
    pass


def _positive_integer(value: Any, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ServingPlanError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ServingPlanError(
            f"{name} must be a non-negative integer"
        )
    return value
