"""Record allocator settings without changing CUDA state or resetting peaks."""

from __future__ import annotations

from collections.abc import Mapping
import math
import os
from typing import Any


SCHEMA_VERSION = "fakegpu.finetune_allocator.v1"
ALLOCATOR_VARIABLES = (
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_NO_CUDA_MEMORY_CACHING",
)


def capture_memory_allocator(torch: Any, *, environ=None) -> dict[str, Any]:
    """Read the initialized CUDA allocator's backend, cap and relevant env vars."""
    env = os.environ if environ is None else environ
    value = {
        "schema_version": SCHEMA_VERSION,
        "backend": torch.cuda.get_allocator_backend(),
        "per_process_memory_fraction": torch.cuda.get_per_process_memory_fraction(0),
        "environment": {name: env.get(name) for name in ALLOCATOR_VARIABLES},
    }
    validate_memory_allocator(value)
    return value


def validate_memory_allocator(value: Mapping[str, Any]) -> None:
    """Require an explicit allocator identity; absent fields are not defaults."""
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {"schema_version", "backend", "per_process_memory_fraction", "environment"}
        or value["schema_version"] != SCHEMA_VERSION
        or not isinstance(value["backend"], str)
        or not value["backend"].strip()
    ):
        raise ValueError("invalid native memory allocator identity")
    fraction = value["per_process_memory_fraction"]
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 < fraction <= 1
    ):
        raise ValueError(
            "native memory allocator fraction must be finite and in (0, 1]"
        )
    environment = value["environment"]
    if (
        not isinstance(environment, Mapping)
        or set(environment) != set(ALLOCATOR_VARIABLES)
        or any(
            item is not None and not isinstance(item, str)
            for item in environment.values()
        )
    ):
        raise ValueError("native memory allocator requires explicit environment values")
