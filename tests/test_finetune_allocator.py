from copy import deepcopy
from types import SimpleNamespace

import pytest

from fakegpu._finetune_allocator import (
    ALLOCATOR_VARIABLES,
    capture_memory_allocator,
    validate_memory_allocator,
)


def _identity():
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_allocator_backend=lambda: "native",
            get_per_process_memory_fraction=lambda device: 0.5,
        )
    )
    return capture_memory_allocator(torch, environ={})


def test_capture_only_reads_backend_fraction_and_explicit_environment():
    calls = []

    def fraction(device):
        calls.append(device)
        return 0.5

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_allocator_backend=lambda: "native",
            get_per_process_memory_fraction=fraction,
        )
    )
    environment = {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    value = capture_memory_allocator(torch, environ=environment)
    assert calls == [0]
    assert value["backend"] == "native"
    assert value["per_process_memory_fraction"] == 0.5
    assert value["environment"] == {
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_CUDA_ALLOC_CONF": None,
        "PYTORCH_NO_CUDA_MEMORY_CACHING": None,
    }
    assert environment == {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    before = deepcopy(value)
    validate_memory_allocator(value)
    assert value == before


@pytest.mark.parametrize(
    "value", [0, -1, 1.01, True, None, "0.5", float("nan"), float("inf")]
)
def test_invalid_fraction_is_rejected(value):
    allocator = _identity()
    allocator["per_process_memory_fraction"] = value
    with pytest.raises(ValueError, match="fraction"):
        validate_memory_allocator(allocator)


@pytest.mark.parametrize(
    "field", ["schema_version", "backend", "environment", "per_process_memory_fraction"]
)
def test_missing_allocator_fields_are_not_silently_defaulted(field):
    value = _identity()
    del value[field]
    with pytest.raises(ValueError):
        validate_memory_allocator(value)


@pytest.mark.parametrize("field", ALLOCATOR_VARIABLES)
def test_missing_environment_values_are_not_defaulted(field):
    value = _identity()
    del value["environment"][field]
    with pytest.raises(ValueError, match="environment"):
        validate_memory_allocator(value)


def test_environment_values_and_unknown_fields_are_rejected():
    value = _identity()
    value["environment"]["PYTORCH_ALLOC_CONF"] = 0
    with pytest.raises(ValueError, match="environment"):
        validate_memory_allocator(value)
    value = _identity()
    value["extra"] = True
    with pytest.raises(ValueError, match="identity"):
        validate_memory_allocator(value)
