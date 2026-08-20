from __future__ import annotations

from types import SimpleNamespace

from fakegpu._cross_device import _wrap_tensor_binary_op


def test_binary_wrapper_preserves_reflected_operator_protocol() -> None:
    class FakeTensor:
        device_index = 0

        def __add__(self, other):
            return wrapped(self, other)

    class CustomValue:
        def __radd__(self, other):
            return ("reflected", other)

    def original(self, other):
        return NotImplemented

    wrapped = _wrap_tensor_binary_op(
        original,
        "__add__",
        SimpleNamespace(Tensor=FakeTensor),
    )

    result = FakeTensor() + CustomValue()
    assert result[0] == "reflected"
