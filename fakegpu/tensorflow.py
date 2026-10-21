"""Explicit CPU execution for TensorFlow code using GPU device scopes.

This adapter does not create TensorFlow GPU devices or simulate CUDA kernels,
GPU memory, communication, or execution time. Tensor and device discovery APIs
continue to report TensorFlow's actual devices.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Iterator


@contextmanager
def simulate(*, device_count: int = 1) -> Iterator[None]:
    """Run ``tf.device('/GPU:N')`` scopes on CPU for the configured indexes.

    TensorFlow is an optional dependency imported on entry. The replacement of
    ``tf.device`` is process-wide: enter this scope before starting worker
    threads. Graphs traced inside the scope retain their CPU placement.
    """
    if isinstance(device_count, bool) or not isinstance(device_count, int) or device_count < 1:
        raise ValueError("device_count must be a positive integer")

    import tensorflow as tf

    original_device = tf.device

    @wraps(original_device)
    def cpu_device(device_name):
        if isinstance(device_name, str):
            spec = tf.DeviceSpec.from_string(device_name)
            if spec.device_type == "GPU":
                index = spec.device_index or 0
                if not 0 <= index < device_count:
                    raise ValueError(
                        f"invalid simulated GPU index {index}; available: {device_count}"
                    )
                device_name = spec.replace(device_type="CPU", device_index=0).to_string()
        return original_device(device_name)

    tf.device = cpu_device
    try:
        with original_device("/CPU:0"):
            yield
    finally:
        tf.device = original_device
