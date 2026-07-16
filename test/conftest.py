"""Pytest collection policy for standalone integration scripts.

These files are invoked directly by ``ftest`` or shell runners. They expose a
``main()`` function (or parameterized helper functions), not pytest tests, and
some require a platform-specific native library during module import.
"""

collect_ignore = [
    "test_comparison.py",
    "test_cudart_basic.py",
    "test_cuda_direct.py",
    "test_privateuse1_basic.py",
    "test_privateuse1_device_state.py",
    "test_privateuse1_load_aliasing.py",
    "test_privateuse1_module_aliasing.py",
    "test_privateuse1_ops.py",
    "test_privateuse1_runtime.py",
    "test_privateuse1_training.py",
]
