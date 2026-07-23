"""B3-1: Runtime overhead benchmark.

Measures the per-call overhead introduced by FakeGPU's torch_patch
redirect layer compared to the ordinary CPU tensor path.  The test
**asserts** version-calibrated relative bounds for compute-heavy ops and
absolute bounds for redirect calls.

Run standalone for readable output::

    python test/test_benchmark_overhead.py
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"

import unittest

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WARMUP = 100
_TENSOR_SIZE = 256


def _torch_minor_version() -> tuple[int, int]:
    version = torch.__version__.split("+", 1)[0]
    major, minor, *_rest = version.split(".")
    return (int(major), int(minor))


_STRICT_REDIRECT_BOUNDS = _torch_minor_version() >= (2, 9)
_FORWARD_PASS_MAX_RATIO = 8.0 if _torch_minor_version() >= (2, 11) else 4.0
_MATMUL_MAX_RATIO = 3.5
_TRAIN_STEP_MAX_RATIO = 5.0


def _bench(fn, n: int = 1000, warmup: int = _WARMUP) -> float:
    """Return the median batched wall-clock time per call in **microseconds**."""
    for _ in range(warmup):
        fn()

    # A single timing window is vulnerable to scheduler pauses when this file
    # runs near the end of the full suite.  Three equal-sized windows preserve
    # the requested total call count while discarding one timing outlier.
    quotient, remainder = divmod(n, 3)
    batch_sizes = [
        quotient + (index < remainder)
        for index in range(3)
        if quotient + (index < remainder) > 0
    ]
    timings: list[float] = []
    for batch_size in batch_sizes:
        t0 = time.perf_counter()
        for _ in range(batch_size):
            fn()
        timings.append((time.perf_counter() - t0) / batch_size * 1e6)
    return statistics.median(timings)


# ---------------------------------------------------------------------------
# Baseline measurements (ordinary CPU tensor path)
# ---------------------------------------------------------------------------
# The reference is captured immediately before the FakeGPU timings in
# setUpClass.  Measuring it during pytest collection can leave a multi-minute
# gap between the two samples in the full suite, making CPU frequency and
# scheduler changes look like redirect overhead.

_baseline: dict[str, float] = {}


def _measure_baseline():
    a = torch.randn(_TENSOR_SIZE, _TENSOR_SIZE)
    b = torch.randn(_TENSOR_SIZE, _TENSOR_SIZE)
    model = torch.nn.Sequential(
        torch.nn.Linear(_TENSOR_SIZE, _TENSOR_SIZE * 2),
        torch.nn.ReLU(),
        torch.nn.Linear(_TENSOR_SIZE * 2, 10),
    )
    x = torch.randn(32, _TENSOR_SIZE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    y_labels = torch.randint(0, 10, (32,))

    _baseline["create"] = _bench(
        lambda: torch.randn(_TENSOR_SIZE, _TENSOR_SIZE), n=5000
    )
    _baseline["matmul"] = _bench(lambda: torch.matmul(a, b))
    _baseline["to_cpu"] = _bench(lambda: a.to("cpu"), n=10000)
    _baseline["forward"] = _bench(lambda: model(x))

    def _train_step():
        out = model(x)
        loss = criterion(out, y_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    _baseline["train_step"] = _bench(_train_step, n=500)


# Now patch ---------------------------------------------------------------
import fakegpu  # noqa: E402

fakegpu.patch_torch()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOverheadBounds(unittest.TestCase):
    """Assert that FakeGPU overhead stays within tolerable bounds."""

    @classmethod
    def setUpClass(cls):
        # Keep the CPU and FakeGPU measurements adjacent so they see the same
        # process load. CPU tensors still take the ordinary, non-FakeCudaTensor
        # path after patching.
        _measure_baseline()

        # Pre-compute FakeGPU timings once for all assertions.
        a = torch.randn(_TENSOR_SIZE, _TENSOR_SIZE, device="cuda")
        b = torch.randn(_TENSOR_SIZE, _TENSOR_SIZE, device="cuda")
        model = torch.nn.Sequential(
            torch.nn.Linear(_TENSOR_SIZE, _TENSOR_SIZE * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(_TENSOR_SIZE * 2, 10),
        ).cuda()
        x = torch.randn(32, _TENSOR_SIZE, device="cuda")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        y_labels = torch.randint(0, 10, (32,), device="cuda")
        to_source = torch.randn(_TENSOR_SIZE, _TENSOR_SIZE)

        cls.fg: dict[str, float] = {}
        cls.fg["create"] = _bench(
            lambda: torch.randn(_TENSOR_SIZE, _TENSOR_SIZE, device="cuda"), n=5000
        )
        cls.fg["matmul"] = _bench(lambda: torch.matmul(a, b))
        cls.fg["to_cuda"] = _bench(
            lambda: to_source.to("cuda"), n=5000
        )
        cls.fg["forward"] = _bench(lambda: model(x))

        def _train_step():
            out = model(x)
            loss = criterion(out, y_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cls.fg["train_step"] = _bench(_train_step, n=500)
        cls.fg["device_ctx"] = _bench(
            lambda: torch.cuda.device(0).__enter__() or None, n=10000
        )
        cls.fg["mem_query"] = _bench(torch.cuda.memory_allocated, n=50000)

    # -- Relative overhead tests (< 50 % regression) ----------------------

    def _assert_relative(self, key_fg: str, key_base: str, max_ratio: float = 1.5):
        fg = self.fg[key_fg]
        base = _baseline[key_base]
        ratio = fg / base if base > 0 else 0
        self.assertLessEqual(
            ratio,
            max_ratio,
            f"FakeGPU {key_fg} ({fg:.1f} µs) > {max_ratio}x baseline "
            f"{key_base} ({base:.1f} µs) — ratio {ratio:.2f}x",
        )

    def test_tensor_creation_overhead(self):
        """Tensor creation overhead < 3x vs baseline (FakeCudaTensor __torch_function__ dispatch)."""
        self._assert_relative("create", "create", max_ratio=3.0)

    def test_matmul_overhead(self):
        """Matmul overhead stays bounded with allocator accounting enabled."""
        self._assert_relative("matmul", "matmul", max_ratio=_MATMUL_MAX_RATIO)

    def test_forward_pass_overhead(self):
        """Forward pass overhead stays within the version-calibrated bound."""
        self._assert_relative("forward", "forward", max_ratio=_FORWARD_PASS_MAX_RATIO)

    def test_train_step_overhead(self):
        """Full training step overhead < 5x vs baseline (FakeCudaTensor dispatch)."""
        if not _STRICT_REDIRECT_BOUNDS:
            self.skipTest(
                "Strict train-step overhead bound is calibrated for torch >= 2.9; "
                "older minors are functionally compatible but measurably slower."
            )
        self._assert_relative("train_step", "train_step", max_ratio=_TRAIN_STEP_MAX_RATIO)

    # -- Absolute overhead tests -------------------------------------------

    def test_to_cuda_redirect_fast(self):
        """Tensor.to('cuda') redirect < 100 µs."""
        if not _STRICT_REDIRECT_BOUNDS:
            self.skipTest(
                "Strict Tensor.to('cuda') redirect bound is calibrated for torch >= 2.9; "
                "older minors are functionally compatible but slower."
            )
        self.assertLess(
            self.fg["to_cuda"],
            _baseline["to_cpu"] + 100,
            f".to('cuda') ({self.fg['to_cuda']:.1f} µs) adds > 100 µs "
            f"over .to('cpu') ({_baseline['to_cpu']:.1f} µs)",
        )

    def test_device_context_fast(self):
        """torch.cuda.device(0) context switch < 10 µs."""
        self.assertLess(self.fg["device_ctx"], 10.0)

    def test_memory_query_fast(self):
        """memory_allocated() < 5 µs."""
        self.assertLess(self.fg["mem_query"], 5.0)


class TestOverheadReport(unittest.TestCase):
    """Print a human-readable summary (always passes)."""

    def test_print_report(self):
        fg = TestOverheadBounds.fg
        print("\n")
        print("=" * 62)
        print("  FakeGPU Runtime Overhead Report")
        print("=" * 62)
        hdr = f"{'Operation':<32} {'Baseline':>10} {'FakeGPU':>10} {'Ratio':>8}"
        print(hdr)
        print("-" * 62)
        pairs = [
            ("Tensor create 256x256", "create", "create"),
            ("Matmul 256x256", "matmul", "matmul"),
            ("Forward pass (32x256→10)", "forward", "forward"),
            ("Full train step", "train_step", "train_step"),
        ]
        for label, base_key, fg_key in pairs:
            base = _baseline[base_key]
            fgv = fg[fg_key]
            ratio = fgv / base if base > 0 else 0
            print(f"  {label:<30} {base:>8.1f}µs {fgv:>8.1f}µs {ratio:>7.2f}x")
        print("-" * 62)
        print(f"  {'Tensor.to(cuda) redirect':<30} {'N/A':>10} {fg['to_cuda']:>8.1f}µs")
        print(f"  {'device ctx switch':<30} {'N/A':>10} {fg['device_ctx']:>8.1f}µs")
        print(f"  {'memory_allocated()':<30} {'N/A':>10} {fg['mem_query']:>8.2f}µs")
        print("=" * 62)


if __name__ == "__main__":
    unittest.main(verbosity=2)
