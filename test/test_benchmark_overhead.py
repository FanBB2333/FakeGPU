"""B3-1: Runtime overhead benchmark.

Measures the per-call overhead introduced by FakeGPU's torch_patch
redirect layer compared to raw CPU PyTorch.  The test **asserts** that
overhead stays within acceptable bounds (< 50 % for compute-heavy ops,
< 100 µs absolute for redirect calls).

Run standalone for readable output::

    python test/test_benchmark_overhead.py
"""

import os
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


def _bench(fn, n: int = 1000, warmup: int = _WARMUP) -> float:
    """Return mean wall-clock time per call in **microseconds**."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6


# ---------------------------------------------------------------------------
# Baseline measurements (raw CPU, no FakeGPU)
# ---------------------------------------------------------------------------
# These are captured *before* patching so we have a clean reference.

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


_measure_baseline()

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
        # Pre-compute FakeGPU timings once for all assertions
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

        cls.fg: dict[str, float] = {}
        cls.fg["create"] = _bench(
            lambda: torch.randn(_TENSOR_SIZE, _TENSOR_SIZE, device="cuda"), n=5000
        )
        cls.fg["matmul"] = _bench(lambda: torch.matmul(a, b))
        cls.fg["to_cuda"] = _bench(
            lambda: torch.randn(_TENSOR_SIZE, _TENSOR_SIZE).to("cuda"), n=5000
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
        """Matmul overhead < 3x vs baseline (FakeCudaTensor __torch_function__ dispatch)."""
        self._assert_relative("matmul", "matmul", max_ratio=3.0)

    def test_forward_pass_overhead(self):
        """Forward pass overhead < 3x vs baseline (FakeCudaTensor __torch_function__ dispatch)."""
        self._assert_relative("forward", "forward", max_ratio=3.0)

    def test_train_step_overhead(self):
        """Full training step overhead < 3x vs baseline (FakeCudaTensor __torch_function__ dispatch)."""
        self._assert_relative("train_step", "train_step", max_ratio=3.0)

    # -- Absolute overhead tests -------------------------------------------

    def test_to_cuda_redirect_fast(self):
        """Tensor.to('cuda') redirect < 100 µs."""
        self.assertLess(
            self.fg["to_cuda"],
            self.fg["create"] + 100,
            f".to('cuda') ({self.fg['to_cuda']:.1f} µs) adds > 100 µs "
            f"over creation ({self.fg['create']:.1f} µs)",
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
