"""P7-2 & P7-4: Device propagation chains and multi-thread safety.

P7-2: Validates that clone/contiguous/detach chains work correctly
under FakeGPU (these operations are patched to propagate through the
redirect layer without error).

P7-4: Validates that concurrent FakeGPU operations from multiple
threads do not crash or produce corrupt results.
"""

import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("FAKEGPU_TERMINAL_REPORT", "0")

import unittest

import fakegpu

fakegpu.patch_torch()

import torch


# ======================================================================
# P7-2: clone / contiguous / detach chains
# ======================================================================


class TestDevicePropagationChains(unittest.TestCase):
    """clone/contiguous/detach on 'cuda' tensors."""

    def test_clone_preserves_data(self):
        x = torch.randn(4, 4, device="cuda")
        c = x.clone()
        self.assertTrue(torch.equal(x, c))
        self.assertNotEqual(c.data_ptr(), x.data_ptr())

    def test_contiguous_noop_on_contiguous(self):
        x = torch.randn(4, 4, device="cuda")
        self.assertTrue(x.contiguous().is_contiguous())

    def test_contiguous_on_transpose(self):
        x = torch.randn(3, 5, device="cuda").t()
        self.assertFalse(x.is_contiguous())
        y = x.contiguous()
        self.assertTrue(y.is_contiguous())
        self.assertEqual(y.shape, (5, 3))

    def test_detach_stops_grad(self):
        x = torch.randn(3, 3, device="cuda", requires_grad=True)
        d = x.detach()
        self.assertFalse(d.requires_grad)

    def test_chain_clone_detach_contiguous(self):
        x = torch.randn(3, 4, device="cuda", requires_grad=True).t()
        result = x.clone().detach().contiguous()
        self.assertTrue(result.is_contiguous())
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.shape, (4, 3))

    def test_clone_to_cuda(self):
        x = torch.randn(3, 3)
        y = x.clone().to("cuda")
        self.assertTrue(torch.equal(x, y))

    def test_detach_clone_preserves_values(self):
        x = torch.randn(3, 3, device="cuda", requires_grad=True)
        y = x.detach().clone()
        self.assertTrue(torch.equal(x.detach(), y))
        self.assertFalse(y.requires_grad)

    def test_contiguous_view(self):
        x = torch.randn(2, 3, 4, device="cuda").permute(2, 0, 1)
        y = x.contiguous().view(-1)
        self.assertEqual(y.shape, torch.Size([24]))


# ======================================================================
# P7-4: Multi-thread safety
# ======================================================================


class TestMultiThreadSafety(unittest.TestCase):
    """Concurrent FakeGPU operations must not crash."""

    def _run_threads(self, target, n_threads=4, **kwargs):
        errors: list[str] = []

        def _wrapper():
            try:
                target()
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")

        threads = [threading.Thread(target=_wrapper) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(errors, [], f"Thread errors: {errors}")

    def test_concurrent_tensor_creation(self):
        """4 threads each create 100 tensors on different devices."""

        def work():
            for _ in range(100):
                x = torch.randn(32, 32, device="cuda:0")
                _ = x + 1

        self._run_threads(work)

    def test_concurrent_memory_queries(self):
        """8 threads query memory stats concurrently."""

        def work():
            for _ in range(500):
                torch.cuda.memory_allocated()
                torch.cuda.memory_reserved()

        self._run_threads(work, n_threads=8)

    def test_concurrent_device_switching(self):
        """4 threads rapidly switch devices."""

        def work():
            for i in range(200):
                torch.cuda.set_device(i % 8)
                torch.cuda.current_device()

        self._run_threads(work)

    def test_concurrent_forward_passes(self):
        """4 threads run forward passes on the same model."""
        model = torch.nn.Linear(10, 2).cuda()

        def work():
            for _ in range(50):
                x = torch.randn(8, 10, device="cuda")
                _ = model(x)

        self._run_threads(work)


if __name__ == "__main__":
    unittest.main()
