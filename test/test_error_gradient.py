"""Error simulation: gradient computation errors (E7-1..E7-3).

These are pure validation tests — FakeGPU should NOT suppress
PyTorch's native gradient error messages. No implementation changes needed.
"""
import gc
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["FAKEGPU_DEVICE_COUNT"] = "2"
os.environ["FAKEGPU_PROFILES"] = "a100:2"


class TestGradientErrors(unittest.TestCase):
    torch = None

    @classmethod
    def setUpClass(cls):
        import fakegpu
        fakegpu.patch_torch()
        import torch
        cls.torch = torch

    def test_e7_1_backward_twice_without_retain(self):
        """E7-1: loss.backward() twice without retain_graph -> RuntimeError."""
        torch = self.torch
        x = torch.randn(3, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        with self.assertRaisesRegex(RuntimeError, r"backward through the graph a second time|Trying to backward"):
            y.backward()

    def test_e7_2_requires_grad_on_integer(self):
        """E7-2: requires_grad on integer tensor -> RuntimeError."""
        torch = self.torch
        with self.assertRaisesRegex(RuntimeError, r"floating point|complex dtype"):
            torch.tensor([1, 2, 3], dtype=torch.long).requires_grad_(True)

    def test_e7_3_backward_on_non_scalar(self):
        """E7-3: backward on non-scalar without gradient arg -> RuntimeError."""
        torch = self.torch
        x = torch.randn(3, requires_grad=True)
        y = x ** 2  # non-scalar
        with self.assertRaisesRegex(RuntimeError, r"grad can be implicitly created only for scalar"):
            y.backward()


if __name__ == "__main__":
    unittest.main()
