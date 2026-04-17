"""Error simulation: cross-device tensor operations (E1-1..E1-5)."""
import gc
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["FAKEGPU_DEVICE_COUNT"] = "4"
os.environ["FAKEGPU_PROFILES"] = "a100:4"
os.environ["FAKEGPU_CROSS_DEVICE_CHECK"] = "1"
os.environ["FAKEGPU_MEMORY_TRACKING"] = "1"


class TestCrossDeviceErrors(unittest.TestCase):
    torch = None

    @classmethod
    def setUpClass(cls):
        import fakegpu
        fakegpu.patch_torch()
        import torch
        cls.torch = torch

    def test_e1_1_add_different_devices(self):
        """E1-1: a + b across cuda:0 and cuda:1."""
        torch = self.torch
        a = torch.randn(3, device="cuda:0")
        b = torch.randn(3, device="cuda:1")
        with self.assertRaisesRegex(RuntimeError, r"same device"):
            _ = a + b

    def test_e1_2_model_forward_cross_device(self):
        """E1-2: model on cuda:0, input on cuda:1."""
        torch = self.torch
        model = torch.nn.Linear(3, 3)
        # Move model to cuda:0
        model = model.to("cuda:0")
        x = torch.randn(2, 3, device="cuda:1")
        with self.assertRaisesRegex(RuntimeError, r"same device"):
            model(x)

    def test_e1_3_cat_cross_device(self):
        """E1-3: torch.cat across devices."""
        torch = self.torch
        a = torch.randn(3, device="cuda:0")
        b = torch.randn(3, device="cuda:1")
        with self.assertRaisesRegex(RuntimeError, r"same device"):
            torch.cat([a, b])

    def test_e1_4_cross_entropy_cross_device(self):
        """E1-4: F.cross_entropy with output on cuda:0, target on cuda:1."""
        torch = self.torch
        import torch.nn.functional as F
        output = torch.randn(2, 5, device="cuda:0")
        target = torch.randint(0, 5, (2,), device="cuda:1")
        with self.assertRaisesRegex(RuntimeError, r"same device"):
            F.cross_entropy(output, target)

    def test_e1_5_same_device_no_error(self):
        """E1-5: a + b on same device -> no error."""
        torch = self.torch
        a = torch.randn(3, device="cuda:0")
        b = torch.randn(3, device="cuda:0")
        try:
            c = a + b
        except RuntimeError:
            self.fail("RuntimeError raised for same-device operation")


if __name__ == "__main__":
    unittest.main()
