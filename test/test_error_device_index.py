"""Error simulation: device index out-of-bounds (E3-1..E3-4)."""
import gc
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Configure BEFORE importing fakegpu/torch
os.environ["FAKEGPU_DEVICE_COUNT"] = "2"
os.environ["FAKEGPU_PROFILES"] = "a100:2"


class TestDeviceIndexErrors(unittest.TestCase):
    torch = None

    @classmethod
    def setUpClass(cls):
        import fakegpu
        fakegpu.patch_torch()
        import torch
        cls.torch = torch

    def test_e3_1_set_device_out_of_bounds(self):
        """E3-1: set_device(5) with 2 devices -> RuntimeError."""
        with self.assertRaisesRegex(RuntimeError, r"invalid device ordinal"):
            self.torch.cuda.set_device(5)

    def test_e3_2_tensor_to_invalid_device(self):
        """E3-2: randn(device='cuda:99') with 2 devices -> RuntimeError."""
        with self.assertRaisesRegex(RuntimeError, r"invalid device ordinal"):
            self.torch.randn(3, device="cuda:99")

    def test_e3_3_get_device_properties_invalid(self):
        """E3-3: get_device_properties(10) with 2 devices -> RuntimeError."""
        with self.assertRaisesRegex(RuntimeError, r"invalid device ordinal"):
            self.torch.cuda.get_device_properties(10)

    def test_e3_4_set_device_valid(self):
        """E3-4: set_device(0) with 2 devices -> no error."""
        try:
            self.torch.cuda.set_device(0)
            self.torch.cuda.set_device(1)
        except RuntimeError:
            self.fail("set_device raised RuntimeError for valid device index")


if __name__ == "__main__":
    unittest.main()
