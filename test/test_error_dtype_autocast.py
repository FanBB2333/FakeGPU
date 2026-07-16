"""Error simulation: dtype autocast mismatch (E4-1..E4-3)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# We test V100 (compute 7.0) profile to trigger bfloat16 rejection.
# Then test A100 (compute 8.0) to verify bfloat16 is allowed.
os.environ["FAKEGPU_DEVICE_COUNT"] = "1"
os.environ["FAKEGPU_STRICT_COMPAT"] = "1"


class TestAutocastDtypeV100(unittest.TestCase):
    """E4-1, E4-3: V100 (compute 7.0) should reject bfloat16."""
    torch = None

    @classmethod
    def setUpClass(cls):
        os.environ["FAKEGPU_PROFILES"] = "v100:1"
        import fakegpu
        fakegpu.patch_torch()
        import torch
        cls.torch = torch

    def test_e4_1_autocast_bf16_on_v100(self):
        """E4-1: autocast bfloat16 on V100 -> RuntimeError."""
        torch = self.torch
        with self.assertRaisesRegex(RuntimeError, r"does not support bfloat16|bfloat16"):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                a = torch.randn(3, 3, device="cuda:0")
                b = torch.randn(3, 3, device="cuda:0")
                _ = a @ b

    def test_e4_2_autocast_fp16_on_v100(self):
        """E4-3: autocast float16 on V100 -> no error (fp16 always ok)."""
        torch = self.torch
        try:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                a = torch.randn(3, 3, device="cuda:0")
                b = torch.randn(3, 3, device="cuda:0")
                _ = a @ b
        except RuntimeError as e:
            if "bfloat16" in str(e):
                self.fail(f"V100 should support float16: {e}")


class TestAutocastDtypeA100(unittest.TestCase):
    """E4-2: A100 (compute 8.0) should allow bfloat16."""
    torch = None

    @classmethod
    def setUpClass(cls):
        os.environ["FAKEGPU_PROFILES"] = "a100:1"
        import fakegpu
        fakegpu.patch_torch()
        import torch
        cls.torch = torch

    def test_e4_2_autocast_bf16_on_a100(self):
        """E4-2: autocast bfloat16 on A100 -> no error."""
        torch = self.torch
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                a = torch.randn(3, 3, device="cuda:0")
                b = torch.randn(3, 3, device="cuda:0")
                _ = a @ b
        except RuntimeError as e:
            if "bfloat16" in str(e):
                self.fail(f"A100 should support bfloat16: {e}")

    @classmethod
    def tearDownClass(cls):
        os.environ["FAKEGPU_PROFILES"] = "a100:1"
        import fakegpu
        fakegpu.patch_torch()


if __name__ == "__main__":
    unittest.main()
