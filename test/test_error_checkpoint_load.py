"""Error simulation: checkpoint load device errors (E5-1..E5-3)."""
import gc
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["FAKEGPU_DEVICE_COUNT"] = "2"
os.environ["FAKEGPU_PROFILES"] = "a100:2"


class TestCheckpointLoadErrors(unittest.TestCase):
    torch = None

    @classmethod
    def setUpClass(cls):
        import fakegpu
        fakegpu.patch_torch()
        import torch
        cls.torch = torch

    def test_e5_1_load_map_location_invalid_device(self):
        """E5-1: load with map_location='cuda:5', count=2 -> RuntimeError."""
        torch = self.torch
        model = torch.nn.Linear(3, 3)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        with self.assertRaisesRegex(RuntimeError, r"invalid device ordinal"):
            torch.load(buf, map_location="cuda:5")

    def test_e5_2_load_map_location_none_warning(self):
        """E5-2: load with map_location=None -> works (normalized to cpu)."""
        torch = self.torch
        model = torch.nn.Linear(3, 3)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        # Should work without error (maps to cpu)
        state = torch.load(buf, map_location=None, weights_only=True)
        self.assertIsInstance(state, dict)

    def test_e5_3_load_map_location_cpu(self):
        """E5-3: load with map_location='cpu' -> success."""
        torch = self.torch
        model = torch.nn.Linear(3, 3)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        state = torch.load(buf, map_location="cpu", weights_only=True)
        self.assertIsInstance(state, dict)


if __name__ == "__main__":
    unittest.main()
