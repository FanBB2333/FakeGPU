"""Error simulation: OOM precise simulation (E2-1..E2-5)."""
import gc
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["FAKEGPU_DEVICE_COUNT"] = "2"
os.environ["FAKEGPU_PROFILES"] = "a100:2"
os.environ["FAKEGPU_MEMORY_TRACKING"] = "1"


class TestOOMErrors(unittest.TestCase):
    torch = None
    _SMALL_MEM = 10 * 1024**2  # 10 MB for testing (avoids allocating real 80GB)

    @classmethod
    def setUpClass(cls):
        import fakegpu
        fakegpu.patch_torch()
        import torch
        cls.torch = torch
        # Override tracker limits to small values for testing.
        # The tracker was initialized with profile defaults (80GB).
        # We shrink it so OOM tests don't require allocating real 80GB of RAM.
        import fakegpu.torch_patch as tp
        if tp._memory_tracker is not None:
            tp._memory_tracker._total = [cls._SMALL_MEM] * 2

    def setUp(self):
        # Force GC to reclaim any leftover tensors between tests
        gc.collect()

    def test_e2_1_single_allocation_exceeds_capacity(self):
        """E2-1: create tensor larger than device memory -> OutOfMemoryError."""
        torch = self.torch
        total = self._SMALL_MEM
        # Try to allocate 2x total
        num_floats = (total * 2) // 4  # float32 = 4 bytes
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            torch.randn(num_floats, device="cuda:0")

    def test_e2_2_cumulative_oom(self):
        """E2-2: allocate 60% then another 60% -> second fails."""
        torch = self.torch
        total = self._SMALL_MEM
        sixty_pct_floats = int((total * 0.6) // 4)
        a = torch.randn(sixty_pct_floats, device="cuda:0")
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            b = torch.randn(sixty_pct_floats, device="cuda:0")
        del a

    def test_e2_3_gc_reclaims_memory(self):
        """E2-3: allocate, del, gc, re-allocate same size -> success."""
        torch = self.torch
        total = self._SMALL_MEM
        half_floats = int((total * 0.5) // 4)
        a = torch.randn(half_floats, device="cuda:0")
        del a
        gc.collect()
        # Should succeed after GC
        try:
            b = torch.randn(half_floats, device="cuda:0")
            del b
        except torch.cuda.OutOfMemoryError:
            self.fail("OOM after GC reclaimed memory")

    def test_e2_4_memory_allocated_tracks_live_tensors(self):
        """E2-4: memory_allocated returns actual sum of live tensor sizes."""
        torch = self.torch
        gc.collect()
        before = torch.cuda.memory_allocated(0)
        a = torch.randn(1000, device="cuda:0")
        after = torch.cuda.memory_allocated(0)
        self.assertGreater(after, before)
        # float32 * 1000 = 4000 bytes
        self.assertGreaterEqual(after - before, 4000)
        del a
        gc.collect()

    def test_e2_5_mem_get_info_reflects_allocations(self):
        """E2-5: mem_get_info free = total - allocated."""
        torch = self.torch
        gc.collect()
        free_before, total = torch.cuda.mem_get_info(0)
        a = torch.randn(1000, device="cuda:0")
        free_after, total2 = torch.cuda.mem_get_info(0)
        self.assertEqual(total, total2)
        self.assertLess(free_after, free_before)
        del a
        gc.collect()


if __name__ == "__main__":
    unittest.main()
