"""P5-1: DataLoader pin_memory=True compatibility tests.

Validates that the common training pattern of pin_memory + non_blocking
transfer works correctly under FakeGPU torch_patch.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import fakegpu
fakegpu.patch_torch()

import torch
from torch.utils.data import DataLoader, TensorDataset


class TestDataLoaderPinMemory(unittest.TestCase):
    """Ensure DataLoader pin_memory=True does not crash under FakeGPU."""

    def setUp(self):
        self.dataset = TensorDataset(
            torch.randn(64, 10),
            torch.randint(0, 2, (64,)),
        )

    # ------------------------------------------------------------------
    # Basic functionality
    # ------------------------------------------------------------------

    def test_pin_memory_basic(self):
        """pin_memory=True creates batches without error."""
        loader = DataLoader(self.dataset, batch_size=16, pin_memory=True)
        x, y = next(iter(loader))
        self.assertEqual(x.shape, torch.Size([16, 10]))
        self.assertEqual(y.shape, torch.Size([16]))

    def test_pin_memory_with_num_workers(self):
        """pin_memory=True + num_workers=2 does not hang or crash."""
        loader = DataLoader(
            self.dataset, batch_size=16, pin_memory=True, num_workers=2
        )
        x, _ = next(iter(loader))
        self.assertEqual(x.shape[0], 16)

    def test_pin_memory_device_kwarg(self):
        """pin_memory_device='cuda' is accepted."""
        loader = DataLoader(
            self.dataset,
            batch_size=16,
            pin_memory=True,
            pin_memory_device="cuda",
        )
        x, _ = next(iter(loader))
        self.assertEqual(x.shape[0], 16)

    # ------------------------------------------------------------------
    # Real-world transfer patterns
    # ------------------------------------------------------------------

    def test_pin_memory_to_cuda(self):
        """pin_memory batch can be moved to 'cuda:0'."""
        loader = DataLoader(self.dataset, batch_size=16, pin_memory=True)
        x, y = next(iter(loader))
        x = x.to("cuda:0")
        y = y.to("cuda:0")
        # Under FakeGPU these stay on CPU but the call must not raise
        self.assertEqual(x.shape, torch.Size([16, 10]))

    def test_pin_memory_non_blocking(self):
        """pin_memory + non_blocking=True transfer — the most common pattern."""
        loader = DataLoader(self.dataset, batch_size=16, pin_memory=True)
        x, y = next(iter(loader))
        x = x.to("cuda:0", non_blocking=True)
        y = y.to("cuda:0", non_blocking=True)
        self.assertEqual(x.shape[0], 16)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def test_training_loop_with_pin_memory(self):
        """Complete 2-epoch training with pin_memory + non_blocking."""
        model = torch.nn.Linear(10, 2).cuda()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        loader = DataLoader(
            self.dataset, batch_size=16, pin_memory=True, shuffle=True
        )

        for _ in range(2):
            for x, y in loader:
                x = x.to("cuda:0", non_blocking=True)
                y = y.to("cuda:0", non_blocking=True)
                loss = criterion(model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.assertIsInstance(loss.item(), float)

    # ------------------------------------------------------------------
    # Custom collate
    # ------------------------------------------------------------------

    def test_custom_collate_with_pin_memory(self):
        """pin_memory works with a custom collate_fn."""

        def my_collate(batch):
            xs, ys = zip(*batch)
            return torch.stack(xs), torch.tensor(ys)

        loader = DataLoader(
            self.dataset,
            batch_size=16,
            pin_memory=True,
            collate_fn=my_collate,
        )
        x, y = next(iter(loader))
        self.assertEqual(x.shape, torch.Size([16, 10]))


if __name__ == "__main__":
    unittest.main()
