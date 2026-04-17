"""P5-2: torch.compile compatibility tests.

Validates that ``torch.compile`` works under FakeGPU torch_patch with
the ``eager`` backend.  The ``inductor`` (default) backend attempts real
CUDA code-generation and is expected to fail — users should use
``backend='eager'`` or ``'aot_eager'`` when running under FakeGPU.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import fakegpu
fakegpu.patch_torch()

import torch


class TestTorchCompileEager(unittest.TestCase):
    """torch.compile with backend='eager' under FakeGPU."""

    # ------------------------------------------------------------------
    # Basic compilation
    # ------------------------------------------------------------------

    def test_compile_simple_function(self):
        """Compile a free function and run it on 'cuda' tensors."""

        @torch.compile(backend="eager")
        def fn(x, y):
            return x + y * 2

        x = torch.randn(4, 4, device="cuda")
        y = torch.randn(4, 4, device="cuda")
        out = fn(x, y)
        self.assertEqual(out.shape, torch.Size([4, 4]))

    def test_compile_module(self):
        """Compile an nn.Module."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        ).cuda()
        compiled = torch.compile(model, backend="eager")
        out = compiled(torch.randn(8, 10, device="cuda"))
        self.assertEqual(out.shape, torch.Size([8, 2]))

    def test_compile_fullgraph(self):
        """fullgraph=True captures the full graph without breaks."""

        @torch.compile(fullgraph=True, backend="eager")
        def matmul_fn(a, b):
            return torch.matmul(a, b)

        a = torch.randn(4, 4, device="cuda")
        b = torch.randn(4, 4, device="cuda")
        out = matmul_fn(a, b)
        self.assertEqual(out.shape, torch.Size([4, 4]))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def test_compile_training_loop(self):
        """Full training with a compiled model."""
        model = torch.nn.Linear(10, 2).cuda()
        compiled = torch.compile(model, backend="eager")
        optimizer = torch.optim.Adam(compiled.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(5):
            x = torch.randn(16, 10, device="cuda")
            y = torch.randint(0, 2, (16,), device="cuda")
            loss = criterion(compiled(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.assertIsInstance(loss.item(), float)

    # ------------------------------------------------------------------
    # Compile + autocast
    # ------------------------------------------------------------------

    def test_compile_with_autocast(self):
        """Compiled model inside autocast context."""
        model = torch.nn.Linear(10, 2).cuda()
        compiled = torch.compile(model, backend="eager")
        x = torch.randn(8, 10, device="cuda")

        with torch.amp.autocast("cuda"):
            out = compiled(x)
        self.assertEqual(out.shape, torch.Size([8, 2]))

    def test_compile_with_legacy_autocast(self):
        """Compiled model inside legacy torch.cuda.amp.autocast()."""
        model = torch.nn.Linear(10, 2).cuda()
        compiled = torch.compile(model, backend="eager")
        x = torch.randn(8, 10, device="cuda")

        with torch.cuda.amp.autocast():
            out = compiled(x)
        self.assertEqual(out.shape, torch.Size([8, 2]))


class TestTorchCompileInductorExpectedFailure(unittest.TestCase):
    """Inductor backend is expected to fail (needs real CUDA)."""

    @unittest.expectedFailure
    def test_inductor_raises(self):
        """Default backend (inductor) cannot compile for fake GPUs."""
        model = torch.nn.Linear(10, 2).cuda()
        compiled = torch.compile(model)
        compiled(torch.randn(8, 10, device="cuda"))


if __name__ == "__main__":
    unittest.main()
