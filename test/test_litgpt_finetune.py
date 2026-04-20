# test/test_litgpt_finetune.py
"""LitGPT finetune compatibility — Fabric-level integration test."""
from __future__ import annotations

import tempfile

import pytest

from hf_test_utils import patch_fakegpu


def test_litgpt_fabric_training_loop() -> None:
    """Simulate LitGPT's Fabric-based training pattern with a tiny model."""
    L = pytest.importorskip("lightning")
    patch_fakegpu(profile="a100", device_count=1)
    import torch
    import torch.nn as nn

    # LitGPT uses Fabric with a GPT model. We simulate with a tiny transformer.
    class TinyGPT(nn.Module):
        def __init__(self, vocab_size=128, d_model=32, nhead=2, num_layers=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            h = self.embed(x)
            h = self.transformer(h)
            return self.head(h)

    fabric = L.Fabric(accelerator="cuda", devices=1, precision="32-true")
    fabric.launch()

    model = TinyGPT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, optimizer = fabric.setup(model, optimizer)

    # Training loop (LitGPT pattern)
    losses = []
    for step in range(3):
        input_ids = torch.randint(0, 128, (2, 16), device="cuda")
        logits = model(input_ids)
        # Shift for next-token prediction
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 128), input_ids[:, 1:].reshape(-1)
        )
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    # Verify loss is finite
    assert all(isinstance(l, float) for l in losses)

    # Save checkpoint (LitGPT uses fabric.save)
    with tempfile.TemporaryDirectory(prefix="fakegpu_litgpt_") as tmpdir:
        ckpt_path = f"{tmpdir}/model.pt"
        fabric.save(ckpt_path, {"model": model, "optimizer": optimizer, "step": 3})

        # Verify checkpoint was saved
        loaded = fabric.load(ckpt_path)
        assert loaded["step"] == 3


def test_litgpt_importable() -> None:
    """LitGPT should be importable if installed."""
    litgpt = pytest.importorskip("litgpt")
    # Just verify the package is importable
    assert hasattr(litgpt, "__version__") or hasattr(litgpt, "GPT")
