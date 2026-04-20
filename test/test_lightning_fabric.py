# test/test_lightning_fabric.py
"""Lightning Fabric on fake CUDA — basic integration tests."""
from __future__ import annotations

import pytest

from hf_test_utils import patch_fakegpu


class TestLightningFabricBasic:
    """Test that Lightning Fabric correctly detects and uses fake CUDA."""

    def test_fabric_cuda_accelerator_selection(self) -> None:
        """Fabric should auto-select CUDAAccelerator when FakeGPU is active."""
        L = pytest.importorskip("lightning")
        patch_fakegpu(profile="a100", device_count=1)
        import torch

        fabric = L.Fabric(accelerator="cuda", devices=1, precision="32-true")
        fabric.launch()

        # Verify CUDA accelerator was selected
        assert fabric.device.type == "cuda"

    def test_fabric_model_setup(self) -> None:
        """Model and optimizer should be placed on fake CUDA via fabric.setup()."""
        L = pytest.importorskip("lightning")
        patch_fakegpu(profile="a100", device_count=1)
        import torch

        fabric = L.Fabric(accelerator="cuda", devices=1, precision="32-true")
        fabric.launch()

        model = torch.nn.Linear(32, 16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model, optimizer = fabric.setup(model, optimizer)

        # Parameters should report cuda device
        param_device = next(model.parameters()).device
        assert param_device.type == "cuda"

    def test_fabric_training_step(self) -> None:
        """A complete forward-backward-step cycle should work on fake CUDA."""
        L = pytest.importorskip("lightning")
        patch_fakegpu(profile="a100", device_count=1)
        import torch

        fabric = L.Fabric(accelerator="cuda", devices=1, precision="32-true")
        fabric.launch()

        model = torch.nn.Linear(32, 16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model, optimizer = fabric.setup(model, optimizer)

        x = torch.randn(4, 32, device="cuda")
        y = model(x)
        loss = y.sum()
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        assert isinstance(loss.item(), float)

    def test_fabric_bf16_precision(self) -> None:
        """Fabric with bf16-true precision should work on fake CUDA."""
        L = pytest.importorskip("lightning")
        patch_fakegpu(profile="a100", device_count=1)
        import torch

        fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-true")
        fabric.launch()

        model = torch.nn.Linear(32, 16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model, optimizer = fabric.setup(model, optimizer)

        x = torch.randn(4, 32, device="cuda")
        y = model(x)
        loss = y.sum()
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        assert isinstance(loss.item(), float)
