"""Minimal integration smoke for the custom Phase 2 torch build."""

import os
import sys
import tempfile
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import fakegpu
import torch


def main() -> None:
    assert os.environ.get("TORCH_FAKEGPU_ENABLE") == "1"
    assert hasattr(fakegpu, "__version__")
    assert torch.cuda.is_available() is True
    assert torch.cuda.device_count() >= 1

    device = torch.device("cuda")
    model = torch.nn.Linear(4, 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(8, 4, device=device)
    y = model(x)
    loss = y.sum()

    opt.zero_grad()
    loss.backward()
    opt.step()

    y_cpu = y.to("cpu")
    with torch.cuda.device(2):
        ctx_tensor = torch.randn(1, device="cuda")
    state = OrderedDict()
    base = torch.randn(2, 2)
    state["w"] = base
    state["alias"] = base
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(state, f.name)
        loaded = torch.load(f.name, map_location="cuda:2")

    assert next(model.parameters()).device.type == "cuda"
    assert y.device.type == "cuda"
    assert y.is_cuda is True
    assert y_cpu.device.type == "cpu"
    assert y_cpu.is_cuda is False
    assert ctx_tensor.device.index == 2
    assert loaded["w"].device.index == 2
    assert loaded["w"] is loaded["alias"]

    print("phase2 custom torch smoke passed")


if __name__ == "__main__":
    main()
