"""Phase 2 bridge smoke for fakegpu.torch_patch.patch().

Requires an installed custom torch build that ships ``torch.fakegpu``.
The goal is to keep old ``patch(); import torch`` workflows working while
switching them to Phase 2 fake-CUDA semantics.
"""

import os
import sys
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from fakegpu.torch_patch import patch

patch()

import torch


def main() -> None:
    importlib.import_module("torch.fakegpu")

    x = torch.randn(4, device="cuda")
    model = torch.nn.Linear(4, 2).cuda()
    legacy = torch.cuda.FloatTensor(2, 3)

    assert x.device.type == "cuda"
    assert x.is_cuda is True
    assert next(model.parameters()).device.type == "cuda"
    assert next(model.parameters()).is_cuda is True
    assert legacy.device.type == "cuda"
    assert legacy.is_cuda is True
    assert legacy.dtype == torch.float32

    print("phase2 patch bridge passed")


if __name__ == "__main__":
    main()
