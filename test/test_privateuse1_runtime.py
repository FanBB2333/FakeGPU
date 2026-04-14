"""Runtime smoke checks for the FakeGPU PrivateUse1 backend."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from fakegpu.privateuse1 import init_privateuse1


def main() -> None:
    init_privateuse1()

    model = torch.nn.Linear(4, 2).fgpu()
    x = torch.randn(3, 4).to("fgpu")

    with torch.amp.autocast(device_type="fgpu", enabled=False):
        y = model(x)

    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)

    torch.save({"state": model.state_dict()}, path)
    loaded = torch.load(path, map_location=torch.device("fgpu"))

    assert "state" in loaded
    assert y.device.type == "fgpu"

    print("privateuse1 runtime smoke passed")


if __name__ == "__main__":
    main()
