"""Regression tests for fgpu map_location container and alias preservation."""

import os
import sys
import tempfile
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from fakegpu.privateuse1 import init_privateuse1


def main() -> None:
    init_privateuse1()

    shared = torch.randn(2, 3)
    payload = OrderedDict(
        [
            ("first", shared),
            ("nested", [shared, shared]),
            ("state", OrderedDict([("weight", shared), ("alias", shared)])),
        ]
    )

    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(payload, path)

    loaded = torch.load(path, map_location=torch.device("fgpu"))

    assert isinstance(loaded, OrderedDict)
    assert isinstance(loaded["state"], OrderedDict)
    assert loaded["first"].device.type == "fgpu"
    assert loaded["nested"][0] is loaded["nested"][1]
    assert loaded["state"]["weight"] is loaded["state"]["alias"]
    assert loaded["first"] is loaded["state"]["weight"]

    print("privateuse1 load-aliasing smoke passed")


if __name__ == "__main__":
    main()
