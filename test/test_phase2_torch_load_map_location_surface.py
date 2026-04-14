"""torch.load map_location surface for the custom Phase 2 torch build."""

import os
import sys
import tempfile
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch


def main() -> None:
    assert os.environ.get("TORCH_FAKEGPU_ENABLE") == "1"

    shared = torch.randn(2, 3)
    payload = OrderedDict(
        [
            ("first", shared),
            ("nested", [shared, shared]),
            ("state", OrderedDict([("weight", shared), ("alias", shared)])),
        ]
    )

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(payload, f.name)
        loaded = torch.load(f.name, map_location={"cpu": "cuda:3"})
        loaded_via_device = torch.load(
            f.name, map_location={torch.device("cpu"): torch.device("cuda:2")}
        )

    assert isinstance(loaded, OrderedDict)
    assert isinstance(loaded["state"], OrderedDict)
    assert loaded["first"].device.type == "cuda"
    assert loaded["first"].device.index == 3
    assert loaded["first"].is_cuda is True
    assert loaded["nested"][0] is loaded["nested"][1]
    assert loaded["state"]["weight"] is loaded["state"]["alias"]
    assert loaded["first"] is loaded["state"]["weight"]

    assert loaded_via_device["first"].device.type == "cuda"
    assert loaded_via_device["first"].device.index == 2
    assert loaded_via_device["first"].is_cuda is True
    assert loaded_via_device["nested"][0] is loaded_via_device["nested"][1]
    assert loaded_via_device["state"]["weight"] is loaded_via_device["state"]["alias"]
    assert loaded_via_device["first"] is loaded_via_device["state"]["weight"]

    print("phase2 torch.load map_location surface passed")


if __name__ == "__main__":
    main()
