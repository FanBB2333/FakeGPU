"""Check native probe setup without using a physical GPU or network."""

from types import SimpleNamespace

import fakegpu
from native import hybrid_visibility
from scripts.update_nvidia_gpu_catalog import _parse_models


def test_hybrid_probe_keeps_the_real_baseline_independent(monkeypatch):
    modes = []
    environments = []

    def native_env(**kwargs):
        modes.append(kwargs["mode"])
        return dict(kwargs["base_env"], PROBE_MODE=kwargs["mode"])

    def run(_command, **kwargs):
        environments.append(kwargs["env"].get("PROBE_MODE"))
        return SimpleNamespace(returncode=0, stdout='GPU_PROBE={"signature":"reference","peak":1}', stderr="")

    monkeypatch.setattr(fakegpu, "env", native_env)
    monkeypatch.setattr(hybrid_visibility.subprocess, "run", run)
    monkeypatch.setattr(hybrid_visibility.sys, "argv", ["hybrid_visibility.py"])
    hybrid_visibility.main()
    assert modes == ["passthrough", "hybrid"]
    assert environments == [None, "passthrough", "hybrid"]


def test_catalog_parser_reads_separate_model_rows():
    table = "<table><tbody><tr><td>8.0</td><td>A100<br>A30</td></tr></tbody></table>"
    assert _parse_models(table) == {"A100": "8.0", "A30": "8.0"}
