from __future__ import annotations

import os

from fakegpu._stage import stage


def test_nested_stage_restores_outer_marker(monkeypatch) -> None:
    monkeypatch.delenv("FAKEGPU_PREFLIGHT_STAGE", raising=False)

    with stage("outer"):
        assert os.environ["FAKEGPU_PREFLIGHT_STAGE"] == "outer"
        with stage("inner"):
            assert os.environ["FAKEGPU_PREFLIGHT_STAGE"] == "inner"
        assert os.environ["FAKEGPU_PREFLIGHT_STAGE"] == "outer"

    assert "FAKEGPU_PREFLIGHT_STAGE" not in os.environ
