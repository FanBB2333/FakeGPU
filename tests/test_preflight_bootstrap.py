from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import fakegpu
import fakegpu._preflight_bootstrap as bootstrap


def test_keyboard_interrupt_uses_shell_exit_code_130(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    report_path = tmp_path / "child.json"
    monkeypatch.setenv("FAKEGPU_PREFLIGHT_CHILD_REPORT", str(report_path))
    monkeypatch.setattr(
        fakegpu,
        "init",
        lambda **kwargs: SimpleNamespace(backend="fakecuda"),
    )
    monkeypatch.setattr(
        bootstrap,
        "_snapshot_fakecuda",
        lambda: {"tracking_confidence": "C3", "devices": []},
    )

    with pytest.raises(KeyboardInterrupt):
        bootstrap.main(["-c", "raise KeyboardInterrupt()"])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["exit_code"] == 130
    assert report["exception"]["type"] == "KeyboardInterrupt"


def test_child_report_write_error_does_not_mask_target_exception(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        fakegpu,
        "init",
        lambda **kwargs: SimpleNamespace(backend="fakecuda"),
    )

    def fail_write(**kwargs: object) -> None:
        raise OSError("read-only report directory")

    monkeypatch.setattr(bootstrap, "_write_child_report", fail_write)

    with pytest.raises(ValueError, match="target failed"):
        bootstrap.main(["-c", "raise ValueError('target failed')"])

    assert "could not write child report" in capsys.readouterr().err
