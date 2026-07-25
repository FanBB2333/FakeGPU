from __future__ import annotations

import re
from pathlib import Path

from fakegpu import __version__


ROOT = Path(__file__).resolve().parent.parent


def test_package_version_is_semantic() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_cpp_report_version_matches_package_version() -> None:
    header = (ROOT / "src/core/version.hpp").read_text(encoding="utf-8")
    match = re.search(r'#define\s+FAKEGPU_VERSION\s+"([^"]+)"', header)

    assert match is not None
    assert match.group(1) == __version__


def test_changelog_contains_current_version() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert f"## v{__version__} " in changelog
