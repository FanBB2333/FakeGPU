from __future__ import annotations

import pytest

from fakegpu.structured_io import StructuredDataError, load_mapping


def test_invalid_yaml_is_reported_as_structured_data_error(tmp_path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "invalid.yaml"
    path.write_text("key:\n\t- invalid-indent\n", encoding="utf-8")

    with pytest.raises(StructuredDataError, match="cannot parse"):
        load_mapping(path)
