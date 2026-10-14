from __future__ import annotations

import gzip
import json

import pytest

from fakegpu.structured_io import StructuredDataError, load_mapping, write_json


def test_invalid_yaml_is_reported_as_structured_data_error(tmp_path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "invalid.yaml"
    path.write_text("key:\n\t- invalid-indent\n", encoding="utf-8")

    with pytest.raises(StructuredDataError, match="cannot parse"):
        load_mapping(path)


def test_compressed_json_has_same_document_and_deterministic_bytes(tmp_path):
    payload = {"中文": [1, {"nested": True}], "rows": list(range(100))}
    plain, first, second = [
        tmp_path / name for name in ("plain.json", "a.json.gz", "b.JSON.GZ")
    ]
    for path in (plain, first, second):
        assert write_json(path, payload) == path
        assert load_mapping(path) == payload
    assert first.read_bytes() == second.read_bytes()
    assert gzip.decompress(first.read_bytes()) == plain.read_bytes()


@pytest.mark.parametrize(
    "data",
    [
        b"not gzip",
        gzip.compress(b"{}")[:-4],
        gzip.compress(b"\xff"),
        gzip.compress(b"[1]"),
        gzip.compress(b"{bad"),
    ],
)
def test_compressed_json_reports_invalid_data(tmp_path, data):
    path = tmp_path / "broken.json.gz"
    path.write_bytes(data)
    with pytest.raises(StructuredDataError):
        load_mapping(path)


def test_compressed_write_failure_preserves_destination_and_cleans_temporary(tmp_path):
    path = tmp_path / "report.json.gz"
    write_json(path, {"valid": 1})
    before = path.read_bytes()
    with pytest.raises(TypeError):
        write_json(path, {"invalid": object()})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_plain_json_format_is_unchanged(tmp_path):
    path = tmp_path / "report.json"
    payload = {"b": [1, 2], "a": "值"}
    write_json(path, payload)
    assert path.read_text() == json.dumps(payload, indent=2, sort_keys=True) + "\n"
