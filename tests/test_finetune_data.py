from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from fakegpu.finetune_data import (
    FinetuneDataError,
    load_local_tokenizer,
    profile_finetune_dataset,
    profile_tokenized_records,
)


class WordTokenizer:
    chat_template = "test template v1"
    bos_token_id = 1

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        ids = [10 + index for index, _ in enumerate(text.split())]
        if kwargs["add_special_tokens"]:
            ids = [self.bos_token_id, *ids]
        return {"input_ids": ids}

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return (
            [1] + [10 for message in messages for _ in message["content"].split()] + [2]
        )


def write_rows(tmp_path: Path, rows, name="train.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_full_scan_text_truncation_padding_and_provenance(tmp_path):
    path = write_rows(tmp_path, [{"text": "a b"}, {"text": "a b c d e"}])
    tokenizer = WordTokenizer()
    report = profile_finetune_dataset(path, sequence_length=4, tokenizer=tokenizer)
    assert report["source_records"] == 2
    assert report["training_inputs"] == {
        "num_samples": 2,
        "sequence_length": 4,
        "exact_effective_count": False,
        "count_basis": "profiled_preprocessing_scenario",
    }
    assert report["tokens"]["raw_tokens"] == 9
    assert report["tokens"]["retained_tokens"] == 7
    assert report["tokens"]["truncated_tokens"] == 2
    assert report["tokens"]["truncated_records"] == 1
    assert report["lengths"]["raw"]["mean"] == 4.5
    assert report["lengths"]["raw"]["p50"] == 3
    assert report["lengths"]["raw"]["p95"] == 6
    assert report["padding"]["padding_tokens"] == 1
    assert report["padding"]["padding_fraction"] == 1 / 8
    assert report["provenance"]["full_scan"] is True
    assert (
        report["provenance"]["files"][0]["sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert report["provenance"]["tokenizer"]["content_identified"] is False
    assert tokenizer.calls[0][1] == {
        "add_special_tokens": True,
        "truncation": False,
        "padding": False,
    }
    json.dumps(report, allow_nan=False)


def test_messages_apply_template_once_without_generation_prompt(tmp_path):
    messages = [
        {"role": "user", "content": "two words"},
        {"role": "assistant", "content": "reply"},
    ]
    path = write_rows(tmp_path, [{"messages": messages}])
    tokenizer = WordTokenizer()
    report = profile_finetune_dataset(path, sequence_length=8, tokenizer=tokenizer)
    assert tokenizer.calls == [
        (messages, {"tokenize": True, "add_generation_prompt": False})
    ]
    assert report["tokens"]["raw_tokens"] == 5
    assert report["provenance"]["tokenization"]["add_special_tokens"] is None


def test_processed_rows_are_exact_without_repacking_and_track_masks_labels(tmp_path):
    path = write_rows(
        tmp_path,
        [
            {
                "input_ids": [1, 2, 3, 0],
                "attention_mask": [1, 1, 2, 0],
                "labels": [-100, 2, 3, -100],
            },
            {
                "input_ids": [0, 4, 5],
                "attention_mask": [0, 1, 1],
                "labels": [-100, -100, 5],
            },
        ],
    )
    report = profile_finetune_dataset(
        path, sequence_length=8, processed=True, source_framework="llamafactory"
    )
    assert report["source_framework"] == "llamafactory"
    assert report["training_inputs"]["exact_effective_count"] is True
    assert report["training_inputs"]["num_samples"] == 2
    assert report["tokens"]["supervised_tokens"] == 3
    assert report["tokens"]["raw_tokens"] == 5
    assert report["tokens"]["truncated_tokens"] == 0
    assert report["padding"]["source_padding_tokens"] == 2
    assert report["tokens"]["attention_mask_records"] == 2
    assert report["padding"]["padding_tokens"] == 11
    assert report["lengths"]["stored_input"]["total"] == 7
    assert report["provenance"]["tokenizer"] is None
    assert any("not verified" in note for note in report["limitations"])


@pytest.mark.parametrize(
    "options", [{"packing": "concatenate"}, {"record_format": "text"}]
)
def test_processed_rejects_secondary_preprocessing(tmp_path, options):
    path = write_rows(tmp_path, [{"input_ids": [1]}])
    with pytest.raises(FinetuneDataError, match="processed"):
        profile_finetune_dataset(path, sequence_length=4, processed=True, **options)


def test_processed_rejects_truncation_even_for_existing_padding(tmp_path):
    path = write_rows(tmp_path, [{"input_ids": [1, 0, 0], "attention_mask": [1, 0, 0]}])
    with pytest.raises(FinetuneDataError, match="processed input_ids length 3 exceeds"):
        profile_finetune_dataset(path, sequence_length=2, processed=True)


@pytest.mark.parametrize(
    "drop, separator, sequences, tokens, remainder, padding",
    [
        (False, None, 3, 10, 0, 2),
        (True, None, 2, 8, 2, 0),
        (False, 9, 3, 11, 0, 1),
    ],
)
def test_generic_concatenation_preserves_long_examples_and_crosses_files(
    tmp_path, drop, separator, sequences, tokens, remainder, padding
):
    left = write_rows(tmp_path, [{"input_ids": [1, 2, 3, 4, 5, 6, 7]}], "left.jsonl")
    right = write_rows(
        tmp_path,
        [{"input_ids": [0, 8, 9, 10], "attention_mask": [0, 1, 1, 1]}],
        "right.jsonl",
    )
    report = profile_finetune_dataset(
        [left, right],
        sequence_length=4,
        packing="concatenate",
        drop_remainder=drop,
        separator_token_id=separator,
    )
    assert report["source_records"] == 2
    assert report["training_inputs"]["num_samples"] == sequences
    assert report["training_inputs"]["exact_effective_count"] is False
    assert report["tokens"]["truncated_tokens"] == 0
    assert report["tokens"]["dropped_remainder_tokens"] == remainder
    assert report["padding"]["nonpadding_tokens"] == tokens
    assert report["padding"]["padding_tokens"] == padding
    assert report["lengths"]["effective"]["count"] == sequences
    assert report["lengths"]["effective"]["total"] == tokens
    assert any("not LLaMA-Factory or Axolotl" in note for note in report["limitations"])


def test_dropping_all_tokens_is_explicit_empty_output(tmp_path):
    path = write_rows(tmp_path, [{"input_ids": [1, 2]}])
    report = profile_finetune_dataset(
        path, sequence_length=4, packing="concatenate", drop_remainder=True
    )
    assert report["training_inputs"]["num_samples"] == 0
    assert report["tokens"]["dropped_remainder_tokens"] == 2
    assert report["lengths"]["effective"]["p50"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "row, error",
    [
        ([], "JSON object"),
        ({"input_ids": [True]}, "integer IDs"),
        ({"input_ids": [[1]]}, "flat list"),
        ({"input_ids": [-1]}, "integer IDs"),
        ({"input_ids": []}, "at least one"),
        ({"input_ids": [1], "attention_mask": [0]}, "at least one"),
        ({"input_ids": [1, 2], "attention_mask": [1]}, "attention_mask length"),
        ({"input_ids": [1], "attention_mask": None}, "attention_mask must"),
        ({"input_ids": [1], "labels": [1, 2]}, "labels length"),
        ({"input_ids": [1], "labels": [-1]}, "labels must"),
        ({"text": 7}, "must be a string"),
        ({"messages": []}, "non-empty message list"),
        (
            {"messages": [{"role": "user", "content": [{"type": "image"}]}]},
            "multimodal",
        ),
        ({"text": "a", "input_ids": [1]}, "exactly one"),
    ],
)
def test_malformed_rows_report_path_and_line(tmp_path, row, error):
    path = write_rows(tmp_path, [row])
    with pytest.raises(FinetuneDataError, match=error) as caught:
        profile_finetune_dataset(path, sequence_length=4, tokenizer=WordTokenizer())
    assert f"{path}:1:" in str(caught.value)


@pytest.mark.parametrize(
    "bad_line",
    [
        '{"input_ids": [1], "input_ids": [2]}',
        '{"input_ids": [NaN]}',
        '{"input_ids": [1], "score": Infinity}',
        '{"input_ids": [1], "score": 1e999}',
        "{bad json",
        "null",
    ],
)
def test_malformed_json_is_not_hidden_after_large_prefix(tmp_path, bad_line):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"input_ids": [1]}\n' * 1001 + bad_line + "\n")
    with pytest.raises(FinetuneDataError) as caught:
        profile_finetune_dataset(path, sequence_length=4)
    assert f"{path}:1002:" in str(caught.value)


def test_invalid_unicode_and_empty_file(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(b"\xff\n")
    with pytest.raises(FinetuneDataError, match=r":1:"):
        profile_finetune_dataset(path, sequence_length=4)
    path.write_bytes(b"\n \n")
    with pytest.raises(FinetuneDataError, match="no records"):
        profile_finetune_dataset(path, sequence_length=4)


def test_explicit_format_allows_metadata_and_auto_rejects_mixed_rows(tmp_path):
    path = write_rows(tmp_path, [{"text": "a", "input_ids": [1, 2]}])
    report = profile_finetune_dataset(path, sequence_length=4, processed=True)
    assert report["tokens"]["raw_tokens"] == 2
    path = write_rows(tmp_path, [{"text": "a"}, {"input_ids": [1]}])
    with pytest.raises(FinetuneDataError, match="mixed record formats"):
        profile_finetune_dataset(path, sequence_length=4, tokenizer=WordTokenizer())


def test_custom_text_field_and_no_special_tokens(tmp_path):
    path = write_rows(tmp_path, [{"body": "a b c"}])
    report = profile_finetune_dataset(
        path,
        sequence_length=4,
        tokenizer=WordTokenizer(),
        text_field="body",
        add_special_tokens=False,
    )
    assert report["tokens"]["raw_tokens"] == 3


def test_partial_labels_are_not_reported_as_dataset_supervision(tmp_path):
    path = write_rows(tmp_path, [{"input_ids": [1], "labels": [1]}, {"input_ids": [2]}])
    report = profile_finetune_dataset(path, sequence_length=4, processed=True)
    assert report["tokens"]["labeled_records"] == 1
    assert report["tokens"]["supervised_tokens"] is None


def test_content_identity_changes_with_source_bytes_and_is_path_independent(tmp_path):
    left = write_rows(tmp_path, [{"input_ids": [1]}], "left.jsonl")
    right = write_rows(tmp_path, [{"input_ids": [1]}], "right.jsonl")

    def identity(path):
        return profile_finetune_dataset(path, sequence_length=4)["provenance"][
            "content_sha256"
        ]

    assert identity(left) == identity(right)
    right.write_text('{"input_ids": [2]}\n')
    assert identity(left) != identity(right)


def test_local_loader_is_optional_and_disables_downloads_and_custom_code(
    tmp_path, monkeypatch
):
    calls = []
    tokenizer = WordTokenizer()

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return tokenizer

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)),
    )
    assert load_local_tokenizer(tmp_path) is tokenizer
    assert calls == [
        (str(tmp_path), {"local_files_only": True, "trust_remote_code": False})
    ]
    with pytest.raises(FinetuneDataError, match="existing local directory"):
        load_local_tokenizer("nonexistent-org/not-a-local-tokenizer")


def test_tokenizer_fingerprint_hashes_assets_and_template_excludes_weights(
    tmp_path, monkeypatch
):
    directory = tmp_path / "tokenizer"
    directory.mkdir()
    (directory / "tokenizer.json").write_text('{"version":1}')
    weights = directory / "model.safetensors"
    weights.write_bytes(b"do not hash model weights")
    tokenizer = WordTokenizer()
    tokenizer.name_or_path = str(directory)
    path = write_rows(tmp_path, [{"text": "a"}])

    def identity():
        return profile_finetune_dataset(path, sequence_length=4, tokenizer=tokenizer)[
            "provenance"
        ]["tokenizer"]

    before = identity()
    assert before["content_identified"] is True
    assert [asset["path"] for asset in before["files"]] == ["tokenizer.json"]
    weights.write_bytes(b"changed weights")
    assert identity()["identity_sha256"] == before["identity_sha256"]
    (directory / "tokenizer.json").write_text('{"version":2}')
    assert identity()["identity_sha256"] != before["identity_sha256"]
    tokenizer.chat_template = "test template v2"
    assert identity()["chat_template_sha256"] != before["chat_template_sha256"]


@pytest.mark.parametrize(
    "options, error",
    [
        ({"sequence_length": 0}, "sequence_length"),
        ({"sequence_length": True}, "sequence_length"),
        ({"packing": "framework"}, "packing must"),
        ({"drop_remainder": True}, "require concatenate"),
        ({"packing": "concatenate", "separator_token_id": -1}, "separator_token_id"),
        ({"record_format": "alpaca"}, "record_format"),
        ({"processed": 1}, "boolean"),
        ({"text_field": "input_ids"}, "distinct"),
    ],
)
def test_invalid_options_fail_before_dataset_scan(tmp_path, options, error):
    path = write_rows(tmp_path, [{"input_ids": [1]}])
    with pytest.raises(FinetuneDataError, match=error):
        profile_finetune_dataset(path, **({"sequence_length": 4} | options))


def test_paths_and_missing_tokenizer_fail_explicitly(tmp_path):
    path = write_rows(tmp_path, [{"text": "a"}])
    with pytest.raises(FinetuneDataError, match="require a tokenizer"):
        profile_finetune_dataset(path, sequence_length=4)
    with pytest.raises(FinetuneDataError, match="duplicate dataset path"):
        profile_finetune_dataset([path, path], sequence_length=4)
    with pytest.raises(FinetuneDataError, match="at least one"):
        profile_finetune_dataset([], sequence_length=4)


def test_native_records_are_scanned_once_and_match_file_statistics(tmp_path):
    rows = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5, 0], "attention_mask": [1, 1, 0], "labels": [4, 5, -100]},
    ]

    class SinglePass:
        calls = 0

        def __iter__(self):
            self.calls += 1
            assert self.calls == 1
            yield from rows

    data = SinglePass()
    native = profile_tokenized_records(
        data, sequence_length=4, source_framework="axolotl"
    )
    path = write_rows(tmp_path, rows)
    file_report = profile_finetune_dataset(
        path, sequence_length=4, processed=True, source_framework="axolotl"
    )
    assert data.calls == 1
    for key in ("tokens", "lengths", "padding", "training_inputs", "source_records"):
        assert native[key] == file_report[key]
    assert native["source_framework"] == "axolotl"
    assert native["provenance"]["source"] == "native_in_memory"
    assert native["provenance"]["files"] == []
    assert native["provenance"]["full_scan"] is True
    assert native["provenance"]["content_hash_format"] == "canonical_json_rows_v1"
    canonical = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )
    assert (
        native["provenance"]["content_sha256"]
        == hashlib.sha256(canonical.encode()).hexdigest()
    )


def test_native_content_hash_is_sensitive_to_order_labels_and_metadata():
    rows = [{"input_ids": [1], "labels": [-100]}, {"input_ids": [2], "labels": [2]}]

    def digest(records):
        return profile_tokenized_records(records, sequence_length=4)["provenance"][
            "content_sha256"
        ]

    before = digest(rows)
    assert digest(list(reversed(rows))) != before
    changed_labels = [{"input_ids": [1], "labels": [1]}, rows[1]]
    assert digest(changed_labels) != before
    assert digest([dict(rows[0], source="different"), rows[1]]) != before
    assert digest([dict(reversed(list(row.items()))) for row in rows]) == before


@pytest.mark.parametrize(
    "rows, error",
    [
        ([], "no records"),
        ([{}], "missing 'input_ids'"),
        ([[]], "each record must be a mapping"),
        ([{"input_ids": [1, 2, 3]}], "exceeds sequence_length"),
        ([{"input_ids": [1], "meta": float("nan")}], "JSON-compatible"),
        ([{"input_ids": [1], "meta": object()}], "JSON-compatible"),
        ({"input_ids": [1]}, "iterable of tokenized mappings"),
    ],
)
def test_native_invalid_records_fail_explicitly(rows, error):
    with pytest.raises(FinetuneDataError, match=error):
        profile_tokenized_records(rows, sequence_length=2)
