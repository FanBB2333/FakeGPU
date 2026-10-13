from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fakegpu._cli import BUILTIN_COMMANDS, CommandSpec
from fakegpu.finetune_data import (
    FinetuneDataError,
    profile_finetune_dataset,
    profile_tokenized_records,
)
from fakegpu.finetune_data_cli import main
from fakegpu.finetune_data_report import apply_dataset_report


@pytest.fixture(autouse=True)
def registered_command(monkeypatch):
    monkeypatch.setitem(
        BUILTIN_COMMANDS,
        "profile-finetune-data",
        CommandSpec(".finetune_data_cli", "main"),
    )


@pytest.fixture
def data_path(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"input_ids": [1,2,3]}\n{"input_ids": [4,5]}\n')
    return path


@pytest.fixture
def dataset_report(data_path):
    return profile_finetune_dataset(data_path, sequence_length=8, processed=True)


@pytest.fixture
def training():
    return {
        "source_framework": "huggingface",
        "num_samples": 1024,
        "sequence_length": 8,
        "packing": False,
        "assumptions": ["Existing assumption."],
    }


def test_cli_processed_json_stdout(data_path, capsys):
    assert (
        main(
            [
                "--data",
                str(data_path),
                "--sequence-length",
                "8",
                "--processed",
                "--json",
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert output.err == ""
    report = json.loads(output.out)
    assert report["source_records"] == 2
    assert report["training_inputs"]["exact_effective_count"] is True
    assert report["padding"]["padding_tokens"] == 11


@pytest.mark.parametrize("flag", ["--json", "--json-path"])
def test_cli_json_artifact(data_path, tmp_path, capsys, flag):
    output = tmp_path / "reports" / "dataset.json"
    assert (
        main(
            [
                "--data",
                str(data_path),
                "--sequence-length",
                "8",
                "--processed",
                flag,
                str(output),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == ""
    assert json.loads(output.read_text())["tokens"]["raw_tokens"] == 5


def test_cli_multiple_files_and_explicit_generic_packing(data_path, tmp_path, capsys):
    other = tmp_path / "other.jsonl"
    other.write_text('{"input_ids": [6,7]}\n')
    assert (
        main(
            [
                "--data",
                str(data_path),
                "--data",
                str(other),
                "--sequence-length",
                "4",
                "--packing",
                "concatenate",
                "--separator-token-id",
                "9",
                "--drop-remainder",
                "--json",
                "-",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["source_records"] == 3
    assert report["training_inputs"]["num_samples"] == 2
    assert report["tokens"]["separator_tokens"] == 2
    assert report["tokens"]["dropped_remainder_tokens"] == 1
    assert report["training_inputs"]["exact_effective_count"] is False


def test_cli_plain_text_surfaces_count_scope(data_path, capsys):
    assert main(["--data", str(data_path), "--sequence-length", "8"]) == 0
    output = capsys.readouterr().out
    assert "Dataset rows: 2; output sequences: 2" in output
    assert "profiled_preprocessing_scenario" in output
    assert "not verified effective counts" in output


def test_cli_custom_text_and_local_tokenizer_flags(
    data_path, tmp_path, capsys, monkeypatch
):
    data_path.write_text('{"body": "three plain words"}\n')
    calls = []

    def tokenize(text, **kwargs):
        calls.append((text, kwargs))
        return {"input_ids": [1, 2, 3]}

    def load(path):
        assert path == str(tmp_path)
        return tokenize

    monkeypatch.setattr("fakegpu.finetune_data.load_local_tokenizer", load)
    assert (
        main(
            [
                "--data",
                str(data_path),
                "--sequence-length",
                "8",
                "--format",
                "text",
                "--text-field",
                "body",
                "--tokenizer",
                str(tmp_path),
                "--no-special-tokens",
                "--source-framework",
                "huggingface",
                "--json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["tokens"]["raw_tokens"] == 3
    assert report["source_framework"] == "huggingface"
    assert calls == [
        (
            "three plain words",
            {"add_special_tokens": False, "truncation": False, "padding": False},
        )
    ]


@pytest.mark.parametrize(
    "options, message",
    [
        (["--sequence-length", "0"], "sequence_length"),
        (["--processed", "--packing", "concatenate"], "processed rows"),
        (["--format", "unknown"], "invalid choice"),
        (["--drop-remainder"], "require concatenate"),
    ],
)
def test_cli_invalid_inputs_are_usage_errors(data_path, capsys, options, message):
    with pytest.raises(SystemExit) as caught:
        main(["--data", str(data_path), "--sequence-length", "8", *options])
    assert caught.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert message in output.err
    assert "Traceback" not in output.err


def test_apply_replaces_configured_estimate_and_preserves_inputs(
    dataset_report, training
):
    before_report, before_training = deepcopy(dataset_report), deepcopy(training)
    result = apply_dataset_report(training, dataset_report)
    assert result["num_samples"] == 2
    assert result["sequence_length"] == 8
    assert result["overridden_num_samples"] == 1024
    assert "replaces configured num_samples=1024" in result["dataset_limitations"][-1]
    assert result["assumptions"][0] == "Existing assumption."
    assert result["dataset_report"] == dataset_report
    assert training == before_training
    assert dataset_report == before_report
    result["dataset_report"]["tokens"]["raw_tokens"] = 100
    assert dataset_report == before_report


def test_apply_fills_unspecified_values_and_accepts_matching_overrides(
    dataset_report, training
):
    training.update(num_samples=None, sequence_length=None)
    result = apply_dataset_report(
        training, dataset_report, num_samples=2, sequence_length=8
    )
    assert result["num_samples"] == 2
    assert result["sequence_length"] == 8
    assert result["overridden_num_samples"] is None


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"num_samples": 3}, "explicit num_samples conflicts"),
        ({"sequence_length": 4}, "explicit sequence_length conflicts"),
        ({"num_samples": True}, "num_samples must be an integer"),
        ({"sequence_length": 0}, "sequence_length must be an integer"),
    ],
)
def test_explicit_overrides_must_agree(dataset_report, training, overrides, message):
    with pytest.raises(FinetuneDataError, match=message):
        apply_dataset_report(training, dataset_report, **overrides)


def test_configured_sequence_cap_must_agree(dataset_report, training):
    training["sequence_length"] = 16
    with pytest.raises(FinetuneDataError, match="configured sequence_length conflicts"):
        apply_dataset_report(training, dataset_report)


def test_apply_preserves_raw_scenario_limitations(data_path, training):
    report = profile_finetune_dataset(data_path, sequence_length=8)
    result = apply_dataset_report(training, report)
    assert result["dataset_report"]["training_inputs"]["exact_effective_count"] is False
    assert any(
        "raw-data preprocessing scenario" in item
        for item in result["dataset_limitations"]
    )


def test_framework_provenance_must_agree(dataset_report, training):
    dataset_report["source_framework"] = "llamafactory"
    with pytest.raises(FinetuneDataError, match="source_framework conflicts"):
        apply_dataset_report(training, dataset_report)


def test_final_processed_file_rows_can_describe_already_packed_sequences(
    dataset_report, training
):
    training["packing"] = True
    result = apply_dataset_report(training, dataset_report)
    assert result["packing"] is True
    assert result["num_samples"] == 2


@pytest.mark.parametrize("framework", ["huggingface", "llamafactory", "axolotl"])
def test_runtime_dataset_rows_never_claim_final_packing_counts(training, framework):
    training.update(source_framework=framework, packing=True)
    report = profile_tokenized_records(
        [{"input_ids": [1, 2]}], sequence_length=8, source_framework=framework
    )
    with pytest.raises(FinetuneDataError, match="runtime sampler or collator"):
        apply_dataset_report(training, report)


def test_native_unpacked_rows_can_be_applied(training):
    report = profile_tokenized_records([{"input_ids": [1, 2]}], sequence_length=8)
    assert apply_dataset_report(training, report)["num_samples"] == 1


@pytest.mark.parametrize("packing", [True, False])
def test_generic_packing_is_not_substituted_for_framework_processing(
    data_path, training, packing
):
    report = profile_finetune_dataset(
        data_path, sequence_length=8, packing="concatenate"
    )
    training["packing"] = packing
    with pytest.raises(FinetuneDataError, match="packing|packed"):
        apply_dataset_report(training, report)


def test_raw_count_cannot_be_used_when_framework_packing_enabled(data_path, training):
    report = profile_finetune_dataset(data_path, sequence_length=8)
    training["packing"] = True
    with pytest.raises(FinetuneDataError, match="requires already processed"):
        apply_dataset_report(training, report)


@pytest.mark.parametrize(
    "path, value, message",
    [
        (("schema_version",), "old", "schema_version"),
        (("provenance", "full_scan"), False, "full_scan"),
        (("provenance", "full_scan"), 1, "full_scan"),
        (("training_inputs", "num_samples"), 0, "num_samples"),
        (("training_inputs", "num_samples"), 3, "num_samples"),
        (("training_inputs", "sequence_length"), 9, "padded_tokens"),
        (("source_records",), 3, "num_samples"),
        (("training_inputs", "exact_effective_count"), False, "matching booleans"),
        (("training_inputs", "count_basis"), "verified_framework", "count_basis"),
        (("tokens", "raw_tokens"), 99, "truncation counts"),
        (("tokens", "truncated_tokens"), 1, "truncation counts"),
        (("tokens", "separator_tokens"), 1, "separator_tokens"),
        (("tokens", "dropped_remainder_tokens"), 1, "dropped_remainder_tokens"),
        (("tokens", "labeled_records"), 3, "token/row"),
        (("tokens", "supervised_tokens"), 1, "supervised_tokens"),
        (("tokens", "attention_mask_records"), 3, "attention_mask_records"),
        (("lengths", "raw", "total"), 99, "raw length count/total"),
        (("lengths", "effective", "count"), 3, "effective length count/total"),
        (("lengths", "effective", "max"), 9, "effective length range"),
        (("lengths", "effective", "mean"), 7, "effective.mean"),
        (("lengths", "raw", "p95"), 99, "percentiles"),
        (("padding", "padding_tokens"), 0, "padding_tokens"),
        (("padding", "nonpadding_tokens"), 4, "nonpadding_tokens"),
        (("padding", "source_padding_tokens"), 1, "stored_input length count/total"),
        (("padding", "padding_fraction"), float("nan"), "padding_fraction"),
        (("provenance", "files", 0, "records"), 3, "file counts"),
        (("provenance", "files", 0, "sha256"), "a" * 64, "content identity"),
        (("provenance", "content_sha256"), "malformed", "SHA-256"),
        (("limitations",), "missing list", "list of strings"),
    ],
)
def test_edited_report_core_inconsistencies_are_rejected(
    dataset_report, training, path, value, message
):
    target = dataset_report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(FinetuneDataError, match=message):
        apply_dataset_report(training, dataset_report)


def test_missing_sections_fail_as_value_error(dataset_report, training):
    del dataset_report["lengths"]
    with pytest.raises(FinetuneDataError, match="lengths must be a mapping"):
        apply_dataset_report(training, dataset_report)
