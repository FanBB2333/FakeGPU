"""Validate and bind measured dataset reports to estimator settings."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from .finetune_data import SCHEMA_VERSION, FinetuneDataError
from .structured_io import is_sha256_hex


def apply_dataset_report(
    training_config: Mapping[str, Any],
    dataset_report: Mapping[str, Any],
    *,
    num_samples: int | None = None,
    sequence_length: int | None = None,
    packed_sampling_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind consistent measured counts to a copy of normalized training settings.

    This validates report arithmetic and provenance structure; it cannot prove
    the authenticity of an edited report without re-reading the original data.
    Explicit overrides must agree. A configured sample estimate is replaced
    with the report count and the previous value is recorded. Sequence caps
    must agree because changing the cap changes tokenization and padding.
    """
    if not isinstance(training_config, Mapping):
        raise FinetuneDataError("training_config must be a mapping")
    _validate_report(dataset_report)
    inputs = dataset_report["training_inputs"]
    for key, override in (
        ("num_samples", num_samples),
        ("sequence_length", sequence_length),
    ):
        if override is not None:
            _integer(override, key, minimum=1)
            if override != inputs[key]:
                raise FinetuneDataError(f"explicit {key} conflicts with dataset report")
    configured_length = training_config.get("sequence_length")
    if configured_length is not None:
        _integer(configured_length, "training sequence_length", minimum=1)
        if configured_length != inputs["sequence_length"]:
            raise FinetuneDataError(
                "configured sequence_length conflicts with dataset report; profile using the training cap"
            )
    framework = training_config.get("source_framework")
    source_framework = dataset_report.get("source_framework")
    if (
        source_framework is not None
        and framework is not None
        and source_framework != framework
    ):
        raise FinetuneDataError("source_framework conflicts with dataset report")
    packing = training_config.get("packing", False)
    if not isinstance(packing, bool):
        raise FinetuneDataError("training packing must be a boolean")
    processed = dataset_report["processed"]
    if packing and not processed:
        raise FinetuneDataError(
            "framework packing requires already processed sequence counts; raw or generic concatenate scenarios cannot establish framework packed counts"
        )
    if packing and dataset_report["provenance"]["source"] == "native_in_memory":
        if packed_sampling_plan is None:
            raise FinetuneDataError(
                "native_in_memory rows cannot establish final packed counts when packing may occur in the runtime sampler or collator; supply final processed JSONL sequences"
            )
        from ._finetune_sampler import PACKED_SCHEMA_VERSION, validate_sampler_plan

        sampled = validate_sampler_plan(packed_sampling_plan)
        if (
            sampled["schema_version"] != PACKED_SCHEMA_VERSION
            or training_config.get("sampler_observation_version")
            != PACKED_SCHEMA_VERSION
            or sampled["framework"] != framework
            or sampled["dataset"]
            != {
                "num_samples": inputs["num_samples"],
                "content_sha256": dataset_report["provenance"].get("content_sha256"),
            }
        ):
            raise FinetuneDataError(
                "packed native counts require matching verified source rows and native bins"
            )
    if not packing and dataset_report["packing"]["strategy"] == "concatenate":
        raise FinetuneDataError(
            "generic concatenate report conflicts with training packing=false; profile the actual processed training sequences"
        )
    previous_count = training_config.get("num_samples")
    if previous_count is not None:
        _integer(previous_count, "training num_samples", minimum=1)
    assumptions = training_config.get("assumptions", [])
    if not isinstance(assumptions, list) or not all(
        isinstance(item, str) for item in assumptions
    ):
        raise FinetuneDataError("training assumptions must be a list of strings")
    limitations = list(dataset_report["limitations"])
    if not processed:
        limitations.append(
            "Dataset counts are an explicit raw-data preprocessing scenario; they are not verified framework-effective training sequences."
        )
    overwritten = (
        previous_count
        if previous_count is not None and previous_count != inputs["num_samples"]
        else None
    )
    if overwritten is not None:
        limitations.append(
            f"Dataset report replaces configured num_samples={overwritten} with measured num_samples={inputs['num_samples']}."
        )
    result = deepcopy(dict(training_config))
    result.update(
        {
            "num_samples": inputs["num_samples"],
            "sequence_length": inputs["sequence_length"],
            "dataset_report": deepcopy(dict(dataset_report)),
            "dataset_limitations": limitations,
            "overridden_num_samples": overwritten,
            "assumptions": [*assumptions, *limitations],
        }
    )
    return result


def _validate_report(report: Mapping[str, Any]) -> None:
    if (
        not isinstance(report, Mapping)
        or report.get("schema_version") != SCHEMA_VERSION
    ):
        raise FinetuneDataError(
            f"dataset report schema_version must be {SCHEMA_VERSION}"
        )
    inputs = _mapping(report, "training_inputs")
    provenance = _mapping(report, "provenance")
    tokens = _mapping(report, "tokens")
    lengths = _mapping(report, "lengths")
    padding = _mapping(report, "padding")
    packing = _mapping(report, "packing")
    if provenance.get("full_scan") is not True:
        raise FinetuneDataError("dataset report requires full_scan=true")
    records = _integer(report.get("source_records"), "source_records", minimum=1)
    count = _integer(inputs.get("num_samples"), "num_samples", minimum=1)
    cap = _integer(inputs.get("sequence_length"), "sequence_length", minimum=1)
    processed = report.get("processed")
    if (
        not isinstance(processed, bool)
        or inputs.get("exact_effective_count") is not processed
    ):
        raise FinetuneDataError(
            "processed and exact_effective_count must be matching booleans"
        )
    expected_basis = (
        "provided_processed_sequences"
        if processed
        else "profiled_preprocessing_scenario"
    )
    if inputs.get("count_basis") != expected_basis:
        raise FinetuneDataError(
            "dataset report count_basis conflicts with processed status"
        )
    format_name = report.get("record_format")
    if format_name not in {"text", "messages", "input_ids"} or (
        processed and format_name != "input_ids"
    ):
        raise FinetuneDataError(
            "dataset report record_format conflicts with processed status"
        )
    source_framework = report.get("source_framework")
    if source_framework is not None and (
        not isinstance(source_framework, str) or not source_framework.strip()
    ):
        raise FinetuneDataError("dataset source_framework must be a non-empty string")
    strategy = packing.get("strategy")
    drop = packing.get("drop_remainder")
    if strategy not in {"none", "concatenate"} or not isinstance(drop, bool):
        raise FinetuneDataError("dataset report has invalid packing settings")
    separator = packing.get("separator_token_id")
    if separator is not None:
        _integer(separator, "separator_token_id")
    if (processed and strategy != "none") or (
        strategy == "none" and (drop or separator is not None)
    ):
        raise FinetuneDataError("dataset report packing settings conflict")
    raw = _integer(tokens.get("raw_tokens"), "raw_tokens", minimum=1)
    retained = _integer(tokens.get("retained_tokens"), "retained_tokens", minimum=1)
    truncated = _integer(tokens.get("truncated_tokens"), "truncated_tokens")
    truncated_rows = _integer(tokens.get("truncated_records"), "truncated_records")
    separators = _integer(tokens.get("separator_tokens"), "separator_tokens")
    discarded = _integer(
        tokens.get("dropped_remainder_tokens"), "dropped_remainder_tokens"
    )
    labeled = _integer(tokens.get("labeled_records"), "labeled_records")
    if (
        retained + truncated != raw
        or truncated_rows > records
        or bool(truncated_rows) != bool(truncated)
    ):
        raise FinetuneDataError("dataset report truncation counts are inconsistent")
    if truncated < truncated_rows or labeled > records:
        raise FinetuneDataError("dataset report token/row counts are inconsistent")
    supervised = tokens.get("supervised_tokens")
    if supervised is not None:
        _integer(supervised, "supervised_tokens")
        if supervised > retained or labeled != records or strategy != "none":
            raise FinetuneDataError("dataset report supervised_tokens is inconsistent")
    elif labeled == records and strategy == "none":
        raise FinetuneDataError(
            "dataset report supervised_tokens is missing for fully labeled data"
        )
    mask_records = tokens.get("attention_mask_records")
    if format_name == "input_ids":
        if _integer(mask_records, "attention_mask_records") > records:
            raise FinetuneDataError(
                "dataset report attention_mask_records exceeds source_records"
            )
    elif mask_records is not None:
        raise FinetuneDataError("raw text report cannot declare attention_mask_records")
    expected_separators = records - 1 if separator is not None else 0
    if separators != expected_separators:
        raise FinetuneDataError("dataset report separator_tokens is inconsistent")
    available = retained + separators
    if strategy == "none":
        if count != records or discarded:
            raise FinetuneDataError(
                "unpacked dataset report num_samples or dropped_remainder_tokens is inconsistent"
            )
    else:
        complete, remainder = divmod(available, cap)
        if (
            count != complete + int(not drop and remainder > 0)
            or discarded != (remainder if drop else 0)
            or truncated
        ):
            raise FinetuneDataError(
                "concatenate dataset report counts are inconsistent"
            )
    expected_tokens = available - discarded
    source_padding = _integer(
        padding.get("source_padding_tokens"), "source_padding_tokens"
    )
    _lengths(_mapping(lengths, "raw"), count=records, total=raw, name="raw")
    _lengths(
        _mapping(lengths, "stored_input"),
        count=records,
        total=raw + source_padding,
        name="stored_input",
    )
    _lengths(
        _mapping(lengths, "effective"),
        count=count,
        total=expected_tokens,
        name="effective",
        cap=cap,
    )
    if processed and (truncated or lengths["stored_input"]["max"] > cap):
        raise FinetuneDataError(
            "processed dataset report cannot truncate rows or exceed sequence_length"
        )
    if padding.get("assumption") != "fixed_sequence_length":
        raise FinetuneDataError("dataset report requires fixed_sequence_length padding")
    for key, expected in (
        ("padded_tokens", count * cap),
        ("nonpadding_tokens", expected_tokens),
        ("padding_tokens", count * cap - expected_tokens),
    ):
        if _integer(padding.get(key), key) != expected:
            raise FinetuneDataError(f"dataset report {key} is inconsistent")
    _number_equal(
        padding.get("padding_fraction"),
        (count * cap - expected_tokens) / (count * cap),
        "padding_fraction",
    )
    limitations = report.get("limitations")
    if not isinstance(limitations, list) or not all(
        isinstance(value, str) for value in limitations
    ):
        raise FinetuneDataError("dataset report limitations must be a list of strings")
    _provenance(provenance, records, processed)


def _provenance(provenance: Mapping[str, Any], records: int, processed: bool) -> None:
    _digest(provenance.get("content_sha256"), "content_sha256")
    files = provenance.get("files")
    if not isinstance(files, list):
        raise FinetuneDataError("dataset report provenance.files must be a list")
    source = provenance.get("source")
    if source == "native_in_memory":
        if (
            files
            or not processed
            or provenance.get("content_hash_format") != "canonical_json_rows_v1"
        ):
            raise FinetuneDataError("native_in_memory provenance is inconsistent")
        return
    if (
        source != "local_jsonl"
        or not files
        or provenance.get("content_hash_format") != "ordered_file_sha256_and_size"
    ):
        raise FinetuneDataError(
            "dataset report requires local_jsonl or native_in_memory provenance"
        )
    file_records = 0
    identities = []
    for item in files:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("path"), str)
            or not item["path"]
        ):
            raise FinetuneDataError("dataset report contains invalid file provenance")
        _digest(item.get("sha256"), "file sha256")
        size = _integer(item.get("size_bytes"), "file size_bytes")
        file_records += _integer(item.get("records"), "file records")
        _integer(item.get("blank_lines"), "file blank_lines")
        identities.append({"sha256": item["sha256"], "size_bytes": size})
    digest = hashlib.sha256(
        json.dumps(
            identities, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if file_records != records or digest != provenance["content_sha256"]:
        raise FinetuneDataError(
            "dataset report file counts or content identity are inconsistent"
        )


def _mapping(parent: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = parent.get(name)
    if not isinstance(value, Mapping):
        raise FinetuneDataError(f"dataset report {name} must be a mapping")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinetuneDataError(f"{name} must be an integer >= {minimum}")
    return value


def _number_equal(value: Any, expected: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise FinetuneDataError(f"dataset report {name} is inconsistent")


def _digest(value: Any, name: str) -> None:
    if not is_sha256_hex(value):
        raise FinetuneDataError(
            f"dataset report {name} must be a lowercase SHA-256 digest"
        )


def _lengths(
    summary: Mapping[str, Any],
    *,
    count: int,
    total: int,
    name: str,
    cap: int | None = None,
) -> None:
    if (
        _integer(summary.get("count"), f"{name}.count", 1) != count
        or _integer(summary.get("total"), f"{name}.total", 1) != total
    ):
        raise FinetuneDataError(
            f"dataset report {name} length count/total is inconsistent"
        )
    low = _integer(summary.get("min"), f"{name}.min", 1)
    high = _integer(summary.get("max"), f"{name}.max", 1)
    if (
        low > high
        or not low * count <= total <= high * count
        or (cap is not None and high > cap)
    ):
        raise FinetuneDataError(f"dataset report {name} length range is inconsistent")
    _number_equal(summary.get("mean"), total / count, f"{name}.mean")
    if summary.get("percentile_method") != "nearest_rank":
        raise FinetuneDataError(
            f"dataset report {name} percentile method is unsupported"
        )
    previous = low
    for percentile in ("p50", "p90", "p95", "p99"):
        value = _integer(summary.get(percentile), f"{name}.{percentile}", 1)
        if not previous <= value <= high:
            raise FinetuneDataError(
                f"dataset report {name} percentiles are inconsistent"
            )
        previous = value


__all__ = ["apply_dataset_report"]
