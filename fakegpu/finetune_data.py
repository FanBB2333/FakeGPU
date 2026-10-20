"""Stream local JSONL data into explicit, reproducible token-count scenarios.

This module does not emulate a training framework's dataset preparation. Supply
its exported, processed ``input_ids`` rows with ``processed=True`` to measure the
actual row count. Raw text uses a caller-supplied or strictly local tokenizer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "fakegpu.finetune_data.v1"


class FinetuneDataError(ValueError):
    pass


def load_local_tokenizer(path: str | Path) -> Any:
    """Load an optional Transformers tokenizer without downloads or custom code."""
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise FinetuneDataError("tokenizer_path must be an existing local directory")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise FinetuneDataError(
            "Raw text tokenization requires transformers or a supplied tokenizer"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False
        )
    except (OSError, ValueError, TypeError, ImportError) as exc:
        raise FinetuneDataError(f"Cannot load local tokenizer: {exc}") from exc


def profile_finetune_dataset(
    paths: str | Path | Sequence[str | Path],
    *,
    sequence_length: int,
    tokenizer: Any = None,
    tokenizer_path: str | Path | None = None,
    record_format: str = "auto",
    text_field: str = "text",
    messages_field: str = "messages",
    packing: str = "none",
    drop_remainder: bool = False,
    add_special_tokens: bool = True,
    separator_token_id: int | None = None,
    processed: bool = False,
    source_framework: str | None = None,
) -> dict[str, Any]:
    """Fully profile local JSONL text, messages, or tokenized training rows.

    Raw rows are right-truncated, then budgeted at fixed ``sequence_length``.
    Generic ``concatenate`` packing instead joins all tokens, optionally inserts
    the requested separator between rows, and splits fixed-length sequences.
    It does not reproduce a framework's packing. ``processed=True`` requires
    input_ids and forbids truncation or secondary packing. See the report's
    limitations for the distinction between scenarios and processed row counts.
    """
    return _profile_dataset(
        paths,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        record_format=record_format,
        text_field=text_field,
        messages_field=messages_field,
        packing=packing,
        drop_remainder=drop_remainder,
        add_special_tokens=add_special_tokens,
        separator_token_id=separator_token_id,
        processed=processed,
        source_framework=source_framework,
    )


def profile_tokenized_records(
    records: Iterable[Mapping[str, Any]],
    *,
    sequence_length: int,
    source_framework: str | None = None,
) -> dict[str, Any]:
    """Scan already processed rows once, without writing an intermediate file.

    The iterable must contain JSON-compatible mappings with flat input_ids and
    optional attention_mask/labels. Content identity hashes ordered canonical
    JSON rows, including metadata. Rows wider than sequence_length are rejected.
    Callers must select the effective training split before invoking this API.
    """
    if not isinstance(records, Iterable) or isinstance(records, (str, bytes, Mapping)):
        raise FinetuneDataError("records must be an iterable of tokenized mappings")
    return _profile_dataset(
        [],
        sequence_length=sequence_length,
        processed=True,
        source_framework=source_framework,
        native_records=records,
    )


def _profile_dataset(
    paths: str | Path | Sequence[str | Path],
    *,
    sequence_length: int,
    tokenizer: Any = None,
    tokenizer_path: str | Path | None = None,
    record_format: str = "auto",
    text_field: str = "text",
    messages_field: str = "messages",
    packing: str = "none",
    drop_remainder: bool = False,
    add_special_tokens: bool = True,
    separator_token_id: int | None = None,
    processed: bool = False,
    source_framework: str | None = None,
    native_records: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fully scan JSONL files; never extrapolate a sample to the whole dataset.

    ``none`` truncates raw rows on the right and budgets padding to the given
    sequence length. ``concatenate`` removes masked padding, joins all tokens
    without per-row truncation, and divides them into fixed-length sequences.
    A separator is inserted *between* rows only when explicitly requested.
    This generic packing algorithm is not LLaMA-Factory or Axolotl packing.

    ``processed=True`` requires input_ids, preserves row boundaries, and rejects
    rows wider than the cap. It attests only to the provided rows; the caller is
    responsible for exporting the effective training split from the framework.
    Optional labels count nonignored, nonpadding positions before causal shift.
    Percentiles use the exact nearest-rank convention; memory is proportional
    to unique lengths plus the largest individual record, not all token data.
    """
    _integer(sequence_length, "sequence_length", minimum=1)
    for name, value in (
        ("processed", processed),
        ("drop_remainder", drop_remainder),
        ("add_special_tokens", add_special_tokens),
    ):
        if not isinstance(value, bool):
            raise FinetuneDataError(f"{name} must be a boolean")
    if record_format not in {"auto", "text", "messages", "input_ids"}:
        raise FinetuneDataError(
            "record_format must be auto, text, messages or input_ids"
        )
    if packing not in {"none", "concatenate"}:
        raise FinetuneDataError("packing must be none or concatenate")
    if processed and (packing != "none" or record_format not in {"auto", "input_ids"}):
        raise FinetuneDataError(
            "processed rows require input_ids without secondary packing"
        )
    if packing == "none" and (drop_remainder or separator_token_id is not None):
        raise FinetuneDataError(
            "drop_remainder and separator_token_id require concatenate packing"
        )
    if separator_token_id is not None:
        _integer(separator_token_id, "separator_token_id")
    if not all(isinstance(name, str) and name for name in (text_field, messages_field)):
        raise FinetuneDataError(
            "text_field and messages_field must be non-empty strings"
        )
    if len({text_field, messages_field, "input_ids"}) != 3:
        raise FinetuneDataError(
            "text_field, messages_field and input_ids must be distinct"
        )
    if source_framework is not None and (
        not isinstance(source_framework, str) or not source_framework.strip()
    ):
        raise FinetuneDataError("source_framework must be a non-empty string")
    if tokenizer is not None and tokenizer_path is not None:
        raise FinetuneDataError("supply tokenizer or tokenizer_path, not both")
    if tokenizer_path is not None:
        tokenizer = load_local_tokenizer(tokenizer_path)
    source_paths = _paths(paths) if native_records is None else []
    tokenizer_identity = _tokenizer_identity(tokenizer, tokenizer_path)

    raw_lengths: Counter[int] = Counter()
    effective_lengths: Counter[int] = Counter()
    stored_lengths: Counter[int] = Counter()
    files: list[dict[str, Any]] = []
    requested_format = "input_ids" if processed else record_format
    selected_format = requested_format
    records = raw_tokens = retained_tokens = source_padding = truncated_records = 0
    labeled_records = supervised_tokens = attention_mask_records = 0

    native_provenance: dict[str, Any] = {}
    rows = (
        _jsonl_records(source_paths, files)
        if native_records is None
        else _native_records(native_records, native_provenance)
    )
    for location, row in rows:
        try:
            current_format = _record_format(
                row, requested_format, text_field, messages_field
            )
            if current_format == "input_ids" and "attention_mask" in row:
                attention_mask_records += 1
            if selected_format == "auto":
                selected_format = current_format
            elif current_format != selected_format:
                raise FinetuneDataError(
                    "mixed record formats are unsupported; select one format explicitly"
                )
            ids, mask, labels = _tokens(
                row,
                current_format,
                tokenizer,
                text_field,
                messages_field,
                add_special_tokens,
            )
            width = len(ids)
            if not width or not any(mask):
                raise FinetuneDataError(
                    "each row must contain at least one nonpadding token"
                )
            if processed and width > sequence_length:
                raise FinetuneDataError(
                    f"processed input_ids length {width} exceeds sequence_length {sequence_length}"
                )
            effective_mask = (
                mask if packing == "concatenate" else mask[:sequence_length]
            )
            raw_count, kept_count = sum(mask), sum(effective_mask)
            if not kept_count:
                raise FinetuneDataError(
                    "right truncation would leave a row with only padding"
                )
            raw_lengths[raw_count] += 1
            effective_lengths[kept_count] += 1
            stored_lengths[width] += 1
            raw_tokens += raw_count
            retained_tokens += kept_count
            source_padding += width - raw_count
            truncated_records += int(raw_count > kept_count)
            if labels is not None:
                labeled_records += 1
                supervised_tokens += sum(
                    active and label != -100
                    for active, label in zip(effective_mask, labels)
                )
            records += 1
        except (ValueError, TypeError, UnicodeError) as exc:
            raise FinetuneDataError(f"{location}: {exc}") from exc
    if not records:
        raise FinetuneDataError("dataset contains no records")

    separator_tokens = records - 1 if separator_token_id is not None else 0
    available_tokens = retained_tokens + separator_tokens
    discarded_tokens = 0
    sequences = records
    if packing == "concatenate":
        sequences, remainder = divmod(available_tokens, sequence_length)
        if drop_remainder:
            discarded_tokens = remainder
        else:
            sequences += int(remainder > 0)
        effective_lengths = Counter(
            {sequence_length: available_tokens // sequence_length}
        )
        if remainder and not drop_remainder:
            effective_lengths[remainder] += 1
    nonpadding_tokens = available_tokens - discarded_tokens
    padded_tokens = sequences * sequence_length
    limitations = [
        "Padding budgets every output sequence to sequence_length; dynamic batch padding, shuffling and distributed sampling are not modeled.",
        "No filtering, train/validation splitting, resampling or framework preprocessing is performed.",
    ]
    if processed:
        limitations.append(
            "The exact count describes the supplied processed rows; source-framework execution and training-split provenance are not verified."
        )
    else:
        limitations.append(
            "Counts describe the explicitly selected preprocessing scenario and are not verified effective counts for a training framework."
        )
    if packing == "concatenate":
        limitations.append(
            "Generic token concatenation can split examples and is not LLaMA-Factory or Axolotl packing; label statistics after packing are unavailable."
        )
    if selected_format == "input_ids":
        limitations.append(
            "input_ids are used as supplied; their original tokenizer and template cannot be inferred."
        )
        if attention_mask_records != records:
            limitations.append(
                "Rows without attention_mask count every input_id as nonpadding; padding cannot be inferred from token IDs alone."
            )
    elif tokenizer_identity and not tokenizer_identity["content_identified"]:
        limitations.append(
            "The supplied tokenizer has no verifiable local files or serialized tokenizer state."
        )
    if labeled_records:
        limitations.append(
            "Supervised tokens count nonignored labels at nonpadding positions before any causal loss shift."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source_records": records,
        "record_format": selected_format,
        "processed": processed,
        "source_framework": source_framework,
        "training_inputs": {
            "num_samples": sequences,
            "sequence_length": sequence_length,
            "exact_effective_count": processed,
            "count_basis": "provided_processed_sequences"
            if processed
            else "profiled_preprocessing_scenario",
        },
        "packing": {
            "strategy": packing,
            "drop_remainder": drop_remainder,
            "separator_token_id": separator_token_id,
        },
        "tokens": {
            "raw_tokens": raw_tokens,
            "retained_tokens": retained_tokens,
            "truncated_tokens": raw_tokens - retained_tokens,
            "truncated_records": truncated_records,
            "separator_tokens": separator_tokens,
            "dropped_remainder_tokens": discarded_tokens,
            "supervised_tokens": supervised_tokens
            if labeled_records == records and packing == "none"
            else None,
            "labeled_records": labeled_records,
            "attention_mask_records": attention_mask_records
            if selected_format == "input_ids"
            else None,
        },
        "lengths": {
            "raw": _length_summary(raw_lengths),
            "effective": _length_summary(effective_lengths),
            "stored_input": _length_summary(stored_lengths),
        },
        "padding": {
            "assumption": "fixed_sequence_length",
            "padded_tokens": padded_tokens,
            "nonpadding_tokens": nonpadding_tokens,
            "padding_tokens": padded_tokens - nonpadding_tokens,
            "padding_fraction": (padded_tokens - nonpadding_tokens) / padded_tokens
            if padded_tokens
            else 0.0,
            "source_padding_tokens": source_padding,
        },
        "provenance": {
            "full_scan": True,
            "files": files,
            "source": "local_jsonl" if native_records is None else "native_in_memory",
            "content_sha256": (
                _json_hash([{k: f[k] for k in ("sha256", "size_bytes")} for f in files])
                if native_records is None
                else native_provenance["content_sha256"]
            ),
            "content_hash_format": "ordered_file_sha256_and_size"
            if native_records is None
            else "canonical_json_rows_v1",
            "tokenizer": tokenizer_identity if selected_format != "input_ids" else None,
            "tokenization": {
                "text_field": text_field,
                "messages_field": messages_field,
                "add_special_tokens": add_special_tokens
                if selected_format == "text"
                else None,
                "add_generation_prompt": False
                if selected_format == "messages"
                else None,
                "truncation_side": None
                if processed or packing == "concatenate"
                else "right",
            },
        },
        "limitations": limitations,
    }


def _paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not isinstance(paths, Sequence) or not paths:
        raise FinetuneDataError("paths must contain at least one local JSONL file")
    resolved: list[Path] = []
    for value in paths:
        if not isinstance(value, (str, Path)):
            raise FinetuneDataError("dataset paths must be strings or Paths")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FinetuneDataError(f"dataset path must be a local JSONL file: {path}")
        if path in resolved:
            raise FinetuneDataError(f"duplicate dataset path: {path}")
        resolved.append(path)
    return resolved


def _jsonl_records(
    paths: list[Path], files: list[dict[str, Any]]
) -> Iterator[tuple[str, dict[str, Any]]]:
    for path in paths:
        digest = hashlib.sha256()
        records = size_bytes = blank_lines = 0
        try:
            with path.open("rb") as stream:
                for line_number, line in enumerate(stream, 1):
                    digest.update(line)
                    size_bytes += len(line)
                    if not line.strip():
                        blank_lines += 1
                        continue
                    location = f"{path}:{line_number}"
                    try:
                        row = json.loads(
                            line.decode("utf-8"),
                            object_pairs_hook=_unique_object,
                            parse_constant=_invalid_constant,
                            parse_float=_finite_float,
                        )
                        if not isinstance(row, dict):
                            raise FinetuneDataError("each record must be a JSON object")
                    except (ValueError, TypeError, UnicodeError) as exc:
                        raise FinetuneDataError(f"{location}: {exc}") from exc
                    records += 1
                    yield location, row
        except OSError as exc:
            raise FinetuneDataError(f"Cannot read dataset {path}: {exc}") from exc
        files.append(
            {
                "path": str(path),
                "sha256": digest.hexdigest(),
                "size_bytes": size_bytes,
                "records": records,
                "blank_lines": blank_lines,
            }
        )


def _native_records(
    records: Iterable[Mapping[str, Any]], provenance: dict[str, Any]
) -> Iterator[tuple[str, dict[str, Any]]]:
    digest = hashlib.sha256()
    for index, record in enumerate(records):
        location = f"native record {index + 1}"
        if not isinstance(record, Mapping):
            raise FinetuneDataError(f"{location}: each record must be a mapping")
        row = dict(record)
        try:
            encoded = json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            digest.update(encoded.encode("utf-8"))
            digest.update(b"\n")
        except (ValueError, TypeError, UnicodeError) as exc:
            raise FinetuneDataError(
                f"{location}: records must be JSON-compatible: {exc}"
            ) from exc
        yield location, row
    provenance["content_sha256"] = digest.hexdigest()


def _record_format(
    row: dict[str, Any], requested: str, text: str, messages: str
) -> str:
    fields = {"text": text, "messages": messages, "input_ids": "input_ids"}
    if requested == "auto":
        matches = [name for name, field in fields.items() if field in row]
        if len(matches) != 1:
            raise FinetuneDataError(
                "auto format requires exactly one of text, messages, input_ids; select record_format explicitly for ambiguous records"
            )
        return matches[0]
    if fields[requested] not in row:
        raise FinetuneDataError(
            f"missing {fields[requested]!r} field for {requested} record"
        )
    return requested


def _tokens(
    row: dict[str, Any],
    format_name: str,
    tokenizer: Any,
    text_field: str,
    messages_field: str,
    add_special_tokens: bool,
) -> tuple[list[int], list[int], list[int] | None]:
    if format_name == "input_ids":
        encoded = row
    elif tokenizer is None:
        raise FinetuneDataError(
            "text and messages require a tokenizer or local tokenizer_path"
        )
    elif format_name == "text":
        if not isinstance(row[text_field], str):
            raise FinetuneDataError(f"{text_field} must be a string")
        encoded = tokenizer(
            row[text_field],
            add_special_tokens=add_special_tokens,
            truncation=False,
            padding=False,
        )
    else:
        messages = row[messages_field]
        if not isinstance(messages, list) or not messages:
            raise FinetuneDataError(
                f"{messages_field} must be a non-empty message list"
            )
        for message in messages:
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not message["role"]
                or not isinstance(message.get("content"), str)
            ):
                raise FinetuneDataError(
                    "each message requires a non-empty role and string content; multimodal messages are unsupported"
                )
        method = getattr(tokenizer, "apply_chat_template", None)
        if not callable(method):
            raise FinetuneDataError(
                "messages require a tokenizer with apply_chat_template"
            )
        encoded = method(messages, tokenize=True, add_generation_prompt=False)
    if not isinstance(encoded, Mapping):
        encoded = {"input_ids": encoded}
    ids = _token_list(encoded.get("input_ids"), "input_ids")
    mask = encoded.get("attention_mask")
    if mask is None and "attention_mask" not in encoded:
        mask = [1] * len(ids)
    else:
        mask = _token_list(mask, "attention_mask")
        if len(mask) != len(ids):
            raise FinetuneDataError("attention_mask length must match input_ids")
        # Packed framework exports may use positive segment IDs instead of 1.
        mask = [int(value > 0) for value in mask]
    labels = None
    if "labels" in encoded:
        labels = _token_list(encoded["labels"], "labels", labels=True)
        if len(labels) != len(ids):
            raise FinetuneDataError("labels length must match input_ids")
    return ids, mask, labels


def _token_list(value: Any, name: str, *, labels: bool = False) -> list[int]:
    if not isinstance(value, (list, tuple)) or any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or (token < 0 and not (labels and token == -100))
        for token in value
    ):
        raise FinetuneDataError(
            f"{name} must be a flat list of non-negative integer IDs"
            + (" or -100" if labels else "")
        )
    return list(value)


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinetuneDataError(f"{name} must be an integer >= {minimum}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FinetuneDataError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise FinetuneDataError(f"non-finite JSON constant {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise FinetuneDataError(f"non-finite JSON number {value}")
    return number


def _length_summary(histogram: Counter[int]) -> dict[str, Any]:
    ordered = sorted((length, count) for length, count in histogram.items() if count)
    count = sum(amount for _, amount in ordered)
    total = sum(length * amount for length, amount in ordered)
    percentiles: dict[str, int | None] = {}
    for percentile in (50, 90, 95, 99):
        rank = math.ceil(count * percentile / 100)
        seen = 0
        percentiles[f"p{percentile}"] = None
        for length, amount in ordered:
            seen += amount
            if seen >= rank:
                percentiles[f"p{percentile}"] = length
                break
    return {
        "count": count,
        "total": total,
        "min": ordered[0][0] if ordered else None,
        "max": ordered[-1][0] if ordered else None,
        "mean": total / count if count else None,
        **percentiles,
        "percentile_method": "nearest_rank",
    }


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _tokenizer_identity(
    tokenizer: Any, path: str | Path | None
) -> dict[str, Any] | None:
    if tokenizer is None:
        return None
    directory = Path(path).expanduser().resolve() if path is not None else None
    if directory is None:
        name = getattr(tokenizer, "name_or_path", None)
        if isinstance(name, str) and name and Path(name).is_dir():
            directory = Path(name).resolve()
    assets = []
    if directory is not None:
        filenames = {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "tokenizer.model",
            "spiece.model",
            "sentencepiece.bpe.model",
            "chat_template.jinja",
            "chat_template.json",
            "tekken.json",
        }
        candidates = set(directory.glob("*.tiktoken"))
        candidates.update(directory / name for name in filenames)
        candidates.update(directory.glob("chat_templates/*.jinja"))
        for asset in sorted(candidates):
            if asset.is_file():
                digest = hashlib.sha256()
                with asset.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                assets.append(
                    {
                        "path": str(asset.relative_to(directory)),
                        "sha256": digest.hexdigest(),
                    }
                )
    backend = getattr(tokenizer, "backend_tokenizer", None)
    serialize = getattr(backend, "to_str", None)
    serialized = serialize() if callable(serialize) else None
    special_ids = {}
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            special_ids[name] = value
    template = getattr(tokenizer, "chat_template", None)
    if template is not None and not isinstance(template, (str, dict, list)):
        raise FinetuneDataError("tokenizer chat_template must be JSON-serializable")
    identity = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "local_path": str(directory) if directory else None,
        "files": assets,
        "backend_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        if isinstance(serialized, str)
        else None,
        "chat_template_sha256": _json_hash(template) if template is not None else None,
        "special_token_ids": special_ids,
        "settings": {
            name: getattr(tokenizer, name)
            for name in (
                "add_bos_token",
                "add_eos_token",
                "padding_side",
                "truncation_side",
                "clean_up_tokenization_spaces",
                "split_special_tokens",
            )
            if isinstance(getattr(tokenizer, name, None), (str, bool, int))
        },
        "content_identified": bool(assets or isinstance(serialized, str)),
    }
    identity["identity_sha256"] = _json_hash(
        {k: v for k, v in identity.items() if k != "local_path"}
    )
    return identity


__all__ = [
    "FinetuneDataError",
    "load_local_tokenizer",
    "profile_finetune_dataset",
    "profile_tokenized_records",
]
