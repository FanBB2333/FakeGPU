"""Prove native packed membership before constructing an isolated block mask.

Native preprocessing and collation retain token/label values. Position resets
alone are not treated as evidence of original sample boundaries: flat packed
rows must supply segment IDs, and Axolotl bins retain their source records.
No training-budget gate is opened by this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from operator import index

from ._finetune_segments import segment_ranges, validate_segment_inputs
from .finetune_data import _token_list
from .finetune_estimator import FinetuneEstimateError


def _flat_membership(row):
    ids, labels, mask = (
        row.get(key) for key in ("input_ids", "labels", "attention_mask")
    )
    if not all(isinstance(value, list) for value in (ids, labels, mask)) or not ids:
        raise FinetuneEstimateError(
            "packed source rows require token, label and segment-ID lists"
        )
    if len(ids) != len(labels) or len(ids) != len(mask):
        raise FinetuneEstimateError("packed source lists must have equal lengths")
    ids = _token_list(ids, "input_ids")
    labels = _token_list(labels, "labels", labels=True)
    positions, expected_segment, prior, offset = [], 1, 0, 0
    active = []
    for segment in mask:
        if isinstance(segment, bool) or not isinstance(segment, int) or segment < 0:
            raise FinetuneEstimateError(
                "packed source segment IDs must be nonnegative integers"
            )
        active.append(segment > 0)
        if segment == 0:
            positions.append(0)
            prior = 0
            continue
        if segment != prior:
            if segment != expected_segment:
                raise FinetuneEstimateError(
                    "packed segment IDs must start at one and increase contiguously"
                )
            expected_segment += 1
            offset = 0
        positions.append(offset)
        prior, offset = segment, offset + 1
    spans = segment_ranges(positions, active)
    size = spans[-1][1]
    if "position_ids" in row:
        provided = _token_list(row["position_ids"], "position_ids")
        if len(provided) != len(ids) or provided[:size] != positions[:size]:
            raise FinetuneEstimateError(
                "source position resets disagree with segment IDs"
            )
    if any(value != -100 for value in labels[size:]):
        raise FinetuneEstimateError("packed source padding labels must be ignored")
    return {
        "ids": ids[:size],
        "labels": labels[:size],
        "positions": positions[:size],
        "spans": spans,
    }


def packed_source_membership(features, *, framework):
    """Read flat HF/Factory records or Axolotl's explicit list-of-record bins."""
    if not isinstance(features, list) or not features:
        raise FinetuneEstimateError("packing requires a nonempty native feature batch")
    if framework in {"huggingface", "llamafactory"}:
        if not all(isinstance(row, Mapping) for row in features):
            raise FinetuneEstimateError("flat packed batches require mapping rows")
        return [_flat_membership(row) for row in features]
    if framework != "axolotl":
        raise FinetuneEstimateError("unsupported packed source framework")
    result = []
    for records in features:
        if not isinstance(records, list) or not records:
            raise FinetuneEstimateError(
                "Axolotl packing requires explicit nonempty source bins"
            )
        ids, labels, positions, spans = [], [], [], []
        for row in records:
            source = _flat_membership(row)
            if len(source["ids"]) != len(row["input_ids"]) or len(source["spans"]) != 1:
                raise FinetuneEstimateError(
                    "Axolotl bins require unpadded, unpacked source records"
                )
            start = len(ids)
            ids.extend(source["ids"])
            labels.extend(source["labels"])
            positions.extend(source["positions"])
            spans.append((start, len(ids)))
        result.append(
            {"ids": ids, "labels": labels, "positions": positions, "spans": spans}
        )
    return result


class IsolatedPackingCollator:
    """Wrap native collation with source membership and isolation verification."""

    def __init__(self, native_collator, *, framework, mask_dtype):
        self.native_collator = native_collator
        self.framework = framework
        self.mask_dtype = mask_dtype

    def __call__(self, features):
        import torch

        sources = packed_source_membership(features, framework=self.framework)
        prepared = deepcopy(features)
        if self.framework in {"huggingface", "llamafactory"}:
            # These native text collators pad labels, not arbitrary position IDs.
            # Rebuild only these known positions from the verified source spans.
            for row in prepared:
                row.pop("position_ids", None)
        inputs = self.native_collator(prepared)
        if (
            not isinstance(inputs, Mapping)
            or "input_ids" not in inputs
            or "labels" not in inputs
        ):
            raise FinetuneEstimateError(
                "native packing collator must return text tensors and labels"
            )
        inputs = dict(inputs)
        ids, labels = inputs["input_ids"], inputs["labels"]
        if ids.dtype not in {torch.int32, torch.int64}:
            raise FinetuneEstimateError("native packed token IDs must be integers")
        if (
            ids.ndim != 2
            or tuple(labels.shape) != tuple(ids.shape)
            or len(sources) != ids.shape[0]
        ):
            raise FinetuneEstimateError("native packing changed source bin dimensions")
        batch, width = ids.shape
        positions = torch.zeros_like(ids)
        allowed = torch.zeros(
            (batch, 1, width, width), dtype=torch.bool, device=ids.device
        )
        active = torch.zeros_like(ids, dtype=torch.bool)
        for row, source in enumerate(sources):
            size = len(source["ids"])
            if (
                size > width
                or ids[row, :size].tolist() != source["ids"]
                or labels[row, :size].tolist() != source["labels"]
            ):
                raise FinetuneEstimateError(
                    "native packing changed source token or label membership"
                )
            if bool((labels[row, size:] != -100).any()):
                raise FinetuneEstimateError("native packing must ignore padded labels")
            positions[row, :size] = torch.tensor(
                source["positions"], device=ids.device, dtype=ids.dtype
            )
            active[row, :size] = True
            for start, end in source["spans"]:
                allowed[row, 0, start:end, start:end] = torch.ones(
                    (end - start, end - start), dtype=torch.bool, device=ids.device
                ).tril()
        native_positions = inputs.get("position_ids")
        if native_positions is not None:
            if (
                native_positions.dtype not in {torch.int32, torch.int64}
                or tuple(native_positions.shape) != tuple(positions.shape)
                or not torch.equal(native_positions[active], positions[active])
            ):
                raise FinetuneEstimateError(
                    "native packing position IDs disagree with source membership"
                )
        # Axolotl pads positions with arange. Inactive positions never contribute
        # to a segment, its state, or loss; canonicalize only those positions.
        native_mask = inputs.get("attention_mask")
        if native_mask is None:
            raise FinetuneEstimateError("native packing must retain its attention mask")
        if native_mask.ndim == 2:
            if native_mask.dtype not in {torch.bool, torch.int32, torch.int64} or bool(
                (native_mask < 0).any()
            ):
                raise FinetuneEstimateError(
                    "native packing mask must contain nonnegative integer IDs"
                )
            if tuple(native_mask.shape) != tuple(ids.shape) or not torch.equal(
                native_mask > 0, active
            ):
                raise FinetuneEstimateError(
                    "native packing mask disagrees with source membership"
                )
            if self.mask_dtype not in {torch.float32, torch.float16, torch.bfloat16}:
                raise FinetuneEstimateError(
                    "packed mask dtype must be float32, float16 or bfloat16"
                )
            inputs["attention_mask"] = torch.where(
                allowed, 0.0, torch.finfo(self.mask_dtype).min
            ).to(self.mask_dtype)
        elif native_mask.ndim != 4:
            raise FinetuneEstimateError(
                "native packing requires a 2D or block-causal 4D mask"
            )
        inputs["position_ids"] = positions
        inputs["use_cache"] = False
        actual_spans = validate_segment_inputs(inputs)
        if actual_spans != [source["spans"] for source in sources]:
            raise FinetuneEstimateError(
                "native block mask disagrees with original sample boundaries"
            )
        return inputs


class _PackedBinFetcher:
    """Handle native nested bins on one Dataset; retain ordinary flat fetches."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.original = dataset.__getitems__

    def __call__(self, keys):
        if keys and isinstance(keys[0], list):
            return [[self.dataset[index(item)] for item in group] for group in keys]
        return self.original(keys)


def install_isolated_packing_collation(trainer, *, framework, mask_dtype, replace):
    """Retain source masks before native loader preparation and own block masks."""
    if getattr(trainer, "_train_dataloader", None) is not None:
        raise FinetuneEstimateError(
            "packed collation must be installed before loader preparation"
        )
    native = trainer.data_collator
    if isinstance(native, IsolatedPackingCollator):
        raise FinetuneEstimateError("isolated packed collation is already installed")
    if framework not in {"huggingface", "llamafactory", "axolotl"}:
        raise FinetuneEstimateError("unsupported packed collation framework")
    if framework == "axolotl":
        # Ax derives this argument from attention support, rather than a raw
        # sample_packing_drop_attention_mask setting. The reference path needs
        # original source masks to prove membership before model execution.
        replace(trainer.args, "sample_packing_drop_attention_mask", False)
        dataset = trainer.train_dataset
        replace(dataset, "__getitems__", _PackedBinFetcher(dataset))
    replace(
        trainer,
        "data_collator",
        IsolatedPackingCollator(native, framework=framework, mask_dtype=mask_dtype),
    )
    return {
        "schema_version": "fakegpu.isolated_packing_collation.v1",
        "framework": framework,
        "native_collator_class": f"{type(native).__module__}.{type(native).__qualname__}",
        "constructed_mask_dtype": str(mask_dtype),
        "source_masks_retained": True,
        "padding_positions": "zero_inactive_only",
        "use_cache": False,
        "nested_bin_fetch": "dataset_instance" if framework == "axolotl" else None,
    }


def packed_input_sha256(inputs):
    """Hash actual integer text tensors and floating block-mask bits in LE order."""
    import torch

    validate_segment_inputs(inputs)
    digest = hashlib.sha256(b"fakegpu.packed_batch_inputs.v1\0")
    integer_types = {torch.int32, torch.int64}
    for field in ("input_ids", "attention_mask", "labels", "position_ids"):
        tensor = inputs[field]
        if field != "attention_mask" and (
            tensor.ndim != 2 or tensor.dtype not in integer_types
        ):
            raise FinetuneEstimateError("packed text tensors must be 2D integers")
        array = tensor.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            array = array.view(torch.int16)
        array = array.numpy()
        payload = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(
            order="C"
        )
        header = json.dumps(
            {"field": field, "shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def describe_packed_batch(inputs):
    """Keep physical rows, logical source sequences and token work separate."""
    spans = validate_segment_inputs(inputs)
    shape = list(inputs["input_ids"].shape)
    nonpadding = sum(end - start for row in spans for start, end in row)
    labels = inputs["labels"].detach().cpu().tolist()
    supervised = sum(value != -100 for row in labels for value in row)
    return {
        "schema_version": "fakegpu.packed_batch.v1",
        "tensor_dtypes": {
            field: str(inputs[field].dtype)
            for field in ("input_ids", "attention_mask", "labels", "position_ids")
        },
        "shape": shape,
        "packed_rows": shape[0],
        "logical_sequences": sum(map(len, spans)),
        "segments": [[list(span) for span in row] for row in spans],
        "nonpadding_tokens": nonpadding,
        "padding_tokens": shape[0] * shape[1] - nonpadding,
        "supervised_tokens": supervised,
        "causal_supervised_tokens": supervised - sum(row[0] != -100 for row in labels),
        "input_sha256": packed_input_sha256(inputs),
    }
