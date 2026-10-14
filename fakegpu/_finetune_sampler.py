"""Plan and attest native text batches without executing a model or reading time.

The prepared Trainer dataloader supplies the sampler and collator. Planning uses
an independent sampler and copies of dataset rows; native sampler state and the
Python, NumPy and CPU Torch random streams are restored before returning.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import random

from ._finetune_batches import _descriptor, describe_batch
from .finetune_data import profile_tokenized_records
from .finetune_estimator import FinetuneEstimateError


SCHEMA_VERSION = "fakegpu.finetune_sampler_plan.v1"
LENGTH_SCHEMA_VERSION = "fakegpu.finetune_sampler_plan.v2"
PACKED_SCHEMA_VERSION = "fakegpu.finetune_sampler_plan.v3"
VERIFICATION_SCHEMA_VERSION = "fakegpu.finetune_sampler_verification.v1"
_INPUT_FIELDS = ("input_ids", "attention_mask", "labels", "position_ids")
_WORKER_COLLATORS = {
    "transformers.data.data_collator.DataCollatorForSeq2Seq",
    "llamafactory.data.collator.SFTDataCollatorWith4DAttentionMask",
    "axolotl.utils.collators.batching.DataCollatorForSeq2Seq",
}
_VALIDATED_VERSIONS = {
    "huggingface": (None, "5.6.0", "1.11.0"),
    "llamafactory": ("0.9.5", "5.6.0", "1.11.0"),
    "axolotl": ("0.18.0", "5.14.1", "1.13.0"),
}


def _integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinetuneEstimateError(f"{name} must be an integer >= {minimum}")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise FinetuneEstimateError(f"{name} must be a SHA256 hex digest")
    return value


def _class_name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def validate_length_grouping_seed(seed):
    if seed is not None and (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**63
    ):
        raise ValueError("length_grouping_seed must be an integer in [0, 2**63)")
    return seed


def validate_length_grouping_evaluation(policy, *, seed):
    fields = {
        "schema_version", "kind", "seed", "iteration_seed_rule",
        "grouping_batch_size", "dataset_rows", "lengths_sha256",
    }
    if not isinstance(policy, Mapping) or set(policy) != fields:
        raise FinetuneEstimateError("evaluation grouping policy is incomplete")
    if (
        policy["schema_version"] != "fakegpu.finetune_eval_sampler.v1"
        or policy["kind"] != "fixed_seed_length_grouped"
        or policy["iteration_seed_rule"] != "same_seed_each_invocation"
        or _integer(policy["seed"], "evaluation grouping seed", minimum=0) != seed
    ):
        raise FinetuneEstimateError("evaluation grouping policy differs from its explicit seed")
    _integer(policy["grouping_batch_size"], "evaluation grouping batch size")
    _integer(policy["dataset_rows"], "evaluation grouping dataset rows")
    _sha256(policy["lengths_sha256"], "evaluation grouping lengths")
    return policy


def validate_native_sampler_versions(framework, versions=None):
    if framework not in _VALIDATED_VERSIONS:
        raise FinetuneEstimateError("unsupported native sampler framework")
    packages = ["torch", "transformers", "accelerate"]
    if framework != "huggingface":
        packages.append(framework)
    if versions is None:
        versions = {name: importlib.metadata.version(name) for name in packages}
    required_framework, required_transformers, required_accelerate = (
        _VALIDATED_VERSIONS[framework]
    )
    if any(not isinstance(versions.get(name), str) for name in packages):
        raise FinetuneEstimateError("native sampler software versions are incomplete")
    if (
        versions["transformers"] != required_transformers
        or versions["accelerate"] != required_accelerate
        or (
            required_framework is not None and versions[framework] != required_framework
        )
    ):
        raise FinetuneEstimateError(
            "native sampler planning requires a validated framework, Trainer and Accelerate version"
        )
    return versions


def batch_input_sha256(inputs):
    """Hash the four supported input tensors, including absent fields and dtype.

    Hash payloads use a fixed field order, length-prefixed canonical JSON headers
    and contiguous little-endian integer/bool tensor bytes. No token values are
    retained in reports. GPU callers should account for host copies as metadata
    instrumentation alongside ``describe_batch``.
    """
    import torch

    if not isinstance(inputs, Mapping) or "input_ids" not in inputs:
        raise FinetuneEstimateError("native sampler inputs require input_ids")
    if set(inputs) - set(_INPUT_FIELDS):
        raise FinetuneEstimateError(
            "native sampler attestation supports only input_ids, attention_mask, labels and position_ids"
        )
    digest = hashlib.sha256(b"fakegpu.native_batch_inputs.v1\0")
    integer_types = {
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    for name in _INPUT_FIELDS:
        tensor = inputs.get(name)
        if tensor is None:
            header, payload = {"field": name, "present": False}, b""
        else:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype not in integer_types
                or tensor.ndim != 2
            ):
                raise FinetuneEstimateError(
                    "native sampler input tensors must be two-dimensional integers or booleans"
                )
            array = tensor.detach().cpu().contiguous().numpy()
            array = array.astype(array.dtype.newbyteorder("<"), copy=False)
            header = {
                "field": name,
                "present": True,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            payload = array.tobytes(order="C")
        encoded = _canonical(header).encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


@contextmanager
def _preserve_random_streams(torch):
    import numpy as np

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _sampler_configuration(trainer, loader, torch):
    from accelerate.data_loader import SeedableRandomSampler

    batch_sampler = loader.batch_sampler
    if type(batch_sampler) is not torch.utils.data.BatchSampler:
        raise FinetuneEstimateError(
            "native sampler planning requires an ordinary single-process BatchSampler"
        )
    if batch_sampler.drop_last or loader.drop_last:
        raise FinetuneEstimateError("native sampler planning must retain tail batches")
    sampler = batch_sampler.sampler
    if type(sampler) is torch.utils.data.SequentialSampler:
        return sampler, "sequential", None
    if _class_name(sampler) in {
        "transformers.trainer_pt_utils.LengthGroupedSampler",
        "fakegpu._finetune_length_sampler.SeedableLengthGroupedSampler",
    }:
        from transformers.trainer_pt_utils import LengthGroupedSampler
        from ._finetune_length_sampler import SeedableLengthGroupedSampler

        if type(sampler) is SeedableLengthGroupedSampler:
            validate_length_grouping_seed(sampler.initial_seed)
            if sampler.batch_size != (
                trainer.args.train_batch_size * trainer.args.gradient_accumulation_steps
            ) or len(sampler.lengths) != len(loader.dataset):
                raise FinetuneEstimateError("native length grouping workload is inconsistent")
            return sampler, "seedable_length_grouped", sampler.initial_seed
        if type(sampler) is LengthGroupedSampler:
            raise FinetuneEstimateError(
                "native length grouping uses the global RNG; provide length_grouping_seed for future planning"
            )
    if type(sampler) is not SeedableRandomSampler or sampler.replacement:
        raise FinetuneEstimateError(
            "native sampler planning supports sequential, seedable random or explicitly seeded length grouping without replacement"
        )
    if sampler.num_samples != len(loader.dataset):
        raise FinetuneEstimateError("native random sampler must consume every row once")
    seed = _integer(sampler.initial_seed, "native sampler seed", minimum=0)
    expected = trainer.args.data_seed
    if expected is None:
        expected = trainer.args.seed
    if seed != expected:
        raise FinetuneEstimateError(
            "native sampler seed differs from configured data_seed or seed"
        )
    return sampler, "seedable_random", seed


def _collated_batch(loader, indices):
    inputs = loader.collate_fn([deepcopy(loader.dataset[index]) for index in indices])
    input_sha256 = batch_input_sha256(inputs)
    if any(
        value is not None and value.device.type != "cpu" for value in inputs.values()
    ):
        raise FinetuneEstimateError("native sampler planning requires a CPU collator")
    descriptor = describe_batch(inputs)
    if (
        descriptor["attention_mask_kind"] != "binary_2d"
        or descriptor["labels_kind"] != "integer_2d"
        or descriptor["position_ids_kind"] == "unsupported"
    ):
        raise FinetuneEstimateError(
            "native sampler planning requires binary masks and integer labels"
        )
    return descriptor, input_sha256


def plan_native_batches(trainer, *, max_steps, framework, sequence_length):
    """Build a complete target plan from the real prepared CPU dataloader.

    The caller must supply an unpacked, map-style, single-device text Trainer.
    Default seedable random sampling is supported, including epoch reseeding and
    partial accumulation tails. Resume, iterable datasets, random worker transforms and
    custom/packing samplers are deliberately outside this contract. Length
    grouping requires the explicit epoch-seeded native adapter.
    """
    import torch
    from accelerate.data_loader import SeedableRandomSampler

    _integer(max_steps, "max_steps")
    _integer(sequence_length, "sequence_length")
    versions = validate_native_sampler_versions(framework)
    args = trainer.args
    if args.world_size != 1 or trainer.accelerator.num_processes != 1 or args.n_gpu > 1:
        raise FinetuneEstimateError("native sampler planning requires one device")
    if (
        args.dataloader_num_workers != 0
        and _class_name(trainer.data_collator) not in _WORKER_COLLATORS
    ):
        raise FinetuneEstimateError(
            "native sampler workers require a supported deterministic text collator"
        )
    if getattr(args, "sample_packing", False) or getattr(
        args, "auto_find_batch_size", False
    ):
        raise FinetuneEstimateError(
            "native sampler planning does not support packing or adaptive batch sizes"
        )
    dataset = trainer.train_dataset
    if isinstance(dataset, torch.utils.data.IterableDataset) or not (
        hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")
    ):
        raise FinetuneEstimateError("native sampler planning needs a map-style dataset")
    rows = _integer(len(dataset), "dataset rows")
    accumulation = _integer(args.gradient_accumulation_steps, "gradient accumulation")
    with _preserve_random_streams(torch):
        loader = trainer.get_train_dataloader()
        if len(loader.dataset) != rows:
            raise FinetuneEstimateError(
                "prepared native dataloader changed dataset size"
            )
        native_sampler, kind, seed = _sampler_configuration(trainer, loader, torch)
        batch_size = _integer(loader.batch_sampler.batch_size, "native batch size")
        if batch_size != args.per_device_train_batch_size:
            raise FinetuneEstimateError("native batch size differs from configuration")
        profile = profile_tokenized_records(
            dataset, sequence_length=sequence_length, source_framework=framework
        )
        if kind == "sequential":
            sampler = torch.utils.data.SequentialSampler(loader.dataset)
        elif kind == "seedable_length_grouped":
            from ._finetune_length_sampler import SeedableLengthGroupedSampler

            sampler = SeedableLengthGroupedSampler(
                native_sampler.batch_size, native_sampler.lengths, seed=seed
            )
        else:
            sampler = SeedableRandomSampler(
                loader.dataset,
                data_seed=seed,
                generator=torch.Generator(device="cpu"),
            )
        microsteps_per_epoch = math.ceil(rows / batch_size)
        updates_per_epoch = math.ceil(microsteps_per_epoch / accumulation)
        updates, epoch_orders = [], []
        for epoch in range(math.ceil(max_steps / updates_per_epoch)):
            if kind != "sequential":
                sampler.set_epoch(epoch)
            order = list(sampler)
            if len(order) != rows or sorted(order) != list(range(rows)):
                raise FinetuneEstimateError(
                    "native sampler does not consume every row once"
                )
            epoch_orders.append(_digest(order))
            steps = min(updates_per_epoch, max_steps - len(updates))
            for position in range(steps):
                batches = []
                start = position * accumulation
                for microstep in range(
                    start, min(start + accumulation, microsteps_per_epoch)
                ):
                    indices = order[
                        microstep * batch_size : (microstep + 1) * batch_size
                    ]
                    descriptor, input_sha256 = _collated_batch(loader, indices)
                    # A second independent row copy catches collator mutations or
                    # stochastic augmentation before making an execution claim.
                    if _collated_batch(loader, indices) != (descriptor, input_sha256):
                        raise FinetuneEstimateError(
                            "native sampler planning requires deterministic collation"
                        )
                    batches.append(
                        {
                            **descriptor,
                            "epoch_index": epoch + 1,
                            "epoch_microstep": microstep + 1,
                            "row_indices": indices,
                            "input_sha256": input_sha256,
                        }
                    )
                updates.append({"optimizer_step": len(updates) + 1, "batches": batches})
    plan = {
        "schema_version": LENGTH_SCHEMA_VERSION
        if kind == "seedable_length_grouped" else SCHEMA_VERSION,
        "framework": framework,
        "dataset": {
            "num_samples": rows,
            "content_sha256": profile["provenance"]["content_sha256"],
        },
        "sampler": {
            "kind": kind,
            "class": _class_name(native_sampler),
            "seed": seed,
            "epoch_seed_rule": "seed + zero_based_epoch" if seed is not None else None,
            "epoch_order_sha256": epoch_orders,
            "versions": versions,
        },
        "collator_class": _class_name(trainer.data_collator),
        "dataloader": {
            "num_workers": args.dataloader_num_workers,
            "prefetch_factor": args.dataloader_prefetch_factor,
            "persistent_workers": args.dataloader_persistent_workers,
            "pin_memory": args.dataloader_pin_memory,
            "drop_last": args.dataloader_drop_last,
        },
        "micro_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "updates_per_epoch": updates_per_epoch,
        "updates": updates,
        "workload": {
            "optimizer_steps": max_steps,
            "microsteps": sum(len(update["batches"]) for update in updates),
            "samples": sum(
                batch["shape"][0] for update in updates for batch in update["batches"]
            ),
        },
    }
    if kind == "seedable_length_grouped":
        lengths = list(native_sampler.lengths)
        plan["sampler"]["grouping"] = {
            "batch_size": native_sampler.batch_size,
            "lengths": lengths,
            "lengths_sha256": _digest(lengths),
        }
    plan["plan_sha256"] = _digest(plan)
    return validate_sampler_plan(plan)


def validate_sampler_plan(plan):
    """Validate a serialized CPU batch plan without importing Torch."""
    if isinstance(plan, Mapping) and plan.get("schema_version") == PACKED_SCHEMA_VERSION:
        from ._finetune_packed_sampler import validate_packed_sampler_plan

        return validate_packed_sampler_plan(plan)
    if not isinstance(plan, Mapping) or plan.get("schema_version") not in {
        SCHEMA_VERSION, LENGTH_SCHEMA_VERSION
    }:
        raise FinetuneEstimateError("unsupported native sampler plan schema")
    fields = {
        "schema_version",
        "framework",
        "dataset",
        "sampler",
        "collator_class",
        "dataloader",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "updates_per_epoch",
        "updates",
        "workload",
        "plan_sha256",
    }
    if set(plan) != fields:
        raise FinetuneEstimateError(
            "native sampler plan fields are incomplete or unsupported"
        )
    copied = deepcopy(dict(plan))
    recorded = copied.pop("plan_sha256", None)
    if recorded != _digest(copied):
        raise FinetuneEstimateError(
            "native sampler plan SHA256 differs from its contents"
        )
    rows = _integer(plan["dataset"]["num_samples"], "dataset rows")
    if set(plan["dataset"]) != {"num_samples", "content_sha256"}:
        raise FinetuneEstimateError("native sampler dataset identity is incomplete")
    _sha256(plan["dataset"]["content_sha256"], "dataset content SHA256")
    sampler = plan["sampler"]
    sampler_fields = {
        "kind",
        "class",
        "seed",
        "epoch_seed_rule",
        "epoch_order_sha256",
        "versions",
    }
    if plan["schema_version"] == LENGTH_SCHEMA_VERSION:
        sampler_fields.add("grouping")
    if set(sampler) != sampler_fields:
        raise FinetuneEstimateError("native sampler identity is incomplete")
    validate_native_sampler_versions(plan["framework"], sampler["versions"])
    if sampler["kind"] == "sequential":
        expected_class = "torch.utils.data.sampler.SequentialSampler"
        if sampler["seed"] is not None or sampler["epoch_seed_rule"] is not None:
            raise FinetuneEstimateError(
                "sequential sampler must not claim an epoch seed"
            )
    elif sampler["kind"] == "seedable_random":
        expected_class = "accelerate.data_loader.SeedableRandomSampler"
        _integer(sampler["seed"], "native sampler seed", minimum=0)
        if sampler["epoch_seed_rule"] != "seed + zero_based_epoch":
            raise FinetuneEstimateError("unsupported native sampler epoch seed rule")
    elif sampler["kind"] == "seedable_length_grouped":
        expected_class = "fakegpu._finetune_length_sampler.SeedableLengthGroupedSampler"
        if plan["schema_version"] != LENGTH_SCHEMA_VERSION:
            raise FinetuneEstimateError("seeded length grouping requires sampler plan v2")
        validate_length_grouping_seed(sampler["seed"])
        if sampler["seed"] is None or sampler["epoch_seed_rule"] != "seed + zero_based_epoch":
            raise FinetuneEstimateError("seeded length grouping requires an epoch seed")
        grouping = sampler["grouping"]
        if set(grouping) != {"batch_size", "lengths", "lengths_sha256"}:
            raise FinetuneEstimateError("native length grouping identity is incomplete")
        _integer(grouping["batch_size"], "native grouping batch size")
        if not isinstance(grouping["lengths"], list) or len(grouping["lengths"]) != rows:
            raise FinetuneEstimateError("native grouping lengths differ from the dataset size")
        for length in grouping["lengths"]:
            _integer(length, "native grouping length")
        if grouping["lengths_sha256"] != _digest(grouping["lengths"]):
            raise FinetuneEstimateError("native grouping length SHA256 is inconsistent")
    else:
        raise FinetuneEstimateError("unsupported native sampler kind")
    if sampler["class"] != expected_class:
        raise FinetuneEstimateError("native sampler class differs from sampler kind")
    if plan["schema_version"] == LENGTH_SCHEMA_VERSION and sampler["kind"] != "seedable_length_grouped":
        raise FinetuneEstimateError("sampler plan v2 requires seeded length grouping")
    if not isinstance(plan["collator_class"], str) or not plan["collator_class"]:
        raise FinetuneEstimateError("native sampler collator identity is missing")
    loader = plan["dataloader"]
    if set(loader) != {
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "pin_memory",
        "drop_last",
    }:
        raise FinetuneEstimateError("native sampler dataloader identity is incomplete")
    workers = _integer(loader["num_workers"], "dataloader workers", minimum=0)
    if workers and plan["collator_class"] not in _WORKER_COLLATORS:
        raise FinetuneEstimateError("native sampler worker collator is unsupported")
    if loader["prefetch_factor"] is not None:
        _integer(loader["prefetch_factor"], "dataloader prefetch factor")
    if (
        any(
            type(loader[name]) is not bool
            for name in ("persistent_workers", "pin_memory", "drop_last")
        )
        or loader["drop_last"]
    ):
        raise FinetuneEstimateError("native sampler dataloader flags are invalid")
    batch_size = _integer(plan["micro_batch_size"], "micro batch size")
    accumulation = _integer(
        plan["gradient_accumulation_steps"], "gradient accumulation"
    )
    if sampler["kind"] == "seedable_length_grouped" and (
        sampler["grouping"]["batch_size"] != batch_size * accumulation
    ):
        raise FinetuneEstimateError("native grouping batch size differs from accumulation")
    per_epoch = math.ceil(math.ceil(rows / batch_size) / accumulation)
    if plan["updates_per_epoch"] != per_epoch:
        raise FinetuneEstimateError("native sampler updates per epoch is inconsistent")
    steps = _integer(plan["workload"]["optimizer_steps"], "optimizer steps")
    if len(plan["updates"]) != steps:
        raise FinetuneEstimateError("native sampler update count is inconsistent")
    if set(plan["workload"]) != {"optimizer_steps", "microsteps", "samples"}:
        raise FinetuneEstimateError("native sampler workload fields are invalid")
    if not isinstance(sampler["epoch_order_sha256"], list) or len(
        sampler["epoch_order_sha256"]
    ) != math.ceil(steps / per_epoch):
        raise FinetuneEstimateError("native sampler epoch order count is inconsistent")
    for digest in sampler["epoch_order_sha256"]:
        _sha256(digest, "epoch order SHA256")
    microsteps = samples = 0
    orders = [[] for _ in sampler["epoch_order_sha256"]]
    for index, update in enumerate(plan["updates"]):
        if (
            set(update) != {"optimizer_step", "batches"}
            or update["optimizer_step"] != index + 1
        ):
            raise FinetuneEstimateError("native sampler updates must be consecutive")
        epoch, position = divmod(index, per_epoch)
        start = position * accumulation
        expected_count = min(accumulation, math.ceil(rows / batch_size) - start)
        if len(update["batches"]) != expected_count:
            raise FinetuneEstimateError(
                "native sampler accumulation tail is inconsistent"
            )
        for offset, batch in enumerate(update["batches"]):
            descriptor = _descriptor(batch)
            if set(batch) != set(descriptor) | {
                "epoch_index",
                "epoch_microstep",
                "row_indices",
                "input_sha256",
            }:
                raise FinetuneEstimateError("native sampler batch fields are invalid")
            if (batch["epoch_index"], batch["epoch_microstep"]) != (
                epoch + 1,
                start + offset + 1,
            ):
                raise FinetuneEstimateError(
                    "native sampler batch epoch position is inconsistent"
                )
            indices = batch["row_indices"]
            expected_size = min(batch_size, rows - (start + offset) * batch_size)
            if descriptor["shape"][0] != expected_size:
                raise FinetuneEstimateError(
                    "native sampler batch tail size is inconsistent"
                )
            if len(indices) != descriptor["shape"][0] or any(
                _integer(value, "row index", minimum=0) >= rows for value in indices
            ):
                raise FinetuneEstimateError("native sampler row indices are invalid")
            if len(set(indices)) != len(indices):
                raise FinetuneEstimateError(
                    "native sampler batch repeats a dataset row"
                )
            orders[epoch].extend(indices)
            _sha256(batch["input_sha256"], "native sampler input SHA256")
            microsteps += 1
            samples += descriptor["shape"][0]
    for epoch, order in enumerate(orders):
        if len(order) != len(set(order)):
            raise FinetuneEstimateError("native sampler epoch repeats a dataset row")
        if sampler["kind"] == "sequential" and order != list(range(len(order))):
            raise FinetuneEstimateError(
                "native sequential sampler row order is invalid"
            )
        if (
            len(order) == rows
            and _digest(order) != sampler["epoch_order_sha256"][epoch]
        ):
            raise FinetuneEstimateError(
                "native sampler epoch order SHA256 is inconsistent"
            )
    if (microsteps, samples) != (
        plan["workload"]["microsteps"],
        plan["workload"]["samples"],
    ):
        raise FinetuneEstimateError("native sampler workload totals are inconsistent")
    return deepcopy(dict(plan))


def validate_sampling_verification(plan, updates, verification):
    """Check a report's consumed input evidence against its CPU plan prefix."""
    plan = validate_sampler_plan(plan)
    if not updates or len(updates) > len(plan["updates"]):
        raise FinetuneEstimateError(
            "native sampling verification has an invalid update count"
        )
    hashes = []
    samples = 0
    for actual, expected in zip(updates, plan["updates"]):
        if actual["optimizer_step"] != expected["optimizer_step"] or len(
            actual["batches"]
        ) != len(expected["batches"]):
            raise FinetuneEstimateError(
                "native sampling verification update differs from plan"
            )
        for batch, planned in zip(actual["batches"], expected["batches"]):
            fields = ("epoch_index", "epoch_microstep", "row_indices", "input_sha256")
            if plan["schema_version"] == PACKED_SCHEMA_VERSION:
                fields = (*fields, "source_bins")
            if _descriptor(batch) != _descriptor(planned) or any(
                batch.get(field) != planned[field]
                for field in fields
            ):
                raise FinetuneEstimateError(
                    "native sampling verification batch differs from plan"
                )
            hashes.append(batch["input_sha256"])
            samples += batch["shape"][0]
    expected = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "optimizer_steps": len(updates),
        "microsteps": len(hashes),
        "samples": samples,
        "input_sha256": _digest(hashes),
        "matched": True,
    }
    if verification != expected:
        raise FinetuneEstimateError(
            "native sampling verification summary differs from consumed inputs"
        )
    return deepcopy(verification)


class NativeBatchMatcher:
    """Compare consumed native inputs against a precomputed target or prefix."""

    def __init__(self, plan, *, expected_updates=None):
        self.plan = validate_sampler_plan(plan)
        count = self.plan["workload"]["optimizer_steps"]
        if expected_updates is None:
            expected_updates = count
        _integer(expected_updates, "expected_updates")
        if expected_updates > count:
            raise FinetuneEstimateError("native matcher prefix exceeds planned updates")
        self.expected_updates = expected_updates
        self.batches = [
            batch
            for update in self.plan["updates"][:expected_updates]
            for batch in update["batches"]
        ]
        self.consumed = 0
        self.actual_hashes = []

    def describe(self, inputs):
        if self.plan["schema_version"] == PACKED_SCHEMA_VERSION:
            from ._finetune_packed_batches import describe_packed_core

            return describe_packed_core(inputs)
        return describe_batch(inputs)

    def observe(self, inputs, *, epoch_index, epoch_microstep, descriptor=None):
        if self.consumed >= len(self.batches):
            raise FinetuneEstimateError(
                "native training consumed more batches than planned"
            )
        expected = self.batches[self.consumed]
        if (epoch_index, epoch_microstep) != (
            expected["epoch_index"],
            expected["epoch_microstep"],
        ):
            raise FinetuneEstimateError(
                "native sampler epoch position differs from CPU plan"
            )
        actual = self.describe(inputs) if descriptor is None else descriptor
        if _descriptor(actual) != _descriptor(expected):
            raise FinetuneEstimateError("native batch descriptor differs from CPU plan")
        if self.plan["schema_version"] == PACKED_SCHEMA_VERSION:
            from ._finetune_packed_batches import packed_input_sha256

            digest = packed_input_sha256(inputs)
        else:
            digest = batch_input_sha256(inputs)
        if digest != expected["input_sha256"]:
            raise FinetuneEstimateError(
                "native batch tensor contents differ from CPU plan"
            )
        self.consumed += 1
        self.actual_hashes.append(digest)
        return deepcopy(expected)

    def finish(self, *, expected_updates=None):
        if expected_updates is None:
            expected_updates = self.expected_updates
        _integer(expected_updates, "expected_updates")
        if expected_updates > self.expected_updates:
            raise FinetuneEstimateError("native matcher prefix exceeds planned updates")
        batches = [
            batch
            for update in self.plan["updates"][:expected_updates]
            for batch in update["batches"]
        ]
        if self.consumed != len(batches):
            raise FinetuneEstimateError(
                "native training did not consume the complete planned prefix"
            )
        return {
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "plan_sha256": self.plan["plan_sha256"],
            "optimizer_steps": expected_updates,
            "microsteps": self.consumed,
            "samples": sum(batch["shape"][0] for batch in batches),
            "input_sha256": _digest(self.actual_hashes),
            "matched": True,
        }
