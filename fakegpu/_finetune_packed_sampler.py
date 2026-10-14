"""Native CPU packing counts with independent epoch-seeded source sampling.

This inspects actual native bins, tails and collation without fitting timings.
Dataset rows, logical sequences and executed
tensor rows have distinct counts. Original sampler and global RNG state stay
unchanged while epochs are reconstructed.
"""

from __future__ import annotations

from copy import copy, deepcopy
from collections.abc import Mapping
import math

from ._finetune_packed_batches import (
    describe_packed_batch,
    describe_packed_core,
    packed_source_membership,
)
from ._finetune_sampler import (
    _class_name,
    _digest,
    _integer,
    _sha256,
    _preserve_random_streams,
    validate_native_sampler_versions,
)
from .finetune_estimator import FinetuneEstimateError

PACKED_SCHEMA_VERSION = "fakegpu.finetune_sampler_plan.v3"


def plan_native_packed_batches(trainer, *, max_steps, framework, dataset_profile):
    """Seal actual native bins and an exact CPU tensor prefix for a fixed horizon."""
    _integer(max_steps, "packed max_steps")
    observed = inspect_packed_epochs(
        trainer, framework=framework, max_steps=max_steps, include_core=True
    )
    args = trainer.args
    accumulation = args.gradient_accumulation_steps
    updates = []
    epochs = []
    for epoch in observed["epochs"]:
        if epoch["microsteps"] != observed["native_declared_microsteps_per_epoch"]:
            raise FinetuneEstimateError(
                "packed horizon requires actual epoch lengths matching the native declaration"
            )
        epochs.append([batch["source_bins"] for batch in epoch["batches"]])
        for start in range(0, len(epoch["batches"]), accumulation):
            if len(updates) == max_steps:
                break
            batches = []
            for microstep, batch in enumerate(
                epoch["batches"][start : start + accumulation], start + 1
            ):
                batches.append(
                    {
                        **batch["core_descriptor"],
                        "epoch_index": epoch["epoch_index"],
                        "epoch_microstep": microstep,
                        "source_bins": batch["source_bins"],
                        "row_indices": [
                            index for group in batch["source_bins"] for index in group
                        ],
                        "input_sha256": batch["input_sha256"],
                    }
                )
            updates.append({"optimizer_step": len(updates) + 1, "batches": batches})
    plan = {
        "schema_version": PACKED_SCHEMA_VERSION,
        "framework": framework,
        "dataset": {
            "num_samples": observed["dataset_rows"],
            "content_sha256": dataset_profile["provenance"]["content_sha256"],
        },
        "sampler": {
            "kind": "seedable_packed_random",
            "class": observed["source_sampler"],
            "seed": observed["source_seed"],
            "epoch_seed_rule": observed["epoch_seed_rule"],
            "epoch_order_sha256": [
                epoch["source_order_sha256"] for epoch in observed["epochs"]
            ],
            "versions": observed["versions"],
        },
        "collator_class": _class_name(trainer.data_collator),
        "dataloader": {
            name: getattr(args, "dataloader_" + name)
            for name in (
                "num_workers",
                "prefetch_factor",
                "persistent_workers",
                "pin_memory",
                "drop_last",
            )
        },
        "micro_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": accumulation,
        "updates_per_epoch": observed["native_declared_optimizer_updates_per_epoch"],
        "packing": {
            "mode": "qwen35_reference.v1",
            "batch_sampler": observed["native_batch_sampler"],
            "native_drop_last": observed["native_drop_last"],
            "microsteps_per_epoch": observed["native_declared_microsteps_per_epoch"],
            "row_logical_sequences": observed["row_logical_sequences"],
            "epoch_source_bins": epochs,
        },
        "updates": updates,
        "workload": {
            "optimizer_steps": len(updates),
            "microsteps": sum(len(update["batches"]) for update in updates),
            "samples": sum(
                batch["shape"][0] for update in updates for batch in update["batches"]
            ),
        },
    }
    plan["plan_sha256"] = _digest(plan)
    return validate_packed_sampler_plan(plan)


def validate_packed_sampler_plan(plan):
    """Validate packed native schedules and exact emitted prefixes with stdlib only."""
    from ._finetune_batches import _descriptor

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
        "packing",
        "updates",
        "workload",
        "plan_sha256",
    }
    if (
        not isinstance(plan, Mapping)
        or set(plan) != fields
        or plan["schema_version"] != PACKED_SCHEMA_VERSION
    ):
        raise FinetuneEstimateError("unsupported packed sampler plan fields or schema")
    content = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan["plan_sha256"] != _digest(content):
        raise FinetuneEstimateError("packed sampler plan SHA256 is inconsistent")
    dataset = plan["dataset"]
    if not isinstance(dataset, Mapping) or set(dataset) != {
        "num_samples",
        "content_sha256",
    }:
        raise FinetuneEstimateError("packed dataset identity is incomplete")
    rows = _integer(dataset["num_samples"], "packed dataset rows")
    _sha256(dataset["content_sha256"], "packed dataset content")
    sampler = plan["sampler"]
    if not isinstance(sampler, Mapping) or set(sampler) != {
        "kind",
        "class",
        "seed",
        "epoch_seed_rule",
        "epoch_order_sha256",
        "versions",
    }:
        raise FinetuneEstimateError("packed source sampler identity is incomplete")
    validate_native_sampler_versions(plan["framework"], sampler["versions"])
    if (
        sampler["kind"] != "seedable_packed_random"
        or sampler["class"] != "accelerate.data_loader.SeedableRandomSampler"
        or sampler["epoch_seed_rule"] != "seed + zero_based_epoch"
    ):
        raise FinetuneEstimateError(
            "packed plan requires the explicit independent source sampler"
        )
    if _integer(sampler["seed"], "packing seed", minimum=0) >= 2**63:
        raise FinetuneEstimateError("packing seed must be less than 2**63")
    if (
        plan["collator_class"]
        != "fakegpu._finetune_packed_batches.IsolatedPackingCollator"
    ):
        raise FinetuneEstimateError("packed plan requires verified isolated collation")
    loader = plan["dataloader"]
    if not isinstance(loader, Mapping) or set(loader) != {
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "pin_memory",
        "drop_last",
    }:
        raise FinetuneEstimateError("packed dataloader identity is incomplete")
    _integer(loader["num_workers"], "packed workers", minimum=0)
    if loader["prefetch_factor"] is not None:
        _integer(loader["prefetch_factor"], "packed prefetch factor")
    if any(
        not isinstance(loader[key], bool)
        for key in ("persistent_workers", "pin_memory", "drop_last")
    ):
        raise FinetuneEstimateError("packed dataloader flags must be boolean")
    packing = plan["packing"]
    if (
        not isinstance(packing, Mapping)
        or set(packing)
        != {
            "mode",
            "batch_sampler",
            "native_drop_last",
            "microsteps_per_epoch",
            "row_logical_sequences",
            "epoch_source_bins",
        }
        or packing["mode"] != "qwen35_reference.v1"
    ):
        raise FinetuneEstimateError("packed execution geometry policy is incomplete")
    expected_class = (
        "axolotl.utils.samplers.multipack.MultipackBatchSampler"
        if plan["framework"] == "axolotl"
        else "torch.utils.data.sampler.BatchSampler"
    )
    if packing["batch_sampler"] != expected_class or not isinstance(
        packing["native_drop_last"], bool
    ):
        raise FinetuneEstimateError(
            "packed batch sampler or tail policy is unsupported"
        )
    if (
        plan["framework"] != "axolotl"
        and packing["native_drop_last"] != loader["drop_last"]
    ):
        raise FinetuneEstimateError("flat packed tail policy differs from its loader")
    batch_size = _integer(plan["micro_batch_size"], "packed configured batch size")
    accumulation = _integer(plan["gradient_accumulation_steps"], "packed accumulation")
    microsteps = _integer(packing["microsteps_per_epoch"], "packed epoch microsteps")
    per_epoch = math.ceil(microsteps / accumulation)
    if _integer(plan["updates_per_epoch"], "packed epoch updates") != per_epoch:
        raise FinetuneEstimateError(
            "packed optimizer epoch length differs from actual microsteps"
        )
    logical = packing["row_logical_sequences"]
    if not isinstance(logical, list) or len(logical) != rows:
        raise FinetuneEstimateError(
            "packed row logical counts differ from dataset size"
        )
    for count in logical:
        _integer(count, "packed row logical sequence count")
    updates = plan["updates"]
    if not isinstance(updates, list) or not updates:
        raise FinetuneEstimateError("packed plan requires optimizer updates")
    epochs = packing["epoch_source_bins"]
    if (
        not isinstance(epochs, list)
        or len(epochs) != math.ceil(len(updates) / per_epoch)
        or not isinstance(sampler["epoch_order_sha256"], list)
        or len(sampler["epoch_order_sha256"]) != len(epochs)
    ):
        raise FinetuneEstimateError("packed epoch bin horizon differs from its updates")
    for epoch, order_hash in zip(epochs, sampler["epoch_order_sha256"]):
        if not isinstance(epoch, list) or len(epoch) != microsteps:
            raise FinetuneEstimateError(
                "packed epoch actual length differs from native declaration"
            )
        consumed = []
        for bins in epoch:
            if not isinstance(bins, list) or not bins or len(bins) > batch_size:
                raise FinetuneEstimateError("packed native bin dimensions are invalid")
            for group in bins:
                if not isinstance(group, list) or not group:
                    raise FinetuneEstimateError("packed native bins cannot be empty")
                for index in group:
                    if _integer(index, "packed source index", minimum=0) >= rows:
                        raise FinetuneEstimateError(
                            "packed source index is outside its dataset"
                        )
                    consumed.append(index)
                if plan["framework"] != "axolotl" and len(group) != 1:
                    raise FinetuneEstimateError(
                        "flat packed datasets require one dataset row per bin"
                    )
        if (
            len(set(consumed)) != len(consumed)
            or (not packing["native_drop_last"] and len(consumed) != rows)
            or _digest(consumed) != order_hash
        ):
            raise FinetuneEstimateError(
                "packed epoch source order, duplicates or dropped rows are inconsistent"
            )
    total_microsteps = total_samples = 0
    for number, update in enumerate(updates, 1):
        if (
            not isinstance(update, Mapping)
            or set(update) != {"optimizer_step", "batches"}
            or _integer(update["optimizer_step"], "packed optimizer step") != number
        ):
            raise FinetuneEstimateError("packed optimizer updates must be consecutive")
        epoch, position = divmod(number - 1, per_epoch)
        start = position * accumulation
        bins = epochs[epoch][start : start + accumulation]
        if not isinstance(update["batches"], list) or len(update["batches"]) != len(
            bins
        ):
            raise FinetuneEstimateError(
                "packed accumulation window differs from native bins"
            )
        for offset, (batch, source_bins) in enumerate(
            zip(update["batches"], bins), start + 1
        ):
            core = _descriptor(batch)
            if set(batch) != set(core) | {
                "epoch_index",
                "epoch_microstep",
                "source_bins",
                "row_indices",
                "input_sha256",
            }:
                raise FinetuneEstimateError(
                    "packed batch has unsupported or missing fields"
                )
            indices = [index for group in source_bins for index in group]
            if (
                core["attention_mask_kind"] != "block_causal_4d"
                or core["shape"][0] != len(source_bins)
                or core["packed"]["logical_sequences"]
                != sum(logical[index] for index in indices)
            ):
                raise FinetuneEstimateError(
                    "packed tensor geometry differs from source membership"
                )
            if (
                _integer(batch.get("epoch_index"), "packed epoch index") != epoch + 1
                or _integer(batch.get("epoch_microstep"), "packed epoch microstep")
                != offset
                or batch.get("source_bins") != source_bins
                or batch.get("row_indices") != indices
            ):
                raise FinetuneEstimateError(
                    "packed consumed prefix differs from native epoch positions"
                )
            _sha256(batch.get("input_sha256"), "packed input SHA256")
            for index in batch["row_indices"]:
                _integer(index, "packed row index", minimum=0)
            for group in batch["source_bins"]:
                for index in group:
                    _integer(index, "packed bin index", minimum=0)
            total_microsteps += 1
            total_samples += core["shape"][0]
    if not isinstance(plan["workload"], Mapping):
        raise FinetuneEstimateError("packed workload requires integer counters")
    for field in ("optimizer_steps", "microsteps", "samples"):
        _integer(plan["workload"].get(field), f"packed workload {field}")
    if plan["workload"] != {
        "optimizer_steps": len(updates),
        "microsteps": total_microsteps,
        "samples": total_samples,
    }:
        raise FinetuneEstimateError(
            "packed workload totals differ from consumed prefix"
        )
    return deepcopy(dict(plan))


def install_packed_random_sampling(trainer, *, framework, seed, replace):
    """Replace only native random source sampling, retaining the native packer."""
    import torch
    from accelerate.data_loader import SeedableRandomSampler

    validate_native_sampler_versions(framework)
    _integer(seed, "packing seed", minimum=0)
    if seed >= 2**63:
        raise FinetuneEstimateError("packing seed must be less than 2**63")
    if getattr(trainer, "_train_dataloader", None) is not None:
        raise FinetuneEstimateError(
            "packing seed must be installed before loader preparation"
        )
    if hasattr(trainer, "_fakegpu_packing_seed"):
        raise FinetuneEstimateError("independent packing seed is already installed")

    def seeded(base, dataset):
        if type(base) is not torch.utils.data.RandomSampler or base.replacement:
            raise FinetuneEstimateError(
                "packing seed requires native random sampling without replacement"
            )
        if base.num_samples != len(dataset):
            raise FinetuneEstimateError(
                "packing source sampler must cover its whole dataset"
            )
        return SeedableRandomSampler(
            dataset, data_seed=seed, generator=torch.Generator(device="cpu")
        )

    if framework == "axolotl":
        from axolotl.utils.samplers.multipack import MultipackBatchSampler

        if not trainer.args.sample_packing or trainer.args.pretraining:
            raise FinetuneEstimateError(
                "Axolotl packed sampling requires SFT sample_packing"
            )
        original = trainer._create_multipack_sampler

        def create(base_sampler, dataset):
            result = original(
                base_sampler=seeded(base_sampler, dataset), dataset=dataset
            )
            if type(result) is not MultipackBatchSampler:
                raise FinetuneEstimateError(
                    "packing requires the native MultipackBatchSampler"
                )
            return result

        replace(trainer, "_create_multipack_sampler", create)
    else:
        original = trainer._get_train_sampler

        def sampler(train_dataset=None):
            dataset = trainer.train_dataset if train_dataset is None else train_dataset
            return seeded(original(dataset), dataset)

        replace(trainer, "_get_train_sampler", sampler)
    # Accelerate replaces RandomSampler subclasses during loader preparation.
    # Its explicit data_seed must agree with the independently installed source.
    replace(trainer.accelerator.dataloader_config, "data_seed", seed)
    replace(trainer, "_fakegpu_packing_seed", seed)


def inspect_packed_epochs(
    trainer, *, framework, epochs=None, max_steps=None, include_core=False
):
    """Record actual native CPU bin membership and collated work for full epochs."""
    import torch
    import numpy as np
    from accelerate.data_loader import SeedableRandomSampler

    versions = validate_native_sampler_versions(framework)
    if (epochs is None) == (max_steps is None):
        raise FinetuneEstimateError("provide one packing inspection duration")
    _integer(epochs if epochs is not None else max_steps, "packing duration")
    args = trainer.args
    explicit_seed = getattr(trainer, "_fakegpu_packing_seed", None)
    if explicit_seed is None:
        raise FinetuneEstimateError(
            "packing inspection requires the independent packing seed"
        )
    if args.world_size != 1 or args.n_gpu > 1 or trainer.accelerator.num_processes != 1:
        raise FinetuneEstimateError("packing inspection requires one device")
    with _preserve_random_streams(torch):
        loader = trainer.get_train_dataloader()
        native = loader.batch_sampler
        if framework == "axolotl":
            from axolotl.utils.samplers.multipack import MultipackBatchSampler

            if type(native) is not MultipackBatchSampler:
                raise FinetuneEstimateError(
                    "packing inspection requires native multipack bins"
                )
        elif type(native) is not torch.utils.data.BatchSampler:
            raise FinetuneEstimateError(
                "flat packing inspection requires native ordinary batches"
            )
        base = native.sampler
        if (
            type(base) is not SeedableRandomSampler
            or base.replacement
            or base.initial_seed != explicit_seed
        ):
            raise FinetuneEstimateError(
                "packing inspection requires the independent packing seed"
            )
        if len(loader.dataset) != len(trainer.train_dataset):
            raise FinetuneEstimateError("prepared packing loader changed dataset size")
        dataset_rows = len(loader.dataset)
        logical_per_row = [
            len(
                packed_source_membership(
                    [[loader.dataset[index]]]
                    if framework == "axolotl"
                    else [loader.dataset[index]],
                    framework=framework,
                )[0]["spans"]
            )
            for index in range(dataset_rows)
        ]
        declared_microsteps = len(loader)
        if not declared_microsteps:
            raise FinetuneEstimateError("native packing drops every batch")
        if epochs is None:
            epochs = math.ceil(
                max_steps
                / math.ceil(declared_microsteps / args.gradient_accumulation_steps)
            )
        sampler = copy(native)
        sampler.sampler = SeedableRandomSampler(
            loader.dataset,
            data_seed=base.initial_seed,
            generator=torch.Generator(device="cpu"),
        )
        results = []
        for epoch in range(epochs):
            if framework == "axolotl":
                sampler.set_epoch(epoch)
            sampler.sampler.set_epoch(epoch)
            batches, consumed = [], []
            for indices in sampler:
                bins = (
                    deepcopy(indices)
                    if framework == "axolotl"
                    else [[index] for index in indices]
                )
                flattened = [index for group in bins for index in group]
                if any(
                    isinstance(index, bool)
                    or not isinstance(index, (int, np.integer))
                    or not 0 <= index < dataset_rows
                    for index in flattened
                ):
                    raise FinetuneEstimateError(
                        "native packing emitted invalid source row indices"
                    )
                bins = [[int(index) for index in group] for group in bins]
                flattened = [index for group in bins for index in group]
                if not bins or any(not group for group in bins):
                    raise FinetuneEstimateError("native packing emitted an empty bin")
                features = (
                    [
                        [deepcopy(loader.dataset[index]) for index in group]
                        for group in bins
                    ]
                    if framework == "axolotl"
                    else [deepcopy(loader.dataset[index]) for index in flattened]
                )
                inputs = loader.collate_fn(deepcopy(features))
                descriptor = describe_packed_batch(inputs)
                if (
                    describe_packed_batch(loader.collate_fn(deepcopy(features)))
                    != descriptor
                ):
                    raise FinetuneEstimateError(
                        "packing inspection requires deterministic collation"
                    )
                if descriptor["packed_rows"] != len(bins):
                    raise FinetuneEstimateError(
                        "native collation changed its number of packed bins"
                    )
                batches.append(
                    {
                        "source_bins": bins,
                        "source_dataset_rows": len(flattened),
                        **descriptor,
                    }
                )
                if include_core:
                    batches[-1]["core_descriptor"] = describe_packed_core(inputs)
                consumed.extend(flattened)
            if not batches or len(consumed) != len(set(consumed)):
                raise FinetuneEstimateError(
                    "native packing repeated source rows or produced no work"
                )
            dropped = sorted(set(range(dataset_rows)) - set(consumed))
            if dropped and not native.drop_last:
                raise FinetuneEstimateError(
                    "native packing lost rows despite retaining tails"
                )
            results.append(
                {
                    "epoch_index": epoch + 1,
                    "microsteps": len(batches),
                    "optimizer_updates": math.ceil(
                        len(batches) / args.gradient_accumulation_steps
                    ),
                    "source_dataset_rows": len(consumed),
                    "dropped_dataset_row_indices": dropped,
                    "dropped_logical_sequences": sum(
                        logical_per_row[index] for index in dropped
                    ),
                    "source_order_sha256": _digest(consumed),
                    "batches": batches,
                    "packed_rows": sum(batch["packed_rows"] for batch in batches),
                    "logical_sequences": sum(
                        batch["logical_sequences"] for batch in batches
                    ),
                    "executed_token_slots": sum(
                        batch["shape"][0] * batch["shape"][1] for batch in batches
                    ),
                    "nonpadding_tokens": sum(
                        batch["nonpadding_tokens"] for batch in batches
                    ),
                }
            )
    return {
        "schema_version": "fakegpu.native_packing_epochs.v1",
        "framework": framework,
        "versions": versions,
        "dataset_rows": dataset_rows,
        "dataset_logical_sequences": sum(logical_per_row),
        **({"row_logical_sequences": logical_per_row} if include_core else {}),
        "configured_micro_batch_size": args.per_device_train_batch_size,
        "source_sampler": _class_name(base),
        "source_seed": base.initial_seed,
        "epoch_seed_rule": "seed + zero_based_epoch",
        "native_batch_sampler": _class_name(native),
        "native_drop_last": native.drop_last,
        "native_declared_microsteps_per_epoch": declared_microsteps,
        "native_declared_optimizer_updates_per_epoch": math.ceil(
            declared_microsteps / args.gradient_accumulation_steps
        ),
        "epochs": results,
    }
