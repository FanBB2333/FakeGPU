"""Native CPU packing counts with independent epoch-seeded source sampling.

This inspects actual native bins, tails and collation; it does not fit timings
or open packed training budgets. Dataset rows, logical sequences and executed
tensor rows have distinct counts. Original sampler and global RNG state stay
unchanged while epochs are reconstructed.
"""

from __future__ import annotations

from copy import copy, deepcopy
import math

from ._finetune_packed_batches import describe_packed_batch, packed_source_membership
from ._finetune_sampler import (
    _class_name,
    _digest,
    _integer,
    _preserve_random_streams,
    validate_native_sampler_versions,
)
from .finetune_estimator import FinetuneEstimateError


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


def inspect_packed_epochs(trainer, *, framework, epochs):
    """Record actual native CPU bin membership and collated work for full epochs."""
    import torch
    import numpy as np
    from accelerate.data_loader import SeedableRandomSampler

    versions = validate_native_sampler_versions(framework)
    _integer(epochs, "packing epochs")
    args = trainer.args
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
        if type(base) is not SeedableRandomSampler or base.replacement:
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
