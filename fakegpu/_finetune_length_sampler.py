"""Opt-in epoch-seeded length grouping using the installed Trainer algorithm.

This native-only module imports the optional training stack. The ordinary
framework sampler remains untouched unless a caller explicitly supplies a seed.
"""

from __future__ import annotations

import torch
from transformers.trainer_pt_utils import LengthGroupedSampler

from .finetune_estimator import FinetuneEstimateError
from ._finetune_sampler import _digest


class SeedableLengthGroupedSampler(LengthGroupedSampler):
    """Keep native grouping while separating sampling from the model RNG."""

    def __init__(self, batch_size, lengths, *, seed):
        super().__init__(
            batch_size,
            lengths=list(lengths),
            generator=torch.Generator(device="cpu"),
        )
        self.initial_seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        self.generator.manual_seed(self.initial_seed + self.epoch)
        yield from super().__iter__()
        self.set_epoch(self.epoch + 1)


class FixedLengthGroupedSampler(SeedableLengthGroupedSampler):
    """Use one independent ordering for every evaluation, including cached loaders."""

    def __iter__(self):
        self.generator.manual_seed(self.initial_seed)
        yield from LengthGroupedSampler.__iter__(self)


def install_length_grouping(trainer, *, seed, replace):
    """Replace only a known native length sampler; the observer owns restoration."""
    if getattr(trainer.args, "train_sampling_strategy", None) != "group_by_length":
        raise FinetuneEstimateError(
            "length_grouping_seed requires native train_sampling_strategy=group_by_length"
        )
    strategy = getattr(trainer.args.eval_strategy, "value", trainer.args.eval_strategy)
    evaluation = strategy != "no" or trainer.args.do_eval or trainer.eval_dataset is not None
    if evaluation and (trainer.eval_dataset is None or isinstance(trainer.eval_dataset, dict)):
        raise FinetuneEstimateError(
            "seeded length grouping requires one finite evaluation dataset"
        )
    if getattr(trainer, "_train_dataloader", None) is not None or getattr(
        trainer, "_eval_dataloaders", None
    ):
        raise FinetuneEstimateError(
            "seeded length grouping must be installed before dataloader preparation"
        )
    original = trainer._get_train_sampler

    def seeded(train_dataset=None):
        native = original(train_dataset)
        if type(native) is not LengthGroupedSampler:
            raise FinetuneEstimateError(
                "length_grouping_seed requires an ordinary native LengthGroupedSampler"
            )
        return SeedableLengthGroupedSampler(
            native.batch_size, native.lengths, seed=seed
        )

    # Validate before changing the Trainer; later calls validate their sampler too.
    seeded(trainer.train_dataset)
    policy = None
    if evaluation:
        original_eval = trainer._get_eval_sampler

        def fixed(eval_dataset):
            native = original_eval(eval_dataset)
            if type(native) is not LengthGroupedSampler:
                raise FinetuneEstimateError(
                    "seeded evaluation requires an ordinary native LengthGroupedSampler"
                )
            return FixedLengthGroupedSampler(native.batch_size, native.lengths, seed=seed)

        native = fixed(trainer.eval_dataset)
        policy = {
            "schema_version": "fakegpu.finetune_eval_sampler.v1",
            "kind": "fixed_seed_length_grouped",
            "seed": seed,
            "iteration_seed_rule": "same_seed_each_invocation",
            "grouping_batch_size": native.batch_size,
            "dataset_rows": len(native.lengths),
            "lengths_sha256": _digest(native.lengths),
        }
        replace(trainer, "_get_eval_sampler", fixed)
    replace(trainer, "_get_train_sampler", seeded)
    return policy
