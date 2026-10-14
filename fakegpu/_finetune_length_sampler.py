"""Opt-in epoch-seeded length grouping using the installed Trainer algorithm.

This native-only module imports the optional training stack. The ordinary
framework sampler remains untouched unless a caller explicitly supplies a seed.
"""

from __future__ import annotations

import torch
from transformers.trainer_pt_utils import LengthGroupedSampler

from .finetune_estimator import FinetuneEstimateError


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


def install_length_grouping(trainer, *, seed, replace):
    """Replace only a known native length sampler; the observer owns restoration."""
    if getattr(trainer.args, "train_sampling_strategy", None) != "group_by_length":
        raise FinetuneEstimateError(
            "length_grouping_seed requires native train_sampling_strategy=group_by_length"
        )
    strategy = getattr(trainer.args.eval_strategy, "value", trainer.args.eval_strategy)
    if strategy != "no" or trainer.args.do_eval or trainer.eval_dataset is not None:
        raise FinetuneEstimateError(
            "seeded length grouping currently requires evaluation disabled"
        )
    if getattr(trainer, "_train_dataloader", None) is not None:
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
    replace(trainer, "_get_train_sampler", seeded)
