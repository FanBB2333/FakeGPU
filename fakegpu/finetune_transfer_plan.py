"""Future native schedules from independently verified CPU sampler plans."""

from __future__ import annotations

from ._finetune_transfer_plan import transfer_plan_from_run, validate_transfer_plan
from .finetune_estimator import FinetuneEstimateError
from .finetune_pipeline_plan import _assemble_schedule, _target_steps


def plan_finetuning_transfer(run, *, epochs=None, max_steps=None):
    """Select a future duration within a native report's saved sampling horizon.

    No target timing is consumed. Increasing the dataset or planning horizon needs
    a new CPU plan and target-specific startup measurements, obtained through the
    native calibration workflow. A complete observed training epoch is unnecessary.
    """
    plan = transfer_plan_from_run(run)
    sampling = plan["sampling_plan"]
    per_epoch = sampling["updates_per_epoch"]
    steps = _target_steps(epochs, max_steps, per_epoch)
    if steps > len(sampling["updates"]):
        raise FinetuneEstimateError(
            "target duration exceeds the saved sampling horizon; run a new calibration with a larger planning horizon"
        )
    return validate_transfer_plan(
        _assemble_schedule(
            run,
            plan,
            sampling["updates"][:steps],
            epochs=epochs,
            max_steps=max_steps,
            per_epoch=per_epoch,
            dataset_rows=sampling["dataset"]["num_samples"],
            sampler=sampling["sampler"]["kind"],
        )
    )
