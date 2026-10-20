"""Plan future native updates and events without reading a target run's timings.

The first native schedule adapter supports explicitly sequential, unpacked
map-style text training in the locally validated framework versions. A complete
recorded epoch supplies actual collator statistics, including its tail batches.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math

from ._finetune_batches import _descriptor
from ._finetune_pipeline_plan import pipeline_plan_from_run, validate_pipeline_plan
from .finetune_estimator import FinetuneEstimateError


_NATIVE_SCHEDULES = {
    "huggingface": (None, "5.6.0", "disable_shuffling"),
    "llamafactory": ("0.9.5", "5.6.0", "disable_shuffling"),
    "axolotl": ("0.18.0", "5.14.1", "curriculum_sampling"),
}


def _integer(value, field, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinetuneEstimateError(f"{field} must be an integer >= {minimum}")
    return value


def _epoch_template(run):
    config = run["training_config"]
    framework = run["framework"]
    if framework not in _NATIVE_SCHEDULES:
        raise FinetuneEstimateError(
            "no validated native future schedule for this framework"
        )
    version, transformers_version, sampler_flag = _NATIVE_SCHEDULES[framework]
    software = config["software_versions"]
    if (version is not None and software.get(framework) != version) or software.get(
        "transformers"
    ) != transformers_version:
        raise FinetuneEstimateError(
            "native future schedule requires a validated framework/Trainer version"
        )
    raw = run["native_config"]
    if raw.get(sampler_flag) is not True:
        raise FinetuneEstimateError(
            f"future epoch planning requires {sampler_flag}=true"
        )
    if (
        config.get("packing") is not False
        or config.get("dataloader_drop_last") is not False
    ):
        raise FinetuneEstimateError(
            "future epoch planning requires unpacked training retaining tail batches"
        )
    _validate_native_schedule_options(run)
    count = _integer(config["num_samples"], "dataset num_samples")
    batch_size = _integer(config["micro_batch_size"], "micro_batch_size")
    accumulation = _integer(
        config["gradient_accumulation_steps"], "gradient_accumulation_steps"
    )
    microsteps = math.ceil(count / batch_size)
    epochs = defaultdict(list)
    for update in run["observation"]["updates"]:
        for batch in update["batches"]:
            epochs[batch["epoch_index"]].append(batch)
    complete = [batches for batches in epochs.values() if len(batches) == microsteps]
    if not complete:
        raise FinetuneEstimateError(
            "future planning requires an observed complete epoch"
        )
    template = [_descriptor(batch) for batch in complete[0]]
    if sum(batch["shape"][0] for batch in template) != count:
        raise FinetuneEstimateError(
            "observed complete epoch does not consume the dataset once"
        )
    expected_sizes = [batch_size] * (microsteps - 1) + [
        count - batch_size * (microsteps - 1)
    ]
    if [batch["shape"][0] for batch in template] != expected_sizes:
        raise FinetuneEstimateError(
            "observed epoch batch sizes differ from sequential non-dropping sampling"
        )
    if any(
        [_descriptor(batch) for batch in batches] != template[: len(batches)]
        for batches in epochs.values()
    ):
        raise FinetuneEstimateError(
            "recorded epochs do not share a deterministic batch descriptor sequence"
        )
    return template, accumulation


def _validate_native_schedule_options(run):
    raw = run["native_config"]
    if any(
        raw.get(field)
        for field in (
            "callbacks",
            "plugins",
            "streaming",
            "resume_from_checkpoint",
            "early_stopping_steps",
            "early_stopping_patience",
            "enable_torch_profiler",
            "profile_modules",
        )
    ):
        raise FinetuneEstimateError(
            "future epoch planning does not support callbacks, plugins, early stopping or resume"
        )


def _schedule_calls(run, steps, updates_per_epoch):
    schedule = run["schedule"]
    if (
        schedule.get("eval_delay") != 0
        or schedule.get("load_best_model_at_end") is not False
    ):
        raise FinetuneEstimateError(
            "future schedule requires eval_delay=0 and no best-model reload"
        )
    if type(schedule.get("eval_on_start")) is not bool:
        raise FinetuneEstimateError("future schedule requires recorded eval_on_start")
    resolved = {}
    for kind in ("eval", "save"):
        strategy = schedule[f"{kind}_strategy"]
        if strategy not in {"no", "steps", "epoch"}:
            raise FinetuneEstimateError("unsupported future event strategy")
        if strategy == "steps":
            value = schedule[f"requested_{kind}_steps"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise FinetuneEstimateError(
                    "future event interval must be finite and positive"
                )
            if value < 1:
                resolved[kind] = math.ceil(steps * value)
            elif value == int(value):
                resolved[kind] = int(value)
            else:
                raise FinetuneEstimateError("event interval >= 1 must be an integer")
    calls = []
    if schedule["eval_on_start"]:
        calls.append((0, "evaluate"))
    for step in range(1, steps + 1):
        # Trainer processes step callbacks before epoch callbacks, even when
        # both happen at the same final update. Mixed strategies preserve that
        # order: a step checkpoint can precede an epoch evaluation.
        for phase in ("steps", "epoch"):
            for kind, operation in (("eval", "evaluate"), ("save", "_save_checkpoint")):
                if schedule[f"{kind}_strategy"] != phase:
                    continue
                interval = resolved[kind] if phase == "steps" else updates_per_epoch
                if step == steps or step % interval == 0:
                    calls.append((step, operation))
    return calls, resolved


def plan_finetuning_pipeline(run, *, epochs=None, max_steps=None, num_samples=None):
    """Project a sequential native dataset onto future epochs or optimizer steps.

    The calibration dataset and execution settings remain bound to their source
    identity. Different data need a new native calibration. This function never
    consumes a holdout report, predicts timing, or fabricates sampler identities.
    """
    plan = pipeline_plan_from_run(run)
    template, accumulation = _epoch_template(run)
    dataset_rows = run["training_config"]["num_samples"]
    if num_samples is not None and _integer(num_samples, "num_samples") != dataset_rows:
        raise FinetuneEstimateError(
            "changed dataset size requires matching native dataset calibration"
        )
    if (epochs is None) == (max_steps is None):
        raise FinetuneEstimateError("provide exactly one of epochs or max_steps")
    per_epoch = math.ceil(len(template) / accumulation)
    if max_steps is not None:
        steps = _integer(max_steps, "max_steps")
    else:
        if (
            isinstance(epochs, bool)
            or not isinstance(epochs, (int, float))
            or not math.isfinite(epochs)
            or epochs <= 0
        ):
            raise FinetuneEstimateError("epochs must be finite and positive")
        steps = math.ceil(epochs * per_epoch)
    planned_updates = []
    for step in range(1, steps + 1):
        epoch, position = divmod(step - 1, per_epoch)
        start = position * accumulation
        planned_updates.append(
            {
                "optimizer_step": step,
                "batches": [
                    dict(
                        deepcopy(batch),
                        epoch_index=epoch + 1,
                        epoch_microstep=start + index + 1,
                    )
                    for index, batch in enumerate(
                        template[start : start + accumulation]
                    )
                ],
            }
        )
    return validate_pipeline_plan(
        _assemble_schedule(
            run,
            plan,
            planned_updates,
            epochs=epochs,
            max_steps=max_steps,
            per_epoch=per_epoch,
            dataset_rows=dataset_rows,
            sampler="native_sequential",
        )
    )


def _assemble_schedule(
    run, plan, planned_updates, *, epochs, max_steps, per_epoch, dataset_rows, sampler
):
    """Attach the same native event schedule to independently planned inputs."""
    steps = len(planned_updates)
    calls, resolved = _schedule_calls(run, steps, per_epoch)
    policy = plan["execution_policy"]
    # Trainer resolves fractional logging_steps for every logging strategy.
    interval = policy["requested_logging_steps"]
    policy["resolved_logging_steps"] = (
        math.ceil(steps * interval) if interval < 1 else int(interval)
    )
    # Check the supported schedule against source execution before extrapolating.
    source_calls, _ = _schedule_calls(
        run, plan["workload"]["optimizer_steps"], per_epoch
    )
    actual_calls = [
        (event["optimizer_step"], event["descriptor"]["operation"])
        for event in plan["root_events"]
        if event["descriptor"]["inside_train"]
    ]
    if actual_calls != source_calls:
        raise FinetuneEstimateError(
            "recorded events differ from the supported native schedule"
        )
    event_templates = defaultdict(list)
    finals = []
    for event in plan["root_events"]:
        descriptor = event["descriptor"]
        if descriptor["inside_train"]:
            event_templates[descriptor["operation"]].append(descriptor)
        else:
            finals.append(descriptor)
    by_step = defaultdict(list)
    for step, operation in calls:
        by_step[step].append(operation)
    events, updates = [], []
    operations = [{"kind": "trainer_start"}]
    seen = set()
    pending = []

    def add_event(step, operation=None, final=None):
        if final is not None:
            descriptor = deepcopy(final)
        else:
            phase = (
                "initial" if step == 0 else "repeated" if operation in seen else "first"
            )
            choices = [
                row
                for row in event_templates.get(operation, [])
                if row["phase"] == phase
            ]
            if not choices or any(row != choices[0] for row in choices):
                raise FinetuneEstimateError(
                    f"source lacks consistent {phase} {operation} workload"
                )
            descriptor = deepcopy(choices[0])
            if step:
                seen.add(operation)
        event_id = len(events) + 1
        events.append(
            {"event_id": event_id, "optimizer_step": step, "descriptor": descriptor}
        )
        operations.append({"kind": descriptor["kind"], "event_id": event_id})
        if descriptor["inside_train"]:
            pending.append(event_id)

    for operation in by_step[0]:
        add_event(0, operation)
    for step in range(1, steps + 1):
        batches = deepcopy(planned_updates[step - 1]["batches"])
        updates.append(
            {
                "optimizer_step": step,
                "warmup": step <= plan["warmup"]["optimizer_steps"],
                "batches": batches,
                "preceding_event_ids": pending.copy(),
            }
        )
        pending.clear()
        operations.append({"kind": "optimizer_update", "optimizer_step": step})
        for operation in by_step[step]:
            add_event(step, operation)
    operations.append({"kind": "trainer_end"})
    for descriptor in finals:
        add_event(steps, final=descriptor)
    plan.update(
        source="cpu_native_schedule",
        updates=updates,
        root_events=events,
        operations=operations,
        workload={
            "optimizer_steps": steps,
            "microsteps": sum(len(row["batches"]) for row in updates),
            "samples": sum(
                batch["shape"][0] for row in updates for batch in row["batches"]
            ),
        },
        projection={
            "source_run_uuid": run["run_identity"]["uuid"],
            "epochs": epochs,
            "max_steps": max_steps,
            "dataset_rows": dataset_rows,
            "updates_per_epoch": per_epoch,
            "resolved_event_steps": resolved,
            "sampler": sampler,
        },
    )
    return plan
