"""Explicit empirical shape transfer over verified native sampler plans.

Dataset identity remains attached to startup. Work reuse removes only training
data locators and the sampler seed; model initialization and optimizer choices
remain part of execution identity. This is an empirical approximation, not an
assertion that different token distributions have identical costs.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

from ._finetune_model import native_config_signature, native_execution_signature
from ._finetune_pipeline_plan import (
    SCHEMA_VERSION as PIPELINE_SCHEMA,
    _describe_components,
    _digest,
    pipeline_plan_from_run,
    validate_pipeline_plan,
)
from ._finetune_sampler import (
    LENGTH_SCHEMA_VERSION,
    PACKED_SCHEMA_VERSION,
    SCHEMA_VERSION as SAMPLER_SCHEMA,
    _integer,
    validate_length_grouping_evaluation,
    validate_sampler_plan,
    validate_sampling_verification,
)
from .finetune_estimator import FinetuneEstimateError
from .finetune_pipeline_plan import _validate_native_schedule_options

SCHEMA_VERSION = "fakegpu.finetune_transfer_plan.v1"


def _sampler_identity(sampling):
    sampler = sampling["sampler"]
    identity = {
        field: deepcopy(sampler[field])
        for field in ("kind", "class", "epoch_seed_rule", "versions")
    }
    if sampler["kind"] == "seedable_length_grouped":
        # Grouping changes epoch setup work. First support retains the complete
        # native length sequence and grouping width rather than pooling sizes.
        identity["grouping"] = deepcopy(sampler["grouping"])
    if sampling["schema_version"] == PACKED_SCHEMA_VERSION:
        identity["seed"] = sampler["seed"]
        identity["packing"] = {
            key: deepcopy(value)
            for key, value in sampling["packing"].items()
            if key != "epoch_source_bins"
        }
    return identity


def _execution_identity(run, sampling):
    if sampling["schema_version"] == PACKED_SCHEMA_VERSION:
        return {
            "signature": deepcopy(run["observation"]["signature"]),
            "sampler": _sampler_identity(sampling),
            "collator_class": sampling["collator_class"],
            "dataloader": deepcopy(sampling["dataloader"]),
        }
    raw = deepcopy(run["native_config"])
    raw.pop("data_seed", None)
    if run["framework"] == "huggingface":
        raw.pop("train_dataset", None)
        raw.pop("num_samples", None)
    elif run["framework"] == "llamafactory":
        for name in ("dataset", "dataset_dir", "max_samples"):
            raw.pop(name, None)
    else:
        # Preserve each dataset's formatting and tokenization options. Only its
        # input file location changes; test/eval datasets remain fully bound.
        for dataset in raw.get("datasets", []):
            if dataset.get("path") not in {"json", "csv", "text", "parquet", "arrow"}:
                dataset.pop("path", None)
            dataset.pop("data_files", None)
    observed = run["observation"]["observed_model"]
    signature = deepcopy(run["observation"]["signature"])
    signature.pop("data_signature")
    signature["native_execution_signature"] = native_execution_signature(
        native_config_signature(raw), observed
    )
    return {
        "signature": signature,
        "sampler": _sampler_identity(sampling),
        "collator_class": sampling["collator_class"],
        "dataloader": deepcopy(sampling["dataloader"]),
    }


def transfer_plan_from_run(run):
    """Require both native timing evidence and verified consumed input hashes."""
    _validate_native_schedule_options(run)
    return _transfer_plan(run, pipeline_plan_from_run(run))


def _transfer_plan(run, plan):
    if plan["signature"].get("sampler_observation_version") not in {
        SAMPLER_SCHEMA,
        LENGTH_SCHEMA_VERSION,
        PACKED_SCHEMA_VERSION,
    }:
        raise FinetuneEstimateError(
            "transfer budgets require a verified native sampling plan; rerun with a planning horizon"
        )
    sampling = validate_sampler_plan(run.get("sampling_plan"))
    validate_sampling_verification(
        sampling,
        run["observation"]["updates"],
        run["observation"].get("sampling_verification"),
    )
    if (
        sampling["collator_class"]
        != run["observation"]["observed_model"]["collator_class"]
    ):
        raise FinetuneEstimateError("sampling plan differs from the actual collator")
    identity = _execution_identity(run, sampling)
    plan.update(
        schema_version=SCHEMA_VERSION,
        sampling_plan=sampling,
        transfer_identity=identity,
        transfer_identity_sha256=_digest(identity),
    )
    plan["setup_workload"]["sampling_plan_sha256"] = sampling["plan_sha256"]
    return validate_transfer_plan(plan)


def validate_transfer_plan(plan):
    """Validate the timing-free protocol and its exact sampled input prefix."""
    if not isinstance(plan, Mapping) or plan.get("schema_version") != SCHEMA_VERSION:
        raise FinetuneEstimateError("unsupported native transfer plan schema")
    validate_pipeline_plan({**plan, "schema_version": PIPELINE_SCHEMA})
    sampling = validate_sampler_plan(plan.get("sampling_plan"))
    signature = plan["signature"]
    if sampling["schema_version"] == PACKED_SCHEMA_VERSION:
        _integer(signature.get("packing_seed"), "packing_seed", minimum=0)
        if (
            signature.get("packing_seed") != sampling["sampler"]["seed"]
            or signature.get("packing") is not True
        ):
            raise FinetuneEstimateError(
                "packed source seed or execution flag differs from its signature"
            )
        native = {
            key: value
            for key, value in sampling["packing"].items()
            if key
            in {
                "microsteps_per_epoch",
                "batch_sampler",
                "native_drop_last",
                "row_logical_sequences",
            }
        }
        if plan["setup_workload"].get("native_packing") != native:
            raise FinetuneEstimateError(
                "packed setup epoch counts differ from the sampled bins"
            )
    elif "packing_seed" in signature or "packing_reference" in signature:
        raise FinetuneEstimateError("packing policy requires sampler plan v3")
    if "length_grouping_evaluation" in signature:
        if sampling["schema_version"] != LENGTH_SCHEMA_VERSION:
            raise FinetuneEstimateError(
                "evaluation grouping requires a v2 length sampler"
            )
        validate_length_grouping_evaluation(
            signature["length_grouping_evaluation"], seed=sampling["sampler"]["seed"]
        )
    if sampling["schema_version"] == LENGTH_SCHEMA_VERSION:
        _integer(
            signature.get("length_grouping_seed"), "length_grouping_seed", minimum=0
        )
    if (
        signature.get("sampler_observation_version") != sampling["schema_version"]
        or ("length_grouping_seed" in signature)
        != (sampling["schema_version"] == LENGTH_SCHEMA_VERSION)
        or (
            sampling["schema_version"] == LENGTH_SCHEMA_VERSION
            and signature.get("length_grouping_seed") != sampling["sampler"]["seed"]
        )
        or any(
            signature["software_versions"].get(name) != version
            for name, version in sampling["sampler"]["versions"].items()
        )
        or sampling["framework"] != signature["source_framework"]
        or sampling["dataset"]["content_sha256"] != signature["data_signature"]
        or sampling["dataset"]["num_samples"]
        != plan["setup_workload"]["dataset_num_samples"]
        or sampling["micro_batch_size"] != signature["micro_batch_size"]
        or sampling["gradient_accumulation_steps"]
        != signature["gradient_accumulation_steps"]
        or sampling["plan_sha256"] != plan["setup_workload"].get("sampling_plan_sha256")
    ):
        raise FinetuneEstimateError(
            "transfer sampling plan differs from its workload identity"
        )
    updates = plan["updates"]
    if len(updates) > len(sampling["updates"]) or any(
        actual["optimizer_step"] != expected["optimizer_step"]
        or actual["batches"] != expected["batches"]
        for actual, expected in zip(updates, sampling["updates"])
    ):
        raise FinetuneEstimateError(
            "transfer updates differ from the sampled input prefix"
        )
    identity = plan.get("transfer_identity")
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"signature", "sampler", "collator_class", "dataloader"}
        or _digest(identity) != plan.get("transfer_identity_sha256")
    ):
        raise FinetuneEstimateError(
            "transfer execution identity is incomplete or changed"
        )
    expected = deepcopy(signature)
    if sampling["schema_version"] != PACKED_SCHEMA_VERSION:
        expected.pop("data_signature")
    expected["native_execution_signature"] = identity["signature"].get(
        "native_execution_signature"
    )
    if (
        identity["signature"] != expected
        or any(
            identity[field] != sampling[field]
            for field in ("collator_class", "dataloader")
        )
        or identity["sampler"] != _sampler_identity(sampling)
    ):
        raise FinetuneEstimateError("transfer execution identity differs from the plan")
    return deepcopy(dict(plan))


def transfer_signature(plan):
    return deepcopy(plan["transfer_identity"])


def _execution_window(plan, update, context):
    from ._finetune_batches import execution_batch_shape

    microsteps = (
        plan["setup_workload"]["native_packing"]["microsteps_per_epoch"]
        if "native_packing" in plan["setup_workload"]
        else math.ceil(
            plan["setup_workload"]["dataset_num_samples"]
            / plan["signature"]["micro_batch_size"]
        )
    )
    return {
        "batches": [execution_batch_shape(batch) for batch in update["batches"]],
        "context": context,
        "epoch_context": {
            "starts_epoch": update["batches"][0]["epoch_microstep"] == 1,
            "ends_epoch": update["batches"][-1]["epoch_microstep"] == microsteps,
        },
    }


def describe_transfer_components(plan):
    """Preserve the timeline partition, replacing only work grouping rules."""
    seen = set()

    def window(plan, update, context):
        result = _execution_window(plan, update, context)
        first_uses = []
        for batch in result["batches"]:
            identity = _digest(batch)
            first_uses.append(identity not in seen)
            seen.add(identity)
        result["first_shape_use"] = first_uses
        return result

    return _describe_components(plan, window)
