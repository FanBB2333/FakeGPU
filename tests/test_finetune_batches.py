from collections import UserDict
from copy import deepcopy
from types import SimpleNamespace

import pytest

from fakegpu._finetune_batches import (
    SCHEMA_VERSION,
    describe_batch,
    summarize_batch_updates,
    validate_batch_profile,
)


class Tensor:
    """Optional tensor metadata without importing a training dependency."""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return deepcopy(self.rows)


def _batch(mask=None, *, labels=None, positions=None, shape=(2, 4)):
    inputs = UserDict({"input_ids": SimpleNamespace(shape=shape)})
    for key, value in (
        ("attention_mask", mask),
        ("labels", labels),
        ("position_ids", positions),
    ):
        if value is not None:
            inputs[key] = Tensor(value)
    return describe_batch(inputs)


def _known_batch():
    return _batch(
        [[1, 1, 1, 0], [0, 1, 1, 1]],
        labels=[[1, -100, 2, -100], [-100, 3, 4, 5]],
        positions=[[0, 1, 0, 0], [0, 0, 1, 2]],
    )


def _update(index, batches, *, warmup=False, seconds=1.0):
    return {
        "optimizer_step": index,
        "batches": deepcopy(batches),
        "microsteps": len(batches),
        "samples": sum(batch["shape"][0] for batch in batches),
        "warmup": warmup,
        "seconds": seconds,
    }


def _observation():
    batch = _known_batch()
    updates = [
        _update(1, [batch, batch], warmup=True, seconds=5.0),
        _update(2, [batch, batch], seconds=2.0),
        _update(3, [batch, batch], seconds=3.0),
        _update(4, [batch], seconds=1.5),
    ]
    profile = summarize_batch_updates(updates)
    return {
        "signature": {"batch_observation_version": SCHEMA_VERSION},
        "updates": updates,
        "batch_profile": profile,
        "workload": dict(profile["totals"], shape_counts=profile["shape_counts"]),
    }


def test_binary_mask_labels_causal_shift_and_position_zeros_are_observed():
    batch = _known_batch()
    assert batch["shape"] == [2, 4]
    assert batch["token_slots"] == 8
    assert batch["row_nonpadding_tokens"] == [3, 3]
    assert batch["nonpadding_tokens"] == 6
    assert batch["padding_tokens"] == 2
    assert batch["row_supervised_tokens"] == [2, 3]
    assert batch["supervised_tokens"] == 5
    # Left padding does not justify dropping the first active label: ordinary
    # causal shift drops tensor column zero, not the first active position.
    assert batch["row_causal_supervised_tokens"] == [1, 3]
    assert batch["causal_supervised_tokens"] == 4
    assert batch["position_reset_offsets"] == [[0, 2], [1]]
    assert batch["labels_present"] is True


def test_all_padding_single_token_rows_and_padding_labels():
    batch = _batch([[0], [1]], labels=[[7], [8]], positions=[[0], [0]], shape=(2, 1))
    assert batch["row_supervised_tokens"] == [0, 1]
    assert batch["row_causal_supervised_tokens"] == [0, 0]
    assert batch["position_reset_offsets"] == [[], [0]]
    summarize_batch_updates([_update(1, [batch])])


@pytest.mark.parametrize(
    "mask", [None, [[1, 1, 2, 2], [1, 1, 1, 0]], [[1.0, float("nan"), 0.0, 0.0]] * 2]
)
def test_missing_or_nonbinary_masks_keep_all_token_counts_unknown(mask):
    batch = _batch(mask, labels=[[1, 2, 3, 4]] * 2, positions=[[0, 1, 2, 3]] * 2)
    assert batch["attention_mask_kind"] == (
        "missing" if mask is None else "unsupported"
    )
    for field in (
        "row_nonpadding_tokens",
        "nonpadding_tokens",
        "padding_tokens",
        "row_supervised_tokens",
        "supervised_tokens",
        "row_causal_supervised_tokens",
        "causal_supervised_tokens",
        "position_reset_offsets",
    ):
        assert batch[field] is None
    assert batch["labels_kind"] == "integer_2d"
    summarize_batch_updates([_update(1, [batch])])


def test_four_dimensional_mask_is_not_misread_as_token_activity():
    inputs = {
        "input_ids": SimpleNamespace(shape=(1, 4)),
        "attention_mask": SimpleNamespace(shape=(1, 1, 4, 4)),
    }
    batch = describe_batch(inputs)
    assert batch["attention_mask_kind"] == "unsupported"
    assert batch["nonpadding_tokens"] is None


@pytest.mark.parametrize(
    "mask", [[[True, True, False, False]] * 2, [[1.0, 1.0, 0.0, 0.0]] * 2]
)
def test_boolean_and_float_binary_masks_are_supported(mask):
    assert _batch(mask)["nonpadding_tokens"] == 4


@pytest.mark.parametrize(
    "values",
    [[[1, 2]], [[True, 2, 3, 4]] * 2, [[1.0, 2.0, 3.0, 4.0]] * 2, [[1, -2, 3, 4]] * 2],
)
def test_bad_label_representations_are_explicitly_unsupported(values):
    batch = _batch([[1, 1, 1, 1]] * 2, labels=values)
    assert batch["labels_present"]
    assert batch["labels_kind"] == "unsupported"
    assert batch["supervised_tokens"] is None
    summarize_batch_updates([_update(1, [batch])])


@pytest.mark.parametrize("shape", [(2,), (1, 2, 3), (0, 4), (True, 4)])
def test_invalid_input_shape_is_rejected(shape):
    with pytest.raises(ValueError, match="shape"):
        _batch(shape=shape)


def test_profile_preserves_order_and_partial_accumulation_windows():
    batch = _known_batch()
    shorter = _batch([[1, 1]], labels=[[1, 2]], shape=(1, 2))
    updates = [
        _update(1, [batch, shorter], warmup=True, seconds=8),
        _update(2, [batch, shorter], seconds=3),
        _update(3, [shorter, batch], seconds=4),
        _update(4, [batch, shorter], seconds=5),
        _update(5, [shorter], seconds=2),
    ]
    for index, update in enumerate(updates):
        for offset, item in enumerate(update["batches"], 1):
            item.update(epoch_index=index + 1, epoch_microstep=offset)
    profile = summarize_batch_updates(updates)
    groups = profile["window_groups"]
    assert len(groups) == 3
    assert groups[0]["optimizer_update_seconds"] == [3.0, 5.0]
    assert groups[0]["seconds"] == {"min": 3, "median": 4, "max": 5}
    assert groups[0]["warmup_updates"] == 1
    assert groups[0]["measured_updates"] == 2
    assert groups[0]["sufficient"] is True
    assert groups[1]["sufficient"] is False
    assert groups[2]["batches"] == [shorter]
    assert "epoch_index" not in groups[0]["batches"][0]
    assert profile["totals"]["microsteps"] == 9
    assert profile["totals"]["samples"] == 13
    assert profile["totals"]["token_slots"] == 42
    assert profile["totals"]["nonpadding_tokens"] == 34
    assert profile["totals"]["padding_tokens"] == 8
    assert profile["shape_counts"] == [
        {"shape": [1, 2], "microsteps": 5},
        {"shape": [2, 4], "microsteps": 4},
    ]
    # A group's timing describes the whole window, never either component shape.
    assert not any("seconds" in shape for shape in profile["shape_counts"])


def test_unknown_batch_makes_total_unknown_and_reports_coverage():
    profile = summarize_batch_updates([_update(1, [_known_batch(), _batch()])])
    assert profile["totals"]["token_slots"] == 16
    for field in (
        "nonpadding_tokens",
        "padding_tokens",
        "supervised_tokens",
        "causal_supervised_tokens",
    ):
        assert profile["totals"][field] is None
    assert profile["coverage"]["nonpadding_microsteps"] == 1
    assert profile["coverage"]["supervised_microsteps"] == 1
    assert profile["coverage"]["position_reset_microsteps"] == 1


def test_warmup_only_group_keeps_timing_unavailable():
    profile = summarize_batch_updates([_update(1, [_known_batch()], warmup=True)])
    group = profile["window_groups"][0]
    assert group["seconds"] is None
    assert group["sufficient"] is False
    assert profile["totals"]["measured_optimizer_steps"] == 0


def test_profile_and_batches_are_independent_owned_values():
    updates = [_update(1, [_known_batch()])]
    profile = summarize_batch_updates(updates)
    profile["window_groups"][0]["batches"][0]["shape"][0] = 50
    assert updates[0]["batches"][0]["shape"] == [2, 4]


def test_validate_profile_accepts_complete_evidence_and_legacy_reports():
    validate_batch_profile(_observation())
    validate_batch_profile({"updates": [{"seconds": 1.0}]})


@pytest.mark.parametrize("field", ["batch_profile", "batches", "both"])
def test_evidence_cannot_be_partially_deleted_or_downgraded(field):
    observation = _observation()
    if field in ("batch_profile", "both"):
        observation.pop("batch_profile")
    if field in ("batches", "both"):
        for update in observation["updates"]:
            update.pop("batches")
    with pytest.raises(ValueError, match="both batches and batch_profile"):
        validate_batch_profile(observation)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("token_slots", 9, "token_slots"),
        ("row_nonpadding_tokens", [3], "one count per row"),
        ("row_nonpadding_tokens", [True, 3], "integer"),
        ("nonpadding_tokens", 7, "row counts"),
        ("padding_tokens", 3, "padding_tokens"),
        ("supervised_tokens", 6, "row counts"),
        ("row_supervised_tokens", [4, 1], "exceeds"),
        ("causal_supervised_tokens", 3, "row counts"),
        ("labels_present", False, "labels_present"),
        ("attention_mask_kind", "missing", "must be unknown"),
        ("labels_kind", "unsupported", "requires a binary mask"),
        ("position_ids_kind", "missing", "position resets require"),
        ("position_reset_offsets", [[2, 0], [1]], "sorted, unique"),
        ("position_reset_offsets", [[0, 4], [1]], "in bounds"),
        ("position_reset_offsets", [[False], [1]], "integer"),
    ],
)
def test_batch_arithmetic_and_metadata_are_validated(field, value, match):
    observation = _observation()
    observation["updates"][0]["batches"][0][field] = value
    with pytest.raises(ValueError, match=match):
        validate_batch_profile(observation)


def test_causal_shift_cannot_discard_a_packed_sequence_start_per_segment():
    batch = _known_batch()
    batch["row_causal_supervised_tokens"] = [0, 3]
    batch["causal_supervised_tokens"] = 3
    with pytest.raises(ValueError, match="at most one label"):
        summarize_batch_updates([_update(1, [batch])])


@pytest.mark.parametrize(
    "field,value",
    [
        ("samples", 5),
        ("microsteps", 3),
        ("warmup", 1),
        ("optimizer_step", 2),
        ("seconds", float("nan")),
    ],
)
def test_update_totals_and_types_are_validated(field, value):
    observation = _observation()
    observation["updates"][0][field] = value
    with pytest.raises(ValueError, match="native batch"):
        validate_batch_profile(observation)


@pytest.mark.parametrize(
    "wall,overhead",
    [(3.0, 1.0), (float("inf"), 1.0), (6.0, -1.0), (6.0, True), (4.0, 5.0)],
)
def test_instrumentation_cannot_corrupt_update_seconds(wall, overhead):
    observation = _observation()
    observation["updates"][0].update(
        wall_seconds=wall, batch_instrumentation_seconds=overhead
    )
    with pytest.raises(ValueError, match="native batch"):
        validate_batch_profile(observation)


def test_valid_instrumentation_cost_is_checked_but_does_not_change_grouping():
    observation = _observation()
    for index, update in enumerate(observation["updates"]):
        update.update(
            wall_seconds=update["seconds"] + index, batch_instrumentation_seconds=index
        )
    validate_batch_profile(observation)


@pytest.mark.parametrize("index,microstep", [(1, 2), (2, 1), (True, 1)])
def test_epoch_metadata_must_start_at_one_and_be_contiguous(index, microstep):
    observation = _observation()
    observation["updates"][0]["batches"][0].update(
        epoch_index=index, epoch_microstep=microstep
    )
    with pytest.raises(ValueError, match="epoch"):
        validate_batch_profile(observation)


def test_profile_recomputation_detects_aggregate_tampering():
    observation = _observation()
    observation["batch_profile"]["window_groups"][0]["seconds"]["median"] = 1
    with pytest.raises(ValueError, match="batch_profile differs"):
        validate_batch_profile(observation)


@pytest.mark.parametrize(
    "field", ["samples", "microsteps", "warmup_microsteps", "measured_optimizer_steps"]
)
def test_workload_counters_must_match_batch_ledger(field):
    observation = _observation()
    observation["workload"][field] += 1
    with pytest.raises(ValueError, match="workload." + field):
        validate_batch_profile(observation)


def test_workload_shapes_must_match_batch_ledger():
    observation = _observation()
    observation["workload"]["shape_counts"] = [{"shape": [1, 8], "microsteps": 7}]
    with pytest.raises(ValueError, match="shape_counts disagrees"):
        validate_batch_profile(observation)


@pytest.mark.parametrize("field", ["optimizer_update_seconds", "seconds_per_microstep"])
@pytest.mark.parametrize(
    "bad_value", [[2.0, 3.0], [2.0, 3.0, float("nan")], [10.0, 10.0, 10.0]]
)
def test_separate_timing_arrays_cannot_override_recorded_updates(field, bad_value):
    observation = _observation()
    observation[field] = bad_value
    with pytest.raises(ValueError, match=field):
        validate_batch_profile(observation)


def test_separate_timing_arrays_preserve_partial_accumulation_divisor():
    observation = _observation()
    observation["optimizer_update_seconds"] = [2.0, 3.0, 1.5]
    observation["seconds_per_microstep"] = [1.0, 1.5, 1.5]
    validate_batch_profile(observation)


@pytest.mark.parametrize("shape,fixed", [((2, 4), False), ((1, 8), True), ((2, 4), 1)])
def test_fixed_shape_flag_is_derived_from_observed_shapes_and_signature(shape, fixed):
    observation = _observation()
    observation["signature"].update(micro_batch_size=shape[0], sequence_length=shape[1])
    observation["workload"]["fixed_shape"] = fixed
    with pytest.raises(ValueError, match="fixed_shape"):
        validate_batch_profile(observation)


@pytest.mark.parametrize("shape,fixed", [((2, 4), True), ((1, 8), False)])
def test_fixed_shape_flag_accepts_dynamic_and_fixed_measurements(shape, fixed):
    observation = _observation()
    observation["signature"].update(micro_batch_size=shape[0], sequence_length=shape[1])
    observation["workload"]["fixed_shape"] = fixed
    validate_batch_profile(observation)
