from copy import deepcopy
from datetime import datetime, timezone

import pytest

from fakegpu.finetune_window_estimator import estimate_finetuning_windows
from native import finetune_window_matrix as matrix
from test_finetune_run_estimator import native_run as native_run
from test_finetune_window_estimator import _run


def test_window_suffix_independently_matches_api(native_run):
    calibrations = [_run(native_run, 1), _run(native_run, 2, seconds=3.0)]
    target = _run(native_run, 3, seconds=2.2, epochs=5)
    projection = estimate_finetuning_windows(calibrations, target)
    suffix = matrix._reference_comparison(
        calibrations, target, after_epoch=3, projection=projection
    )
    assert suffix["epoch_start"] == 4
    assert suffix["epoch_end"] == 5
    assert suffix["optimizer_steps"] == 4
    assert suffix["microsteps"] == 6
    assert suffix["training_update_seconds"]["predicted"] == 6.0
    assert suffix["training_update_seconds"]["actual"] == pytest.approx(6.6)
    # This small unit fixture repeats the same descriptor order every epoch;
    # the CUDA matrix separately refuses to count that as an unseen-order case.
    assert not suffix["descriptor_stream_differs_from_calibration_prefix"]


def test_window_suffix_rejects_insufficient_independent_coverage(native_run):
    calibrations = [_run(native_run, 1, epochs=2), _run(native_run, 2)]
    with pytest.raises(ValueError, match="lacks repeated independent calibration"):
        matrix._reference_comparison(
            calibrations,
            _run(native_run, 3, epochs=5),
            after_epoch=3,
            projection={},
        )


@pytest.mark.parametrize("field", ["predicted_seconds", "observed_seconds", "eligible"])
def test_window_suffix_rejects_projection_drift(native_run, field):
    calibrations = [_run(native_run, 1), _run(native_run, 2, seconds=1.1)]
    target = _run(native_run, 3, epochs=5)
    projection = estimate_finetuning_windows(calibrations, target)
    update = next(
        row for row in projection["target_updates"] if row["epoch_index"] == 4
    )
    if field == "predicted_seconds":
        update[field]["expected"] *= 2
    elif field == "eligible":
        update[field] = False
    else:
        update[field] *= 2
    with pytest.raises(ValueError, match="production"):
        matrix._reference_comparison(
            calibrations, target, after_epoch=3, projection=projection
        )


def test_child_identity_matches_process_and_detects_reuse():
    start = datetime(2026, 10, 13, tzinfo=timezone.utc).timestamp()
    report = {
        "run_identity": {
            "uuid": "unique-run",
            "pid": 123,
            "started_at_utc": "2026-10-13T00:00:01+00:00",
        }
    }
    seen = set()
    assert (
        matrix._check_process_identity(
            report, {"pid": 123}, started_at=start, finished_at=start + 2, seen=seen
        )
        == report["run_identity"]
    )
    with pytest.raises(ValueError, match="reused"):
        matrix._check_process_identity(
            report, {"pid": 123}, started_at=start, finished_at=start + 2, seen=seen
        )
    for key, value, match in (
        ("pid", 456, "actual child"),
        ("started_at_utc", "2026-10-14T00:00:01+00:00", "outside"),
        ("started_at_utc", "2026-10-13T00:00:01+01:00", "UTC"),
    ):
        changed = deepcopy(report)
        changed["run_identity"][key] = value
        with pytest.raises(ValueError, match=match):
            matrix._check_process_identity(
                changed,
                {"pid": 123},
                started_at=start,
                finished_at=start + 2,
                seen=set(),
            )
