r"""Validate native training windows after periodic evaluation/checkpoint events.

The event matrix runs two independent calibrations and two longer holdouts using
the source framework configuration. This layer independently checks complete
ordered windows together with each event kind's latest boundary and distance.

Example:
    python tests/native/finetune_event_window_matrix.py \
        --framework llamafactory --config /tmp/event-windows.json \
        --python /tmp/factory/venv/bin/python --output-dir /tmp/event-windows

The fixture should contain repeated real batch descriptors, retaining epoch tail
batches. No sampling order is fabricated and no CUDA training starts on import.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

if __package__:
    from . import finetune_event_matrix as event_matrix
else:
    import finetune_event_matrix as event_matrix


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reference_windows(run):
    """Replay observed preceding IDs; preserve full same-step event boundaries."""
    root_events = {
        event["event_id"]: event
        for event in run["observation"]["phase_events"]
        if event["parent_event_id"] is None
    }
    descriptors = {}
    for key, events in event_matrix._reference_groups(run).items():
        for event in events:
            descriptors[event["event_id"]] = json.loads(key)
    latest = {"evaluation": None, "checkpoint": None}
    seen, trace = set(), []
    for update in run["observation"]["updates"]:
        boundaries = defaultdict(list)
        for event_id in update["preceding_event_ids"]:
            event = root_events[event_id]
            boundaries[event["optimizer_step"]].append(event)
        for step, events in boundaries.items():
            for kind in {event["kind"] for event in events}:
                latest[kind] = (
                    step,
                    [descriptors[event["event_id"]] for event in events],
                )
        context = {
            kind: None
            if boundary is None
            else {
                "events": boundary[1],
                "optimizer_updates_since_boundary": update["optimizer_step"]
                - boundary[0]
                - 1,
            }
            for kind, boundary in latest.items()
        }
        key = _canonical(
            {
                "batches": [
                    {
                        name: value
                        for name, value in batch.items()
                        if name not in {"epoch_index", "epoch_microstep"}
                    }
                    for batch in update["batches"]
                ],
                "context": context,
            }
        )
        first = key not in seen
        seen.add(key)
        trace.append(
            {
                "optimizer_step": update["optimizer_step"],
                "key": key,
                "eligible": not first and not update["warmup"],
                "seconds": update["seconds"],
                "immediately_after_event": any(
                    value is not None and value["optimizer_updates_since_boundary"] == 0
                    for value in context.values()
                ),
            }
        )
    return trace


def _reference_comparison(calibrations, target, projection, *, calibration_steps):
    samples = []
    for calibration in calibrations:
        groups = defaultdict(list)
        for row in _reference_windows(calibration):
            if row["eligible"]:
                groups[row["key"]].append(row["seconds"])
        samples.append(groups)
    target_trace = _reference_windows(target)
    projected_trace = {
        row["optimizer_step"]: row for row in projection["target_updates"]
    }
    if set(projected_trace) != {row["optimizer_step"] for row in target_trace}:
        raise ValueError("production event-window update set differs from reference")
    retained = []
    for row in target_trace:
        projected = projected_trace[row["optimizer_step"]]
        expected_id = hashlib.sha256(row["key"].encode()).hexdigest()
        if (
            projected["eligible"] != row["eligible"]
            or projected["window_id"] != expected_id
            or projected["immediately_after_event"] != row["immediately_after_event"]
        ):
            raise ValueError(
                "production event-window context differs from independent reference"
            )
        if not row["eligible"]:
            continue
        if any(len(groups.get(row["key"], [])) < 2 for groups in samples):
            raise ValueError(
                "event-window reference lacks repeated independent calibration"
            )
        predicted = statistics.mean(
            statistics.mean(groups[row["key"]]) for groups in samples
        )
        if not math.isclose(
            predicted,
            projected["predicted_seconds"]["expected"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "production event-window timing differs from independent reference"
            )
        if not math.isclose(
            row["seconds"], projected["observed_seconds"], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                "production event-window observation differs from target evidence"
            )
        retained.append(dict(row, predicted=predicted))

    def comparison(rows):
        if not rows:
            raise ValueError(
                "event-window reference subset contains no eligible updates"
            )
        predicted = sum(row["predicted"] for row in rows)
        actual = sum(row["seconds"] for row in rows)
        return {
            "optimizer_updates": len(rows),
            "optimizer_steps": [row["optimizer_step"] for row in rows],
            "predicted_seconds": predicted,
            "observed_seconds": actual,
            "absolute_percentage_error": abs(predicted - actual) / actual * 100,
        }

    return {
        "whole_trace": comparison(retained),
        "longer_holdout_suffix": comparison(
            [row for row in retained if row["optimizer_step"] > calibration_steps]
        ),
        "immediately_after_event": comparison(
            [row for row in retained if row["immediately_after_event"]]
        ),
        "suffix_immediately_after_event": comparison(
            [
                row
                for row in retained
                if row["immediately_after_event"]
                and row["optimizer_step"] > calibration_steps
            ]
        ),
        "eligible_window_groups": len({row["key"] for row in retained}),
        "minimum_calibration_samples_per_group_per_run": min(
            len(groups[row["key"]]) for groups in samples for row in retained
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework", required=True, choices=("llamafactory", "axolotl")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default="rtx-pro-5000-blackwell")
    parser.add_argument("--calibration-steps", type=int, default=72)
    parser.add_argument("--holdout-steps", type=int, default=101)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    command = []
    for name, value in vars(args).items():
        if name == "prepare_only":
            if value:
                command.append("--prepare-only")
        else:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    source = Path(__file__)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    result = event_matrix.main(command)
    directory = Path(args.output_dir).expanduser().resolve()
    summary_path = directory / "event-window-summary.json"
    summary = {
        "schema_version": "fakegpu.finetune_event_window_matrix.v1",
        "framework": args.framework,
        "status": "prepared" if args.prepare_only else "running",
        "sampling_plan": vars(args),
        "event_matrix_summary": str(directory / "summary.json"),
        "matrix_source_sha256": source_hash,
        "holdout_comparisons": {},
        "limitations": [
            "This compares observed update windows and their recorded event context, not future event or sampler schedules.",
            "Initial and first combined windows and global warmup are excluded independently in every process.",
            "Retaining event kinds and distance does not establish controlled hardware, allocator or filesystem state.",
            "Update and event comparisons are reported separately; no total wall-time or VRAM forecast is claimed.",
        ],
    }
    try:
        if result:
            raise ValueError(
                "native event matrix failed; its original evidence is retained"
            )
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
            raise ValueError(
                "event-window matrix source changed during native execution"
            )
        if not args.prepare_only:
            from fakegpu.finetune_window_estimator import estimate_finetuning_windows

            base = json.loads((directory / "summary.json").read_text())
            reports = {
                name: json.loads(Path(row["report"]).read_text())
                for name, row in base["runs"].items()
            }
            calibrations = [reports[f"calibration-{index}"] for index in (1, 2)]
            for name in ("holdout-1", "holdout-2"):
                target = reports[name]
                projection = estimate_finetuning_windows(
                    calibrations, target, event_context=True
                )
                summary["holdout_comparisons"][name] = {
                    "estimate": projection,
                    "independent_reference": _reference_comparison(
                        calibrations,
                        target,
                        projection,
                        calibration_steps=args.calibration_steps,
                    ),
                }
            summary["status"] = "completed"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        summary.update(status="failed", error=str(exc))
        event_matrix._write_json(summary_path, summary)
        print(
            f"Event-window matrix failed: {exc}; evidence: {summary_path}",
            file=sys.stderr,
        )
        return 1
    event_matrix._write_json(summary_path, summary)
    print(f"Native event-window evidence written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
