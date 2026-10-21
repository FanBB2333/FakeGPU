"""Observe native evaluation/save call trees without inferring event schedules."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
from pathlib import Path
import time

from ._finetune_batches import describe_batch
from .structured_io import canonical_json as _canonical


def _dataset_identity(dataset):
    """Fingerprint actual finite map-style rows, retaining no token contents."""
    if (
        dataset is None
        or isinstance(dataset, Mapping)
        or not hasattr(dataset, "__len__")
        or not hasattr(dataset, "__getitem__")
    ):
        return {"status": "unknown", "reason": "not_one_map_style_dataset"}
    # Arbitrary __getitem__ implementations can advance random transforms or
    # consume external data. Only scan already materialized rows or an unformatted
    # Arrow Dataset; actual prediction inputs are independently fingerprinted.
    arrow_dataset = (
        type(dataset).__module__ == "datasets.arrow_dataset"
        and type(dataset).__name__ == "Dataset"
        and getattr(dataset, "format", {}).get("type") is None
    )
    if type(dataset) not in (list, tuple) and not arrow_dataset:
        return {"status": "unknown", "reason": "unverified_map_style_access"}
    digest = hashlib.sha256()
    size = len(dataset)
    for index in range(size):
        row = dataset[index]
        if not isinstance(row, Mapping) or "input_ids" not in row:
            return {"status": "unknown", "reason": "not_processed_text_rows"}
        try:
            encoded = _canonical(dict(row)).encode()
        except (TypeError, ValueError):
            return {"status": "unknown", "reason": "rows_not_canonical_json"}
        digest.update(encoded)
        digest.update(b"\n")
    return {
        "status": "known",
        "content_sha256": digest.hexdigest(),
        "num_samples": size,
    }


def _prediction_content(inputs):
    if not isinstance(inputs, Mapping) or "input_ids" not in inputs:
        return None
    content = {}
    for key in ("input_ids", "attention_mask", "labels", "position_ids"):
        value = inputs.get(key)
        if value is not None:
            if not callable(getattr(value, "detach", None)):
                return None
            content[key] = value.detach().cpu().tolist()
    try:
        return (_canonical(content) + "\n").encode()
    except (TypeError, ValueError):
        return None


def _argument(args, kwargs, name, position, default=None):
    return kwargs.get(name, args[position] if len(args) > position else default)


class NativeEventObserver:
    def __init__(self, observer, run_started):
        self.observer = observer
        self.run_started = run_started
        self.ledger = []
        self.stack = []
        self.pending = []
        self.instrumentation_seconds = 0.0
        self._save_directories = {}
        self._consumed_digests = {}

    @contextmanager
    def _metadata(self):
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.instrumentation_seconds += elapsed
            self.observer.instrumentation_seconds += elapsed
            for event in self.stack:
                event["instrumentation_seconds"] += elapsed

    def take_preceding_events(self):
        events, self.pending = self.pending, []
        return events

    def attach(self, trainer):
        for name, kind in (
            ("evaluate", "evaluation"),
            ("predict", "evaluation"),
            ("_save_checkpoint", "checkpoint"),
            ("save_model", "checkpoint"),
        ):
            self._wrap(trainer, name, kind)
        self._wrap(trainer.model, "save_pretrained", "checkpoint")
        self._wrap_prediction(trainer)

    def _evaluation_workload(self, operation, args, kwargs):
        trainer = self.observer.trainer
        name = "eval_dataset" if operation == "evaluate" else "test_dataset"
        dataset = _argument(args, kwargs, name, 0)
        if operation == "evaluate" and dataset is None:
            dataset = getattr(trainer, "eval_dataset", None)
        trainer_args = getattr(trainer, "args", None)
        return {
            "dataset": _dataset_identity(dataset),
            "batches": [],
            "unobserved_batches": 0,
            "consumed_content_status": "known",
            "consumed_content_sha256": None,
            "prediction_loss_only": [],
            "predict_with_generate": getattr(
                trainer_args, "predict_with_generate", False
            ),
            "compute_metrics": getattr(trainer, "compute_metrics", None) is not None,
            "eval_batch_size": getattr(
                trainer_args, "per_device_eval_batch_size", None
            ),
            "dataloader_drop_last": getattr(trainer_args, "dataloader_drop_last", None),
        }

    def _checkpoint_workload(self):
        return {
            "save_only_model": getattr(
                getattr(self.observer.trainer, "args", None), "save_only_model", None
            ),
            "artifact_kind": "unknown",
            "files_status": "unknown",
            "files": [],
            "total_file_bytes": None,
            "output_directories": [],
        }

    def _save_destination(self, operation, args, kwargs):
        trainer = self.observer.trainer
        trainer_args = getattr(trainer, "args", None)
        if operation == "save_pretrained":
            directory = _argument(args, kwargs, "save_directory", 0)
        elif operation == "save_model":
            directory = _argument(
                args, kwargs, "output_dir", 0, getattr(trainer_args, "output_dir", None)
            )
        else:
            # Nested save_model/save_pretrained supplies the actual output path.
            directory = None
        if directory is None:
            return
        directory = Path(directory).expanduser().absolute()
        for event in self.stack:
            if event["kind"] == "checkpoint":
                self._save_directories[event["event_id"]].add(directory)

    def _inventory(self, event):
        directories = sorted(self._save_directories.pop(event["event_id"]))
        if not directories and event["operation"] == "_save_checkpoint":
            output = getattr(
                getattr(self.observer.trainer, "args", None), "output_dir", None
            )
            if output is not None:
                directories = [
                    Path(output).expanduser().absolute()
                    / f"checkpoint-{event['optimizer_step']}"
                ]
        # Inspect only top-level files in paths actually passed to save methods.
        # A final model directory may also contain old checkpoint subdirectories.
        workload = event["workload"]
        workload["output_directories"] = [str(path) for path in directories]
        if not directories or any(not path.is_dir() for path in directories):
            return
        files = []
        try:
            for index, root in enumerate(directories):
                for path in sorted(root.iterdir()):
                    if path.is_file():
                        relative = path.relative_to(root).as_posix()
                        if len(directories) > 1:
                            relative = f"output_{index + 1}/{relative}"
                        files.append({"path": relative, "bytes": path.stat().st_size})
        except OSError:
            workload["files_status"] = "unavailable"
            return
        names = {Path(item["path"]).name for item in files}
        adapter = any(name.startswith("adapter_model.") for name in names)
        model = any(
            name.startswith(("model", "pytorch_model"))
            and name.endswith((".safetensors", ".bin"))
            for name in names
        )
        workload.update(
            files_status="known",
            files=files,
            total_file_bytes=sum(item["bytes"] for item in files),
            artifact_kind="mixed"
            if adapter and model
            else "adapter"
            if adapter
            else "model"
            if model
            else "unknown",
        )

    def _wrap(self, obj, operation, kind):
        original = getattr(obj, operation, None)
        if original is None:
            return

        def measured(*args, **kwargs):
            owner = self.observer
            owner.torch.cuda.synchronize()
            owner.record_peaks()
            state = getattr(owner.trainer, "state", None)
            started = time.perf_counter()
            event = {
                "event_id": len(self.ledger) + 1,
                "parent_event_id": self.stack[-1]["event_id"] if self.stack else None,
                "kind": kind,
                "operation": operation,
                "inside_train": owner._inside_train,
                "optimizer_step": getattr(state, "global_step", 0),
                "epoch": getattr(state, "epoch", None),
                "start_seconds": started - self.run_started,
                "instrumentation_seconds": 0.0,
                "status": "failed",
                "workload": {},
            }
            self.ledger.append(event)
            self.stack.append(event)
            try:
                with self._metadata():
                    if kind == "evaluation":
                        event["workload"] = self._evaluation_workload(
                            operation, args, kwargs
                        )
                        self._consumed_digests[event["event_id"]] = hashlib.sha256()
                    else:
                        event["workload"] = self._checkpoint_workload()
                        self._save_directories[event["event_id"]] = set()
                        self._save_destination(operation, args, kwargs)
                result = original(*args, **kwargs)
                event["status"] = "succeeded"
                return result
            finally:
                owner.torch.cuda.synchronize()
                owner.record_peaks()
                if kind == "checkpoint":
                    with self._metadata():
                        self._inventory(event)
                if kind == "evaluation":
                    digest = self._consumed_digests.pop(event["event_id"], None)
                    if (
                        digest is not None
                        and event["workload"]["consumed_content_status"] == "known"
                    ):
                        event["workload"]["consumed_content_sha256"] = (
                            digest.hexdigest()
                        )
                ended = time.perf_counter()
                event["end_seconds"] = ended - self.run_started
                event["wall_seconds"] = ended - started
                event["seconds"] = (
                    event["wall_seconds"] - event["instrumentation_seconds"]
                )
                self.stack.pop()
                if event["parent_event_id"] is None:
                    owner.phases[kind] += event["wall_seconds"]
                    owner.events[kind] += 1
                    if event["inside_train"]:
                        owner.during_train_phases[kind] += event["wall_seconds"]
                        owner.during_train_events[kind] += 1
                        self.pending.append(event["event_id"])

        self.observer._replace(obj, operation, measured)

    def _wrap_prediction(self, trainer):
        original = getattr(trainer, "prediction_step", None)
        if original is None:
            return

        def measured(*args, **kwargs):
            events = [event for event in self.stack if event["kind"] == "evaluation"]
            if not events:
                return original(*args, **kwargs)
            inputs = _argument(args, kwargs, "inputs", 1)
            self.observer.torch.cuda.synchronize()
            with self._metadata():
                try:
                    batch = describe_batch(inputs)
                except ValueError:
                    batch = None
                encoded = _prediction_content(inputs)
            result = original(*args, **kwargs)
            self.observer.torch.cuda.synchronize()
            with self._metadata():
                loss_only = _argument(args, kwargs, "prediction_loss_only", 2)
                for event in events:
                    workload = event["workload"]
                    workload["prediction_loss_only"].append(loss_only)
                    if encoded is None:
                        workload["consumed_content_status"] = "unknown"
                    else:
                        self._consumed_digests[event["event_id"]].update(encoded)
                    if batch is None:
                        workload["unobserved_batches"] += 1
                    else:
                        workload["batches"].append(batch)
            return result

        self.observer._replace(trainer, "prediction_step", measured)
