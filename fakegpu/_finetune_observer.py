"""Temporary instrumentation of the Trainer instance created by a framework."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import math
import time

from ._finetune_batches import describe_batch


class NativeTrainerObserver:
    def __init__(self, torch, *, warmup_steps, on_trainer, on_train_begin):
        self.torch = torch
        self.warmup_steps = warmup_steps
        self.on_trainer = on_trainer
        self.on_train_begin = on_train_begin
        self.trainer = None
        self.outcome = None
        self.updates = []
        self.shapes = Counter()
        self.microsteps = self.samples = 0
        self.instrumentation_seconds = 0.0
        self.batch_instrumentation_seconds = 0.0
        self.trainer_started = self.trainer_finished = None
        self.phases = Counter()
        self.events = Counter()
        self.during_train_phases = Counter()
        self.during_train_events = Counter()
        self._restores = []
        self._phase_depth = 0
        self._inside_train = False
        self._update_started = None
        self._update_microsteps = self._update_samples = 0
        self._update_batches = []
        self._update_batch_seconds = 0.0
        self._epoch_index = 0
        self._epoch_microstep = 0
        self._losses = []
        self._before_weights = {}
        self.parameter_changes = None
        self.peak_allocated = self.peak_reserved = 0

    def record_peaks(self):
        # Native memory trackers may reset allocator peaks between phases.
        self.peak_allocated = max(
            self.peak_allocated, self.torch.cuda.max_memory_allocated()
        )
        self.peak_reserved = max(
            self.peak_reserved, self.torch.cuda.max_memory_reserved()
        )

    def _replace(self, obj, name, replacement):
        present = name in obj.__dict__
        old = obj.__dict__.get(name)
        setattr(obj, name, replacement)
        self._restores.append((obj, name, present, old))

    def restore(self):
        for obj, name, present, old in reversed(self._restores):
            if present:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        self._restores.clear()

    def _phase_wrapper(self, obj, name, phase):
        original = getattr(obj, name, None)
        if original is None:
            return

        def measured(*args, **kwargs):
            outer = self._phase_depth == 0
            if outer:
                self.torch.cuda.synchronize()
                self.record_peaks()
                started = time.perf_counter()
            self._phase_depth += 1
            try:
                return original(*args, **kwargs)
            finally:
                self._phase_depth -= 1
                if outer:
                    self.torch.cuda.synchronize()
                    self.record_peaks()
                    elapsed = time.perf_counter() - started
                    self.phases[phase] += elapsed
                    self.events[phase] += 1
                    if self._inside_train:
                        self.during_train_phases[phase] += elapsed
                        self.during_train_events[phase] += 1

        self._replace(obj, name, measured)

    def attach(self, trainer, callback_base):
        if self.trainer is not None:
            raise RuntimeError(
                "native runner supports exactly one Trainer.train invocation"
            )
        self.trainer = trainer
        self.record_peaks()
        if not hasattr(trainer, "get_batch_samples"):
            raise RuntimeError(
                "this Trainer does not expose accumulation-window prefetch"
            )
        started = time.perf_counter()
        self.on_trainer(trainer)
        self.instrumentation_seconds += time.perf_counter() - started
        batch_samples = trainer.get_batch_samples
        training_step = trainer.training_step

        def measured_batches(*args, **kwargs):
            if self._update_started is not None:
                raise RuntimeError(
                    "previous optimizer update has no matching step-end event"
                )
            self.torch.cuda.synchronize()
            self._update_started = time.perf_counter()
            return batch_samples(*args, **kwargs)

        def measured_step(*args, **kwargs):
            inputs = kwargs.get("inputs", args[1] if len(args) > 1 else None)
            if not isinstance(inputs, Mapping) or "input_ids" not in inputs:
                raise RuntimeError(
                    "native runner requires text input_ids in every microbatch"
                )
            shape = tuple(int(size) for size in inputs["input_ids"].shape)
            if len(shape) != 2 or min(shape) < 1:
                raise RuntimeError(
                    "native runner requires nonempty two-dimensional text batches"
                )
            # Wait for preceding training work before starting the metadata timer:
            # a host copy must not subtract unfinished GPU compute from the sample.
            self.torch.cuda.synchronize()
            metadata_started = time.perf_counter()
            batch = describe_batch(inputs)
            self.torch.cuda.synchronize()
            metadata_seconds = time.perf_counter() - metadata_started
            self.instrumentation_seconds += metadata_seconds
            self.batch_instrumentation_seconds += metadata_seconds
            self._update_batch_seconds += metadata_seconds
            result = training_step(*args, **kwargs)
            # Only training_step calls that returned successfully count as consumed.
            self._epoch_microstep += 1
            batch["epoch_index"] = self._epoch_index or 1
            batch["epoch_microstep"] = self._epoch_microstep
            self._update_batches.append(batch)
            self.shapes[shape] += 1
            self._losses.append(result.detach())
            self.microsteps += 1
            self.samples += shape[0]
            self._update_microsteps += 1
            self._update_samples += shape[0]
            return result

        self._replace(trainer, "get_batch_samples", measured_batches)
        self._replace(trainer, "training_step", measured_step)
        for name, phase in (
            ("evaluate", "evaluation"),
            ("predict", "evaluation"),
            ("save_model", "checkpoint"),
            ("_save_checkpoint", "checkpoint"),
        ):
            self._phase_wrapper(trainer, name, phase)
        # Axolotl's final save calls the model directly, outside Trainer.train.
        self._phase_wrapper(trainer.model, "save_pretrained", "checkpoint")
        observer = self

        class TimingCallback(callback_base):
            def on_epoch_begin(self, args, state, control, **kwargs):
                observer._epoch_index += 1
                observer._epoch_microstep = 0

            def on_train_begin(self, args, state, control, **kwargs):
                if state.global_step:
                    raise ValueError(
                        "resumed training is unsupported by the native runner"
                    )
                started = time.perf_counter()
                observer.on_train_begin(trainer)
                observer._before_weights = observer._weight_samples()
                observer.instrumentation_seconds += time.perf_counter() - started

            def on_step_begin(self, args, state, control, **kwargs):
                if observer._update_started is None:
                    raise RuntimeError(
                        "optimizer update bypassed observed batch prefetch"
                    )

            def on_step_end(self, args, state, control, **kwargs):
                observer._finish_update(state.global_step)

            def on_train_end(self, args, state, control, **kwargs):
                started = time.perf_counter()
                after = observer._weight_samples()
                changed = sum(
                    value != after.get(name)
                    for name, value in observer._before_weights.items()
                )
                observer.parameter_changes = {
                    "method": "sha256_of_up_to_32_evenly_spaced_values_per_trainable_tensor",
                    "sampled_tensors": len(after),
                    "changed_sampled_tensors": changed,
                    "has_changed_trainable_parameters": changed > 0,
                }
                observer.instrumentation_seconds += time.perf_counter() - started

        self.callback = TimingCallback()
        trainer.add_callback(self.callback)

    def _weight_samples(self):
        fingerprints = {}
        for name, parameter in self.trainer.model.named_parameters():
            if parameter.requires_grad and parameter.numel():
                flat = parameter.detach().reshape(-1)
                stride = max(1, flat.numel() // 32)
                sample = flat[::stride][:32].contiguous().cpu()
                fingerprints[name] = hashlib.sha256(
                    sample.view(self.torch.uint8).numpy().tobytes()
                ).hexdigest()
        return fingerprints

    def _finish_update(self, step):
        self.torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - self._update_started
        elapsed = wall_seconds - self._update_batch_seconds
        self.record_peaks()
        if self._update_microsteps < 1 or step != len(self.updates) + 1:
            raise RuntimeError(
                "native optimizer-update and microstep accounting diverged"
            )
        if getattr(self.trainer.accelerator, "optimizer_step_was_skipped", False):
            raise RuntimeError(
                "optimizer skipped an update; no valid timing evidence produced"
            )
        started = time.perf_counter()
        losses = [float(loss.item()) for loss in self._losses]
        if not all(math.isfinite(loss) for loss in losses):
            raise RuntimeError("native training produced a nonfinite microstep loss")
        self.instrumentation_seconds += time.perf_counter() - started
        self.updates.append(
            {
                "optimizer_step": step,
                "seconds": elapsed,
                "wall_seconds": wall_seconds,
                "batch_instrumentation_seconds": self._update_batch_seconds,
                "microsteps": self._update_microsteps,
                "samples": self._update_samples,
                "batches": self._update_batches,
                "losses": losses,
                "warmup": step <= self.warmup_steps,
            }
        )
        self._update_started = None
        self._update_microsteps = self._update_samples = 0
        self._update_batches = []
        self._update_batch_seconds = 0.0
        self._losses = []


@contextmanager
def capture_native_trainer(trainer_base, callback_base, observer):
    """Restore the shared entrypoint and all instance methods even on failure."""
    original = trainer_base.train
    original_reset = observer.torch.cuda.reset_peak_memory_stats

    def measured_reset(*args, **kwargs):
        # TrainerMemoryTracker can reset during Trainer.__init__, before we
        # ever see the instance. Preserve the preceding model-loading peak.
        observer.record_peaks()
        return original_reset(*args, **kwargs)

    def measured_train(trainer, *args, **kwargs):
        observer.attach(trainer, callback_base)
        observer.torch.cuda.synchronize()
        observer.trainer_started = time.perf_counter()
        observer._inside_train = True
        try:
            observer.outcome = original(trainer, *args, **kwargs)
            return observer.outcome
        finally:
            observer.torch.cuda.synchronize()
            observer.record_peaks()
            observer.trainer_finished = time.perf_counter()
            observer._inside_train = False

    trainer_base.train = measured_train
    observer.torch.cuda.reset_peak_memory_stats = measured_reset
    try:
        yield observer
    finally:
        trainer_base.train = original
        observer.torch.cuda.reset_peak_memory_stats = original_reset
        if observer.trainer is not None and hasattr(observer, "callback"):
            observer.trainer.remove_callback(observer.callback)
        observer.restore()
