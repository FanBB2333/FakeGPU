"""The ``fakegpu run`` side that lives in each Python process of the run.

``fakegpu/_bootstrap/sitecustomize.py`` calls :func:`install` at interpreter
startup. FakeCUDA starts when the process first imports torch, before torch
returns to the importing code, so a process that never imports torch pays
only for this module. A process that uses a simulated GPU keeps its memory
record in ``FAKEGPU_RUN_DIR`` up to date for the summary ``fakegpu run``
prints: every second, on SIGTERM or SIGHUP, and at exit.
"""

from __future__ import annotations

import atexit
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any

RUN_DIR_ENV = "FAKEGPU_RUN_DIR"
LAYOUT_ENV = "FAKEGPU_RUN_LAYOUT"
PHYSICAL_INDICES_ENV = "FAKEGPU_PHYSICAL_INDICES"
RECORD_SCHEMA = "fakegpu.run.process.v1"
RECORD_INTERVAL_SECONDS = 1.0

_started = False
_exception: dict[str, str] | None = None
_physical: list[int] | None = None
_lock = threading.RLock()
_final = False


def install() -> None:
    run_dir = os.environ.get(RUN_DIR_ENV)
    if not run_dir:
        return
    if "torch" in sys.modules:
        _start(Path(run_dir))
        return
    sys.meta_path.insert(0, _StartWithTorch(Path(run_dir)))


class _StartWithTorch(importlib.abc.MetaPathFinder):
    """Start FakeCUDA right after ``torch`` itself finishes importing."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if fullname != "torch":
            return None
        sys.meta_path.remove(self)
        spec = None
        for finder in [importlib.machinery.PathFinder, *sys.meta_path]:
            find_spec = getattr(finder, "find_spec", None)
            spec = find_spec(fullname, path, target) if find_spec is not None else None
            if spec is not None:
                break
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        exec_module = spec.loader.exec_module
        run_dir = self.run_dir

        def exec_module_then_start(module: Any) -> None:
            exec_module(module)
            _start(run_dir)

        spec.loader.exec_module = exec_module_then_start
        return spec


def visible_gpus(layout: str | None, mask: str | None) -> tuple[list[str], list[int]] | None:
    """The run's GPUs that ``CUDA_VISIBLE_DEVICES`` leaves visible, and their indices.

    Like CUDA, the mask is read up to its first entry that is not a GPU of
    the run, so ``-1`` or an empty mask hides every GPU. ``None`` means the
    process sees the whole layout.
    """

    profiles = [item for item in (layout or "").split(",") if item]
    if mask is None or not profiles:
        return None
    indices: list[int] = []
    for entry in mask.split(","):
        entry = entry.strip()
        if not entry.isdigit() or int(entry) >= len(profiles) or int(entry) in indices:
            break
        indices.append(int(entry))
    if indices == list(range(len(profiles))):
        return None
    return [profiles[index] for index in indices], indices


def _start(run_dir: Path) -> None:
    global _started, _physical
    if _started:
        return
    _started = True
    options: dict[str, Any] = {}
    visible = visible_gpus(os.environ.get(LAYOUT_ENV), os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible is not None:
        profiles, _physical = visible
        if not profiles:
            return  # CUDA_VISIBLE_DEVICES hides every GPU, as on a real machine.
        options["devices"] = ",".join(profiles)
        # nvidia-smi and the summary show the GPUs under their run indices.
        os.environ[PHYSICAL_INDICES_ENV] = ",".join(map(str, _physical))
    try:
        import warnings

        import fakegpu

        with warnings.catch_warnings():
            # The run's summary says FakeCUDA ran; one notice per worker is noise.
            warnings.filterwarnings("ignore", message="fakegpu.torch_patch: enabled", category=UserWarning)
            fakegpu.init(runtime="fakecuda", **options)
    except Exception as exc:  # The user's command must still run.
        message = f"{type(exc).__name__}: {exc}"
        print(f"fakegpu run: FakeCUDA did not start in pid {os.getpid()}: {message}", file=sys.stderr)
        _write(run_dir, {"start_error": message})
        return
    previous_hook = sys.excepthook

    def remember_exception(kind: type[BaseException], value: BaseException, traceback: Any) -> None:
        global _exception
        _exception = {"type": kind.__name__, "message": str(value)}
        previous_hook(kind, value, traceback)

    sys.excepthook = remember_exception
    atexit.register(_write_usage, run_dir, final=True)
    _write_on_termination(run_dir)
    # A process killed with SIGKILL, or one stuck in a collective when
    # SIGTERM arrives, still leaves a record at most a second old.
    threading.Thread(target=_write_periodically, args=(run_dir,), name="fakegpu-run-record", daemon=True).start()


def _write_on_termination(run_dir: Path) -> None:
    for name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        try:
            previous = None if signum is None else signal.getsignal(signum)
            if previous is not None and previous is not signal.SIG_IGN:
                signal.signal(signum, _write_then(run_dir, previous))
        except (OSError, ValueError):
            pass  # Not the main thread: the periodic record still covers it.


def _write_then(run_dir: Path, previous: Any) -> Any:
    """A handler that writes the record, then does what ``previous`` did."""

    def handler(signum: int, frame: Any) -> None:
        name = signal.Signals(signum).name
        if callable(previous):
            # e.g. FakeCUDA's own nvidia-smi publisher, which then exits.
            _write_usage(run_dir, signal_name=name)
            previous(signum, frame)
            return
        _write_usage(run_dir, final=True, signal_name=name)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    return handler


def _write_periodically(run_dir: Path) -> None:
    while True:
        time.sleep(RECORD_INTERVAL_SECONDS)
        _write_usage(run_dir)


def _forget_in_forked_child() -> None:
    # A child forked after FakeCUDA started, e.g. a data-loader worker,
    # inherits the parent's memory figures; only the parent reports them.
    # The lock may have been held by the record thread, which does not exist
    # in the child.
    global _lock, _final
    _lock = threading.RLock()
    if _started:
        _final = True


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_in_forked_child)


def _write_usage(run_dir: Path, *, final: bool = False, signal_name: str | None = None) -> None:
    global _final
    with _lock:
        if _final:
            return
        _final = final
        from . import torch_patch

        try:
            snapshot = torch_patch.memory_snapshot()
        except Exception as exc:
            if final:
                _write(run_dir, {"start_error": f"memory snapshot failed: {exc}"})
            return
        if _physical is not None:
            for device in snapshot.get("devices", []):
                index = device.get("index")
                if isinstance(index, int) and 0 <= index < len(_physical):
                    device["index"] = _physical[index]
        used = [
            device for device in snapshot.get("devices", [])
            if device.get("allocation_count") or device.get("oom_events")
        ]
        if not used:
            return  # e.g. a launcher or a data-loader worker
        payload: dict[str, Any] = {"snapshot": snapshot, "exception": _exception}
        if signal_name:
            payload["signal"] = signal_name
        _write(run_dir, payload)


def _write(run_dir: Path, payload: dict[str, Any]) -> None:
    record = {
        "schema_version": RECORD_SCHEMA,
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        **payload,
    }
    with _lock:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            target = run_dir / f"process-{os.getpid()}.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
            os.replace(temporary, target)
        except OSError as exc:
            print(f"fakegpu run: could not write the memory record: {exc}", file=sys.stderr)
