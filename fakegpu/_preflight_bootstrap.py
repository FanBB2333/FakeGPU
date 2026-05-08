from __future__ import annotations

import json
import os
import runpy
import sys
import traceback
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        print("fakegpu preflight bootstrap: missing Python target", file=sys.stderr)
        return 2

    init_result: Any = None
    exception: BaseException | None = None
    exit_code = 0
    try:
        import fakegpu

        init_result = fakegpu.init(runtime="fakecuda")
        _run_python_target(args)
    except SystemExit as exc:
        exit_code = _system_exit_code(exc)
        if exit_code != 0:
            exception = exc
        raise
    except BaseException as exc:
        exception = exc
        exit_code = 1
        raise
    finally:
        _write_child_report(init_result=init_result, exception=exception, exit_code=exit_code)

    return exit_code


def _run_python_target(args: list[str]) -> None:
    target = args[0]
    if target == "-c":
        if len(args) < 2:
            raise SystemExit("argument expected for the -c option")
        code = args[1]
        sys.argv = ["-c", *args[2:]]
        globals_dict = {
            "__name__": "__main__",
            "__package__": None,
            "__spec__": None,
            "__file__": "<string>",
        }
        exec(compile(code, "<string>", "exec"), globals_dict)
        return

    if target == "-m":
        if len(args) < 2:
            raise SystemExit("argument expected for the -m option")
        module_name = args[1]
        sys.argv = [module_name, *args[2:]]
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
        return

    script = str(Path(target))
    sys.argv = [script, *args[1:]]
    runpy.run_path(script, run_name="__main__")


def _write_child_report(*, init_result: Any, exception: BaseException | None, exit_code: int) -> None:
    path_text = os.environ.get("FAKEGPU_PREFLIGHT_CHILD_REPORT")
    if not path_text:
        return

    payload: dict[str, Any] = {
        "runtime": "fakecuda",
        "backend": getattr(init_result, "backend", None),
        "stage": os.environ.get("FAKEGPU_PREFLIGHT_STAGE"),
        "exit_code": int(exit_code),
        "devices": _snapshot_fakecuda_devices(),
    }
    if exception is not None:
        payload["exception"] = {
            "type": type(exception).__name__,
            "message": str(exception),
            "traceback": "".join(traceback.format_exception(type(exception), exception, exception.__traceback__)),
        }

    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _snapshot_fakecuda_devices() -> list[dict[str, Any]]:
    try:
        from fakegpu import torch_patch as tp
    except Exception:
        return []

    tracker = getattr(tp, "_memory_tracker", None)
    profiles = list(getattr(tp, "_DEVICE_PROFILES", []))
    devices: list[dict[str, Any]] = []
    for index, profile in enumerate(profiles):
        total_memory = int(profile.get("total_memory", 0))
        current_memory = 0
        peak_memory = 0
        allocation_count = 0
        if tracker is not None:
            current_memory = int(tracker.memory_allocated(index))
            peak_memory = int(tracker.max_memory_allocated(index))
            alloc_calls = getattr(tracker, "_alloc_calls", [])
            if index < len(alloc_calls):
                allocation_count = int(alloc_calls[index])

        devices.append(
            {
                "index": index,
                "name": str(profile.get("name", "")),
                "profile_id": str(profile.get("profile_id", "")),
                "total_memory": total_memory,
                "current_memory": current_memory,
                "peak_memory": peak_memory,
                "allocation_count": allocation_count,
            }
        )
    return devices


def _system_exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
