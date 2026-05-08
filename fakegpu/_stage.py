from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Mark a preflight stage in the current process.

    The marker is intentionally lightweight: it updates an environment variable
    that can be read at process exit, and optionally appends JSONL events when
    ``FAKEGPU_PREFLIGHT_STAGE_LOG`` is set by the preflight runner.
    """

    stage_name = str(name)
    os.environ["FAKEGPU_PREFLIGHT_STAGE"] = stage_name
    _append_stage_event(stage_name, "enter")
    try:
        yield
    except BaseException as exc:
        os.environ["FAKEGPU_PREFLIGHT_STAGE"] = stage_name
        _append_stage_event(stage_name, "error", type(exc).__name__, str(exc))
        raise
    else:
        os.environ["FAKEGPU_PREFLIGHT_STAGE"] = stage_name
        _append_stage_event(stage_name, "exit")


def _append_stage_event(
    stage_name: str,
    event: str,
    exception_type: str | None = None,
    exception_message: str | None = None,
) -> None:
    path_text = os.environ.get("FAKEGPU_PREFLIGHT_STAGE_LOG")
    if not path_text:
        return

    payload: dict[str, object] = {
        "time": time.time(),
        "stage": stage_name,
        "event": event,
    }
    if exception_type is not None:
        payload["exception_type"] = exception_type
    if exception_message is not None:
        payload["exception_message"] = exception_message

    try:
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        return
