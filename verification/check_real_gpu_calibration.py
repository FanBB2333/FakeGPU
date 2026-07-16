#!/usr/bin/env python3
"""Public CLI for validating generic real-GPU calibration reports."""

from __future__ import annotations

try:
    from .check_rtx3090ti_calibration import main
except ImportError:
    from check_rtx3090ti_calibration import main


if __name__ == "__main__":
    raise SystemExit(main())
