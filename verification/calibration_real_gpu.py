#!/usr/bin/env python3
"""Public CLI for generic real-GPU versus fakecuda calibration."""

from __future__ import annotations

try:
    from .calibration_rtx3090ti import main
except ImportError:
    from calibration_rtx3090ti import main


if __name__ == "__main__":
    raise SystemExit(main())
