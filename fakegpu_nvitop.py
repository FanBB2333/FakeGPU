#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# IMPORTANT: Disable GC as early as possible, before any other imports
import gc
gc.disable()

"""Wrapper script to run nvitop with FakeGPU libraries.

This script provides GPU monitoring using nvitop with FakeGPU's simulated GPUs.
It uses gc.disable() and os._exit(0) to avoid Python cleanup crashes.

Usage:
    LD_PRELOAD=./build/libnvidia-ml.so.1:... python3 fakegpu_nvitop.py --once
    # Or use the shell wrapper:
    ./fakegpu_nvitop.sh --once
"""

import os
import sys


def main():
    """Run nvitop with FakeGPU, handling exit properly."""
    from nvitop.cli import main as nvitop_main
    
    try:
        exit_code = nvitop_main()
    except SystemExit as e:
        exit_code = e.code if e.code is not None else 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        exit_code = 1
    
    # Force exit to avoid Python GC cleanup crash with FakeGPU
    os._exit(exit_code if isinstance(exit_code, int) else 0)


if __name__ == '__main__':
    main()
