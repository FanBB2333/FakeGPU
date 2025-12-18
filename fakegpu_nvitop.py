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
    # Use TUI's Device class which has fewer SNAPSHOT_KEYS and works better with FakeGPU
    from nvitop.tui.library.device import Device
    from nvitop.tui import TUI
    
    # Get devices using TUI's Device class
    devices = Device.all()
    
    # Check for --once flag
    once_mode = '--once' in sys.argv or '-1' in sys.argv
    
    # Create and run TUI
    tui = TUI(devices, [], no_unicode=False)
    
    if once_mode:
        tui.print()
    else:
        print("Interactive mode not supported with FakeGPU. Use --once flag.")
        print("Example: ./fakegpu_nvitop.sh --once")
    
    # Force exit to avoid Python GC cleanup crash with FakeGPU
    os._exit(0)


if __name__ == '__main__':
    main()
