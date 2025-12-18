#!/usr/bin/env python3
"""Wrapper script to run nvitop with FakeGPU libraries.

This script provides GPU monitoring using nvitop with FakeGPU's simulated GPUs.
It forces exit with os._exit(0) to avoid Python cleanup segfaults.
"""

import os
import sys

# Ensure we're using FakeGPU libraries
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, 'build')

# Check if we're already preloaded
ld_preload = os.environ.get('LD_PRELOAD', '')
if 'libnvidia-ml.so' not in ld_preload or 'build/' in ld_preload:
    # Already using our libs, continue
    pass


def main():
    # Import after env setup
    from nvitop.cli import main as nvitop_main
    
    # Run nvitop with provided arguments
    try:
        exit_code = nvitop_main()
    except SystemExit as e:
        exit_code = e.code if e.code is not None else 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        exit_code = 1
    
    # Force exit to avoid Python GC cleanup crash with FakeGPU
    # This is necessary because FakeGPU's shared libraries can interfere
    # with Python's garbage collector during normal cleanup
    os._exit(exit_code if exit_code else 0)


if __name__ == '__main__':
    main()
