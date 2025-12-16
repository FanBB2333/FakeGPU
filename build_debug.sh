#!/bin/bash
# Build FakeGPU with logging enabled (for debugging)
cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=ON
cmake --build build
