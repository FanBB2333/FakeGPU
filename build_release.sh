#!/bin/bash
# Build FakeGPU without logging (for production use)
cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=OFF
cmake --build build
