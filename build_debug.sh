#!/bin/bash
# Build FakeGPU with logging enabled (for debugging)
if [[ "$(uname -s)" == "Darwin" ]]; then
    cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
else
    cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=ON
fi
cmake --build build
