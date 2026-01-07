#!/bin/bash
# Build FakeGPU without logging (for production use)
if [[ "$(uname -s)" == "Darwin" ]]; then
    cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
else
    cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=OFF
fi
cmake --build build
