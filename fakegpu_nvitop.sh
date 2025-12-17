#!/bin/bash
# FakeGPU nvitop 包装脚本
# 使用Python API避免TUI模式的段错误

LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libcublas.so.12 \
python3 test_nvitop_wrapper.py "$@"
