#!/bin/bash

export LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH
export LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libcublas.so.12

# 只显示非调试信息的输出
python3 test/test_load_qwen2_5_fixed.py 2>&1 | grep -v "^\[Fake\|^\[Global\|^\[Monitor"
