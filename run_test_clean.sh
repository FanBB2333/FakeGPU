#!/bin/bash
# Run PyTorch test with minimal output
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libcublas.so.12 \
python3 test/test_pytorch_with_cublas.py 2>&1 | grep -v -E "(FakeCUDART|FakeCUBLAS|GlobalState|Monitor)"
