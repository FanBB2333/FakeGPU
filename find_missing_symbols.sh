#!/bin/bash
# Find all missing CUDA symbols from PyTorch libraries

echo "=== Missing CUDA symbols in PyTorch libraries ==="
for lib in /home/l1ght/anaconda3/envs/fakegpu/lib/python3.12/site-packages/torch/lib/*.so; do
    echo "Checking $lib..."
    nm -D "$lib" 2>/dev/null | grep "U cuda" | grep "@libcudart" | awk '{print $2}' | sed 's/@libcudart.so.12//' | sort -u
done | sort -u
