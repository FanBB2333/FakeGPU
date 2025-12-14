#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== FakeGPU Transformers DDP 测试脚本 ==="
echo "项目根目录: $PROJECT_ROOT"

if [ ! -f "$PROJECT_ROOT/build/libfake_gpu.so" ]; then
    echo "错误: 未找到 libfake_gpu.so，请先构建项目"
    echo "运行: cmake -S . -B build && cmake --build build"
    exit 1
fi

export LD_LIBRARY_PATH="$PROJECT_ROOT/build:$LD_LIBRARY_PATH"
export LD_PRELOAD="$PROJECT_ROOT/build/libfake_gpu.so"

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "LD_PRELOAD: $LD_PRELOAD"
echo ""

if [ "$1" == "--ddp" ]; then
    echo "=== 运行DDP多卡训练模式 ==="
    NUM_GPUS=${2:-2}
    echo "使用GPU数量: $NUM_GPUS"

    torchrun --nproc_per_node=$NUM_GPUS \
        "$SCRIPT_DIR/test_transformers.py" \
        --epochs 2 \
        --batch-size 4 \
        --num-samples 100 \
        --log-interval 5
else
    echo "=== 运行单卡训练模式 ==="
    python "$SCRIPT_DIR/test_transformers.py" \
        --epochs 2 \
        --batch-size 4 \
        --num-samples 100 \
        --log-interval 5
fi

echo ""
echo "=== 检查生成的报告 ==="
if [ -f "$PROJECT_ROOT/fake_gpu_report.json" ]; then
    echo "报告文件已生成: $PROJECT_ROOT/fake_gpu_report.json"
    cat "$PROJECT_ROOT/fake_gpu_report.json"
else
    echo "警告: 未找到报告文件"
fi
