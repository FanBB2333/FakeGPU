#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== FakeGPU Transformers DDP 测试脚本 ==="
echo "项目根目录: $PROJECT_ROOT"

REQUIRED_LIBS=(
    "libcublas.so.12"
    "libcudart.so.12"
    "libcuda.so.1"
    "libnvidia-ml.so.1"
)

for lib in "${REQUIRED_LIBS[@]}"; do
    if [ ! -f "$PROJECT_ROOT/build/$lib" ]; then
        echo "错误: 未找到 build/$lib，请先构建项目"
        echo "运行: cmake -S . -B build && cmake --build build"
        exit 1
    fi
done

if [ ! -x "$PROJECT_ROOT/fgpu" ]; then
    echo "错误: 未找到可执行的 $PROJECT_ROOT/fgpu"
    echo "运行: cmake -S . -B build && cmake --build build"
    exit 1
fi

echo "使用统一入口: $PROJECT_ROOT/fgpu"
echo ""

if [ "$1" == "--ddp" ]; then
    echo "=== 运行DDP多卡训练模式 ==="
    NUM_GPUS=${2:-2}
    echo "使用GPU数量: $NUM_GPUS"

    "$PROJECT_ROOT/fgpu" torchrun --nproc_per_node=$NUM_GPUS \
        "$SCRIPT_DIR/test_transformers.py" \
        --epochs 2 \
        --batch-size 4 \
        --num-samples 100 \
        --log-interval 5
else
    echo "=== 运行单卡训练模式 ==="
    "$PROJECT_ROOT/fgpu" python "$SCRIPT_DIR/test_transformers.py" \
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
