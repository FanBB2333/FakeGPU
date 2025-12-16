#!/bin/bash

set -e

echo "================================================================="
echo "Qwen2.5 完整对比测试：真实GPU vs FakeGPU"
echo "================================================================="
echo ""

# 创建输出目录
mkdir -p test/output

# 测试1：真实GPU（如果可用）
echo "================================================================="
echo "测试 1/2: 在真实GPU上运行"
echo "================================================================="
echo ""

if nvidia-smi &> /dev/null; then
    echo "检测到真实GPU，开始测试..."
    echo ""
    python3 test/test_load_qwen2_5.py 2>&1 | tee test/output/real_gpu_output.txt
    REAL_GPU_RESULT=$?
    echo ""
    echo "真实GPU测试输出已保存到 test/output/real_gpu_output.txt"
else
    echo "未检测到真实GPU，跳过此测试"
    REAL_GPU_RESULT=999
fi

echo ""
echo ""

# 测试2：FakeGPU
echo "================================================================="
echo "测试 2/2: 在FakeGPU上运行"
echo "================================================================="
echo ""
echo "使用fakeGPU模拟GPU操作..."
echo ""

LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libcublas.so.12 \
timeout 300 python3 test/test_load_qwen2_5.py 2>&1 | tee test/output/fakegpu_output.txt
FAKE_GPU_RESULT=$?

echo ""
echo "FakeGPU测试输出已保存到 test/output/fakegpu_output.txt"
echo ""
echo ""

# 结果对比
echo "================================================================="
echo "测试结果对比"
echo "================================================================="
echo ""

if [ $REAL_GPU_RESULT -eq 999 ]; then
    echo "真实GPU: 未测试（无GPU可用）"
elif [ $REAL_GPU_RESULT -eq 0 ] && grep -q "TEST PASSED" test/output/real_gpu_output.txt; then
    echo "✓ 真实GPU: 测试通过"
    echo "  生成的token: $(grep "Generated token:" test/output/real_gpu_output.txt | cut -d':' -f2)"
else
    echo "✗ 真实GPU: 测试失败"
    echo "  最后20行输出:"
    tail -n 20 test/output/real_gpu_output.txt | sed 's/^/    /'
fi

echo ""

if [ $FAKE_GPU_RESULT -eq 0 ] && grep -q "TEST PASSED" test/output/fakegpu_output.txt; then
    echo "✓ FakeGPU: 测试通过"
    echo "  生成的token: $(grep "Generated token:" test/output/fakegpu_output.txt | cut -d':' -f2)"
else
    echo "✗ FakeGPU: 测试失败"
    echo "  最后20行输出:"
    tail -n 20 test/output/fakegpu_output.txt | sed 's/^/    /'
fi

echo ""
echo "================================================================="
echo "内存使用报告"
echo "================================================================="
if [ -f fake_gpu_report.json ]; then
    echo ""
    echo "FakeGPU内存使用情况 (来自 fake_gpu_report.json):"
    python3 -c "
import json
with open('fake_gpu_report.json', 'r') as f:
    report = json.load(f)
    for i, dev in enumerate(report['devices']):
        if dev['used_memory_peak'] > 0 or dev['used_memory_current'] > 0:
            print(f\"  GPU {i}: {dev['name']}\")
            print(f\"    总内存: {dev['total_memory'] / 1024**3:.2f} GB\")
            print(f\"    峰值使用: {dev['used_memory_peak'] / 1024**3:.2f} GB\")
            print(f\"    当前使用: {dev['used_memory_current'] / 1024**3:.2f} GB\")
            print()
    "
else
    echo "未找到 fake_gpu_report.json"
fi

echo ""
echo "================================================================="
echo "结论"
echo "================================================================="
echo ""
if [ $FAKE_GPU_RESULT -eq 0 ] && grep -q "TEST PASSED" test/output/fakegpu_output.txt; then
    echo "✓ FakeGPU库成功运行Qwen2.5模型推理！"
    echo ""
    echo "说明："
    echo "  - 模型成功加载到模拟的GPU设备上"
    echo "  - 成功执行了forward pass"
    echo "  - 成功生成了一个token"
    echo "  - 虽然生成的token可能不正确（因为没有真实计算），"
    echo "    但这证明了FakeGPU库能够让GPU依赖的代码运行起来"
else
    echo "✗ FakeGPU测试失败，需要进一步调试"
fi
echo ""
echo "完整输出文件位置:"
if [ $REAL_GPU_RESULT -ne 999 ]; then
    echo "  - test/output/real_gpu_output.txt"
fi
echo "  - test/output/fakegpu_output.txt"
echo "  - fake_gpu_report.json"
echo ""
echo "================================================================="
