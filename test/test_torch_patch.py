"""Quick test of FakeGPU torch_patch module."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from fakegpu.torch_patch import patch
patch()

import torch

print('=== Basic checks ===')
print(f'cuda available: {torch.cuda.is_available()}')
print(f'device count: {torch.cuda.device_count()}')
print(f'current device: {torch.cuda.current_device()}')
print(f'device name: {torch.cuda.get_device_name(0)}')
print(f'device capability: {torch.cuda.get_device_capability(0)}')

print('\n=== Tensor creation with device=cuda ===')
x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
print(f'tensor(device=cuda): {x}, device={x.device}')

y = torch.randn(3, 3, device='cuda')
print(f'randn(device=cuda): shape={y.shape}, device={y.device}')

z = torch.zeros(4, device='cuda:0')
print(f'zeros(device=cuda:0): {z}, device={z.device}')

w = torch.ones(2, 3, device='cuda')
print(f'ones(device=cuda): shape={w.shape}, device={w.device}')

e = torch.empty(5, device='cuda')
print(f'empty(device=cuda): shape={e.shape}, device={e.device}')

print('\n=== .cuda() and .to() ===')
a = torch.randn(3)
b = a.cuda()
print(f'.cuda(): device={b.device}')

c = a.to('cuda')
print(f".to('cuda'): device={c.device}")

d = a.to(torch.device('cuda', 0))
print(f'.to(device(cuda,0)): device={d.device}')

e2 = a.to('cuda', dtype=torch.float16)
print(f".to('cuda', dtype=fp16): device={e2.device}, dtype={e2.dtype}")

# dtype-only .to() should still work
f = a.to(torch.float64)
print(f'.to(float64): dtype={f.dtype}')

print('\n=== Module operations ===')
model = torch.nn.Linear(10, 5)
model.cuda()
print(f'model.cuda(): weight device={model.weight.device}')

model2 = torch.nn.Linear(10, 5)
model2.to('cuda')
print(f"model.to('cuda'): weight device={model2.weight.device}")

print('\n=== Forward pass ===')
inp = torch.randn(3, 10, device='cuda')
out = model(inp)
print(f'output: shape={out.shape}, device={out.device}')
print(f'output values (first row): {out[0].tolist()}')

print('\n=== Stream / Event ===')
s = torch.cuda.Stream()
ev = torch.cuda.Event(enable_timing=True)
with torch.cuda.stream(s):
    x2 = torch.randn(3)
ev.record(s)
ev.synchronize()
print('stream/event OK')

print('\n=== Misc ===')
torch.cuda.synchronize()
torch.cuda.manual_seed(42)
torch.cuda.empty_cache()
print(f'memory_allocated: {torch.cuda.memory_allocated()}')
print(f'mem_get_info: {torch.cuda.mem_get_info()}')
print(f'is_bf16_supported: {torch.cuda.is_bf16_supported()}')

print('\n=== Legacy cuda tensor types ===')
lt = torch.cuda.FloatTensor(3, 4)
print(f'cuda.FloatTensor(3,4): shape={lt.shape}, dtype={lt.dtype}, device={lt.device}')

print('\n=== CNN model ===')
cnn = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(16, 32, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool2d(1),
    torch.nn.Flatten(),
    torch.nn.Linear(32, 10),
).cuda()
img = torch.randn(2, 3, 32, 32, device='cuda')
logits = cnn(img)
print(f'CNN output: shape={logits.shape}, device={logits.device}')
print(f'CNN logits[0]: {logits[0].tolist()}')

print('\n=== ALL TESTS PASSED ===')
