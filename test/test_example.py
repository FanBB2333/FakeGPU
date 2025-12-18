import torch

# All PyTorch CUDA operations are intercepted by FakeGPU
device = torch.device('cuda:0')
# print the name of cuda:0
print(torch.cuda.get_device_name(device))
x = torch.randn(100, 100, device=device)
y = torch.randn(100, 100, device=device)
z = x @ y  # Matrix multiplication

# Simple neural network
model = torch.nn.Linear(100, 50).to(device)
output = model(x)
print("Pass all test")