"""Simulate a typical PyTorch CUDA training loop using FakeGPU.

This is the kind of code a user would write on a machine with a real GPU.
With fakegpu.torch_patch, it should run unmodified on CPU-only macOS.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ---- This is the only added line a user needs ----
from fakegpu.torch_patch import patch; patch()
# ---------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. Typical device selection pattern
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# 2. Define a model (ResNet-like block)
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.block1 = ResBlock(32)
        self.block2 = ResBlock(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ============================================================
# 3. Move model to GPU
# ============================================================
model = SimpleNet(num_classes=10).to(device)
print(f"Model on device: {next(model.parameters()).device}")

# ============================================================
# 4. Typical training loop
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Fake dataset (would normally be DataLoader)
batch_size = 16
num_batches = 5

print(f"\nTraining for {num_batches} batches...")
model.train()
for batch_idx in range(num_batches):
    # Simulate loading data to GPU
    images = torch.randn(batch_size, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)

    # Forward
    logits = model(images)
    loss = criterion(logits, labels)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Typical per-batch logging
    _, predicted = logits.max(1)
    correct = predicted.eq(labels).sum().item()
    accuracy = correct / batch_size
    print(f"  Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f} | Acc: {accuracy:.2%}")

# ============================================================
# 5. Evaluation
# ============================================================
model.eval()
with torch.no_grad():
    test_images = torch.randn(32, 3, 32, 32, device=device)
    test_logits = model(test_images)
    probs = F.softmax(test_logits, dim=1)
    print(f"\nEval: output shape={test_logits.shape}, probs sum={probs.sum(dim=1).mean():.4f}")

# ============================================================
# 6. Save / load checkpoint (common pattern)
# ============================================================
ckpt_path = "/tmp/fakegpu_test_ckpt.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": 1,
}, ckpt_path)
print(f"\nCheckpoint saved to {ckpt_path}")

# Reload
ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
print(f"Checkpoint loaded, epoch={ckpt['epoch']}")

# ============================================================
# 7. Mixed precision (autocast)
# ============================================================
print("\nTesting autocast...")
with torch.amp.autocast(device_type='cuda'):
    mixed_out = model(torch.randn(4, 3, 32, 32, device=device))
    print(f"  autocast output dtype: {mixed_out.dtype}, shape: {mixed_out.shape}")

# ============================================================
# 8. DataParallel (should not crash)
# ============================================================
print("\nTesting DataParallel...")
try:
    dp_model = nn.DataParallel(model)
    dp_out = dp_model(torch.randn(4, 3, 32, 32, device=device))
    print(f"  DataParallel output: shape={dp_out.shape}")
except Exception as e:
    print(f"  DataParallel error (expected): {type(e).__name__}: {e}")

print("\n=== TRAINING SIMULATION COMPLETE ===")
