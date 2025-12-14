import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
import argparse


class DummyTextDataset(Dataset):
    def __init__(self, num_samples=100, seq_length=128, vocab_size=50257):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        labels = input_ids.clone()
        return {
            'input_ids': input_ids,
            'labels': labels
        }


def train_single_gpu(args):
    print("=== 单GPU训练模式 ===")

    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("错误: CUDA不可用，使用CPU训练")
        device = torch.device('cpu')
    else:
        print(f"GPU数量: {torch.cuda.device_count()}")
        device = torch.device('cuda:0')

    print(f"使用设备: {device}")

    config = GPT2Config(
        vocab_size=50257,
        n_positions=128,
        n_embd=256,
        n_layer=4,
        n_head=4,
    )

    print("创建模型...")
    model = GPT2LMHeadModel(config)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    print("创建数据集...")
    dataset = DummyTextDataset(num_samples=args.num_samples, seq_length=128)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print("开始训练...")
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Step {step}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成, 平均Loss: {avg_loss:.4f}")

    print("训练完成!")

    if torch.cuda.is_available():
        print("\n=== GPU内存使用情况 ===")
        for i in range(torch.cuda.device_count()):
            try:
                allocated = torch.cuda.memory_allocated(i) / 1024**2
                reserved = torch.cuda.memory_reserved(i) / 1024**2
                print(f"GPU {i}: 已分配 {allocated:.2f} MB, 已保留 {reserved:.2f} MB")
            except Exception as e:
                print(f"GPU {i}: 无法获取内存信息 ({e})")


def main():
    parser = argparse.ArgumentParser(description='测试transformers训练')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--num-samples', type=int, default=100, help='数据集样本数')
    parser.add_argument('--log-interval', type=int, default=10, help='日志打印间隔')

    args = parser.parse_args()

    print("=== FakeGPU Transformers 测试 ===")
    train_single_gpu(args)


if __name__ == "__main__":
    main()
