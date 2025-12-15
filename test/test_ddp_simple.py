import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
import argparse


class SimpleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        return {"loss": loss, "logits": logits}


class DummyTextDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=128, vocab_size=50257):
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


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(args):
    rank, world_size, local_rank = setup_distributed()

    print(f"[Rank {rank}] 初始化完成, World Size: {world_size}, Local Rank: {local_rank}")

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    print(f"[Rank {rank}] 使用设备: {device}")
    print(f"[Rank {rank}] 设备名称: {torch.cuda.get_device_name(local_rank)}")
    print(f"[Rank {rank}] 设备内存: {torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.2f} GB")

    print(f"[Rank {rank}] 创建模型...")
    model = SimpleTransformer(
        vocab_size=50257,
        embed_dim=256,
        num_heads=4,
        num_layers=4
    )
    model = model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"[Rank {rank}] 模型已用DDP包装")

    optimizer = AdamW(model.parameters(), lr=args.lr)

    print(f"[Rank {rank}] 创建数据集...")
    dataset = DummyTextDataset(num_samples=args.num_samples, seq_length=128)

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"[Rank {rank}] 开始训练...")
    model.train()

    for epoch in range(args.epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)

        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % args.log_interval == 0:
                print(f"[Rank {rank}] Epoch {epoch+1}/{args.epochs}, Step {step}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"[Rank {rank}] Epoch {epoch+1} 完成, 平均Loss: {avg_loss:.4f}")

    print(f"[Rank {rank}] 训练完成!")

    if rank == 0:
        print("\n=== GPU内存使用情况 ===")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            print(f"GPU {i}: 已分配 {allocated:.2f} MB, 已保留 {reserved:.2f} MB")

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='测试PyTorch DDP多卡训练')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--num-samples', type=int, default=100, help='数据集样本数')
    parser.add_argument('--log-interval', type=int, default=10, help='日志打印间隔')

    args = parser.parse_args()

    print("=== FakeGPU PyTorch DDP 测试 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"GPU数量: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            except Exception as e:
                print(f"GPU {i}: 无法获取设备名称 ({e})")

    train(args)


if __name__ == "__main__":
    main()
