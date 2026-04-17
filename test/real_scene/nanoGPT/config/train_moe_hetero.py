"""
MoE-GPT heterogeneous config: 2x A100 + 2x V100
FAKEGPU_PROFILES=a100:2,v100:2

V100 does not support BF16 natively, so use float16 as the safe dtype.
"""

n_layer = 4
n_head = 4
n_embd = 128
block_size = 128

num_experts = 4
num_experts_per_tok = 2
expert_parallel = True

max_iters = 50
batch_size = 4
learning_rate = 1e-3
log_interval = 10
dtype = "float16"
