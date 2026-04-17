"""
MoE-GPT homogeneous config: 4x A100
FAKEGPU_PROFILES=a100:4
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
dtype = "bfloat16"
