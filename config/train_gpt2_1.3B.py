# config for training GPT-2 (1.3B)

# these make the total batch size be ~1M
# 8 batch size * 64 block size * 2048 grad_accum_steps = 1,048,576
# Using strong scaling, so make grad_acc_steps multiple of maximum gpu count
batch_size = 8
block_size = 64
gradient_accumulation_steps = 1 * 2048

# USE FOR TESTING = ~0.25M batch size
# batch_size = 8
# block_size = 16
# gradient_accumulation_steps = 1 * 2048

# model - 1.3B from OPT paper table 1 (https://arxiv.org/pdf/2205.01068)
# # model params ~= 12 * n_layer * n_emd**2 + n_embd * vocab_size
n_layer = 24
n_head = 32
n_embd = 2048
learning_rate=2e-4

# max iters
max_iters = 10

use_pccl=True
bucket_cap_mb=32