# disable wandb
wandb_log = False

# disable compile
compile = False

# Batch size settings
batch_size = 1
block_size = 2048
gradient_accumulation_steps = 16

# Training settings
max_iters = 20
lr_decay_iters = 20
max_iters = 20

# Evaluation settings
eval_interval = 1000
eval_iters = 200
log_interval = 1

# model
n_layer = 32
n_embd = 4096
n_head = 32

# Weight decay
weight_decay = 1e-1

# AxoNN
G_intra_d = 16
use_uni_dist = False
uni_dist_low_latency_all_gathers = False
uni_dist_low_latency_reduce_scatters = False
flops_promised = 192e12

# Activation checkpointing
gradient_checkpointing = True