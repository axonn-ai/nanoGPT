# disable wandb
wandb_log = False

# disable compile
compile = False

# Batch size settings
batch_size = 2
block_size = 2048
gradient_accumulation_steps = 64

# Training settings
max_iters = 1000
lr_decay_iters = 1000

# Evaluation settings
eval_interval = 1000
eval_iters = 200
log_interval = 1

# model
n_layer = 32
n_embd = 4096
n_head = 32

flops_promised = 192e12

patch_collectives = False

# Activation checkpointing
gradient_checkpointing = True