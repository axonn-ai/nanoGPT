import argparse
import json
import os

# def create_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", help="model name", choices=["7B"], default="7B")
#     parser.add_argument("--global-batch-size", type=int, default=8192, help="Number of samples in global batch")
#     parser.add_argument("--grad-acc", type=int, default=1, help="Number of grad-acc steps")
#     parser.add_argument("--disable-gradient-checkpointing", action="store_false", dest="gradient_checkpointing", help="Disable gradient checkpointing")
#     parser.set_defaults(gradient_checkpointing=True)
#     parser.add_argument("--gpus", type=int, required=True, help="Number of GPUs")
#     parser.add_argument("--max-iters", type=int, default=20, help="Number of batches to run")
#     parser.add_argument("--seq-len", type=int, default=2048)
#     parser.add_argument("--uni-dist", action="store_true", default=False, help="Use the unified-dist library")
#     parser.add_argument("--uni-dist-low-lat-ag", action="store_true", default=False, help="Use ring-rd algo for all-gathers in uni-dist")
#     parser.add_argument("--uni-dist-low-lat-rs", action="store_true", default=False, help="Use ring-rh algo for reduce-scatters in uni-dist")
#     return parser

def generate_config(args):
    config = f"""
# disable wandb
wandb_log = False

# disable compile
compile = False

# Batch size settings
batch_size = {args.batch_size}
block_size = {args.block_size}
gradient_accumulation_steps = {args.grad_acc}

# Training settings
max_iters = {args.max_iters}
lr_decay_iters = {args.max_iters}
max_iters = {args.max_iters}

# Evaluation settings
eval_interval = 1000
eval_iters = 200
log_interval = 1

# model
n_layer = {args.n_layer}
n_embd = {args.n_embd}
n_head = {args.n_head}

# Weight decay
weight_decay = 1e-1

# AxoNN
G_intra_d = {args.gpus}
use_uni_dist = {args.uni_dist}
uni_dist_low_latency_all_gathers = {args.uni_dist_low_lat_ag}
uni_dist_low_latency_reduce_scatters = {args.uni_dist_low_lat_rs}
flops_promised = 192*1e12

# Activation checkpointing
gradient_checkpointing = {args.gradient_checkpointing}
"""
    return config.strip()

def process_args(args):
    # batch size
    assert args.global_batch_size % args.grad_acc == 0
    global_step_size = args.global_batch_size // args.grad_acc
    assert global_step_size % args.gpus == 0
    # in nanogpt args.batch_size is the local batch size per GPU
    args.batch_size = global_step_size // args.gpus
    # nanogpt's definition of grad_acc is different from the conventional definition
    args.grad_acc *= args.gpus
    # block size is the sequence length of each sample
    args.block_size = args.seq_len

    with open("./model_archs.json") as f:
        model_config = json.load(f)[args.model]

    args.n_layer = model_config["n_layer"]
    args.n_embd = model_config["n_embd"]
    args.n_head = model_config["n_head"]
    return args


def generate_all_configs():
    gpus_list = [256, 512, 1024, 2048]

    variants = [
        {"uni_dist": False, "ag": False, "rs": False}, 
        {"uni_dist": True, "ag": False, "rs": False}, 
        {"uni_dist": True, "ag": True, "rs": False}, 
        {"uni_dist": True, "ag": False, "rs": True}, 
        {"uni_dist": True, "ag": True, "rs": True}
    ]

    model_name = "13B"
    for variant in variants:
        folder_name = (
            f"uni_dist_{str(variant['uni_dist']).lower()}"
            + (f"_ag_{str(variant['ag']).lower()}_rs_{str(variant['rs']).lower()}" if variant['uni_dist'] else "")
        )
        path = f"configs/{model_name}/{folder_name}"
        os.makedirs(path, exist_ok=True)

        for gpus in gpus_list:
            args = argparse.Namespace(
                model=model_name,
                global_batch_size=2048,
                grad_acc=1,
                gradient_checkpointing=True,
                gpus=gpus,
                max_iters=20,
                seq_len=2048,
                uni_dist=variant["uni_dist"],
                uni_dist_low_lat_ag=variant["ag"],
                uni_dist_low_lat_rs=variant["rs"]
            )
            args = process_args(args)
            config_content = generate_config(args)
            config_path = f"{path}/gpus_{gpus}.py"

            with open(config_path, "w") as f:
                f.write(config_content)
    
    print("Config files generated successfully.")

if __name__ == "__main__":
    generate_all_configs()