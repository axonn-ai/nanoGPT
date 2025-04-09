#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH -A m2404_g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --time=00:05:00
#SBATCH --qos=regular


module load nccl
module load cudatoolkit/12.4
source ./venv/bin/activate

## calculating the number of nodes and GPUs
export NNODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=4 ## change as per your machine
export GPUS=$(( NNODES * GPUS_PER_NODE )) 


## pytorch dist variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$GPUS
export CXX=CC 
export CC=cc

#export MPICH_OFI_CXI_COUNTER_REPORT=5

source ./pm_env.sh

export WANDB_MODE="offline"


export PYTHONPATH="$PYTHONPATH:."

GLOBAL_BATCH_SIZE=16
GAS=${2:-1}
GLOBAL_GAS=$(( GAS * WORLD_SIZE ))
MICRO_BS=$(( GLOBAL_BATCH_SIZE / GLOBAL_GAS ))


COMMON_ARGS="--wandb_log=False \
            --compile=False \
            --max_iters=10 \
            --lr_decay_iters=10 \
            --log_interval=1 \
            --gradient_checkpointing=True \
            --gradient_accumulation_steps='${GLOBAL_GAS}' \
            --batch_size='${MICRO_BS}' \
            --block_size=2048
"


if [[ -n "$USE_PCCL" ]]; then
    COMMON_ARGS+=" --use_pccl=True"
fi

echo $COMMON_ARGS

export SCRIPT="python -u train_deepspeed.py $1 $COMMON_ARGS"
source ./comm-venv/bin/activate
run_cmd="srun -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 $SCRIPT" 
echo $run_cmd 
eval $run_cmd 
