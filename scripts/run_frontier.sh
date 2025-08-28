#!/bin/bash
#SBATCH -q normal
#SBATCH -J nanogpt
#SBATCH --gpu-bind none
#SBATCH -t 00:10:00
#SBATCH -A csc547
#SBATCH --exclusive
# Run like: sbatch run_frontier16.sh

echo "start run: $(date)"

export SCRATCH="/lustre/orion/csc547/scratch/keshprad"
export WRKSPC="${SCRATCH}/nanoGPT"
export HF_HOME="${SCRATCH}/.cache/hf"
export TRITON_CACHE_DIR="${SCRATCH}/.cache/triton"
cd $WRKSPC

# load modules
ROCM_VERSION=6.2.4
echo resetting modules:
module reset
echo loading modules:
module load PrgEnv-gnu/8.6.0
module load rocm/${ROCM_VERSION}
module load craype-accel-amd-gfx90a
module load cray-python/3.11.7
module load cray-mpich/8.1.32

module load cpe/24.11
module load Core/24.00
module load ninja
module list
# activate env
source ${WRKSPC}/.venv/bin/activate

NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 8 ))
## master addr and port
# setting variables for torch.distributed
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$GPUS
export OMP_NUM_THREADS=7

## some RCCL env variables
export FI_CXI_ATS=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0
export CUDA_DEVICE_MAX_CONNECTIONS=1
# AWS-OFI-RCCL
export LD_LIBRARY_PATH="${SCRATCH}/aws-ofi-rccl/lib:$LD_LIBRARY_PATH"
# other
export MPICH_GPU_SUPPORT_ENABLED=1
export GPU_MAX_HW_QUEUES=1
export OFI_NCCL_USE_IPV6_TCP=1

SCRIPT="train.py config/train_gpt2_124M.py"

# log start date
echo "start nanoGPT: $(date)"
run_cmd="srun -N $NNODES -n $GPUS --cpu-bind=cores --gpus-per-node=8 --ntasks-per-node=8 scripts/get_rank.sh python -u $SCRIPT"
echo $run_cmd
eval $run_cmd
# log end date
echo "end nanoGPT: $(date)"

echo "end run: $(date)"