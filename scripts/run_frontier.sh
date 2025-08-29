#!/bin/bash
#SBATCH -q normal
#SBATCH -J nanogpt
#SBATCH --gpu-bind none
#SBATCH -t 00:05:00
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
rocm_version="6.4.1"
module load PrgEnv-cray
module load rocm/${rocm_version}
module load cray-mpich/8.1.32
module load cpe/25.03
module load craype-accel-amd-gfx90a
module load cray-python/3.11.7
module load ninja
module list
export CXX=CC 
export CC=cc

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

# mpich gpu support
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_VERBOSE=1
export MPICH_OFI_NIC_POLICY="USER"
export MPICH_OFI_NIC_MAPPING="0:0-1; 1:2-3; 2:4-5; 3:6-7"

# fi variables
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0 
export MPICH_OFI_CXI_COUNTER_VERBOSE=1
# collecting counter data
export MPICH_OFI_CXI_COUNTER_REPORT=5
export HSA_ENABLE_SDMA=0

# other
export MPICH_GPU_SUPPORT_ENABLED=1
export GPU_MAX_HW_QUEUES=1
export OFI_NCCL_USE_IPV6_TCP=1

MASK_0="0x00fe000000000000" # Cores 49-55
MASK_1="0xfe00000000000000" # Cores 57-64
MASK_2="0x0000000000fe0000" # Cores 17-23
MASK_3="0x00000000fe000000" # Cores 25-31
MASK_4="0x00000000000000fe" # Cores 1-7
MASK_5="0x000000000000fe00" # Cores 9-15
MASK_6="0x000000fe00000000" # Cores 33-39
MASK_7="0x0000fe0000000000" # Cores 41-47
CPU_MASK="--cpu-bind=mask_cpu:${MASK_0},${MASK_1},${MASK_2},${MASK_3},${MASK_4},${MASK_5},${MASK_6},${MASK_7}"


SCRIPT="scripts/get_rank.sh python -u train.py config/train_gpt2_124M.py"
# log start date
echo "start nanoGPT: $(date)"
run_cmd="srun -N $NNODES -n $GPUS --ntasks-per-node=8 -c 7 ${CPU_MASK} --mem-bind=map_mem:3,3,1,1,0,0,2,2 $SCRIPT"
echo $run_cmd
eval $run_cmd
# log end date
echo "end nanoGPT: $(date)"

echo "end run: $(date)"