#!/bin/bash
#SBATCH -p batch
#SBATCH -A CSC547
#SBATCH -t 00:15:00
#SBATCH -C nvme

PROJ_NAME="csc547"
export WRKSPC="/lustre/orion/$PROJ_NAME/scratch/$USER/communication"
export ENV_NAME="comm-venv"

module load cray-mpich/8.1.31
module load amd-mixed/6.2.4
module load cpe/24.11
module load craype-accel-amd-gfx90a
#module load cray-python/3.10.10

## calculating the number of nodes and GPUs
export NNODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=8 ## change as per your machine
export GPUS=$(( NNODES * GPUS_PER_NODE )) 

## pytorch dist variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$GPUS

## some RCCL env variables
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_CROSS_NIC=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL="PHB"
## RCCL plugin
export userid=$(whoami)
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/bb/${userid}/aws-ofi-rccl/lib/"

# mpich gpu support
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_VERBOSE=1
export MPICH_OFI_NIC_POLICY="USER"
export MPICH_OFI_NIC_MAPPING="0:0-1; 1:2-3; 2:4-5; 3:6-7"
# fi variables
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0 
#export MPICH_OFI_CXI_COUNTER_VERBOSE=1
#export MPICH_OFI_CXI_COUNTER_REPORT=5
export HSA_ENABLE_SDMA=0


MASK_0="0x00fe000000000000" # Cores 49-55
MASK_1="0xfe00000000000000" # Cores 57-64
MASK_2="0x0000000000fe0000" # Cores 17-23
MASK_3="0x00000000fe000000" # Cores 25-31
MASK_4="0x00000000000000fe" # Cores 1-7
MASK_5="0x000000000000fe00" # Cores 9-15
MASK_6="0x000000fe00000000" # Cores 33-39
MASK_7="0x0000fe0000000000" # Cores 41-47
CPU_MASK="--cpu-bind=mask_cpu:${MASK_0},${MASK_1},${MASK_2},${MASK_3},${MASK_4},${MASK_5},${MASK_6},${MASK_7}"

export WANDB_MODE="offline"


export PYTHONPATH="$PYTHONPATH:."


start_time=$(date +%s)
# echo "broadcasting data to burst buffer"
# sbcast -pf data/openwebtext/train.bin /mnt/bb/ssingh37/train.bin
echo "broadcasting environment"
sbcast $WRKSPC/$ENV_NAME.tar.gz /mnt/bb/$userid/$ENV_NAME.tar.gz
echo "brodcasting aws ofi"
sbcast $WRKSPC/aws-ofi-rccl.tar.gz /mnt/bb/$userid/aws-ofi-rccl.tar.gz
end_time=$(date +%s)

echo "Time taken: $((end_time - start_time)) seconds"

if [ ! -L  $WRKSPC/$ENV_NAME -a ! -d $WRKSPC/$ENV_NAME ]; then
	rm $WRKSPC/$ENV_NAME
	mkdir /mnt/bb/$userid/$ENV_NAME
	ln -s /mnt/bb/$userid/$ENV_NAME $WRKSPC/$ENV_NAME
	rm -rf /mnt/bb/$userid/$ENV_NAME
fi

export SCRIPT="python -u train.py $1"
run_cmd="srun -N $NNODES -n $GPUS --ntasks-per-node=8 -c 7 ${CPU_MASK} --mem-bind=map_mem:3,3,1,1,0,0,2,2  ./runner.sh" 
echo $run_cmd 
eval $run_cmd 


