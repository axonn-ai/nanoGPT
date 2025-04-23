#!/bin/bash
#
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

WRKSPC=/lustre/orion/csc547/scratch/keshprad/nanoGPT
# everything will be installed in $WRKSPC

ENV_NAME="axonn_nanogpt"
# this is the name of your python venv, change if needed

cd $WRKSPC
echo -e "${RED}Creating Python Environment in $WRKSPC:${GREEN}"

# Load modules
module reset
# load modules
rocm_version=6.1.3
module load PrgEnv-cray
module load rocm/${rocm_version}
module load craype-accel-amd-gfx90a
module load cray-python/3.9.13.1
module load cray-mpich/8.1.30
module load cray-hdf5-parallel/1.12.2.9

#Step 1 - activate your venv
python -m venv $WRKSPC/$ENV_NAME
source $WRKSPC/$ENV_NAME/bin/activate
# upgrade pip
pip install -U pip

echo -e "${RED}Installing Dependencies:${GREEN}"
#Step 2 - install torch
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/rocm6.1
MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

#Step 3 - install axonn from source
mkdir -p ${WRKSPC}/repos
cd ${WRKSPC}/repos
git clone https://github.com/axonn-ai/axonn.git
pip install -e axonn

#Step 4 - install other packages
pip install numpy transformers datasets tiktoken wandb tqdm


#Step 5 - AWS-OFI-RCCL plugin
mkdir -p ${WRKSPC}/repos
cd ${WRKSPC}/repos
libfabric_path=/opt/cray/libfabric/1.15.2.0
# Download the plugin repo
git clone --recursive https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl
cd aws-ofi-rccl
# Build the plugin
./autogen.sh
export LD_LIBRARY_PATH=/opt/rocm-$rocm_version/hip/lib:$LD_LIBRARY_PATH
PLUG_PREFIX=$PWD
# cmake configure
CC=hipcc CFLAGS=-I/opt/rocm-$rocm_version/rccl/include ./configure \
    --with-libfabric=$libfabric_path --with-rccl=/opt/rocm-$rocm_version --enable-trace \
    --prefix=$PLUG_PREFIX --with-hip=/opt/rocm-$rocm_version/hip --with-mpi=$MPICH_DIR
# make
make
make install
# Reminder to export the plugin to your path
echo $PLUG_PREFIX
echo "Add the following line in the environment to use the AWS OFI RCCL plugin"
echo "export LD_LIBRARY_PATH="$PLUG_PREFIX"/lib:$""LD_LIBRARY_PATH"
cd $WRKSPC

echo -e "${RED}Your Python Environment is ready. To activate it run the following commands in the SAME order:${NC}"
echo -e "${GREEN}source $WRKSPC/$ENV_NAME/bin/activate${NC}"
echo ""
echo -e "${NC}"