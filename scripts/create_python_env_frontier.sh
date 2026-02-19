#!/bin/bash
#
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

SCRATCH=/lustre/orion/csc547/scratch/keshprad
WRKSPC=${SCRATCH}/nanoGPT
# everything will be installed in $WRKSPC

ENV_NAME=".venv"
# this is the name of your python venv, change if needed

cd $WRKSPC
echo -e "${RED}Creating Python Environment in $WRKSPC:${GREEN}"

# Load modules
module reset
# load modules
rocm_version=6.4.1
module load PrgEnv-cray
module load rocm/${rocm_version}
module load cray-mpich/8.1.32
module load cpe/25.03
module load craype-accel-amd-gfx90a
module load cray-python/3.11.7
module load ninja
module list

#Step 1 - activate your venv
python -m venv $WRKSPC/$ENV_NAME
source $WRKSPC/$ENV_NAME/bin/activate
# upgrade pip
pip install -U pip

echo -e "${RED}Installing Dependencies:${GREEN}"
#Step 2 - install torch
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
pip install --upgrade numpy

#Step 3 build mpi4py
MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

#Step 4 - install other packages
pip install numpy transformers datasets tiktoken wandb tqdm


#Step 5 - AWS-OFI-RCCL plugin
# skip for now as I already have it installed
# echo "Installing RCCL Plugin"
# cd ${SCRATCH}
# git clone --recursive https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl 
# cd aws-ofi-rccl
# libfabric_path=/opt/cray/libfabric/1.22.0
# ./autogen.sh
# export LD_LIBRARY_PATH=/opt/rocm-$rocm_version/lib:$LD_LIBRARY_PATH
# PLUG_PREFIX=$PWD
# CC=hipcc CFLAGS=-I/opt/rocm-$rocm_version/include ./configure \
# 	--with-libfabric=$libfabric_path --with-rccl=/opt/rocm-$rocm_version --enable-trace \
# 	--prefix=$PLUG_PREFIX --with-hip=/opt/rocm-$rocm_version --with-mpi=$MPICH_DIR
# make
# make install
# cd ..

echo -e "${RED}Your Python Environment is ready. To activate it run the following commands in the SAME order:${NC}"
echo -e "${GREEN}source $WRKSPC/$ENV_NAME/bin/activate${NC}"
echo ""
echo -e "${NC}"