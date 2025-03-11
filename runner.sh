#!/bin/bash

# select_gpu_device wrapper script
export LOCAL_RANK=${SLURM_LOCALID}
export RANK=${SLURM_PROCID}
export WORLD_SIZE=${SLURM_NTASKS}

cd /mnt/bb/${userid}/

if [ $LOCAL_RANK == 0 ]; then

	if [ ! -d /mnt/bb/${userid}/${ENV_NAME} ]; then
		#mkdir my_cloned_env
		tar -xf /mnt/bb/${userid}/$ENV_NAME.tar.gz 
	fi

	if [ ! -d /mnt/bb/${userid}/aws-ofi-rccl ]; then
		tar -xf aws-ofi-rccl.tar.gz
	fi	

	touch /mnt/bb/$userid/barrier
fi


while [ ! -f /mnt/bb/$userid/barrier ]
do
		sleep 20
done



. ${WRKSPC}/${ENV_NAME}/bin/activate



# if [ $RANK == 0 ]; then
# 	pip show axonn
# 	echo "============"
# 	pip show lightning
# 	echo "============"
# 	pip show torch
# 	echo "============"
# 	ls /mnt/bb/ssingh37/aws-ofi-rccl/lib
# 	echo "=========="
# 	echo "Current working directory"
# 	pwd
# fi



ulimit -n 131070
ulimit -c 0

if [ $RANK == 0 ]; then
	echo $SCRIPT
fi

cd $WRKSPC/../nanoGPT
eval $SCRIPT