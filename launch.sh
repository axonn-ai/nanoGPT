#!/bin/bash
MODEL="13B"
for nodes in 128 256 512 1024; do
    cmd="USE_PCCL=1 sbatch --nodes=$nodes -o ds_${MODEL}_yaccl_$nodes.out fr_run.sh ./configs/deepspeed/${MODEL}.py"
    echo $cmd
    eval $cmd
    cmd="sbatch --nodes=$nodes -o ds_${MODEL}_$nodes.out fr_run.sh ./configs/deepspeed/${MODEL}.py"
    echo $cmd
    eval $cmd
done