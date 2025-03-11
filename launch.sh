#!/bin/bash
GPUS=$1

for config_file in configs/13B/*/gpus_$GPUS.py; do
    # Extract GPU count from filename (e.g., gpus_128.py -> 128)
    GPUS=$(echo "$config_file" | grep -oP '(?<=gpus_)\d+')
    NODES=$(( (GPUS + 7) / 8 ))  # Calculate nodes (ceiling division)
    
    cmd="sbatch --nodes=$NODES fr_run.sh $config_file"
    echo $cmd
    eval $cmd
done