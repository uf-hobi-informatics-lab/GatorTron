#!/bin/bash

##### resource allocation
#SBATCH --job-name=sts
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexgre@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=2000gb
#SBATCH --time=10:00:00
#SBATCH --output=./2021gatortron/nemo_downstream/sts/log/sts_9b_%j.out
#SBATCH --partition=gpu
#SBATCH --reservation=gatortron
#SBATCH --exclusive


pwd; hostname; date
echo "start STS..."

tag=9b
CONF=./2021gatortron/nemo_downstream/sts/slurm/sts_9b.yaml

CONTAINER=./containers/nemo120.sif

output=./2021gatortron/nemo_downstream/sts/results/${tag}

# nsys profile –o ${SLURM_LOG}/nsys_output \
singularity exec --nv $CONTAINER \
    python ./2021gatortron/nemo_downstream/sts/sts.py \
        --gpus $SLURM_GPUS_PER_TASK \
        --pred_output $output \
        --config $CONF
