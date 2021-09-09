#!/bin/bash

##### resource allocation
#SBATCH --job-name=qa
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexgre@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=2000gb
#SBATCH --time=72:00:00
#SBATCH --output=./2021gatortron/nemo_downstream/qa/log/qa_emr_9b_med_%j.out
#SBATCH --partition=gpu
#SBATCH --reservation=gatortron
#SBATCH --exclusive


pwd; hostname; date
echo "start QA..."

tag=emrqa_med02_9b_50k

CONF=./2021gatortron/nemo_downstream/qa/slurm/emrqa_9b_med.yaml

CONTAINER=./containers/nemo120.sif

output=./2021gatortron/nemo_downstream/qa/results/${tag}

# nsys profile –o ${SLURM_LOG}/nsys_output \
singularity exec --nv $CONTAINER \
    python ./2021gatortron/nemo_downstream/qa/qa_ddp.py \
        --gpus $SLURM_GPUS_PER_TASK \
        --pred_output $output \
        --qa_config $CONF