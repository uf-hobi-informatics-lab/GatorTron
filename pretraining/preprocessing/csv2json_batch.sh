#!/bin/sh

##### resource allocation
#SBATCH --job-name=megatron_preprocess_batch    # Job name
#SBATCH --mail-type=FAIL,END               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=USER@DOMAIN   # Where to send mail  
#SBATCH --nodes=1                # Use one node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128              # Use 1 core
#SBATCH --mem=2000gb                     # Memory limit
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=megatron_csv2json_%A_%a.out   # Standard output and error log
#SBATCH --array=0-19 # 20 tasks in total
#SBATCH --partition=hpg-ai
#SBATCH --reservation=HPGAI

pwd; hostname; date
echo "Pipeline task on processing raw tsv data to json with text preprocessing"

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

INPUT=./data/note_txt_${SLURM_ARRAY_TASK_ID}.tsv
OUTPUT=./output/note_txt_${SLURM_ARRAY_TASK_ID}.json
CHUNK_SIZE=20000
NUM_CPU=$SLURM_CPUS_PER_TASK

echo $INPUT $OUTPUT $NUM_CPU

CONTAINER=./container/megatron.sif

singularity exec --nv $CONTAINER python gatortron_process_data.py \
    --input $INPUT \
    --output $OUTPUT \
    --cpus $NUM_CPU \
    --chunk_size $CHUNK_SIZE

date