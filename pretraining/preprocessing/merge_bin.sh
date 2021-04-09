#!/bin/sh

##### resource allocation
#SBATCH --job-name=megatron_preprocess    # Job name
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=USER@DOMAIN   # Where to send mail
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=4             # Use 1 core
#SBATCH --mem=999gb
#SBATCH --gpus-per-task=1                       # Memory limit
#SBATCH --time=12:00:00               # Time limit hrs:min:sec
#SBATCH --output=bin_%j.out   # Standard output and error log
#SBATCH --partition=hpg-ai
#SBATCH --reservation=HPGAI

pwd; hostname; date

CONTAINER=./containers/pytorch.sif # a container has no megatron and nemo installed

singularity exec $CONTAINER python merge_megatron_preprocessing_bin_files.py \
    --input ./to_merge \
    --output ./uf_full_uf30kcased \
    --output_prefix uf_full_uf30kcased_TEXT \
    --vocab_file ./vocab.txt \
    --tokenizer_type BertWordPieceCase

date