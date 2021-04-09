#!/bin/sh

##### resource allocation
#SBATCH --job-name=json2data   # Job name
#SBATCH --mail-type=FAIL,END               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=USER@DOMAIN  # Where to send mail  
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=40            # Use 1 core
#SBATCH --mem=512gb  
#SBATCH --gpus-per-task=1                   # Memory limit
#SBATCH --time=12:00:00          # Time limit hrs:min:sec
#SBATCH --partition=hpg-ai
#SBATCH --reservation=HPGAI
##SBATCH --exclusive
#SBATCH --output=./j2d_%j.out   # Standard output and error log
#SBATCH --partition=hpg-ai
#SBATCH --reservation=HPGAI


pwd; hostname; date
echo "Pipeline task on processing json to data bin"


KEY='NOTE_TEXT'
CONTAINER=./containers/pytorch.sif # we need a container without megatron installed
VOCAB=./vocab.txt

for i in $(seq 1 21)
do 
    DATA=./data/note_txt_${i}.json
    PREFIX=./data/bin/note_txt_${i}

    singularity exec --nv $CONTAINER python ./Megatron-LM/tools/preprocess_data.py \
        --input $DATA \
        --json-keys $KEY \
        --split-sentences \
        --tokenizer-type BertWordPieceCase \
        --vocab-file $VOCAB \
        --output-prefix $PREFIX \
        --dataset-impl mmap \
        --workers 32 \
        --log-interval 1000
done

date
