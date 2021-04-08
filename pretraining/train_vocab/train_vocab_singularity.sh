#!/bin/sh

##### resource allocation
#SBATCH --job-name=train_vocab    
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=USER@DOMAIN  
#SBATCH --nodes=1
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=32          
#SBATCH --mem=2000gb                   
#SBATCH --time=05:00:00              
#SBATCH --output=train_vocab_%j.out

#### task ####
pwd; hostname; date

echo 'train vocab'

CONTAINER=./<path to container>/megatron.sif

TEXT=/workspace/full_text.txt
OUTPUT=/workspace
VOCAB_SIZE=30000
BERT_HEADER=/workspace/bert_vocab_head.txt
PREF=full_uncased_30k

singularity exec --nv python train_vocab.py \
    --input $TEXT \
    --prefix $PREF \
    --output $OUTPUT \
    --vocab_size $VOCAB_SIZE \
    --threads 32 \
    --lower_case \
    --bert_header $BERT_HEADER
date