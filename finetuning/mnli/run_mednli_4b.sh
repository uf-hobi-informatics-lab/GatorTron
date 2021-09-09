#!/bin/bash

##### resource allocation
#SBATCH --job-name=mednli
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexgre@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128 
#SBATCH --mem=1000gb
#SBATCH --time=24:00:00
#SBATCH --output=./2021gatortron/nemo_downstream/mnli/log/mednli_4b_batch8_%j.out
#SBATCH --partition=gpu
#SBATCH --reservation=gatortron
#SBATCH --exclusive


DISTRIBUTED_ARGS="--nproc_per_node $SLURM_GPUS_PER_TASK \
                  --nnodes 1 \
                  --node_rank 0"

DATAROOT=./2021gatortron/data/mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0/MNLI
TRAIN_DATA=${DATAROOT}/mli_train_v1.tsv
VALID_DATA="${DATAROOT}/mli_dev_v1.tsv \
            ${DATAROOT}/mli_test_v1.tsv"

VOCAB_FILE=./new_vocabs/uf_full_pubmed_wiki_cased_50k/vocab.txt
PRETRAINED_CHECKPOINT=./uf_pubmed_wiki_4b_uf_pubmed_wiki_50k_cased_final
CHECKPOINT_PATH=./2021gatortron/nemo_downstream/mnli/results/4b

CONTAINER=./containers/csv2json.sif

L=48
H=2560
A=40
tensor_par=2
bs=8

singularity exec --nv $CONTAINER python -m torch.distributed.launch $DISTRIBUTED_ARGS \
                ./Megatron-LM-2.2/tasks/main.py \
               --task MNLI \
               --seed 1234 \
               --keep-last \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --tokenizer-type BertWordPieceCase \
               --vocab-file $VOCAB_FILE \
               --epochs 10 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --tensor-model-parallel-size $tensor_par \
               --num-layers $L \
               --hidden-size $H \
               --num-attention-heads $A \
               --micro-batch-size $bs \
               --checkpoint-activations \
               --lr 5.0e-5 \
               --lr-decay-style linear \
               --lr-warmup-fraction 0.065 \
               --seq-length 512 \
               --max-position-embeddings 512 \
               --save-interval 500000 \
               --save $CHECKPOINT_PATH \
               --log-interval 1000 \
               --weight-decay 1.0e-1 \
               --fp16
