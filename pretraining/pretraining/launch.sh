#!/bin/bash

#
# Script to launch a multi-node pytorch.distributed training run.
#
# (c) 2021, Brian J. Stucky
# UF Research Computing
#

#SBATCH --wait-all-nodes=1
#SBATCH --job-name=pretraining
#SBATCH --mail-type=NONE
#SBATCH --mail-user=alexgre@ufl.edu
#SBATCH --nodes=124
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=2000gb
#SBATCH --partition=hpg-ai
#SBATCH --reservation=HPGAI
#SBATCH --exclusive
#SBATCH --time=120:00:00
#SBATCH --output=./log/full_%j.out


#
# Training command specification.
#
VOCAB_FILE=./vocab.txt
CHECKPOINT_PATH=./gatortron_4b_uf30kcased
DATA_PATH=./uf_all_ufvocab_30k_cased_NOTE_TEXT_sentence

#    --tokenizer-type BertWordPieceCase \
BERT_ARGS="--num-layers 48 \
    --hidden-size 2560 \
    --tokenizer-type BertWordPieceCase \
    --num-attention-heads 40 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --lr 0.0001 \
    --lr-decay-iters 500000 \
    --seed 13 \
    --train-iters 1000000 \
    --min-lr 0.00001 \
    --lr-warmup-fraction 0.01 \
    --micro-batch-size 8 \
    --global-batch-size 3968 \
    --vocab-file $(realpath $VOCAB_FILE) \
    --split 949,50,1 \
    --DDP-impl torch \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --tensorboard-dir ./log \
    --fp16"

OUTPUT_ARGS="--log-interval 1000 \
    --save-interval 10000 \
    --eval-interval 5000 \
    --eval-iters 100 \
    --checkpoint-activations"


TRAINING_SCRIPT=./Megatron-LM/pretrain_bert.py
TRAINING_CMD="$TRAINING_SCRIPT \
    $BERT_ARGS \
    $OUTPUT_ARGS \
    --save $(realpath $CHECKPOINT_PATH) \
    --load $(realpath $CHECKPOINT_PATH) \
    --data-path $(realpath $DATA_PATH)"


#
# Python location (if not provided, system default will be used).
#
PYTHON_PATH="singularity exec --nv ./containers/py2103.sif python"


#
# The location of the Pytorch multi-node launch utilities.
#
PT_LAUNCH_UTILS_PATH=.


#
# The remainder of this script should not require modification.
#
source "${PT_LAUNCH_UTILS_PATH}/helper.sh"
#export NCCL_DEBUG=INFO
init_node_info

pwd; hostname; date

echo "Primary node: $PRIMARY"
echo "Primary TCP port: $PRIMARY_PORT"
echo "Secondary nodes: $SECONDARIES"

LAUNCH_CMD="$PYTHON_PATH \
        -m torch.distributed.launch \
              --nproc_per_node=$SLURM_GPUS_PER_TASK \
              --nnodes=$SLURM_JOB_NUM_NODES \
              --node_rank=$SLURM_NODEID \
              --master_addr=$PRIMARY \
              --master_port=$PRIMARY_PORT \
            $TRAINING_CMD"

echo "Running \"$TRAINING_CMD\" on each node..."
run_with_retry "$LAUNCH_CMD"