#!/bin/bash

##### resource allocation
#SBATCH --job-name=ner_training
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexgre@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128 
#SBATCH --mem=2000gb
#SBATCH --time=24:00:00
#SBATCH --output=./2021gatortron/nemo_downstream/nemo_ner/log/9b_i2b22012_%j.out
#SBATCH --partition=gpu
#SBATCH --exclusive


pwd; hostname; date
echo "ner training..."

expr_tag=9B_i2b22012

NEMO_GS=./2021gatortron/data/ner/nemo_ner_2012i2b2_new/
NEMO_PRED=./2021gatortron/nemo_downstream/nemo_ner/results/${expr_tag}/prediction
RESULT=./2021gatortron/nemo_downstream/nemo_ner/results/result_${expr_tag}.txt
py_root=./2021gatortron/nemo_downstream/nemo_ner


CONF=./2021gatortron/nemo_downstream/nemo_ner/shell/i2b22012_9B.yaml

CONTAINER=./containers/nemo120.sif


singularity exec --nv $CONTAINER \
    python $py_root/ner_expr_ddp.py \
    --gpus $SLURM_GPUS_PER_TASK \
    --ner_config $CONF \
    --pred_output $NEMO_PRED \
    --do_train
sleep 10


GS=${NEMO_GS}/test_bio.txt
PRED=${NEMO_PRED}/test_bio.txt
cp ${NEMO_GS}/text_test.txt ${NEMO_PRED}/text_test.txt
singularity exec --nv $CONTAINER python $py_root/nemo2bio.py --nemo $NEMO_GS --bio $GS --type test
sleep 10
singularity exec --nv $CONTAINER python $py_root/nemo2bio.py --nemo $NEMO_PRED --bio $PRED --type test
sleep 10

# f1 measurement
echo "eval based on our eval script:" > $RESULT
singularity exec --nv $CONTAINER python ./scripts/nemo_ner/new_bio_eval.py -f1 $GS -f2 $PRED >> $RESULT
sleep 10
echo "" >> $RESULT
echo "" >> $RESULT


CONLL_OUTPUT=./2021gatortron/nemo_downstream/nemo_ner/results/${expr_tag}/prediction/conll.bio
CONTAINER1=./containers/py2.sif
singularity exec $CONTAINER python $py_root/generate_conll_compatible_eval.py --gs $GS --pred $PRED --output $CONLL_OUTPUT
sleep 10
echo "eval based on conll script:" >> $RESULT
singularity exec $CONTAINER1 python $py_root/conlleval/conlleval.py $CONLL_OUTPUT >> $RESULT
sleep 10

date
