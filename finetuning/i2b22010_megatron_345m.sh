#!/bin/bash

#SBATCH --job-name=ner
#SBATCH --mail-type=ALL
#SBATCH --mail-user=USER@DOMAIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=32 
#SBATCH --mem=512gb
#SBATCH --time=24:00:00
#SBATCH --output=./log/i2b22010_345m_megatron_%j.out
#SBATCH --partition=hpg-ai
#SBATCH --reservation=HPGAI
#SBATCH --exclusive


pwd; hostname; date
echo "ner training..."


expr_tag=i2b22010_345m_megatron


NEMO_GS=./ner_datasets/nemo_ner_2010i2b2
NEMO_PRED=./${expr_tag}/prediction
RESULT=./result_${expr_tag}.txt


CONF=./i2b22010_megatron_345m.yaml
CONTAINER=./<path to container>/nemo.sif


singularity exec --nv $CONTAINER \
    python ner_expr.py \
    --gpus $SLURM_GPUS_PER_TASK \
    --ner_config $CONF \
    --do_train
sleep 10


singularity exec --nv $CONTAINER \
    python ner_expr.py \
    --gpus $SLURM_GPUS_PER_TASK \
    --ner_config $CONF \
    --do_pred \
    --pred_output $NEMO_PRED
sleep 10


GS=${NEMO_GS}/test_bio.txt
PRED=${NEMO_PRED}/test_bio.txt
cp ${NEMO_GS}/text_test.txt ${NEMO_PRED}/text_test.txt
singularity exec --nv $CONTAINER python ./nemo2bio.py --nemo $NEMO_GS --bio $GS --type test
sleep 10
singularity exec --nv $CONTAINER python ./nemo2bio.py --nemo $NEMO_PRED --bio $PRED --type test
sleep 10


echo "eval based on our eval script:" > $RESULT
singularity exec --nv $CONTAINER python ./new_bio_eval.py -f1 $GS -f2 $PRED >> $RESULT
sleep 10
echo "" >> $RESULT
echo "" >> $RESULT


CONLL_OUTPUT=./${expr_tag}/prediction/conll.bio
CONTAINER1=./<path to container>/py2.sif # we need a container with python2 env since conlleval.py was implemented with py2
singularity exec $CONTAINER python generate_conll_compatible_eval.py --gs $GS --pred $PRED --output $CONLL_OUTPUT
sleep 10
echo "eval based on conll script:" >> $RESULT
singularity exec $CONTAINER1 python ./conlleval.py $CONLL_OUTPUT >> $RESULT

date