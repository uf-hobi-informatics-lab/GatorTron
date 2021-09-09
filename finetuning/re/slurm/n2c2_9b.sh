#!/bin/bash

##### resource allocation
#SBATCH --job-name=re
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexgre@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=2000gb
#SBATCH --time=48:00:00
#SBATCH --output=/red/gatortron-phi/workspace/2021gatortron/nemo_downstream/re/log/re_n2c2_9b_%j.out
#SBATCH --partition=gpu
#SBATCH --reservation=gatortron
#SBATCH --exclusive

pwd; hostname; date
echo "start RE..."

tag=n2c2_9b

CONF=/red/gatortron-phi/workspace/2021gatortron/nemo_downstream/re/slurm/n2c2_9b.yaml
CONTAINER=/red/gatortron-phi/workspace/containers/nemo120.sif
output=/red/gatortron-phi/workspace/2021gatortron/nemo_downstream/re/results/${tag}

## training
singularity exec --nv $CONTAINER \
    python /red/gatortron-phi/workspace/2021gatortron/nemo_downstream/re/re_ddp.py \
        --gpus $SLURM_GPUS_PER_TASK \
        --pred_output $output \
        --re_config $CONF
sleep 10


# try to load model.nemo separately and do predicion
# singularity exec --nv $CONTAINER \
#     python /red/gatortron-phi/workspace/2021gatortron/nemo_downstream/re/re_pred_ddp.py \
#         --gpus $SLURM_GPUS_PER_TASK \
#         --pred_output $output \
#         --re_config $CONF
# sleep 10


# singularity exec --nv $CONTAINER \
#     python /red/gatortron-phi/workspace/2021gatortron/nemo_downstream/re/re_ddp_hydra.py
# sleep 10


###################### evaluation based on prediction labels ##################################
gs_set=/red/gatortron-phi/workspace/2021gatortron/data/re/2018n2c2/gold_standard_set.pkl
test_supp=/red/gatortron-phi/workspace/2021gatortron/data/re/2018n2c2/2018n2c2_marker_format_1/testsupp.tsv

singularity exec $CONTAINER python /red/gatortron-phi/workspace/2021gatortron/nemo_downstream/re/brat_eval_res.py \
    --gs $gs_set \
    --pred ${output}/predict_labels.txt \
    --supp $test_supp \
    --output ${output}/res_prf.txt

