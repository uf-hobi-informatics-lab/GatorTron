from nemo.utils.exp_manager import exp_manager
from nemo.collections import nlp as nemo_nlp
import traceback

import os
import wget 
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from pathlib import Path
import argparse


EVAL_FILE = "/red/gatortron-phi/workspace/2021gatortron/nemo_downstream/qa/evaluate-v1.1.py"


def main(args):
    config_path = args.qa_config
    config = OmegaConf.load(config_path)

    config.trainer.gpus = args.gpus
    print(OmegaConf.to_yaml(config))
    print(config.model.nemo_path)

    trainer = pl.Trainer(**config.trainer)
    exp_dir = exp_manager(trainer, config.get("exp_manager", None))
    model_qa = None

    try:
        model_qa = nemo_nlp.models.QAModel.restore_from(config.model.nemo_path)
    except:
        traceback.print_exc()
        config.model.nemo_path = str(exp_dir) + "/checkpoints/default.nemo"
        model_qa = nemo_nlp.models.QAModel.restore_from(config.model.nemo_path)


    output_nbest_file = Path(args.pred_output) / "predict_nbest.json"
    output_prediction_file = Path(args.pred_output) / "predict_prediction.json"

    model_qa.inference(
        file=config.model.test_ds.file,
        batch_size=16,
        num_samples=-1,
        output_nbest_file=output_nbest_file,
        output_prediction_file=output_prediction_file,
    )

    os.system(f"python {EVAL_FILE} --dataset_file {config.model.test_ds.file} --prediction_file {str(output_prediction_file)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1,
                        help="how many gpus to use")
    parser.add_argument("--pred_output", type=str, default=None,
                        help="where we store all the models and prediction")
    parser.add_argument("--qa_config", type=str, required=True,
                        help="config file .yaml")


    global_args = parser.parse_args()
    main(global_args)