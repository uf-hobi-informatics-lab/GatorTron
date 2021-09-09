from nemo.utils.exp_manager import exp_manager
from nemo.collections import nlp as nemo_nlp
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin

import os
import wget 
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from pathlib import Path
import argparse


def main(args):
    config_path = args.qa_config
    config = OmegaConf.load(config_path)

    config.trainer.gpus = args.gpus
    print(OmegaConf.to_yaml(config))

    trainer = pl.Trainer(plugins=[NLPDDPPlugin()], **config.trainer)
    exp_dir = exp_manager(trainer, config.get("exp_manager", None))

    model_qa = nemo_nlp.models.QAModel(cfg=config.model, trainer=trainer)

    trainer.fit(model_qa)

    # model_qa.setup_test_data(test_data_config=config.model.test_ds)
    trainer.test(model_qa)
    # output_nbest_file = Path(args.pred_output) / "predict_nbest.json"
    # output_prediction_file = Path(args.pred_output) / "predict_prediction.json"
    # model_qa.inference(
    #     file=config.model.test_ds.file,
    #     batch_size=16,
    #     num_samples=-1,
    #     output_nbest_file=output_nbest_file,
    #     output_prediction_file=output_prediction_file,
    # )


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
