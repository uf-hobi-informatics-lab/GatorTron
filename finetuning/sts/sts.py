from nemo.utils.exp_manager import exp_manager
from nemo.collections import nlp as nemo_nlp
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.collections.nlp.models import GLUEModel

import os
import wget 
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from pathlib import Path
import argparse

import warnings
warnings.filterwarnings("ignore")


def main(args):
    config_path = args.config
    config = OmegaConf.load(config_path)

    config.trainer.gpus = args.gpus
    print(OmegaConf.to_yaml(config))

    trainer = pl.Trainer(plugins=[NLPDDPPlugin()], **config.trainer)
    exp_dir = exp_manager(trainer, config.get("exp_manager", None))

    model = GLUEModel(cfg=config.model, trainer=trainer)

    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1,
                        help="how many gpus to use")
    parser.add_argument("--pred_output", type=str, default=None,
                        help="where we store all the models and prediction")
    parser.add_argument("--config", type=str, required=True,
                        help="config file .yaml")


    global_args = parser.parse_args()
    main(global_args)
