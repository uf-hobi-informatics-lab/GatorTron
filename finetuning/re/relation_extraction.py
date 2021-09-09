from nemo.collections import nlp as nemo_nlp
from nemo.utils.exp_manager import exp_manager
from nemo.collections.nlp.parts.utils_funcs import tensor2list

import os
import wget 
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

import torch

from pathlib import Path
import argparse


def load_test(test_fn):
    quries = []
    with open(test_fn, "r") as f:
        for line in f.readlines():
            quries.append(line.split("\t")[0].strip())
    return quries


def main(args):
    config_path = args.re_config
    config = OmegaConf.load(config_path)

    config.trainer.gpus = args.gpus
    print(OmegaConf.to_yaml(config))

    trainer = pl.Trainer(**config.trainer)
    exp_dir = exp_manager(trainer, config.get("exp_manager", None))

    model = nemo_nlp.models.TextClassificationModel(config.model, trainer=trainer)

    trainer.fit(model)

    try:
        model.setup_test_data(test_data_config=config.model.test_ds)
        trainer.test(model)
    except Exception as ex:
        pass

    # test_data_loader = model._setup_dataloader_from_config(cfg=config.model.test_ds)

    # all_preds = []

    # device = next(model.parameters()).device
    # model.eval()
    
    # for i, batch in enumerate(test_data_loader):
    #     input_ids, input_type_ids, input_mask, subtokens_mask = batch

    #     logits = model.forward(
    #         input_ids=input_ids.to(device),
    #         token_type_ids=input_type_ids.to(device),
    #         attention_mask=input_mask.to(device),
    #     )

    #     preds = tensor2list(torch.argmax(logits, axis=-1))
    #     all_preds.extend(preds)

    quries = load_test(config.model.test_ds.file_path)
    all_preds = model.classifytext(quries, config.model.test_ds.batch_size, config.model.dataset.max_seq_length)

    output_path = Path(args.pred_output)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path/"predict_labels.txt", "w") as f:
        f.write("\n".join([str(e) for e in all_preds]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_output", type=str, default=None,
                        help="where we store all the models and prediction")
    parser.add_argument("--re_config", type=str, required=True,
                        help="config file .yaml")
    parser.add_argument("--gpus", type=int, default=1,
                        help="how many gpus to use")


    global_args = parser.parse_args()
    main(global_args)
