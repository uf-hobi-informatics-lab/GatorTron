from nemo.collections import nlp as nemo_nlp
from nemo.utils.exp_manager import exp_manager
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
import traceback

import os
import wget 
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from pathlib import Path
import argparse

import sys
sys.path.append(Path(__file__).parent.absolute())

from new_bio_eval import BioEval


def load_gs_labels(label_file):
  labels = []
  with open(label_file, "r") as f:
    lines = f.readlines()
  for line in lines:
    line = line.strip()
    if len(line) < 1:
      continue
    labels.append(line.split(" "))

  return labels


def predict_in_sent(text_file, model_ner, bz=32, combine=True):
    idx2label = {v: k for k, v in model_ner.cfg.label_ids.items()}

    res = []
    with open(text_file, "r") as f:
        lines = f.readlines()
    
    ll = [len(line.split(" ")) for line in lines]
    all_preds = model_ner._infer(lines, bz)

    for idx in ll:
        tmp = [idx2label[each] for each in all_preds[:idx]]
        if combine:
            res.append(" ".join(tmp))
        else:
            res.append(tmp)
        all_preds = all_preds[idx:]

    return res


def main(args):
    config_path = args.ner_config
    config = OmegaConf.load(config_path)

    config.trainer.gpus = args.gpus
    print(OmegaConf.to_yaml(config))

    trainer = pl.Trainer(plugins=[NLPDDPPlugin()], **config.trainer)
    exp_dir = exp_manager(trainer, config.get("exp_manager", None))
    model_ner = None
    
    if args.do_train:
        if not config.pretrained_model:
            model_ner = nemo_nlp.models.TokenClassificationModel(cfg=config.model, trainer=trainer)
        else:
            model_ner = nemo_nlp.models.TokenClassificationModel.restore_from(config.pretrained_model)

        trainer.fit(model_ner)

        try:
            if config.model.nemo_path:
                model_ner.save_to(config.model.nemo_path)
        except:
            traceback.print_exc()
            config.model.nemo_path = Path(str(exp_dir)) / "checkpoints/default.nemo"

        dev_text = Path(config.model.dataset.data_dir) / config.model.test_ds.text_file
        dev_label = Path(config.model.dataset.data_dir) / config.model.test_ds.labels_file 
        dev_preds = predict_in_sent(dev_text, model_ner, bz=config.model.test_ds.batch_size, combine=False)

        try:
            dev_gs = load_gs_labels(dev_label)
            bio_eval = BioEval()
            bio_eval.eval_mem(dev_gs, dev_preds, do_flat=False)
            bio_eval.show_evaluation()
        except:
            pass
        
        dev_preds = predict_in_sent(dev_text, model_ner, bz=config.model.test_ds.batch_size, combine=True)
        pout = Path(args.pred_output)
        pout.mkdir(exist_ok=True)
        ofn = pout / "labels_test.txt"
        with open(ofn, "w") as f:
            f.write("\n".join(dev_preds))

    if args.do_pred and args.pred_output:
        trainer = pl.Trainer(plugins=[NLPDDPPlugin()], **config.trainer)
        exp_dir = exp_manager(trainer, config.get("exp_manager", None))

        try:
            model_ner = nemo_nlp.models.TokenClassificationModel.restore_from(config.model.nemo_path, trainer=trainer, strict=False)
        except:
            traceback.print_exc()
            config.model.nemo_path = str(exp_dir) + "/checkpoints/default.nemo"
            model_ner = nemo_nlp.models.TokenClassificationModel.restore_from(config.model.nemo_path, trainer=trainer, strict=False)
       
        text_file = Path(config.model.dataset.data_dir) / config.model.test_ds.text_file

        res = predict_in_sent(text_file, model_ner, combine=True)

        test_label = Path(config.model.dataset.data_dir) / config.model.test_ds.labels_file
        test_gs = load_gs_labels(test_label)
        bio_eval = BioEval()
        bio_eval.eval_mem(test_gs, res, do_flat=False)
        bio_eval.show_evaluation()
        
        pout = Path(args.pred_output)
        pout.mkdir(exist_ok=True)
        ofn = pout / "labels_test1.txt"
        with open(ofn, "w") as f:
            f.write("\n".join(res))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_output", type=str, default=None,
                        help="where we store all the models and prediction")
    parser.add_argument("--ner_config", type=str, required=True,
                        help="config file .yaml")
    parser.add_argument("--gpus", type=int, default=1,
                        help="how many gpus to use")
    parser.add_argument("--do_train", action='store_true',
                        help="training model")
    parser.add_argument("--do_dev", action='store_true',
                        help="run evaluation using f1 on dev set")
    parser.add_argument("--do_pred", action='store_true',
                        help="run prediction on test data")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="local rank")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="local rank")
    parser.add_argument("--nnodes", type=int, default=1,
                        help="local rank")

    global_args = parser.parse_args()
    main(global_args)