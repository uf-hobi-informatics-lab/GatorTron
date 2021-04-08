## run fine-tuning

### i2b22010_megatron_345m.sh and i2b22010_megatron_345m.yaml
Example code used for execute experiment on i2b22010 dataset with megatron model

### iob_format_nv.py
this script is copied from Nvidia used to convert standard BIO formatted datasets to NeMo compatible BIO datasets

### ner_expr.py
this script is used for training and evaluate (predict) for token classification (NER) tasks based on NeMo

### conlleval.py newbio_eval.py
use for evaluation: micro precision, recall and F10-score

### nemo2bio.py
convert nemo predicted results back to stardard BIO

### generate_conll_compatible_eval.py
merge prediction with gold standard for conlleval.py