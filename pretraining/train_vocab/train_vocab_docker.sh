#! /bin/bash

mapings=<local_path>:/workspace

TEXT=/workspace/full_text.txt
OUTPUT=/workspace
VOCAB_SIZE=30000
BERT_HEADER=/workspace/bert_vocab_head.txt
PREF=full_uncased_30k

docker run -d --ipc=host -v $mapings <docker image ID> python /workspace/train_vocab.py \
    --input $TEXT \
    --prefix $PREF \
    --output $OUTPUT \
    --vocab_size $VOCAB_SIZE \
    --threads 32 \
    --lower_case \
    --bert_header $BERT_HEADER

docker logs -f <docker container ID> &> train_vocab_log.txt &