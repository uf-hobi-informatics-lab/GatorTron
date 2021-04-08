"""
Author: bugface https://github.com/bugface

The script is based on Google's SentencePiece to train a vocab from local corpous
see more details at https://github.com/google/sentencepiece

note: training with a large corpus may take TB-level of RAM; sentencepiece can limit input number of sentences, we did not use 
in this project.
"""


import sentencepiece as spm
from pathlib import Path
import argparse


def read_text(fn):
    with open(fn, "r") as f:
        text = f.read().strip()
    return text


def write_text(text, fn):
    with open(fn, "w") as f:
        f.write(text)


def main(args):
    mn = args.prefix
    output = args.outpu
    data = args.input
    bert_head = args.bert_header
    pref = f"{output}/{mn}"
    vsz = args.vocab_size

    p = Path(f"{pref}")
    p.mkdir(parents=True, exist_ok=True)

    if args.lower_case:
        rule = 'nmt_nfkc_cf'
    else:
        rule = 'nmt_nfkc'

    spm.SentencePieceTrainer.Train(
        f'--input={data} ' \
        '--input_format=text ' \
        f'--model_prefix={pref}/{mn} ' \
        f'--vocab_size={vsz} ' \
        f'--normalization_rule_name={rule} ' \
        '--character_coverage=0.9999 ' \
        '--model_type=bpe ' \
        '--train_extremely_large_corpus=true ' \
        '--self_test_sample_size=100' \
        '--max_sentencepiece_length=128' \
        '--max_sentence_length=33536' \
        '--hard_vocab_limit=false' \
        f'--num_threads={args.threads}'
    )


    bert_header = read_text(bert_head).strip().split("\n")

    exclude = {'[UNK]', "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<unk>", "<s>", "</s>", "<pad>", "<cls>", "<sep>"}
    nv = [each.split("\t")[0] for each in read_text(pref+f"/{mn}.vocab").strip().split("\n")]
    nv = [each for each in nv if each not in exclude]
    nnv = [each.replace("▁", "") if each.startswith("▁") else "##"+each for each in nv]

    bert_vocab = bert_header + nnv

    # output dir
    with open(p/"vocab.txt", "w") as f:
        f.write("\n".join(bert_vocab))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="input text file for training vocab")
    parser.add_argument("--prefix", type=str, required=True,
                        help="the prefix of the output file - vocab name")
    parser.add_argument("--output", type=str, default="./GatorTron/vocabs",
                        help="output path for trained vocab files")
    parser.add_argument("--bert_header", type=str, default="./bert_vocab_head.txt",
                        help="the standard bert vocab special tags - like [CLS] [SEP] [PAD] [unused1-99]")
    parser.add_argument("--vocab_size", default=32000, type=int,
                        help="targeted vocab size")
    parser.add_argument("--threads", default=32, type=int,
                        help="number of threads used for training")
    parser.add_argument("--lower_case", action='store_true',
                        help="set training to use lower case for all text")
    global_args = parser.parse_args()
    main(global_args)