"""
convert csv to json
preprocessing text
"""
import argparse
import traceback
import logging
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial

import sys
sys.path.append("./NLPreprocessing")
sys.path.append("./NLPreprocessing/text_process")
# NLPreprocessing is an internal pakcage 
# NLPreprocessing is for text normalization
# NLPreprocessing is mainly designed for UF notes
# separate punct from words, setence and word tokenizations, special cases handling
# we cannot share the source code for this package
# you can replace with any text normalization tools

from text_process.sentence_tokenization import SentenceBoundaryDetection
from text_process.sentence_tokenization import logger
logger.setLevel(logging.ERROR)


def d2t(x, tok, args):
    return json.dumps({'NOTE_TEXT': tok.sent_tokenizer(txt=x[args.col_name])})


def helper(x, args):
     tok = SentenceBoundaryDetection()
     tok.special = True
     x = x.dropna()
     x = x.drop_duplicates(subset=[args.col_name], ignore_index=True)
     processed = [d2t(v, tok, args) for _, v in x.to_dict('index').items()]
     return processed


def main(args):
    input_fn = args.input
    output_fn = args.output

    try:
        with open(output_fn, "w") as f:
            with ProcessPoolExecutor(max_workers=args.cpus) as exe:
                for chunk in exe.map(partial(helper, args=args), pd.read_csv(input_fn, dtype=str, chunksize=args.chunk_size, sep=args.sep)):
                    [f.write(each + "\n") for each in chunk]
    except Exception as ex:
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="input tsv file")
    parser.add_argument("--output", type=str, required=True,
                        help="output json file")
    parser.add_argument("--sep", default="\t", type=str,
                        help="',' for csv; '\\t' for tsv")
    parser.add_argument("--cpus", default=32, type=int,
                        help="using how many CPUs for parallel")
    parser.add_argument("--chunk_size", default=20000, type=int,
                        help="num of lines read from tsv/csv as chunk for processing")
    parser.add_argument("--col_name", type=str, default="NOTE_TEXT",
                        help="output json file")
    global_args = parser.parse_args()

    main(global_args)