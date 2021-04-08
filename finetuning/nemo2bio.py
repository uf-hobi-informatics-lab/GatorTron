# -*- coding: utf-8 -*-

"""
The script is used to map the bio output from Nemo to standard BIO format 

We need to use the conll evaluation script which only takes standard BIO format to measure the performance

The script will not handle remap splitted sentence back to originial since when we split the long sentence, we did not 
set up a strategy to do this.  We will just treat the nemo dataset as the formal data set used for performance measurement
"""

from pathlib import Path
import argparse


def read_text(fn):
    with open(fn, "r") as f:
        text = f.read()
    return text


def t2l(text):
    return [each.split(" ") for each in text.strip().split("\n")]


def nemo2bio(nemo_dir, bio_file, nty="test"):
    text_nemo = nemo_dir / f"text_{nty}.txt"
    label_nemo = nemo_dir / f"labels_{nty}.txt"
    
    words = t2l(read_text(text_nemo))
    labels = t2l(read_text(label_nemo))
    
    assert len(words) == len(labels), "labels and words have different dim at sentence level"
    
    bio = []
    for ws, ls in zip(words, labels):
        assert len(ws) == len(ls), "labels and words have different dim at word level"
        bio.append(list(zip(ws, ls)))
    
    with open(bio_file, "w") as f:
        for sent in bio:
            f.write("\n".join([f"{w} {l}" for (w, l) in sent]))
            f.write("\n\n")


def main(args):
    nemo_input_dir = Path(args.nemo)
    bio_output = args.bio

    nemo2bio(nemo_input_dir, bio_output, args.type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo", type=str, required=True, help="the nemo output prediction dir (text and labels)")
    parser.add_argument("--bio", type=str, required=True, help="the output file with standard BIO format")
    parser.add_argument("--type", type=str, default="test", help="what kind of data: train, dev, test or a special prefix")
    args = parser.parse_args()
    main(args)