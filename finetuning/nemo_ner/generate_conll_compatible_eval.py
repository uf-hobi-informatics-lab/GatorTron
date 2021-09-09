# run with python2
from pathlib import Path
import argparse


def ff(data):
    nsents = []
    for sent in data.split("\n\n"):
        nsent = []
        for each in sent.split("\n"):
            nsent.append(each.split(" "))
        nsents.append(nsent)
    return nsents


def main(args):
    with open(args.gs, "r") as f1, open(args.pred, "r") as f2:
        d1 = ff(f1.read())
        d2 = ff(f2.read())

    ss = []
    for s1, s2 in zip(d1, d2):
        s = []
        for w1, w2 in zip(s1, s2):
            w1.append(w2[-1])
            s.append(w1)
        ss.append(s)

    with open(args.output, "w") as f:
        for ns in ss:
            f.write("\n".join([" ".join(each) for each in ns]))
            f.write("\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs", type=str, required=True, help="gold standard")
    parser.add_argument("--pred", type=str, required=True, help="prediction")
    parser.add_argument("--output", type=str, default="test", help="output")
    args = parser.parse_args()
    main(args)