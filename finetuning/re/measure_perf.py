from collections import defaultdict
import argparse


class PRF:
    def __init__(self):
        self.tp = 0
        self.fp = 0
    
    def __repr__(self):
        return f"tp: {self.tp}; fp: {self.fp}"


def calc_prf(gs, pred, out, labels=['0', '1'], neg="0"):
    gl = []
    pl = []
    tn_cnt_dict = defaultdict(lambda: 0)
    
    d = dict()
    for l in labels:
        d[l] = PRF()
    
    with open(gs, "r") as f:
        for line in f.readlines():
            l = line.strip().split("\t")[-1]
            gl.append(l)
    with open(pred, "r") as f:
        for line in f.readlines():
            pl.append(line.strip())
    
    assert len(gl) == len(pl)
    
    for k in labels:
        for g, p in zip(gl, pl):
            if k == g:
                tn_cnt_dict[g] += 1

            if g == p == k:
                d[k].tp += 1
            elif g != k and p == k:
                d[k].fp += 1
    
    print(d, tn_cnt_dict)
    
    res = dict()
    ac_tp = 0
    ac_tp_fp = 0
    ac_tp_fn = 0
    for l in labels:
        tp = d[l].tp
        tp_fp = tp + d[l].fp
        pre = tp / tp_fp
        rec = tp / tn_cnt_dict[l]
        if pre != 0 or rec != 0:
            f1 = 2*pre*rec / (pre + rec)
        else:
            f1 = 0
        res[l] = (pre, rec, f1)
        
        if l != neg:
            ac_tp += tp
            ac_tp_fp += tp_fp
            ac_tp_fn += tn_cnt_dict[l]
    
    pre = ac_tp / ac_tp_fp
    rec = ac_tp / ac_tp_fn
    if pre != 0 or rec != 0:
        f1 = 2*pre*rec / (pre + rec)
    else:
        f1 = 0
    
    res['mico_avg'] = (pre, rec, f1)
    
    print(res)

    with open(out, "w") as f:
        for k, v in res.items():
            temp = f"{k} - pre: {v[0]}, rec: {v[1]}, f1: {v[2]}\n"
            f.write(temp)


def main(args):
    calc_prf(args.gs, args.pred, args.output, args.labels, args.neg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True,
                        help="predicted results")
    parser.add_argument("--gs", type=str, required=True,
                        help="gold standard")
    parser.add_argument("--output", type=str, required=True,
                        help="results output")
    parser.add_argument("--neg", type=str, default="0",
                        help="negative category not for evaluation")
    parser.add_argument('--labels', metavar='N', type=str, nargs='+',
                    help='relation categories')


    global_args = parser.parse_args()
    main(global_args)
