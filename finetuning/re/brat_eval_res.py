import argparse
import pickle
from collections import defaultdict


mappings = {
  'Duration': 'Duration-Drug',
  'Route': 'Route-Drug',
  'Strength': 'Strength-Drug',
  'ADE': 'ADE-Drug',
  'Form': 'Form-Drug',
  'Reason': 'Reason-Drug',
  'Dosage': 'Dosage-Drug',
  'Frequency': 'Frequency-Drug'
}


class PRF:
    def __init__(self):
        self.tp = 0
        self.fp = 0
    
    def __repr__(self):
        return f"tp: {self.tp}; fp: {self.fp}"


def calc(tp, tp_fp, tp_tn):
    if tp_fp != 0:
        pre = tp / tp_fp
    else:
        pre = 0
    
    if tp_tn == 0:
        rec =0
    else:
        rec = tp / tp_tn
    
    if pre == 0 and rec == 0:
        f1 = 0
    else:
        f1 = 2*pre*rec / (pre + rec)
    
    return pre, rec, f1


def measure_prf(gss, preds):
    tp_tn_dict = defaultdict(lambda: 0)
    # get tp+tn counts for each category
    for each in gss:
        tp_tn_dict[each[0]] += 1

    tp_fp_cnt = defaultdict(PRF)

    total_tp, total_tp_fp, total_tp_tn = 0, 0, 0
    res = dict()

    for each in preds:
        rel_type = each[0]
        for_eval = (each[0], *each[3:])
        if rel_type == "NonRel":
            rel_type = mappings[each[1]]
        
        if for_eval in gss:
            tp_fp_cnt[rel_type].tp +=1
        else:
            tp_fp_cnt[rel_type].fp +=1

    for k, v in tp_fp_cnt.items():
        tp_tn = tp_tn_dict[k]
        tp = v.tp
        fp = v.fp
        tp_fp = tp + fp

        pre, rec, f1 = calc(tp, tp_fp, tp_tn)
        
        res[k] = (pre, rec, f1)

        total_tp += tp
        total_tp_fp += tp_fp
        total_tp_tn += tp_tn

    res['micro_avg'] = calc(total_tp, total_tp_fp, total_tp_tn)

    return res


def main(args):
    # load supp
    supps = []
    with open(args.supp, "r") as f:
        for line in f.readlines():
            l = line.strip().split("\t")
            supps.append(l[2:])

    preds = []
    # load prediction
    with open(args.pred, "r") as f:
        for line in f.readlines():
            preds.append(line.strip())

    assert len(preds) == len(supps)

    # merge prediction with supp
    # create set of prediction with format (rel_type, en1, en2, en1_id, en2_id, fid)
    preds_for_eval = []
    for p, s in zip(preds, supps):
        if p == "1":
            preds_for_eval.append(s)

    # load gold standard set
    with open(args.gs, "rb") as f:
        gs_set = pickle.load(f)

    #evaluate
    res = measure_prf(gs_set, preds_for_eval)
    print(res)

    # output
    with open(args.output, "w") as f:
        for k, v in res.items():
            temp = f"{k} - pre: {v[0]}, rec: {v[1]}, f1: {v[2]}\n"
            f.write(temp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred", type=str, required=True,
                        help="predicted results")
    parser.add_argument("--supp", type=str, required=True,
                        help="support files to map predicted results")
    parser.add_argument("--gs", type=str, required=True,
                        help="gold standard")
    parser.add_argument("--output", type=str, required=True,
                        help="results output")


    global_args = parser.parse_args()
    main(global_args)