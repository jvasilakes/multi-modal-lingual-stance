import json
import argparse
import warnings

import torch
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_file_1", type=str)
    parser.add_argument("predictions_file_2", type=str)
    return parser.parse_args()


def main(args):
    predictions1 = pd.read_csv(args.predictions_file_1)
    predictions2 = pd.read_csv(args.predictions_file_2)
    assert set(predictions1.stance_targets) == set(predictions2.stance_targets)
    assert len(predictions1) == len(predictions2)
    all_results = {}
    for stance_target in predictions1.stance_targets.unique():
        stance_preds1 = predictions1[predictions1.stance_targets == stance_target]  # noqa
        stance_preds2 = predictions2[predictions2.stance_targets == stance_target]  # noqa
        results = compare_predictions(stance_preds1, stance_preds2)
        all_results[stance_target] = results
    print(json.dumps(all_results, indent=2))


def compare_predictions(df1, df2):
    logit_columns1 = [col for col in df1.columns if col.startswith("logits_")]
    label_set1 = [col.split('_')[-1] for col in logit_columns1]
    logit_columns2 = [col for col in df2.columns if col.startswith("logits_")]
    label_set2 = [col.split('_')[-1] for col in logit_columns2]
    logits1 = df1[logit_columns1].values
    logits2 = df2[logit_columns2].values
    pred_idxs1 = logits1.argmax(1)
    pred_labels1 = [label_set1[i] for i in pred_idxs1]
    pred_idxs2 = logits2.argmax(1)
    pred_labels2 = [label_set2[i] for i in pred_idxs2]

    labels = ['/'.join([lab1, lab2]) for (lab1, lab2) in zip(label_set1, label_set2)]
    ps, rs, fs, _ = precision_recall_fscore_support(
            pred_idxs1, pred_idxs2, labels=range(len(labels)),
            zero_division=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa = cohen_kappa_score(pred_idxs1, pred_idxs2)
    supports = []
    for (lab1, lab2) in zip(label_set1, label_set2):
        supp1 = len([pl for pl in pred_labels1 if pl == lab1])
        supp2 = len([pl for pl in pred_labels2 if pl == lab2])
        supports.append([supp1, supp2])

    ce = cross_entropy(logits1, logits2)
    results = {}
    for (i, lab) in enumerate(labels):
        p = ps[i]
        r = rs[i]
        f = fs[i]
        s = supports[i]
        results[lab] = {"precision": p,
                        "recall": r,
                        "F1": f,
                        "support": s}
    results["kappa"] = kappa
    results["cross_entropy"] = ce
    return results


def cross_entropy(true_logits, pred_logits):
    return torch.nn.functional.cross_entropy(
            torch.tensor(pred_logits), torch.tensor(true_logits)).item()


if __name__ == "__main__":
    main(parse_args())
