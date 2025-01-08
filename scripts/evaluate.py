import os
import json
import argparse

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_files", nargs='+', type=str)
    parser.add_argument("--per_target", action="store_true", default=False,
                        help="Compute metrics per stance target")
    return parser.parse_args()


def main(args):
    predictions = pd.concat([
        pd.read_csv(f) for f in args.predictions_files])
    all_results = {}
    if args.per_target is True:
        all_predictions = []
        names = []
        for stance_target in predictions.stance_targets.unique():
            names.append(stance_target)
            all_predictions.append(predictions[predictions.stance_targets == stance_target])
    else:
        all_predictions = [predictions]
        names = ["predictions"]
    for (name, preds) in zip(names, all_predictions):
        if "ocr_text_coverage" in preds.columns:
            no_text_preds = preds[preds.ocr_text_coverage == 0.0]
            no_text_results = evaluate(no_text_preds)
            with_text_preds = preds[preds.ocr_text_coverage > 0.0]
            with_text_results = evaluate(with_text_preds)
            results = {"no_text_in_image": no_text_results,
                       "text_in_image": with_text_results}
        else:
            results = evaluate(preds)
        all_results[name] = results
    if len(args.predictions_files) == 1:
        pred_file = args.predictions_files[0]
        basename = os.path.basename(pred_file)
        datasplit = os.path.splitext(basename)[0]
        outdir = os.path.join(os.path.dirname(pred_file), "results")
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f"{datasplit}.json")
        with open(outfile, 'w') as outF:
            json.dump(all_results, outF, indent=2)
    else:
        for (name, results) in all_results.items():
            key = list(results.keys())[0]
            nested = isinstance(results[key], dict)
            if nested is False:
                results = [results]
            print(name)
            print(pd.DataFrame(results).T.to_markdown())


def evaluate(df):
    logit_columns = [col for col in df.columns if col.startswith("logits_")]
    label_set = [col.split('_')[-1] for col in logit_columns]
    logits = df[logit_columns].values
    #pred_idxs = logits.argmax(1)
    #pred_labels = [label_set[i] for i in pred_idxs]
    p, r, f, _ = precision_recall_fscore_support(
            #df.gold_labels, pred_labels, average="macro")
            df.gold_labels, df.predicted_label, average="macro")
    support = len(df)

    one_hot = np.zeros((support, len(label_set)))
    gold_idxs = [label_set.index(lab) for lab in df.gold_labels]
    one_hot[np.arange(len(one_hot)), gold_idxs] = 1.
    ce = cross_entropy(one_hot, logits)
    results = {"precision": p,
               "recall": r,
               "F1": f,
               "support": support,
               "cross_entropy": ce}
    return results


def cross_entropy(true_logits, pred_logits):
    return torch.nn.functional.cross_entropy(
            torch.tensor(pred_logits), torch.tensor(true_logits)).item()


if __name__ == "__main__":
    main(parse_args())
