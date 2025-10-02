"""
Perform a McNemar's test between two sets of predictions
to determine whether the models' prediction distributions
are significantly different.
"""

import argparse

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions1", type=str, nargs='+',
                        help="First set of predictions files.")
    parser.add_argument("--predictions2", type=str, nargs='+',
                        help="Second set of predictions files.")
    return parser.parse_args()


def main(args):
    first = read_all_predictions(args.predictions1)
    second = read_all_predictions(args.predictions2)
    pairs = [(first, second)]
    exp_names = ["average"]
    if "ocr_text_coverage" in first.columns and "ocr_text_coverage" in second.columns:  # noqa
        has_text_mask = first["ocr_text_coverage"] > 0
        first_with_text = first[has_text_mask]
        first_no_text = first[~has_text_mask]
        second_with_text = second[has_text_mask]
        second_no_text = second[~has_text_mask]
        pairs = [(first_no_text, second_no_text),
                 (first_with_text, second_with_text)]
        exp_names = ["no_text_in_image", "text_in_image"]
    for (pair, name) in zip(pairs, exp_names):
        print(f"=== {name} ===")
        table = contingency_table(*pair)
        print(table)
        correction = any(table.flatten() < 25)
        result = mcnemar(table, exact=True, correction=correction)
        print(result)
        print()


def read_all_predictions(prediction_files):
    df = pd.DataFrame()
    for f in prediction_files:
        preds = pd.read_csv(f)
        df = pd.concat([df, preds])
    return df


def contingency_table(preds1, preds2):
    y1 = preds1.predicted_label
    gold1 = preds1.gold_labels
    y2 = preds2.predicted_label
    gold2 = preds2.gold_labels
    a = sum(np.logical_and(y1 == gold1, y2 == gold2))
    b = sum(np.logical_and(y1 == gold1, y2 != gold2))
    c = sum(np.logical_and(y1 != gold1, y2 == gold2))
    d = sum(np.logical_and(y1 != gold1, y2 != gold2))
    return np.array([[a, b],
                     [c, d]])


if __name__ == "__main__":
    main(parse_args())
