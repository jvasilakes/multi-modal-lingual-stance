import json
import argparse
from collections import defaultdict

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_files", type=str, nargs='+',
                        help="prediction JSON files to average.")
    parser.add_argument("--compare_to_results_files", type=str, nargs='*',
                        help="prediction JSON files to compare.")
    parser.add_argument("--average_over_targets",
                        action="store_true", default=False,
                        help="Average over stance targets.")
    return parser.parse_args()


def main(args):
    raise ValueError("Deprecated. Use evaluate.py")
    averages = average_results(args.results_files, args.average_over_targets)
    if args.compare_to_results_files is not None:
        comp_averages = average_results(args.compare_to_results_files,
                                        args.average_over_targets)
        comparison = averages - comp_averages
        averages = pd.concat([averages, comp_averages, comparison], axis=1)
        averages.columns = ["results", "comp_results", "delta"]
    averages = averages.round(3)
    averages = averages.reset_index().rename(
            columns={"index": '', "level_0": '', "level_1": ''})
    print(averages.to_markdown(index=False))


def average_results(results_files, average_over_targets=False):
    data = [json.load(open(f)) for f in results_files]

    results_by_stance = defaultdict(list)
    for datum in data:
        for (stance_target, results) in datum.items():
            results_by_stance[stance_target].append(results)

    averages = {}
    for (stance_target, results) in results_by_stance.items():
        dfs = [pd.DataFrame.from_dict(d, orient="index") for d in results]
        nested = isinstance(list(results[0].values())[0], dict)
        if nested is False:
            dfs = [df.T for df in dfs]
        avgs = pd.concat(dfs).groupby(level=0).mean()
        averages[stance_target] = avgs
    # Make the stance target into the first level of a multi index.
    averages = pd.concat(averages)
    if average_over_targets is True:
        if nested is True:
            grouped = averages.groupby(level=1)
            support = grouped.sum()["support"]
            averages = grouped.mean()
            averages["support"] = support
        else:
            support = averages["support"].sum()
            averages = averages.mean(0)
            averages["support"] = support
    return averages


if __name__ == "__main__":
    main(parse_args())
