import os
import json
import argparse
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison_results_dir", type=str,
                        help="Directory containing model/dataset/*.json")
    return parser.parse_args()


def main(args):
    model_nicknames = {
            "InternVL2-8B": "InternVL2",
            "Llama-3.2-11B-Vision-Instruct": "Llama-Vision",
            "Ovis1.6-Gemma2-9B": "Ovis 1.6",
            "Qwen2-VL-7B-Instruct": "Qwen2-VL"}
    langs = ["en", "de", "es", "fr", "hi", "pt", "zh"]
    models = os.listdir(args.comparison_results_dir)

    avg_kappas = defaultdict(lambda: defaultdict(dict))
    for model in models:
        model_dir = os.path.join(args.comparison_results_dir, model)
        for lang1 in langs:
            for lang2 in langs:
                if lang1 == lang2:
                    avg_kappas[model][lang1][lang2] = 1.0
                    continue
                inpath = os.path.join(model_dir, f"*/{lang1}_{lang2}.json")
                infiles = glob(inpath)
                kappas = []
                for results_file in infiles:
                    results = json.load(open(results_file))
                    for (stance_target, metrics) in results.items():
                        kappas.append(metrics["kappa"])
                avg_kappas[model][lang1][lang2] = np.nanmean(kappas)

    heatmap_dfs = {}
    for (model, data) in avg_kappas.items():
        df = pd.DataFrame.from_dict(data)
        heatmap_dfs[model] = df

    fig, axs = plt.subplots(ncols=len(models))
    for (i, model) in enumerate(sorted(models)):
        df = heatmap_dfs[model].round(2)
        df[df == -0.0] = 0.0
        use_cbar = True if i == (len(models) - 1) else False
        sns.heatmap(df, annot=True, ax=axs[i], xticklabels=True,
                    yticklabels=True, cbar=use_cbar, square=True,
                    vmin=0, vmax=1, annot_kws={"fontsize": 12})
        xticklabs = axs[i].get_xticklabels()
        yticklabs = axs[i].get_yticklabels()
        axs[i].set_xticklabels(xticklabs, fontsize=12, rotation=0)
        axs[i].set_yticklabels(yticklabs, fontsize=12)
        model_short = model_nicknames[model]
        axs[i].set_title(model_short, fontsize=15)
        if use_cbar is True:
            cbar = axs[i].collections[0].colorbar
            cbar.ax.tick_params(labelsize=12)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


if __name__ == "__main__":
    main(parse_args())
