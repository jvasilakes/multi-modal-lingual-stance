import argparse

import numpy as np
import matplotlib.pyplot as plt

# Data from the table
models = sorted(["Qwen2-VL", "InternVL2", "Ovis 1.6", "LLama-Vision"])
modes = ["Text", "Image", "Text & Image"]
languages = ["en", "de", "es", "fr", "hi", "pt", "zh", "avg"]

# Scores for each model and mode (Text, Image, Text & Image)

# Prompt in same language as tweet.
scores_prompt_target = {
    "InternVL2": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh",  "avg"]
        "Text":         [0.443, 0.276, 0.191, 0.304, 0.237, 0.339, 0.394, 0.312],
        "Image":        [0.230, 0.268, 0.185, 0.257, 0.302, 0.252, 0.221, 0.245],
        "Text & Image": [0.417, 0.184, 0.254, 0.316, 0.308, 0.234, 0.342, 0.294],
    },
    "LLama-Vision": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh",  "avg"]
        "Text":         [0.224, 0.301, 0.311, 0.222, 0.235, 0.302, 0.327, 0.275],
        "Image":        [0.265, 0.282, 0.304, 0.295, 0.228, 0.290, 0.295, 0.280],
        "Text & Image": [0.333, 0.336, 0.332, 0.217, 0.225, 0.306, 0.340, 0.298],
    },
    "Ovis 1.6": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh",  "avg"]
        "Text":         [0.550, 0.247, 0.383, 0.267, 0.258, 0.356, 0.314, 0.339],
        "Image":        [0.300, 0.247, 0.300, 0.271, 0.189, 0.277, 0.224, 0.258],
        "Text & Image": [0.435, 0.384, 0.405, 0.357, 0.167, 0.358, 0.348, 0.351],
    },
    "Qwen2-VL": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh",  "avg"]
        "Text":         [0.470, 0.360, 0.198, 0.386, 0.288, 0.314, 0.265, 0.326],
        "Image":        [0.283, 0.178, 0.242, 0.309, 0.235, 0.216, 0.193, 0.237],
        "Text & Image": [0.484, 0.386, 0.237, 0.349, 0.205, 0.311, 0.273, 0.321],
    },
}

# Prompt always in English.
scores_prompt_english = {
    "InternVL2": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh",  "avg"]
        "Text":         [0.443, 0.397, 0.419, 0.418, 0.345, 0.412, 0.462, 0.414],
        "Image":        [0.234, 0.233, 0.231, 0.232, 0.231, 0.231, 0.230, 0.232],
        "Text & Image": [0.417, 0.400, 0.396, 0.414, 0.278, 0.412, 0.424, 0.392],
    },
    "LLama-Vision": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh",  "avg"]
        "Text":         [0.224, 0.196, 0.203, 0.291, 0.222, 0.225, 0.229, 0.227],
        "Image":        [0.265, 0.265, 0.265, 0.266, 0.266, 0.265, 0.265, 0.265],
        "Text & Image": [0.333, 0.293, 0.328, 0.406, 0.289, 0.306, 0.294, 0.321],
    },
    "Ovis 1.6": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh",  "avg"]
        "Text":         [0.550, 0.469, 0.465, 0.536, 0.456, 0.491, 0.454, 0.489],
        "Image":        [0.300, 0.301, 0.300, 0.300, 0.300, 0.301, 0.300, 0.300],
        "Text & Image": [0.435, 0.435, 0.436, 0.462, 0.443, 0.447, 0.441, 0.443],
    },
    "Qwen2-VL": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh",  "avg"]
        "Text":         [0.470, 0.451, 0.456, 0.444, 0.424, 0.464, 0.434, 0.449],
        "Image":        [0.283, 0.283, 0.283, 0.283, 0.283, 0.283, 0.283, 0.283],
        "Text & Image": [0.484, 0.479, 0.479, 0.482, 0.449, 0.481, 0.434, 0.470],
    },
}

# lang v English for scores_prompt_english
significances = {
    "InternVL2": {
        #               ["de",   "es",   "fr",  "hi",     "pt",   "zh"]
        "Text":         [1.8e-7, 0.0002, 0.0004, 1.9e-15, 6.4e-6, 0.060],
        "Image":        [1.0,    1.0,    1.0,    1.0,     1.0,    1.0],
        "Text & Image": [1.0,    0.122,  0.331,  0.0002,  0.174,  0.840],
    },
    "LLama-Vision": {
        #               ["de", "es",   "fr",  "hi",   "pt",   "zh"]
        "Text":         [0.186, 0.787, 0.829, 0.040,  0.157, 0.104],
        "Image":        [1.0,   1.0,   1.0,   1.0,    1.0,   1.0],
        "Text & Image": [0.003, 0.327, 0.016, 3.3e-8, 0.023, 3.5e-5],
    },
    "Ovis 1.6": {
        #               ["de",  "es",   "fr",  "hi",  "pt",  "zh"]
        "Text":         [0.080, 0.121, 0.157, 2.7e-8, 0.284, 3.9e-6],
        "Image":        [1.0,   1.0,   1.0,   1.0,    1.0,   1.0],
        "Text & Image": [0.338, 0.447, 0.225, 0.002,  0.363, 0.224],
    },
    "Qwen2-VL": {
        #               ["de", "es",   "fr",  "hi", "pt",   "zh"]
        "Text":         [0.044, 0.010, 0.007, 1.4e-6, 0.119, 0.005],
        "Image":        [1.0,   1.0,   1.0,   1.0,    1.0,   1.0],
        "Text & Image": [0.935, 0.800, 0.932, 0.028,  1.0,   1.9e-7],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, default="prompt_target_language",
                        choices=["prompt_target_language", "prompt_english"])
    return parser.parse_args()


def main(args):
    if args.experiment == "prompt_target_language":
        scores = scores_prompt_target
    elif args.experiment == "prompt_english":
        scores = scores_prompt_english

    # Prepare data for plotting
    x = np.arange(len(models))  # Model indices
    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]
    
    # Plot individual language results for each model with "avg" highlighted
    colors = ["#7fc97f", "#beaed4", "#fdc086"]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
    axi = 0
    for (ax, model) in zip(axs.flatten(), models):
        for i, mode in enumerate(modes):
            # Don't plot average
            xs = np.arange(len(languages)-1) + offsets[i]
            bars = ax.bar(xs, scores[model][mode][:-1], width=bar_width-0.01,
                          label=mode, color=colors[i])
            # Plot markers of statistical significance
            sigs = significances[model][mode]
            english_score = scores[model][mode][0]
            lang_scores = scores[model][mode][1:-1]  # skip English and average
            for (x, score, sig) in zip(xs[1:], lang_scores, sigs):
                marker_xs = []
                if sig <= 0.005:
                    marker_xs = [x-0.05, x+0.05]
                elif sig <= 0.05:
                    marker_xs = [x]
                for mx in marker_xs:
                    ax.plot(mx, score+0.03, marker='*', color="darkorchid", linestyle='', markersize=7)

        # Customization
        ax.set_ylim(0.0, 0.6)
        ax.set_xticks(np.arange(len(languages)-1))
        ax.set_xticklabels(languages[:-1], fontsize=15)
        for lab in ax.get_yticklabels():
            lab.set_fontsize(12)
        if axi % 2 == 0:
            ax.set_ylabel("Macro F1", fontsize=15)
        ax.set_title(model, fontsize=12)
        axi += 1
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=15)
    plt.suptitle(' '.join(args.experiment.split('_')).title())
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(parse_args())
