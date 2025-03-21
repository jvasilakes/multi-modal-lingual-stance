import argparse

import numpy as np
import matplotlib.pyplot as plt

# Data from the table
models = ["InternVL2", "Qwen2-VL", "Ovis 1.6", "LLama-Vision"]
modes = ["Text", "Image", "Text & Image"]
languages = ["en", "de", "es", "fr", "hi", "pt", "zh"]

# Scores for each model and mode (Text, Image, Text & Image)


# Prompt always in English.
scores_prompt_english = {
    "InternVL2": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh"]
        "Text":         [0.501, 0.400, 0.435, 0.433, 0.325, 0.425, 0.469],
        "Image":        [0.264, 0.264, 0.264, 0.264, 0.264, 0.264, 0.264],
        "Text & Image": [0.420, 0.419, 0.401, 0.406, 0.342, 0.407, 0.436],
    },
    "LLama-Vision": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh"],
        "Text":         [0.266, 0.241, 0.260, 0.260, 0.237, 0.281, 0.243],
        "Image":        [0.284, 0.284, 0.284, 0.284, 0.284, 0.284, 0.284],
        "Text & Image": [0.427, 0.394, 0.418, 0.389, 0.350, 0.398, 0.368],
    },
    "Ovis 1.6": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh"],
        "Text":         [0.590, 0.572, 0.574, 0.574, 0.522, 0.583, 0.541],
        "Image":        [0.374, 0.374, 0.374, 0.374, 0.374, 0.374, 0.374],
        "Text & Image": [0.596, 0.586, 0.585, 0.603, 0.564, 0.585, 0.582],
    },
    "Qwen2-VL": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh"],
        "Text":         [0.496, 0.476, 0.468, 0.465, 0.431, 0.478, 0.455],
        "Image":        [0.337, 0.337, 0.337, 0.337, 0.337, 0.337, 0.337],
        "Text & Image": [0.516, 0.520, 0.513, 0.515, 0.477, 0.511, 0.450],
    },
}

# p-values on English vs Text & Image
# other languages vs english
significances = {
    "InternVL2": {
        #               ["en",    "de",   "es",   "fr",   "hi",    "pt",   "zh"]
        "Text":         [6.4e-5,  1.8e-7, 0.0002, 0.0004, 2.9e-15, 6.4e-6, 0.060],
        "Image":        [1.5e-10, 1.0,    1.0,    1.0,    1.0,     1.0,    1.0],
        "Text & Image": [1.0,     1.0,    0.122,  0.331,  0.0002,  0.175,  0.840],
    },
    "LLama-Vision": {
        "Text":         [4.1e-22, 0.186,  0.787,  0.829,  0.040,   0.157,  0.104], 
        "Image":        [2.2e-17, 1.0,    1.0,    1.0,    1.0,     1.0,    1.0],
        "Text & Image": [1.0,     0.004,  0.327,  0.017,  3.3e-8,  0.023,  3.5e-5],
    },
    "Ovis 1.6": {
        "Text":         [0.126,   0.080,  0.121,  0.157,  2.7e-8,  0.284,  3.9e-6],
        "Image":        [5.2e-36, 1.0,    1.0,    1.0,    1.0,     1.0,    1.0],
        "Text & Image": [1.0,     0.338,  0.447,  0.225,  0.002,   0.363,  0.224],
    },
    "Qwen2-VL": {
        "Text":         [0.017,   0.044,  0.010,  0.007,  1.4e-6,  0.119,  0.005],   
        "Image":        [4.4e-21, 1.0,    1.0,    1.0,    1.0,     1.0,    1.0],
        "Text & Image": [1.0,     0.935,  0.800,  0.932,  0.028,   1.0,    1.9e-7],
    },
}




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, default="prompt_english",
                        choices=["prompt_english"])
    return parser.parse_args()


def main(args):
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
            xs = np.arange(len(languages)) + offsets[i]
            bars = ax.bar(xs, scores[model][mode], width=bar_width-0.01,
                          label=mode, color=colors[i])
            # Plot markers of statistical significance
            sigs = significances[model][mode][1:]
            english_score = scores[model][mode][0]
            lang_scores = scores[model][mode][1:]  # skip English
            for (x, score, sig) in zip(xs[1:], lang_scores, sigs):
                marker_xs = []
                if sig <= 0.005:
                    marker_xs = [x-0.05, x+0.05]
                elif sig <= 0.05:
                    marker_xs = [x]
                for mx in marker_xs:
                    ax.plot(mx, score+0.03, marker='*', color="darkorchid", linestyle='', markersize=7)

        # Customization
        ax.set_ylim(0.0, 0.65)
        ax.set_xticks(np.arange(len(languages)))
        ax.set_xticklabels(languages, fontsize=15)
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
