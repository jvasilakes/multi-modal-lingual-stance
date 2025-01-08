import argparse

import numpy as np
import matplotlib.pyplot as plt

# Data from the table
models = ["InternVL2", "Qwen2-VL", "Ovis 1.6", "LLama-Vision"]  # ordered by model size
modes = ["Text", "Image", "Text & Image"]
languages = ["en", "de", "es", "fr", "hi", "pt", "zh", "avg"]

# Macro F1 for each model and mode (Text, Image, Text & Image)

# Prompt always in English.
scores_prompt_english = {
    "InternVL2": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh",  "avg"]
        "Text":         [0.501, 0.276, 0.191, 0.304, 0.237, 0.339, 0.394, 0.312],
        "Image":        [0.262, 0.268, 0.185, 0.257, 0.302, 0.252, 0.221, 0.245],
        "Text & Image": [0.420, 0.184, 0.254, 0.316, 0.308, 0.234, 0.342, 0.294],
    },
    "LLama-Vision": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh",  "avg"]
        "Text":         [0.266, 0.301, 0.311, 0.222, 0.235, 0.302, 0.327, 0.275],
        "Image":        [0.284, 0.282, 0.304, 0.295, 0.228, 0.290, 0.295, 0.280],
        "Text & Image": [0.427, 0.336, 0.332, 0.217, 0.225, 0.306, 0.340, 0.298],
    },
    "Ovis 1.6": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh",  "avg"]
        "Text":         [0.590, 0.247, 0.383, 0.267, 0.258, 0.356, 0.314, 0.339],
        "Image":        [0.374, 0.247, 0.300, 0.271, 0.189, 0.277, 0.224, 0.258],
        "Text & Image": [0.596, 0.384, 0.405, 0.357, 0.167, 0.358, 0.348, 0.351],
    },
    "Qwen2-VL": {
        #               ["en",  "de", "es",   "fr",  "hi", "pt",   "zh",  "avg"]
        "Text":         [0.496, 0.360, 0.198, 0.386, 0.288, 0.314, 0.265, 0.326],
        "Image":        [0.337, 0.178, 0.242, 0.309, 0.235, 0.216, 0.193, 0.237],
        "Text & Image": [0.516, 0.386, 0.237, 0.349, 0.205, 0.311, 0.273, 0.321],
    },
}

# p-values on English vs Text & Image
significances = {
    "InternVL2": {
        "Text": 6.4e-5,
        "Image": 1.5e-10,
        "Text & Image": 1.0,
    },
    "LLama-Vision": {
        "Text": 4.1e-22,
        "Image": 2.2e-17,
        "Text & Image": 1.0,
    },
    "Ovis 1.6": {
        "Text": 0.126,
        "Image": 5.2e-36,
        "Text & Image": 1.0,
    },
    "Qwen2-VL": {
        "Text": 0.017,
        "Image": 4.4e-21,
        "Text & Image": 1.0,
    },
}



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, default="prompt_target_language",
                        choices=["prompt_english", "covered_image_text"])
    return parser.parse_args()


def main(args):
    if args.experiment == "prompt_target_language":
        scores = scores_prompt_target
    elif args.experiment == "prompt_english":
        scores = scores_prompt_english
    elif args.experiment == "covered_image_text":
        scores = scores_covered_image_text

    # Prepare data for plotting
    xs = np.arange(len(models))  # Model indices
    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = ["#7fc97f", "#beaed4", "#fdc086"]
    for i, mode in enumerate(modes):
        #avg_scores = [scores[model][mode][-1] for model in models]
        en_scores = [scores[model][mode][0] for model in models]
        xs_ = xs + offsets[i]
        ax.bar(xs_, en_scores,
               width=bar_width-0.05, label=mode, color=colors[i])
        for (model, x, score) in zip(models, xs_, en_scores):
            txt_img_score = scores[model]["Text & Image"][0]
            sig = significances[model][mode]

            marker_xs = []
            if sig <= 0.005:
                marker_xs = [x-0.03, x+0.03]
            elif sig <= 0.05:
                marker_xs = [x]
            for mx in marker_xs:
                ax.plot(mx, score+0.01, marker='*', color="darkorchid", linestyle='', markersize=10)

    # Customization
    ax.set_xticks(xs)
    ax.set_xticklabels(models, fontsize=20)
    yticklabs = ax.get_yticklabels()
    ax.set_yticklabels(yticklabs, fontsize=15)
    ax.set_ylabel("Macro F1", fontsize=20)
    ax.set_title(' '.join(args.experiment.split('_')).title())
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=20)
    plt.tight_layout()
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    main(parse_args())
