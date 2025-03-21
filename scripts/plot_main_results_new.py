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
