import os
import random
import argparse
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

from config import config
from src.data.util import get_datamodule, modify_labels
from src.modeling.util import get_model


torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    train_parser = subparsers.add_parser("train", help="Run model training")
    train_parser.add_argument(
        "config_file", type=str, help="Path to yaml config file.")

    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument(
        "config_file", type=str, help="Path to yaml config file.")
    predict_parser.add_argument("--split", type=str, default="validation",
                                choices=["train", "validation", "test"])

    return parser.parse_args()


def main(args):
    config.load_yaml(args.config_file)

    start_time = datetime.now()
    print(f"Experiment start: {start_time}")
    print()
    print(config)

    random.seed(config.Experiment.random_seed.value)
    np.random.seed(config.Experiment.random_seed.value)
    torch.manual_seed(config.Experiment.random_seed.value)

    run_kwargs = {}
    if args.command == "train":
        run_fn = run_train
    elif args.command == "predict":
        run_fn = run_predict
        run_kwargs["datasplit"] = args.split
    else:
        raise argparse.ArgumentError(f"Unknown command '{args.command}'.")
    run_fn(config, **run_kwargs)

    end_time = datetime.now()
    print()
    print(f"Experiment end: {end_time}")
    print(f"  Time elapsed: {end_time - start_time}")


def run_train(config):
    raise NotImplementedError()


def run_predict(config, datasplit="validation"):
    datamodule = get_datamodule(config)
    model = get_model(config)

    outdir = config.Experiment.output_dir.value
    os.makedirs(outdir, exist_ok=True)

    all_outputs = {"prompts": [],
                   "images": [],
                   "generated_text": [],
                   "predicted_label": [],
                   "gold_labels": [],
                   "stance_targets": []}
    # We'll format this to save in the dataframe later
    all_logits = []
    labels = datamodule.labels[config.Data.prompt_language.value]
    try:
        # Qwen2 and Meta-Llama use a combined processor
        tokenizer = datamodule.processor.tokenizer
    except AttributeError:
        # InternVL2 uses just a tokenizer
        tokenizer = datamodule.processor
    label_ids = tokenizer(labels, add_special_tokens=False)["input_ids"]
    label_versions = modify_labels(label_ids, tokenizer)
    for example in tqdm(datamodule.splits[datasplit]):
        try:
            enc = model.encode_for_prediction(datamodule.processor, example)
            outputs = model.predict(enc, label_ids=label_versions)
        except KeyboardInterrupt:
            break
        else:
            decoded = datamodule.processor.batch_decode(
                outputs["generated_text"], skip_special_tokens=True)[0]
            all_outputs["generated_text"].append(decoded.strip().lower())
            decoded_label = datamodule.processor.batch_decode(
                outputs["predicted_label_ids"], skip_special_tokens=True)[0]
            all_outputs["predicted_label"].append(decoded_label)
            logits = outputs["label_logits"].squeeze().detach().cpu().tolist()
            all_logits.append(logits)
            all_outputs["prompts"].append(example["message"][0]["content"][1]["text"])
            all_outputs["images"].append(example["message"][0]["content"][0]["image"])
            all_outputs["gold_labels"].append(example["label"])
            all_outputs["stance_targets"].append(example["target_code"])

    preds_dir = os.path.join(outdir, "predictions")
    os.makedirs(preds_dir, exist_ok=True)
    config.yaml(outpath=os.path.join(preds_dir, "config.yaml"))
    out_df = pd.DataFrame(all_outputs)
    all_logits = torch.tensor(all_logits)
    for (lab, lab_logits) in zip(labels, all_logits.T):
        out_df.loc[:, f"logits_{lab}"] = lab_logits
    out_df.to_csv(os.path.join(preds_dir, f"{datasplit}.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
