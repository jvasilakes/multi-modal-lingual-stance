import string

import numpy as np
from PIL import Image


DATASET_REGISTRY = {}


def register_dataset(name):
    def add_to_registry(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return add_to_registry


def get_datamodule(config):
    ds_name = config.Data.name.value
    try:
        ds = DATASET_REGISTRY[ds_name]
    except KeyError:
        raise KeyError(f"Dataset {ds_name} was not found.")
    return ds.from_config(config)


def modify_labels(label_ids, tokenizer):
    label_strs = tokenizer.batch_decode(label_ids)
    versions = {}
    for (lab_ids, lab_str) in zip(label_ids, label_strs):
        mods = [lab_str, f" {lab_str}"]
        mods.append(lab_str.capitalize())  # Capitalize first character.
        mods.append(f" {lab_str.capitalize()}")
        mods.append(lab_str.title())  # Capitalize first character of every token.
        mods.append(f" {lab_str.title()}")
        mods.append(lab_str.upper())
        mods.append(f" {lab_str.upper()}")  # Capitalize all characters.
        lab_id_str = '_'.join([str(i) for i in lab_ids])
        mods = list(set(mods))
        versions[lab_id_str] = tokenizer(
                mods, add_special_tokens=False)["input_ids"]
    return versions


def generate_gaussian_noise(savepath):
    noise = np.random.normal(0, 1**0.5, (500, 500, 3))
    img = Image.fromarray(noise, mode="RGB")
    img.save(savepath)
