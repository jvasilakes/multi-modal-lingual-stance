"""
Test that a model/dataset pair works using the pipeline.
"""


import json
import argparse
from copy import deepcopy

from config import config
import src.modeling.util as M
import src.data.util as D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    return parser.parse_args()


def main(args):
    # load model
    config.load_yaml(args.config)
    model = M.get_model(config)

    # enter image path and prompt
    ds = D.get_datamodule(config)
    ex = ds.splits["validation"][128]
    print(json.dumps(ex, indent=2))
    print()

    #attr_res, prompt, target = model.attribute(ex, ds.processor)
    #print(f"Prompt: {prompt}")
    #print(f"Output: {target}")
    #for (key, val) in attr_res.seq_attr_dict.items():
    #    print(f"{key}: {val:.4f}")
    #input()

    model_inputs = model.encode_for_prediction(ds.processor, ex)

    labels = ds.labels[config.Data.prompt_language.value]
    try:
        tokenizer = ds.processor.tokenizer
    except AttributeError:
        tokenizer = ds.processor
    label_ids = tokenizer(labels, add_special_tokens=False)["input_ids"]
    label_versions = D.modify_labels(label_ids, tokenizer)
    print("\n*** LABELS ***")
    for (lab, versions) in label_versions.items():
        lab_ids = [int(i) for i in lab.split('_')]
        print(tokenizer.convert_ids_to_tokens(lab_ids))
        for v in versions:
            tokens = tokenizer.convert_ids_to_tokens(v)
            print('  ', tokens)
        print()

    outputs = model.predict(model_inputs, label_ids=label_versions)
    out_tokens = tokenizer.convert_ids_to_tokens(outputs["generated_text"][0])
    print(f'Raw output: \n  """{out_tokens}"""')
    print()
    gen_text = tokenizer.batch_decode(outputs["generated_text"],
                                      skip_special_tokens=True)
    pred_label = tokenizer.batch_decode(outputs["predicted_label_ids"],
                                        skip_special_tokens=True)
    prompt = tokenizer.batch_decode(model_inputs["input_ids"],
                                    skip_special_tokens=True)
    print(f'Prompt:\n  """{prompt[0]}"""')
    print(f'Generated Text:\n  """{gen_text[0]}"""')
    print(f'Predicted Label: """{pred_label[0]}"""')
    print("Logits:")
    for (lab, logit) in zip(labels, outputs["label_logits"][0]):
        print(f"  {lab}: {logit:.4f}")


if __name__ == "__main__":
    main(parse_args())
