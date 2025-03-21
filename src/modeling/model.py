from copy import deepcopy

import torch
from PIL import Image
from transformers import (AutoModel,
                          AutoModelForImageTextToText,
                          AutoModelForCausalLM)
from qwen_vl_utils import process_vision_info

#from captum.attr import (LayerIntegratedGradients,
#                         FeatureAblation,
#                         LLMAttribution,
#                         LLMGradientAttribution,
#                         TextTokenInput)

#from .inputs import TextVisionInput
from .util import register_model, MODEL_REGISTRY
from ..image_utils import load_image


def _get_model_class(model_path):
    mapping = {"Qwen/Qwen2-VL-7B-Instruct": AutoModelForImageTextToText,
               "Qwen/Qwen2-VL-2B-Instruct": AutoModelForImageTextToText,
               "OpenGVLab/InternVL2-8B": AutoModel,
               "AIDC-AI/Ovis1.6-Gemma2-9B": AutoModelForCausalLM,
               "meta-llama/Llama-3.2-11B-Vision-Instruct": AutoModelForImageTextToText,  # noqa
               "mistralai/Pixtral-12B-2409": None}
    res = mapping[model_path]
    if res is None:
        raise NotImplementedError(model_path)
    return res


class VLMForClassification(object):

    @classmethod
    def from_config(cls, config):
        return cls(model_path=config.Model.model_path.value)

    def __init__(self, model_path):
        self.model_path = model_path
        model_cls = _get_model_class(model_path)
        self.model = model_cls.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto",
            trust_remote_code=True, )

    def predict(self, inputs, label_ids=None, max_new_tokens=30):
        """
        inputs: dict of input_ids, etc output from processor
        label_ids: {label_str: [[label_ids], [label_ids], ...]}
                   where each label_id is a modified version of label_str,
                   such as upper, lower, and titlecase.
        """
        # Predict
        model_outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                output_scores=True, return_dict_in_generate=True)

        # Get only the generated token IDs, without the prompt.
        generated_trimmed = [
                out_ids[len(in_ids):] for (in_ids, out_ids)
                in zip(inputs["input_ids"], model_outputs.sequences)]
        outputs = {"generated_text": generated_trimmed}

        # If labels are provided, figure out where the model
        # makes its prediction and get logits for each label
        # in the label set.
        logits = None
        if label_ids is not None:
            logits, pred_tokens = self.get_logits_predictions(
                    generated_trimmed, model_outputs.scores, label_ids)
            outputs["label_logits"] = logits
            outputs["predicted_label_ids"] = pred_tokens
        return outputs

    def forward(self, inputs):
        return self.model(**inputs)

    def attribute(self, example, processor, label_ids=None):
        raise NotImplementedError()
        # TODO: we can inspect the integrated gradients wrt to the tweet and the image tokens
        # to determine how much the model uses each. However, I need a lot more VRAM to do this.
        modules = [self.model.model.embed_tokens, self.model.visual.patch_embed]
        #text_lig = LayerIntegratedGradients(self.model, modules[0], device_ids=[0,1,2])
        #llm_attr = LLMGradientAttribution(text_lig, processor.tokenizer)
        text_fa = FeatureAblation(self.model)
        llm_attr = LLMAttribution(text_fa, processor.tokenizer)
        skip_tokens = [tok for (key, tok) in processor.tokenizer.special_tokens_map.items() if key != "additional_special_tokens"]
        add_special_toks = processor.tokenizer.special_tokens_map["additional_special_tokens"]
        skip_tokens.extend([t for t in add_special_toks if t != "<|image_pad|>"])
        encoded = self.encode_for_prediction(processor, example)
        outputs = self.predict(encoded)
        target = processor.tokenizer.batch_decode(outputs["generated_text"])[0]
        prompt = processor.tokenizer.decode(encoded["input_ids"][0])
        inputs = TextTokenInput(prompt, processor.tokenizer, skip_tokens=skip_tokens)
        #target = f"Answer: {example['label']}"
        attr_res = llm_attr.attribute(inputs, target=target)
        return attr_res, prompt, target

    def get_logits_predictions(self, generated_ids, scores_list, label_ids):
        scores = torch.stack(scores_list, dim=1)
        batch_size, seq_len, vocab_size = scores.shape
        pred_idxs, pred_tokens = self.get_prediction_indices(
                generated_ids, label_ids)
        pred_idxs = [i if i is not None else 0 for i in pred_idxs]

        label_logits = []
        minval = torch.finfo().min
        for (label, ids_for_label_versions) in label_ids.items():
            best_version_logits = torch.empty(batch_size, dtype=scores.dtype).fill_(minval)  # noqa
            best_version_logits = best_version_logits.to(scores.device)
            for version_ids in ids_for_label_versions:
                lab_len = len(version_ids)
                # Truncate any indices where the label is longer than the
                # predicted sequence length.
                end_idxs = [idx + lab_len for idx in pred_idxs]
                # If all examples in the batch are too long, skip it.
                if all([end_i >= seq_len for end_i in end_idxs]):
                    continue
                end_idxs = [idx if idx <= seq_len else seq_len
                            for idx in end_idxs]
                # The range of prediction indices for this label version.
                _pred_idxs = torch.stack(
                        [torch.arange(i, end_idx)
                         for (i, end_idx) in zip(pred_idxs, end_idxs)])
                # Expand the indices to the shape of the scores.
                _pred_idxs = _pred_idxs[:, :, None].expand(-1, -1, vocab_size)
                # The logits for each vocab token at the pred idxs.
                pred_logits = scores.gather(1, _pred_idxs.to(scores.device))
                # Mean logits for this label version for each example.
                _pred_logits = pred_logits[:, torch.arange(lab_len), version_ids].mean(1)  # noqa

                # Update the best logits
                best = _pred_logits[_pred_logits > best_version_logits]
                best_version_logits[_pred_logits > best_version_logits] = best
            label_logits.append(best_version_logits)
        label_logits = torch.stack(label_logits, dim=1)
        return label_logits, pred_tokens

    @staticmethod
    def get_prediction_indices(generated_ids, label_ids):
        """
        Searches generated_ids for the first instance
        of any of the label_ids and returns the index of the
        first token.

        generated_ids: (batch, seq_len, vocab_size) The token IDs generated
                       by the model.
        label_ids: {label_str: [[label_ids], [label_ids], ...]
        """
        pred_idxs = []
        pred_tokens = []
        for tids in generated_ids:
            tids = tids.tolist()
            try:
                for (lab, versions) in label_ids.items():
                    found = None
                    for version_ids in versions:
                        start = 0
                        end = start + len(version_ids)
                        while end <= len(tids):
                            matched = tids[start:end] == version_ids
                            if matched is True:
                                found = start
                                raise StopIteration
                            start += 1
                            end += 1
            except StopIteration:
                pass
            pred_idxs.append(found)
            lab_ids = [int(i) for i in lab.split('_')]
            pred_tokens.append(lab_ids)
        return pred_idxs, pred_tokens

    @property
    def device(self):
        return self.model.device


@register_model("qwen2")
class Qwen2VLForClassification(VLMForClassification):

    def apply_template(self, processor, example):
        prompt = processor.apply_chat_template(
                example["message"], tokenize=False,
                add_generation_prompt=True)
        return prompt

    def encode_for_prediction(self, processor, example):
        prompt = self.apply_template(processor, example)
        img = process_vision_info(example["message"])[0]
        encoded = processor(text=[prompt], images=[img],
                            return_tensors="pt", padding=True)
        return encoded.to(self.device)


@register_model("meta-llama")
class MetaLlamaForClassification(VLMForClassification):

    def encode_for_prediction(self, processor, example):
        prompt = processor.apply_chat_template(
                example["message"], tokenize=False,
                add_generation_prompt=True)
        img = process_vision_info(example["message"])[0]
        encoded = processor(text=[prompt], images=[img],
                            return_tensors="pt", padding=True)
        return encoded.to(self.device)


@register_model("internvl2")
class InternVL2ForClassification(VLMForClassification):

    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    TEMPLATE = "{{ bos_token }}'<|im_start|>system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>\n{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'][1]['text'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"  # noqa

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = AutoModel.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True)

    def encode_for_prediction(self, processor, example):
        # This attribute is required by generate()
        self.model.img_context_token_id = processor.convert_tokens_to_ids(
                self.IMG_CONTEXT_TOKEN)
        pixel_values = load_image(example["message"][0]["content"][0]["image"])
        pixel_values = pixel_values.to(device=self.device, dtype=self.model.dtype)
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []  # noqa

        ex_cp = deepcopy(example)
        question = ex_cp["message"][0]["content"][1]["text"]
        ex_cp["message"][0]["content"][1]["text"] = "<image>\n" + question

        query = processor.apply_chat_template(
                ex_cp["message"], tokenize=False,
                chat_template=self.TEMPLATE)
        for num_patches in num_patches_list:
            image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * \
                    self.model.num_image_token * num_patches + self.IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        encoded = processor(query, return_tensors="pt", padding=True)
        model_inputs = {
                "input_ids": encoded["input_ids"].to(self.device),
                "attention_mask": encoded["attention_mask"].to(self.device),
                "pixel_values": pixel_values}
        return model_inputs

    def predict(self, inputs, label_ids=None, max_new_tokens=30):
        """
        inputs: dict of input_ids, etc output from processor
        label_ids: {label_str: [[label_ids], [label_ids], ...]}
                   where each label_id is a modified version of label_str,
                   such as upper, lower, and titlecase.
        """
        # Predict
        model_outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                output_scores=True, return_dict_in_generate=True)
        outputs = {"generated_text": model_outputs.sequences}

        # If labels are provided, figure out where the model
        # makes its prediction and get logits for each label
        # in the label set.
        logits = None
        if label_ids is not None:
            logits, pred_tokens = self.get_logits_predictions(
                    model_outputs.sequences, model_outputs.scores, label_ids)
            outputs["label_logits"] = logits
            outputs["predicted_label_ids"] = pred_tokens
        return outputs


@register_model("ovis")
class OvisForClassification(VLMForClassification):

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16,
                multimodal_max_length=8192, trust_remote_code=True,
                device_map="auto")

    def encode_for_prediction(self, _, example):
        # processor isn't used
        text_tokenizer = self.model.get_text_tokenizer()
        question = example["message"][0]["content"][1]["text"]
        question = "<image>\n" + question
        imgpath = example["message"][0]["content"][0]["image"]
        image = Image.open(imgpath)

        dtype = self.model.dtype
        device = self.model.device
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(question, [image])  # noqa
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        pixel_values = [pixel_values.to(dtype=dtype, device=device)]

        model_inputs = {
                "input_ids": input_ids.unsqueeze(0).to(device),
                "attention_mask": attention_mask.unsqueeze(0).to(device),
                "pixel_values": pixel_values}
        return model_inputs

    def predict(self, inputs, label_ids=None, max_new_tokens=30):
        """
        inputs: dict of input_ids, etc output from processor
        label_ids: {label_str: [[label_ids], [label_ids], ...]}
                   where each label_id is a modified version of label_str,
                   such as upper, lower, and titlecase.
        """
        # Predict
        # For whatever reason Ovis uses "text_input_ids" instead of
        # "input_ids" as the argument to generate(), so we have to
        # pass things in separately rather than just expanding the dict.
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        try:
            pixel_values = inputs["pixel_values"]
        except KeyError:
            pixel_values = torch.zeros((5, 3, 384, 384))
            pixel_values = [pixel_values.to(device=self.device, dtype=self.model.dtype)]
        model_outputs = self.model.generate(
                input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, max_new_tokens=max_new_tokens,
                output_scores=True, return_dict_in_generate=True)
        outputs = {"generated_text": model_outputs.sequences}

        # If labels are provided, figure out where the model
        # makes its prediction and get logits for each label
        # in the label set.
        logits = None
        if label_ids is not None:
            logits, pred_tokens = self.get_logits_predictions(
                    model_outputs.sequences, model_outputs.scores, label_ids)
            outputs["label_logits"] = logits
            outputs["predicted_label_ids"] = pred_tokens
        return outputs


if __name__ == "__main__":
    print("Available Models")
    print("================")
    for name in MODEL_REGISTRY.keys():
        print(f" {name}")
