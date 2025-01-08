import os
import re
import random
from glob import glob

import pandas as pd
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

from .util import register_dataset, generate_gaussian_noise, DATASET_REGISTRY


class MultiModalStanceDataset(object):

    @classmethod
    def from_config(cls, config):
        return cls(datadir=config.Data.datadir.value,
                   prompt_language=config.Data.prompt_language.value,
                   tweet_language=config.Data.tweet_language.value,
                   model_path=config.Model.model_path.value,
                   use_images=config.Data.use_images.value,
                   images_dirname=config.Data.images_dirname.value,
                   use_text=config.Data.use_text.value,
                   use_image_text=config.Data.use_image_text.value)

    def __init__(self,
                 datadir,
                 model_path,
                 prompt_language="en",
                 tweet_language="en",
                 use_images=True,
                 images_dirname="images",
                 use_text=True,
                 use_image_text=False):
        self.datadir = datadir
        self.prompt_language = prompt_language
        self.tweet_language = tweet_language
        self.model_path = model_path
        self.use_images = use_images
        self.images_dirname = images_dirname
        self.use_text = use_text
        self.use_image_text = use_image_text
        try:
            self.processor = AutoProcessor.from_pretrained(
                    self.model_path, trust_remote_code=True)
        except TypeError:
            # Ovis 1.6 doesn't work with AutoProcessor
            self.processor = AutoTokenizer.from_pretrained(
                    self.model_path, trust_remote_code=True)
        train, val, test = self.load()
        self.splits = {"train": train, "validation": val, "test": test}

    @property
    def labels(self):
        # Dataset dependent. Set in child classes.
        """
        Labels in each language
        {"en": [],
         "es": [],
         "fr": [],
         "hi": [],
         "pl": [],
         "cs": [],
         "de": [],
         "ca": [],
         "zh": [],
         "el": [],
         "ru": []
         }
        """
        pass

    @property
    def prompt_templates(self):
        return {"en": "From the image and tweet, determine the stance regarding {target}. The possible stance labels are {label_str}. Answer with the label first, before any explanation. Tweet: {tweet}", # noqa
         "es": "A partir de la imagen y el tweet, determina la postura con respecto a {target}. Las posibles etiquetas de postura son {label_str}. Responde con la etiqueta primero, antes de cualquier explicación. Tweet: {tweet}", # noqa
         "pt": "A partir da imagem e do tweet, determine a postura em relação a {target}. Os possíveis rótulos de postura são {label_str}. Responda com o rótulo primeiro, antes de qualquer explicação. Tweet: {tweet}",  # noqa
         "fr": "À partir de l'image et du tweet, déterminez la position par rapport à {target}. Les libellés de position possibles sont {label_str}. Répondez d'abord avec le libellé, avant toute explication. Tweet: {tweet}", # noqa
         "hi": "छवि और ट्वीट से {target} के बारे में रुख निर्धारित करें। संभावित रुख लेबल {label_str} हैं। किसी भी स्पष्टीकरण से पहले लेबल के साथ उत्तर दें। ट्वीट: {tweet}", # noqa
         "pl": "Określ stanowisko dotyczące {target} na podstawie obrazu i tweeta. Możliwe etykiety stanowiska to {label_str}. Odpowiedz etykietą przed jakimkolwiek wyjaśnieniem. Tweet: {tweet}", # noqa
         "cs": "Určete postoj k {target} z obrázku a tweetu. Možné štítky postojů jsou {label_str}. Před jakýmkoli vysvětlením odpovězte štítkem. Tweet: {tweet}", # noqa
         "de": "Bestimmen Sie die Haltung zu {target} anhand von Bild und Tweet. Mögliche Haltungsbezeichnungen sind {label_str}. Antworten Sie mit der Bezeichnung, bevor Sie eine Erklärung abgeben. Tweet: {tweet}", # noqa
         "ca": "Determina la posició sobre {target} a partir de la imatge i el tuit. Les etiquetes de posició possibles són {label_str}. Respon amb etiqueta abans de qualsevol explicació. Tweet: {tweet}", # noqa
         "zh": "根据图片和推文确定您对 {target} 的立场。可能的立场标签是 {label_str}。在给出任何解释之前，请先回复标签。推文：{tweet}", # noqa
         "el": "Καθορίστε τη στάση σας στο {target} με βάση την εικόνα και το tweet. Οι πιθανές ετικέτες στάσης είναι {label_str}. Απαντήστε στην ετικέτα πριν δώσετε οποιαδήποτε εξήγηση. Tweet: {tweet}",  # noqa 
         "ru": "Определите свою позицию по {target} на основе изображения и твита. Возможные метки позиции: {label_str}. Пожалуйста, ответьте на метку, прежде чем давать какие-либо объяснения. Твит: {tweet}"}  # noqa

    @property
    def prompt_templates_text_only(self):
        return {"en": "From the tweet and the text extracted from the image, determine the stance regarding {target}. The possible stance labels are {label_str}. Answer with the label first, before any explanation. Tweet: {tweet}, Image Text: {image_text}"}  # noqa

    def load(self):
        columns = ["stance_label", "tweet_image"]
        # I didn't plan ahead well enough when creating the data to
        # append "_en" to the English columns, so we have to do this.
        prompt_postfix = '' if self.prompt_language == "en" else f"_{self.prompt_language}"  # noqa
        tweet_postfix = '' if self.tweet_language == "en" else f"_{self.tweet_language}"  # noqa
        columns.extend([f"stance_target{prompt_postfix}",
                        f"tweet_text{tweet_postfix}"])

        train = self.load_split("train", columns)
        val = self.load_split("valid", columns)
        test = self.load_split("test", columns)
        return train, val, test

    def load_split(self, split, columns):
        assert split in ["train", "valid", "test"]
        split_glob = os.path.join(
                self.datadir, f"in-target/*/{split}_translated.csv")
        split_df = pd.DataFrame()
        for split_path in glob(split_glob):
            split_df_part = pd.read_csv(split_path)
            split_df_part.dropna(how="any", inplace=True)
            split_df_part = split_df_part[columns]
            code = split_path.split(os.sep)[-2]
            split_df_part.loc[:, "stance_target_code"] = code
            split_df = pd.concat([split_df, split_df_part])
        split = self.preprocess(split_df)
        return split

    def preprocess(self, df):
        """
        Map labels to the target language.
        Normalize the tweet.
        """
        tweet_col = "tweet_text"
        stance_col = "stance_target"
        if self.prompt_language != "en":
            stance_col += f"_{self.prompt_language}"
        if self.tweet_language != "en":
            tweet_col += f"_{self.tweet_language}"
        tweets = df[tweet_col].apply(self.preprocess_tweet)

        # Get the prompt template
        if self.use_image_text is True:
            prompt_templates = self.prompt_templates_text_only
        else:
            prompt_templates = self.prompt_templates
        try:
            prompt_template = prompt_templates[self.prompt_language]
        except KeyError:
            raise KeyError(f"Prompt language {self.prompt_language} not supported when use_image_text is {self.use_image_text}.")  # noqa

        # Define the labels for the prompt.
        label_set = self.labels[self.prompt_language]
        label_str = ', '.join(label_set[:-1]) + f", or {label_set[-1]}"
        labels = self.map_labels_from_english(df["stance_label"])

        # Get the image paths, or the paths to the extracted text.
        # Load the text extracted from the images
        if self.use_image_text is True:
            imgpaths = df["tweet_image"]
            imgfiles = [os.path.basename(path) for path in imgpaths]
            txtfiles = [os.path.splitext(f)[0] + ".txt" for f in imgfiles]
            imgdir = os.path.dirname(imgpaths.iloc[0])
            preimgdir = os.path.dirname(imgdir)
            imgdir = os.path.join(preimgdir, self.images_dirname)
            txtpaths = [os.path.join(imgdir, fname) for fname in txtfiles]
            image_texts = [open(path).read().strip() for path in txtpaths]

        if self.use_images is True:
            imgpaths = df["tweet_image"]
            if self.images_dirname != "images":
                imgfiles = [os.path.basename(path) for path in imgpaths]
                imgdir = os.path.dirname(imgpaths.iloc[0])
                preimgdir = os.path.dirname(imgdir)
                imgdir = os.path.join(preimgdir, self.images_dirname)
                imgpaths = [os.path.join(imgdir, fname) for fname in imgfiles]
        else:
            # Load Gaussian noise.
            imgdir = os.path.dirname(df["tweet_image"].iloc[0])
            imgpath = os.path.join(imgdir, "noise.jpg")
            generate_gaussian_noise(imgpath)
            imgpaths = [imgpath for _ in range(len(tweets))]

        # Build the prompts
        prompts = []
        for (i, (trg, tweet)) in enumerate(zip(df[stance_col], tweets)):
            if self.use_text is False:
                tweet = ''
            values = {"target": trg, "label_str": label_str, "tweet": tweet}
            if self.use_image_text is True:
                values["image_text"] = image_texts[i]
            prompts.append(prompt_template.format(**values))

        messages = [{"role": "user",
                     "content": [{"type": "image", "image": imgpath},
                                 {"type": "text", "text": prompt}]}
                    for (prompt, imgpath) in zip(prompts, imgpaths)]
        stance_codes = df["stance_target_code"]
        return [{"message": [msg], "label": label, "target_code": trg}
                for (msg, label, trg) in zip(messages, labels, stance_codes)]

    @staticmethod
    def preprocess_tweet(text):
        flags = re.MULTILINE | re.DOTALL
        text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "HTTPURL",
                      text, flags=flags)
        text = re.sub(r"@\w+", "@USER", text, flags=flags)
        return text.replace('\n', '')

    def map_labels_from_english(self, labels):
        if self.prompt_language == "en":
            return [lab.lower() for lab in labels]
        mapping = dict(zip(
            self.labels["en"], self.labels[self.prompt_language])
            )
        return [mapping[lab.lower()] for lab in labels]

    def collate_fn(self, examples, shuffle=False):
        prompts_with_labels = []
        img_inputs = []
        idxs = list(range(len(examples)))
        if shuffle is True:
            random.shuffle(idxs)
        for i in idxs:
            ex = examples[i]
            prompt = self.processor.apply_chat_template(
                    ex["message"], tokenize=False, add_generation_prompt=True)
            prompt_with_label = prompt + ' ' + ex["label"]
            prompts_with_labels.append(prompt_with_label)
            img = process_vision_info(ex["message"])[0]
            img_inputs.append(img)

        batch = self.processor(text=prompts_with_labels, images=img_inputs,
                               return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        if isinstance(self.processor, Qwen2VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [
                    self.processor.tokenizer.convert_tokens_to_ids(
                        self.processor.image_token)
                    ]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch


@register_dataset("mtse")
class MTSEDataset(MultiModalStanceDataset):

    @property
    def labels(self):
        return {"en": ["favor", "against", "neutral"],
                "es": ["a favor", "en contra", "neutral"],
                "pt": ["a favor", "contra", "neutro"],
                "fr": ["pour", "contre", "neutre"],
                "hi": ["के लिए", "खिलाफ", "तटस्थ"],
                "pl": ["za", "przeciw", "neutralny"],
                "cs": ["pro", "proti", "neutrální"],
                "de": ["für", "gegen", "neutral"],
                "ca": ["a favor", "en contra", "neutre"],
                "zh": ["赞成", "反对", "中立"],
                "el": ["Συμφωνώ", "Ενάντια", "Ουδέτερο"],
                "ru": ["Согласен", "Против", "Нейтральный"]
                }


@register_dataset("mccq")
class MCCQDataset(MultiModalStanceDataset):

    @property
    def labels(self):
        return {"en": ["favor", "against", "neutral"],
                "es": ["a favor", "en contra", "neutral"],
                "pt": ["a favor", "contra", "neutro"],
                "fr": ["pour", "contre", "neutre"],
                "hi": ["के लिए", "खिलाफ", "तटस्थ"],
                "pl": ["za", "przeciw", "neutralny"],
                "cs": ["pro", "proti", "neutrální"],
                "de": ["für", "gegen", "neutral"],
                "ca": ["a favor", "en contra", "neutre"],
                "zh": ["赞成", "反对", "中立"],
                "el": ["Συμφωνώ", "Ενάντια", "Ουδέτερο"],
                "ru": ["Согласен", "Против", "Нейтральный"]
                }


@register_dataset("mwtwt")
class MWTWTDataset(MultiModalStanceDataset):

    @property
    def labels(self):
        return {"en": ["support", "refute", "comment", "unrelated"],
                "es": ["apoya", "refuta", "comenta", "no relacionado"],
                "pt": ["apoiar", "refutar", "comentar", "não relacionado"],
                "fr": ["soutenir", "réfuter", "commenter", "sans rapport"],
                "hi": ["समर्थन", "खंडन", "टिप्पणी", "असंबद्ध"],
                "pl": ["wsparcie", "obalanie", "komentarz", "niezwiązane"],
                "cs": ["podpora", "vyvrácení", "komentář", "nesouvisející"],
                "de": ["unterstützen", "widerlegen", "Kommentar", "unabhängig"],  # noqa
                "ca": ["suport", "refutar", "comentar", "no relacionat"],
                "zh": ["支持", "反驳", "评论", "无关"],
                "el": ["υποστήριξη", "διάψευση", "σχόλιο", "άσχετο"],
                "ru": ["поддерживать", "опровергать", "комментировать", "несвязанный"]  # noqa
                }


@register_dataset("mruc")
class MRUCDataset(MultiModalStanceDataset):

    @property
    def labels(self):
        return {"en": ["support", "oppose", "neutral"],
                "es": ["apoyar", "oponerse", "neutral"],
                "pt": ["apoiar", "opor-se", "neutro"],
                "fr": ["soutenir", "s'opposer", "neutre"],
                "hi": ["समर्थन", "विरोध", "तटस्थ"],
                "pl": ["wspierać", "sprzeciwiać się", "neutralny"],
                "cs": ["podpora", "proti", "neutrální"],
                "de": ["unterstützen", "ablehnen", "neutral"],
                "ca": ["suport", "oposar-se", "neutre"],
                "zh": ["支持", "反对", "中立"],
                "el": ["υποστήριξη", "αντίθεση", "ουδέτερο"],
                "ru": ["поддержка", "против", "нейтральный"],
                }


@register_dataset("mtwq")
class MTWQDataset(MultiModalStanceDataset):

    @property
    def labels(self):
        return {"en": ["support", "oppose", "neutral"],
                "es": ["apoyar", "oponerse", "neutral"],
                "pt": ["apoiar", "opor-se", "neutro"],
                "fr": ["soutenir", "s'opposer", "neutre"],
                "hi": ["समर्थन", "विरोध", "तटस्थ"],
                "pl": ["wspierać", "sprzeciwiać się", "neutralny"],
                "cs": ["podpora", "proti", "neutrální"],
                "de": ["unterstützen", "ablehnen", "neutral"],
                "ca": ["suport", "oposar-se", "neutre"],
                "zh": ["支持", "反对", "中立"],
                "el": ["υποστήριξη", "αντίθεση", "ουδέτερο"],
                "ru": ["поддержка", "против", "нейтральный"],
                }

if __name__ == "__main__":
    print("Available Datasets")
    print("==================")
    for name in DATASET_REGISTRY.keys():
        print(f" {name}")
