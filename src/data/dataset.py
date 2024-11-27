import os
import re
from glob import glob

import pandas as pd
from datasets import Dataset, DatasetDict

from transformers import AutoProcessor


class MultiModalStanceDataset(object):

    def __init__(self,
                 datadir,
                 model_path,
                 language="en"):
        self.datadir = datadir
        self.language = language
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        train, val, test = self.load()
        self.splits = DatasetDict(
                {"train": train, "validation": val, "test": test})

    @classmethod
    def from_config(cls, config):
        return cls(datadir=config.Data.datadir.value,
                   language=config.Data.language.value,
                   model_path=config.Model.model_path.value)

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
    def prompt_template(self):
        return {"en": "From the image and tweet, determine the stance regarding {target}. The possible stance labels are {label_str}. Answer with the label first, before any explanation. Tweet: {tweet}", # noqa
         "es": "A partir de la imagen y el tweet, determina la postura con respecto a {target}. Las posibles etiquetas de postura son {label_str}. Responde con la etiqueta primero, antes de cualquier explicación. Tweet: {tweet}", # noqa
         "fr": "À partir de l'image et du tweet, déterminez la position par rapport à {target}. Les libellés de position possibles sont {label_str}. Répondez d'abord avec le libellé, avant toute explication. Tweet: {tweet}", # noqa
         "hi": "छवि और ट्वीट से {target} के बारे में रुख निर्धारित करें। संभावित रुख लेबल {label_str} हैं। किसी भी स्पष्टीकरण से पहले लेबल के साथ उत्तर दें। ट्वीट: {tweet}", # noqa
         "pl": "Określ stanowisko dotyczące {target} na podstawie obrazu i tweeta. Możliwe etykiety stanowiska to {label_str}. Odpowiedz etykietą przed jakimkolwiek wyjaśnieniem. Tweet: {tweet}", # noqa
         "cs": "Určete postoj k {target} z obrázku a tweetu. Možné štítky postojů jsou {label_str}. Před jakýmkoli vysvětlením odpovězte štítkem. Tweet: {tweet}", # noqa
         "de": "Bestimmen Sie die Haltung zu {target} anhand von Bild und Tweet. Mögliche Haltungsbezeichnungen sind {label_str}. Antworten Sie mit der Bezeichnung, bevor Sie eine Erklärung abgeben. Tweet: {tweet}", # noqa
         "ca": "Determina la posició sobre {target} a partir de la imatge i el tuit. Les etiquetes de posició possibles són {label_str}. Respon amb etiqueta abans de qualsevol explicació. Tweet: {tweet}", # noqa
         "zh": "根据图片和推文确定您对 {target} 的立场。可能的立场标签是 {label_str}。在给出任何解释之前，请先回复标签。推文：{tweet}", # noqa
         "el": "Καθορίστε τη στάση σας στο {target} με βάση την εικόνα και το tweet. Οι πιθανές ετικέτες στάσης είναι {label_str}. Απαντήστε στην ετικέτα πριν δώσετε οποιαδήποτε εξήγηση. Tweet: {tweet}",  # noqa 
         "ru": "Определите свою позицию по {target} на основе изображения и твита. Возможные метки позиции: {label_str}. Пожалуйста, ответьте на метку, прежде чем давать какие-либо объяснения. Твит: {tweet}"  # noqa
         }

    def load(self):
        columns = ["stance_label", "tweet_image"]
        if self.language == "en":
            columns.extend(["stance_target", "tweet_text"])
        else:
            columns.extend([f"stance_target_{self.language}",
                            f"tweet_text_{self.language}"])

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
        split = Dataset.from_list(self.preprocess(split_df), split=split)
        return split

    def preprocess(self, df):
        """
        Map labels to the target language.
        Normalize the tweet.
        """
        tweet_col = "tweet_text"
        stance_col = "stance_target"
        if self.language != "en":
            tweet_col += f"_{self.language}"
            stance_col += f"_{self.language}"
        tweets = df[tweet_col].apply(self.preprocess_tweet)
        prompt_template = self.prompt_template[self.language]
        label_set = self.labels[self.language]
        label_str = ', '.join(label_set[:-1]) + f", or {label_set[-1]}"
        prompts = []
        for (trg, tweet) in zip(df[stance_col], tweets):
            values = {"target": trg, "label_str": label_str, "tweet": tweet}
            prompts.append(prompt_template.format(**values))
        labels = self.map_labels_from_english(df["stance_label"])
        messages = [{"role": "user",
                     "content": [{"type": "image"},
                                 {"type": "text", "text": prompt}]}
                    for prompt in prompts]
        formatted_prompts = []
        for mess in messages:
            formatted = self.processor.apply_chat_template(
                    [mess], tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted)
        stance_codes = df["stance_target_code"]
        imgpaths = df["tweet_image"]
        return [{"prompt": prompt, "label": label,
                 "target_code": trg, "image": imgpath}
                for (prompt, label, trg, imgpath) in
                zip(formatted_prompts, labels, stance_codes, imgpaths)]

    def map_labels_from_english(self, labels):
        if self.language == "en":
            return [lab.lower() for lab in labels]
        mapping = dict(zip(self.labels["en"], self.labels[self.language]))
        return [mapping[lab.lower()] for lab in labels]

    @staticmethod
    def preprocess_tweet(text):
        flags = re.MULTILINE | re.DOTALL
        text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "HTTPURL",
                      text, flags=flags)
        text = re.sub(r"@\w+", "@USER", text, flags=flags)
        return text.replace('\n', '')


class MTSEDataset(MultiModalStanceDataset):

    @property
    def labels(self):
        return {"en": ["favor", "against", "neutral"],
                "es": ["a favor", "en contra", "neutral"],
                "fr": ["Pour", "Contre", "Neutre"],
                "hi": ["के लिए", "खिलाफ", "तटस्थ"],
                "pl": ["za", "przeciw", "neutralny"],
                "cs": ["pro", "proti", "neutrální"],
                "de": ["für", "gegen", "neutral"],
                "ca": ["a favor", "en contra", "neutre"],
                "zh": ["赞成", "反对", "中立"],
                "el": ["Συμφωνώ", "Ενάντια", "Ουδέτερο"],
                "ru": ["Согласен", "Против", "Нейтральный"]
                }
