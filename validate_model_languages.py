import argparse

from config import config
from src.modeling.util import get_model
from src.data.util import get_datamodule


QUESTIONS1 = {
        "english": "Describe the image.",
        "german": "Beschreiben Sie das Bild.",
        "spanish": "Describe la imagen.",
        "french": "Décrivez l'image.",
        "hindi": "छवि का वर्णन करें.",
        "portuguese": "Descreva a imagem.",
        "chinese": "描述图像。",
        "udmurt": "суредэз возьматыны.",
        "yucatec maya": "tsol le oochelo'.",
        }


QUESTIONS2 = {
        "english": "What does this animal eat?",
        "german": "Was frisst dieses Tier?",
        "spanish": "¿Qué come este animal?",
        "french": "Que mange cet animal?",
        "hindi": "यह जानवर क्या खाता है?",
        "portuguese": "O que esse animal come?",
        "chinese": "这种动物吃什么？",
        "udmurt": "мар сие та пӧйшур?",
        "yucatec maya": "ba'ax ku jaantik le ba'alche'a'?",
        }


QUESTIONS3 = {
        "english": "What does a cat eat?",
        "german": "Was frisst eine Katze?",
        "spanish": "¿Qué come un gato?",
        "french": "Que mange un chat?",
        "hindi": "बिल्ली क्या खाती है?",
        "portuguese": "O que um gato come?",
        "chinese": "猫吃什么？",
        "udmurt": "Мар сие кошке?",
        "yucatec maya": "Ba'ax ku jaantik juntúul miis?",
        }


QUESTIONS4 = {
        "english": "Translate the following into English: This is my favorite kind of cat, with sleek fur and large ears.",
        "german": "Translate the following into English: Das ist meine Lieblingskatzenart, mit glattem Fell und großen Ohren.",
        "spanish": "Translate the following into English: Este es mi tipo de gato favorito, con pelaje elegante y orejas grandes.",
        "french": "Translate the following into English: C'est mon type de chat préféré, avec une fourrure lisse et de grandes oreilles.",
        "hindi": "Translate the following into English: यह मेरी पसंदीदा बिल्ली है, जिसके बाल चिकने और कान बड़े हैं।",
        "portuguese": "Translate the following into English: Este é meu tipo de gato favorito, com pelo liso e orelhas grandes.",
        "chinese": "Translate the following into English: 这是我最喜欢的猫，毛皮光滑，耳朵大大的。",
        "udmurt": "Translate the following into English: Та мынам яратоно кошкы, ӟырдыт йырсиё но бадӟым пельёсын.",
        "yucatec maya": "Translate the following into English: Lela' in bin yano'ob miiso' jach uts tin wich, yéetel u tso'otsel jats'uts yéetel nukuch xikino'ob.",
        }


QUESTIONS5 = {
        "english": "Translate the following into English: This is my favorite kind of cat, with sleek fur and large ears.",
        "german": "Translate the following into German: This is my favorite kind of cat, with sleek fur and large ears.",
        "spanish": "Translate the following into Spanish: This is my favorite kind of cat, with sleek fur and large ears.",
        "french": "Translate the following into French: This is my favorite kind of cat, with sleek fur and large ears.",
        "hindi": "Translate the following into Hindi: This is my favorite kind of cat, with sleek fur and large ears.",
        "portuguese": "Translate the following into Portuguese: This is my favorite kind of cat, with sleek fur and large ears.",
        "chinese": "Translate the following into Chinese: This is my favorite kind of cat, with sleek fur and large ears.",
        "udmurt": "Translate the following into Udmurt: This is my favorite kind of cat, with sleek fur and large ears.",
        "yucatec maya": "Translate the following into Yucatec Maya: This is my favorite kind of cat, with sleek fur and large ears.",
        }

#IMAGE = "/home/ac1jv/Projects/multi_modal_stance/previous_work/Multi-Modal-Stance-Detection/dataset/Multi-Modal-Stance-Detection/Multi-modal-Twitter-Stance-Election-2020/images/noise.jpg"
IMAGE = "/home/ac1jv/Projects/multi_modal_stance/graphics/abyssinian-main.jpg"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    return parser.parse_args()


def main(args):
    config.load_yaml(args.config_file)
    model = get_model(config)
    ds = get_datamodule(config)
    try:
        tokenizer = ds.processor.tokenizer
    except AttributeError:
        tokenizer = ds.processor
    ex = {'message': [{'role': 'user',
                       'content': [
                           {'type': 'image',
                            'image': IMAGE},
                           {'type': 'text',
                            'text': ''}
                           ]}]
         }
    for (i, question_set) in enumerate([QUESTIONS1, QUESTIONS2, QUESTIONS3, QUESTIONS4, QUESTIONS5]): 
        if i >= 2:
            continue
        print(f"===== {i}: {question_set['english']} ====")
        for (lang, question) in question_set.items():
            ex["message"][0]["content"][1]["text"] = question
            inputs = model.encode_for_prediction(ds.processor, ex)
            if i > 1:  # questions 3,4,5 do not use any image
                keys = list(inputs.keys())
                for key in keys:
                    if key not in ["input_ids", "attention_mask"]:
                        inputs.pop(key)
            outputs = model.predict(inputs, max_new_tokens=100)
            generated = tokenizer.batch_decode(outputs["generated_text"], skip_special_tokens=True)
            print(f" - {lang}: \n    {generated[0]}")
            print()


if __name__ == "__main__":
    main(parse_args())
