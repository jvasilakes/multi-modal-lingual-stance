import os
import json
import argparse
import warnings
from glob import glob

import pandas as pd
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_file", type=str)
    parser.add_argument("--get_ocr_columns_from_predictions",
                        type=str, default=None,
                        help="If specified, just copy the OCR columns from this file.")  # noqa
    return parser.parse_args()


def main(args):
    preds = pd.read_csv(args.predictions_file)
    if args.get_ocr_columns_from_predictions is not None:
        ocr_preds = pd.read_csv(args.get_ocr_columns_from_predictions)
        preds = pd.concat([preds, ocr_preds[["ocr_text", "ocr_text_coverage"]]], axis=1)
    else:
        columns = {"ocr_text": [],
                   "ocr_text_coverage": []}
        for imgpath in preds.images:
            imgdir = os.path.dirname(imgpath)
            if not imgdir.endswith("/images"):
                imgdir = os.path.join(os.path.dirname(imgdir), "images")
            ocrdir = imgdir + "_ocr"
            imgfile = os.path.basename(imgpath)
            name, _ = os.path.splitext(imgfile)
            ocr_fname_glob = name + "*.json"
            ocr_path_glob = os.path.join(ocrdir, ocr_fname_glob)
            ocr_path = glob(ocr_path_glob)
            if len(ocr_path) == 0 or not os.path.isfile(ocr_path[0]):
                raise ValueError(f"Could not find OCR file for {imgpath}")
            ocr_path = ocr_path[0]
            ocr_data = json.load(open(ocr_path))
            fulltext = '. '.join(
                    [box["text"] for box in ocr_data["bounding_boxes"]])
            text_coverage = compute_text_coverage(
                    imgpath, ocr_data["bounding_boxes"])
            columns["ocr_text"].append(fulltext)
            columns["ocr_text_coverage"].append(text_coverage)
        preds = preds.assign(**columns)
    preds_path, preds_ext = os.path.splitext(args.predictions_file)
    outfile = f"{preds_path}_ocr{preds_ext}"
    preds.to_csv(outfile, index=False)


def compute_text_coverage(image_path, bounding_boxes):
    image = Image.open(image_path)
    image_area = image.size[0] * image.size[1]
    total_text_area = 0
    for box_data in bounding_boxes:
        box = box_data["bounding_box"]
        side1 = box[1][0] - box[0][0]
        side2 = box[2][1] - box[1][1]
        total_text_area += side1 * side2
    coverage = total_text_area / image_area
    if coverage > 1.0:
        warnings.warn(f"Coverage > 1 ({coverage:.2f}): {image_path}")
        coverage = 1.0
    return coverage


if __name__ == "__main__":
    main(parse_args())
