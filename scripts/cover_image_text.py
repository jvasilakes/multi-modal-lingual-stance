import os
import json
import argparse
import shlex
import subprocess
import warnings

import numpy as np
from PIL import Image
from tqdm import tqdm


"""
Given a directory of images and corresponding OCR data
cover any and all text in the image with a black box.
The OCR data is obtained using the GATE cloud service
https://cloud.gate.ac.uk/shopfront/displayItem/ml-ocr
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", type=str)
    parser.add_argument("ocr_output_dir", type=str,
                        help="""Contents share the same filenames as
                                the corresponding images + .json.
                                E.g. img.jpg.json""")
    parser.add_argument("outdir", type=str)
    parser.add_argument("--invert", action="store_true", default=False,
                        help="Cover everything OUTSIDE the bounding boxes.")
    return parser.parse_args()


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    for imgfile in tqdm(os.listdir(args.images_dir)):
        ocrfile = f"{imgfile}.json"
        ocrpath = os.path.join(args.ocr_output_dir, ocrfile)
        if os.path.isfile(ocrpath) is False:
            warnings.warn(f"File not found {ocrfile}")
            continue
        outpath = os.path.join(args.outdir, imgfile)
        if os.path.isfile(outpath):
            continue
        imgpath = os.path.join(args.images_dir, imgfile)
        if args.invert is True:
            cover_non_image(imgpath, ocrpath, outpath)
        else:
            cover_image_text(imgpath, ocrpath, outpath)


def cover_image_text(imgfile, ocr_data_file, outfile):
    """
    Cover everything inside the bounding boxes.
    """
    try:
        data = json.load(open(ocr_data_file))
    except:
        print(ocr_data_file)
        input()
    try:
        bounding_boxes = get_bounding_boxes(data)
    except KeyError:
        warnings.warn(f"Improperly formatted JSON in {ocr_data_file}")
        bounding_boxes = []
        return
    if len(bounding_boxes) == 0:
        subprocess.run(["cp", imgfile, outfile])
        return
    draw_cmd = ""
    for box in bounding_boxes:
        x0, y0 = box[0]
        x1, y1 = box[1]
        box_str = f"{x0},{y0} {x1},{y1}"
        draw_cmd += f"-draw 'rectangle {box_str}' "
    cmd = f"magick {imgfile} {draw_cmd} {outfile}"
    args = shlex.split(cmd)
    subprocess.run(args)


def cover_non_image(imgfile, ocr_data_file, outfile):
    """
    Cover everything outside the bounding boxes.
    """
    try:
        data = json.load(open(ocr_data_file))
    except:
        print(ocr_data_file)
        input()
    try:
        bounding_boxes = get_bounding_boxes(data)
    except KeyError:
        warnings.warn(f"Improperly formatted JSON in {ocr_data_file}")
        bounding_boxes = []
        return
    img = Image.open(imgfile)
    if len(bounding_boxes) == 0:
        img = Image.fromarray(np.zeros_like(img))
        img.save(outfile)
        return
    draw_cmd = ""
    box_xs, box_ys = get_pixel_coordinates(bounding_boxes)
    pixels = img.load()
    num_pixels = img.size[0] * img.size[1]
    num_covered = 0
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if x in box_xs or y in box_ys:
                continue
            try:
                pixels[x, y] = (0, 0, 0)
            except TypeError: # B/W image
                pixels[x, y] = 0
            num_covered += 1
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(outfile)


def get_bounding_boxes(ocr_data):
    boxes = []
    for detected_text in ocr_data["bounding_boxes"]:
        box = detected_text["bounding_box"]
        x0, y0 = box[0]
        x1, y1 = box[2]
        boxes.append([[x0, y0], [x1, y1]])
    return boxes


def get_pixel_coordinates(bounding_boxes):
    box_xs = []
    box_ys = []
    for box in bounding_boxes:
        [x0, y0], [x1, y1] = box
        box_xs.extend(range(x0, x1))
        box_ys.extend(range(y0, y1))
    return box_xs, box_ys



if __name__ == "__main__":
    main(parse_args())
