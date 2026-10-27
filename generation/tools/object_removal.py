# Minimal example: build a scene prior by removing masked objects with OmniPaint.
#   python generation/tools/object_removal.py <image> <mask.png> [--out edited.jpg]
import argparse

import cv2
from ObjectRemoval import ObjectRemover, ObjectRemovalConfig

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("mask", help="white=remove, black=keep (e.g. merged_mask.png from gsam_object_prior.py)")
parser.add_argument("--out", default="edited.jpg")
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

# `input_is_bgr=True` if you pass OpenCV images (cv2.imread gives BGR)
remover = ObjectRemover(ObjectRemovalConfig(input_is_bgr=True, device=args.device))

img_bgr = cv2.imread(args.image)
mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)  # white=remove, black=keep

out_pil = remover.remove(img_bgr, mask)
out_pil.save(args.out)
