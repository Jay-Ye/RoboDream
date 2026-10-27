# Minimal example: build an object prior with Grounded-SAM.
#   python generation/tools/gsam_object_prior.py <image> "a mouse." [--out_dir .]
import argparse
import os

import cv2
from GSamObjectPrior import GroundedSamScenePrior, GSamScenePriorConfig

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("object_prompts", nargs="+")
parser.add_argument("--out_dir", default=".")
args = parser.parse_args()

cfg = GSamScenePriorConfig(input_is_bgr=True)
prior = GroundedSamScenePrior(cfg)

img_bgr = cv2.imread(args.image)
out = prior(img_bgr, args.object_prompts)

scene_prior_object_img_bgr = out.scene_prior_object_img_bgr
merged_mask = out.merged_mask
masks = out.mask_list
labels = out.mask_labels

os.makedirs(args.out_dir, exist_ok=True)
cv2.imwrite(os.path.join(args.out_dir, "object_prior.jpg"), scene_prior_object_img_bgr)
cv2.imwrite(os.path.join(args.out_dir, "merged_mask.png"), merged_mask)
