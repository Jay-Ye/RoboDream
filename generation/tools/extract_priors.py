"""
extract_priors.py — Batch scene/object prior extraction tool.

Generates scene priors (_edited.png, inpainted background) and/or object
priors (_object_prior.png, segmented objects) for all source images in a
folder.  The output filenames match the caching convention used by
retrieval_rebirth.py so that pre-computed priors are picked up
automatically and do not need to be recreated at inference time.

Usage:
    python generation/tools/extract_priors.py <input> \
        --object_prompts "a sponge." "a bowl." \
        --prior_type both \
        --device cuda:0

    # White background variant (for paper figures):
    python generation/tools/extract_priors.py <input> \
        --object_prompts "a sponge." \
        --prior_type object \
        --white_background \
        --device cuda:0

Arguments:
    input              Path to an image file or a directory.  Directories are
                       searched recursively; files matching *_edited.png and
                       *_object_prior.png are automatically skipped.
    --object_prompts   One or more text prompts for Grounded-SAM (e.g.
                       "a sponge." "a marker.").  When omitted the object
                       prior is a copy of the source image and the scene prior
                       is its RGB version (no inpainting).
    --prior_type       Which priors to generate: "scene", "object", or "both"
                       (default: "both").
    --white_background Also save a white-background version of the object prior
                       as _object_prior_white.png (does not overwrite the
                       pipeline cache).  Requires --object_prompts.
    --device           Torch device string (default: "cuda:0").
    --overwrite        Re-generate and overwrite existing cached priors.
"""

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Allow running from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from GSamObjectPrior import GroundedSamScenePrior, GSamScenePriorConfig
from ObjectRemoval import ObjectRemover, ObjectRemovalConfig


# ---------------------------------------------------------------------------
# Helpers (mirrored from retrieval_rebirth.py)
# ---------------------------------------------------------------------------

IMG_EXTENSIONS = {"jpg", "JPG", "png", "PNG", "jpeg", "JPEG"}


def get_image_paths(input_path: str) -> list[str]:
    """Return source image paths from a file or directory (recursive)."""
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lstrip(".") in IMG_EXTENSIONS:
            stem = p.stem
            if "_edited" in stem or "_object_prior" in stem:
                return []
            return [str(p)]
        raise ValueError(f"File is not a recognised image type: {input_path}")

    if not p.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    found = []
    for ext in IMG_EXTENSIONS:
        found.extend(glob.glob(str(p / "**" / f"*.{ext}"), recursive=True))

    filtered = []
    for path in found:
        stem = Path(path).stem
        if "_edited" in stem or "_object_prior" in stem:
            continue
        filtered.append(path)

    filtered.sort()
    return filtered


def get_prior_cache_paths(source_path: str) -> tuple[str, str]:
    """Return (edited_path, object_path) matching retrieval_rebirth.py."""
    base = Path(source_path)
    stem = base.stem
    edited_path = base.with_name(f"{stem}_edited.png")
    object_path = base.with_name(f"{stem}_object_prior.png")
    return str(edited_path), str(object_path)


def get_white_bg_path(source_path: str) -> str:
    """Return path for the white-background object prior (paper figure variant)."""
    base = Path(source_path)
    return str(base.with_name(f"{base.stem}_object_prior_white.png"))


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(
    source_path: str,
    object_prompts: list[str],
    prior_type: str,
    gsam_prior: GroundedSamScenePrior | None,
    gsam_white: GroundedSamScenePrior | None,
    remover: ObjectRemover | None,
    overwrite: bool,
) -> None:
    edited_path, object_path = get_prior_cache_paths(source_path)
    white_bg_path = get_white_bg_path(source_path) if gsam_white is not None else None

    need_scene = prior_type in ("scene", "both")
    need_object = prior_type in ("object", "both")

    scene_done = (not need_scene) or (not overwrite and Path(edited_path).exists())
    object_done = (not need_object) or (not overwrite and Path(object_path).exists())
    white_bg_done = (gsam_white is None) or (not overwrite and Path(white_bg_path).exists())

    if scene_done and object_done and white_bg_done:
        print(f"  [SKIP] All requested priors already exist for {source_path}")
        return

    img_bgr = cv2.imread(source_path)
    if img_bgr is None:
        print(f"  [ERROR] Could not read image: {source_path}")
        return

    merged_mask = None
    object_img_bgr = None

    # --- Object prior ---
    if need_object and not object_done:
        if object_prompts:
            print(f"  [GSam] Segmenting objects with prompts: {object_prompts}")
            out = gsam_prior(img_bgr, object_prompts)
            object_img_bgr = out.scene_prior_object_img_bgr
            merged_mask = out.merged_mask
        else:
            object_img_bgr = img_bgr.copy()
        cv2.imwrite(object_path, object_img_bgr)
        print(f"  [SAVED] Object prior → {object_path}")

    # --- White-background object prior (paper figure variant) ---
    if not white_bg_done:
        if object_prompts:
            print(f"  [GSam (white bg)] Segmenting objects with prompts: {object_prompts}")
            out_white = gsam_white(img_bgr, object_prompts)
            cv2.imwrite(white_bg_path, out_white.scene_prior_object_img_bgr)
            print(f"  [SAVED] Object prior (white bg) → {white_bg_path}")
        else:
            print("  [WARNING] --white_background requires --object_prompts; skipping.")

    # --- Scene prior (inpainted background) ---
    if need_scene and not scene_done:
        if object_prompts:
            if merged_mask is None:
                print(f"  [GSam] Segmenting objects (for inpainting mask) with prompts: {object_prompts}")
                out = gsam_prior(img_bgr, object_prompts)
                merged_mask = out.merged_mask
            print("  [Inpaint] Removing objects ...")
            edited_pil = remover.remove(img_bgr, merged_mask)
            edited_rgb = np.array(edited_pil.convert("RGB"))
        else:
            edited_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        Image.fromarray(edited_rgb).save(edited_path)
        print(f"  [SAVED] Scene prior  → {edited_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-extract scene/object priors compatible with retrieval_rebirth.py"
    )
    parser.add_argument(
        "input",
        help="Path to a source image or a directory to search recursively.",
    )
    parser.add_argument(
        "--object_prompts",
        nargs="*",
        default=[],
        help="Text prompts for Grounded-SAM (e.g. 'a sponge.' 'a bowl.'). "
             "When omitted, no segmentation or inpainting is performed.",
    )
    parser.add_argument(
        "--prior_type",
        choices=["scene", "object", "both"],
        default="both",
        help="Which priors to generate (default: both).",
    )
    parser.add_argument(
        "--white_background",
        action="store_true",
        help="Also save a white-background version of the object prior as "
             "_object_prior_white.png (does not overwrite the pipeline cache). "
             "Requires --object_prompts.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device string (default: cuda:0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached priors.",
    )
    args = parser.parse_args()

    object_prompts = [p.strip() for p in args.object_prompts if p.strip()]

    # Collect source images.
    image_paths = get_image_paths(args.input)
    if not image_paths:
        print("[WARNING] No source images found.")
        return

    print(f"Found {len(image_paths)} source image(s).")
    print(f"Prior type       : {args.prior_type}")
    print(f"Prompts          : {object_prompts or '(none)'}")
    print(f"White background : {args.white_background}")
    print(f"Device           : {args.device}")
    print(f"Overwrite        : {args.overwrite}")
    print()

    need_gsam = bool(object_prompts) and (
        args.prior_type in ("object", "both", "scene") or args.white_background
    )
    need_gsam_for_scene = bool(object_prompts) and args.prior_type in ("scene", "both")

    gsam_prior = (
        GroundedSamScenePrior(GSamScenePriorConfig(input_is_bgr=True, device=args.device))
        if need_gsam else None
    )
    gsam_white = (
        GroundedSamScenePrior(GSamScenePriorConfig(input_is_bgr=True, device=args.device, bg_color=(255, 255, 255)))
        if (args.white_background and object_prompts) else None
    )
    remover = (
        ObjectRemover(ObjectRemovalConfig(input_is_bgr=True, device=args.device))
        if need_gsam_for_scene else None
    )

    for i, source_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {source_path}")
        process_image(
            source_path=source_path,
            object_prompts=object_prompts,
            prior_type=args.prior_type,
            gsam_prior=gsam_prior,
            gsam_white=gsam_white,
            remover=remover,
            overwrite=args.overwrite,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
