#!/usr/bin/env python3
"""
Copy edited first-frame images into each episode's scene prior folder.

Example:
  ${ROBODREAM_DATA_DIR}/first_frames_processed/edited/
    AUTOLab+...+2023-10-21-19h-02m-09s_external_cam2.jpg
  -> ${ROBODREAM_DATA_DIR}/processed/droid_traj2demo/real/
       AUTOLab+...+2023-10-21-19h-02m-09s/scene_prior_rgb/external_cam2/
       0000000000_omnipaint.jpg
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm


_DATA_DIR = Path(os.environ.get("ROBODREAM_DATA_DIR", "datasets"))
DEFAULT_SRC_DIR = _DATA_DIR / "first_frames_processed" / "edited"
DEFAULT_EPISODE_ROOT = _DATA_DIR / "processed" / "droid_traj2demo" / "real"
TARGET_FILENAME = "0000000000_omnipaint.jpg"


def parse_image_name(image_path: Path) -> tuple[str, str] | None:
    """
    Return (episode_name, camera_name) parsed from filename.

    Expected pattern: "<episode_name>_external_camX.<ext>"
    """
    name = image_path.stem  # drop extension
    marker = "_external_cam"
    if marker not in name:
        return None
    episode_name, camera_suffix = name.split(marker, 1)
    camera_name = f"external_cam{camera_suffix}"
    if not episode_name or camera_name == "external_cam":
        return None
    return episode_name, camera_name


def copy_images(src_dir: Path, episode_root: Path) -> None:
    if not src_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {src_dir}")
    if not episode_root.exists():
        raise FileNotFoundError(f"Episode root not found: {episode_root}")

    images = sorted(p for p in src_dir.iterdir() if p.is_file())
    if not images:
        print(f"No files found in {src_dir}")
        return

    copied = 0
    skipped = 0
    missing_episodes = 0

    for image_path in tqdm(images):
        parsed = parse_image_name(image_path)
        if parsed is None:
            skipped += 1
            continue

        episode_name, camera_name = parsed
        episode_dir = episode_root / episode_name
        if not episode_dir.exists():
            missing_episodes += 1
            continue

        target_dir = episode_dir / "scene_prior_rgb" / camera_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / TARGET_FILENAME

        shutil.copy2(image_path, target_path)
        copied += 1

    print(
        "Done. copied={}, skipped={}, missing_episodes={}".format(
            copied, skipped, missing_episodes
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy edited first frames into scene_prior_rgb folders."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=DEFAULT_SRC_DIR,
        help="Source directory of edited first frames.",
    )
    parser.add_argument(
        "--episode-root",
        type=Path,
        default=DEFAULT_EPISODE_ROOT,
        help="Root directory containing episode folders.",
    )
    args = parser.parse_args()

    copy_images(args.src_dir, args.episode_root)


if __name__ == "__main__":
    main()
