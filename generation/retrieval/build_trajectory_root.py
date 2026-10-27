"""
Build the retrieval trajectory pool (DROID_TRAJECTORY_ROOT) from the raw DROID release.

For every successful DROID episode (optionally restricted to --episode_list), copies
    <droid_root>/<relative_path>/trajectory.h5 -> <output_root>/<episode_id>/trajectory.h5
and writes its language annotations to <output_root>/<episode_id>/task_descriptions.txt.
Episodes longer than --max_length frames are skipped (the video model handles up to 512 frames).

In the paper the pool was restricted to the episodes used for training (those with robot-only renders
and priors); pass e.g. the processed-episode list from data_processing to reproduce it.

Usage:
    python generation/retrieval/build_trajectory_root.py \
        --droid_root ${DROID_ROOT} \
        --droid_meta_dir ${ROBODREAM_DATA_DIR}/droid_meta \
        --output_root ${ROBODREAM_DATA_DIR}/trajengine \
        [--episode_list episodes.txt] [--symlink]
"""

import argparse
import json
import multiprocessing
import os
import shutil
from functools import partial

import h5py
from tqdm import tqdm

DATA_DIR = os.environ.get("ROBODREAM_DATA_DIR", "datasets")


def parse_args():
    parser = argparse.ArgumentParser(description="Build the DROID trajectory pool used for retrieval")
    parser.add_argument("--droid_root", type=str, default=os.environ.get("DROID_ROOT", "droid"),
                        help="Raw DROID release root (contains <lab>/success/<date>/<episode>/trajectory.h5)")
    parser.add_argument("--droid_meta_dir", type=str, default=os.path.join(DATA_DIR, "droid_meta"),
                        help="Folder with episode_id_to_path.json and droid_language_annotations.json")
    parser.add_argument("--output_root", type=str, default=os.path.join(DATA_DIR, "trajengine"))
    parser.add_argument("--episode_list", type=str, default=None,
                        help="Optional .txt (one episode id per line) or .json (list or dict keyed by episode id)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--symlink", action="store_true", help="Symlink trajectory.h5 instead of copying")
    parser.add_argument("--num_workers", type=int, default=32)
    return parser.parse_args()


def load_episode_list(path):
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        return set(data.keys() if isinstance(data, dict) else data)
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def process_episode(episode_item, args, annotations):
    episode_id, rel_path = episode_item
    trajectory_path = os.path.join(args.droid_root, rel_path, "trajectory.h5")
    if not os.path.exists(trajectory_path):
        # NOTE: in rare cases, the trajectory file is named as trajectory_im128.h5
        trajectory_path = trajectory_path.replace(".h5", "_im128.h5")
        if not os.path.exists(trajectory_path):
            return False
    try:
        with h5py.File(trajectory_path, "r") as f:
            length = len(f['observation']['robot_state']['cartesian_position'])
    except Exception:
        return False
    if length > args.max_length:
        return False

    tgt_episode_path = os.path.join(args.output_root, episode_id)
    os.makedirs(tgt_episode_path, exist_ok=True)
    tgt_trajectory_path = os.path.join(tgt_episode_path, "trajectory.h5")
    if not os.path.exists(tgt_trajectory_path):
        if args.symlink:
            os.symlink(os.path.abspath(trajectory_path), tgt_trajectory_path)
        else:
            shutil.copy(trajectory_path, tgt_trajectory_path)

    with open(os.path.join(tgt_episode_path, "task_descriptions.txt"), "w") as f:
        for task_description in annotations.get(episode_id, {}).values():
            f.write(task_description + "\n")
    return True


def main():
    args = parse_args()
    with open(os.path.join(args.droid_meta_dir, "episode_id_to_path.json"), "r") as f:
        episode_id_to_path = json.load(f)
    with open(os.path.join(args.droid_meta_dir, "droid_language_annotations.json"), "r") as f:
        annotations = json.load(f)

    episode_items = [(k, v) for k, v in episode_id_to_path.items() if "failure" not in v]
    if args.episode_list is not None:
        keep = load_episode_list(args.episode_list)
        episode_items = [item for item in episode_items if item[0] in keep]
    print(f"Number of candidate episodes: {len(episode_items)}")

    os.makedirs(args.output_root, exist_ok=True)
    fn = partial(process_episode, args=args, annotations=annotations)
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(fn, episode_items), total=len(episode_items)))
    print(f"Number of valid episodes: {sum(results)} / {len(episode_items)}")


if __name__ == "__main__":
    main()
