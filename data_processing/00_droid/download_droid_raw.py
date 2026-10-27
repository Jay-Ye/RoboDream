import json
import os
from tqdm import tqdm
import multiprocessing
from functools import partial
import subprocess

import argparse

parser = argparse.ArgumentParser(description="Download raw DROID 1.0.1 episodes with good camera extrinsics (no .svo).")
parser.add_argument("--droid_meta", default=os.path.join(os.environ.get("ROBODREAM_DATA_DIR", "datasets"), "droid_meta"),
                    help="Folder with cam2base_extrinsic_superset.json (huggingface.co/KarlP/droid)")
parser.add_argument("--output_dir", default=os.environ.get("DROID_ROOT", "droid"))
parser.add_argument("--save_episode_list", default=None,
                    help="Optionally dump the selected episodes (episodes_with_good_extrinsics.json) here")
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

path_to_droid_repo = args.droid_meta
dataset_path = args.output_dir

def download_episode(episode_data, dataset_path):
    episode_id, episode_metadata = episode_data
    target_path = os.path.join(dataset_path, episode_metadata['relative_path'])
    os.makedirs(target_path, exist_ok=True)
    gs_path = f"gs://gresearch/robotics/droid_raw/1.0.1/{episode_metadata['relative_path']}"
    exclude_pattern = r".*\.svo$"  # Regex to exclude .svo files

    subprocess.run([
        "gsutil", "-q", "-m", "rsync", "-r",
        "-x", exclude_pattern,
        gs_path, target_path
    ], check=True)

cam2base_extrinsics_path = f"{path_to_droid_repo}/cam2base_extrinsic_superset.json"
with open(cam2base_extrinsics_path, "r") as f:
    cam2base_extrinsics = json.load(f)

episodes_with_good_extrinsics = {}
for episode_id, episode_metadata in cam2base_extrinsics.items():
    if "failure" in episode_metadata['relative_path']:
        continue
    episode_IoUs = []
    reprojection_errors = []
    for key, value in episode_metadata.items():
        if "metric_type" in key and episode_metadata[key] == "IoU":
            episode_IoUs.append(episode_metadata[key.replace("metric_type", "quality_metric")])
        if "metric_type" in key and episode_metadata[key] == "Reprojection_error":
            reprojection_errors.append(episode_metadata[key.replace("metric_type", "quality_metric")])
    if (len(episode_IoUs)+len(reprojection_errors)) >= 2 and all(x > 0.75 for x in episode_IoUs) and all(x < 5 for x in reprojection_errors):
        episodes_with_good_extrinsics[episode_id] = episode_metadata
print("Number of episodes with good extrinsics: ", len(episodes_with_good_extrinsics))
if args.save_episode_list is not None:
    with open(args.save_episode_list, "w") as f:
        json.dump(episodes_with_good_extrinsics, f)

# Create process pool and download in parallel
with multiprocessing.Pool(processes=args.num_workers) as pool:
    list(tqdm(
        pool.imap_unordered(
            partial(download_episode, dataset_path=dataset_path),
            episodes_with_good_extrinsics.items(),
            chunksize=1
        ),
        total=len(episodes_with_good_extrinsics)
    ))

