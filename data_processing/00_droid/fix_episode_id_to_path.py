import os
import json
import shutil
import argparse

parser = argparse.ArgumentParser(description="Patch episode_id_to_path.json with episodes found in the raw DROID folder.")
parser.add_argument("--droid_root", default=os.environ.get("DROID_ROOT", "droid"))
parser.add_argument("--droid_meta", default=os.path.join(os.environ.get("ROBODREAM_DATA_DIR", "datasets"), "droid_meta"))
args = parser.parse_args()

data_path = args.droid_root
path_to_droid_meta = args.droid_meta
episode_id_to_path = json.load(open(os.path.join(path_to_droid_meta, "episode_id_to_path.json")))
episode_path_to_id = {v: k for k, v in episode_id_to_path.items()}
# episode_id_to_task_description = json.load(open(os.path.join(path_to_droid_meta, "droid_language_annotations.json")))

episode_meta = {}
for station in os.listdir(data_path):
    station_dir = os.path.join(data_path, station, "success")
    if not os.path.isdir(station_dir):
        continue
    for date in os.listdir(station_dir):
        date_dir = os.path.join(station_dir, date)
        for episode in os.listdir(date_dir):
            episode_dir = os.path.join(date_dir, episode)
            rel_path = os.path.relpath(episode_dir, data_path)
            if rel_path in episode_path_to_id:
                episode_id = episode_path_to_id[rel_path]
            elif rel_path.replace(":", "_") in episode_path_to_id:
                episode_id = episode_path_to_id[rel_path.replace(":", "_")]
            else:
                if os.path.exists(os.path.join(data_path, rel_path)):
                    json_files = [f for f in os.listdir(os.path.join(data_path, rel_path)) if f.endswith('.json')]
                    if len(json_files) != 1:
                        continue
                    episode_meta = json.load(open(os.path.join(data_path, rel_path, json_files[0])))
                    episode_id = episode_meta['uuid']
                    episode_rel_path = os.path.join(episode_meta['lab'], episode_meta['hdf5_path']).rsplit('/', 1)[0]
                    episode_path_to_id[episode_rel_path] = episode_id

            # if episode_id not in episode_id_to_task_description:
            #     breakpoint()
            #     if os.path.exists(os.path.join(data_path, rel_path, "trajectory.h5")):
            #         with h5py.File(os.path.join(data_path, rel_path, "trajectory.h5"), "r") as f:
            #             for key, value in f.attrs.items():
            #                 print(key, value)
            #             breakpoint()

episode_id_to_path = {v: k for k, v in episode_path_to_id.items()}
with open(os.path.join(path_to_droid_meta, "episode_id_to_path_fixed.json"), "w") as f:
    json.dump(episode_id_to_path, f, indent=4)

shutil.copy(os.path.join(path_to_droid_meta, "episode_id_to_path.json"), os.path.join(path_to_droid_meta, "episode_id_to_path_original.json"))
shutil.copy(os.path.join(path_to_droid_meta, "episode_id_to_path_fixed.json"), os.path.join(path_to_droid_meta, "episode_id_to_path.json"))