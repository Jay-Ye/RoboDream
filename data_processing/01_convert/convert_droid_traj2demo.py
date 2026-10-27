import argparse
import numpy as np
import os
import json
import h5py
import pickle
import cv2
from tqdm import tqdm
import multiprocessing
from functools import partial
from tqdm import tqdm
from file_utils import ensure_s3_prefix_local, s3_prefix_has_objects
from multiprocessing import Pool, cpu_count
import itertools
import shutil
import glob

def load_video(video_path):
    """Frames of an mp4 (or of a folder of *.jpg / its video.mp4) as a (T, H, W, 3) uint8 array, plus fps."""
    import imageio
    if os.path.isdir(video_path):
        if os.path.isfile(os.path.join(video_path, "video.mp4")):
            return load_video(os.path.join(video_path, "video.mp4"))
        frames = [imageio.imread(f) for f in sorted(glob.glob(os.path.join(video_path, "*.jpg")))]
        return np.array(frames), 10
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        frames = vr.get_batch(np.arange(len(vr))).asnumpy()
        try:
            fps = vr.get_avg_fps()
        except Exception:
            fps = 10
        return frames, fps
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return [], 10


CAM2IMAGE_MAP = {
    "ext1_cam_serial": "exterior_image_1_left",
    "ext2_cam_serial": "exterior_image_2_left",
    "wrist_cam_serial": "wrist_image_left",
}
IMAGE2SCENE_CAM_MAP = {
    "exterior_image_1_left": "external_cam1",
    "exterior_image_2_left": "external_cam2",
    "wrist_image_left": "wrist_cam",
}

DATA_DIR = os.environ.get("ROBODREAM_DATA_DIR", "datasets")
parser = argparse.ArgumentParser(description="Convert raw DROID + robot-only renders into the per-episode Processed layout.")
parser.add_argument("--droid_root", default=os.environ.get("DROID_ROOT", "droid"),
                    help="Raw DROID release root (contains <lab>/success/<date>/<episode>/)")
parser.add_argument("--droid_meta", default=os.path.join(DATA_DIR, "droid_meta"),
                    help="Folder with episode_id_to_path.json, droid_language_annotations.json, camera_serials.json")
parser.add_argument("--render_root", default=os.environ.get("ROBODREAM_RENDER_DIR", os.path.join(DATA_DIR, "droid_renders")),
                    help="Robot-only renders from replay/ (<rel_path>/rgb_sim_rollout_masked/<cam>/%%010d.jpg). "
                         "May also be an s3:// prefix; missing episodes are then synced locally into --render_cache.")
parser.add_argument("--render_cache", default=os.path.join(DATA_DIR, "droid_renders_cache"))
parser.add_argument("--output_dir", default=os.path.join(DATA_DIR, "processed", "droid_traj2demo", "real"))
parser.add_argument("--num_workers", type=int, default=64)
args = parser.parse_args()

path_to_droid_meta = args.droid_meta
if args.render_root.startswith("s3://"):
    ext_data_path = args.render_cache
    s3_ext_data_path = args.render_root
else:
    ext_data_path = args.render_root
    s3_ext_data_path = None

data_path = args.droid_root
tgt_path = args.output_dir
os.makedirs(tgt_path, exist_ok=True)
cam_names = ["external_cam1", "external_cam2", "wrist_cam"]
# scene_prior_rgb / scene_prior_object_gsam are filled later by data_processing/03_object_prior and 04_scene_prior
subdirs = ["lowdim", "rgb", "robot_moving_sim", "scene_prior_rgb"]
resize_wh = (640, 360)

def check_episode_wo_robot_moving_video(episode_item):
    episode_id, episode_meta_info = episode_item
    robot_moving_video_rel = os.path.join(episode_meta_info['rel_path'], "rgb_sim_rollout_masked")
    local_path = os.path.join(ext_data_path, robot_moving_video_rel)
    if os.path.exists(local_path) or (s3_ext_data_path is not None and s3_prefix_has_objects(f"{s3_ext_data_path}/{robot_moving_video_rel}")):
        return episode_item
    return None

def fileter_episodes_wo_robot_moving_video(episode_items):
    num_workers = min(32, cpu_count())
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(check_episode_wo_robot_moving_video, episode_items), total=len(episode_items)))
    s3_episodes = [item for item in results if item is not None]
    print(f"number of episodes with robot-only renders / total: {len(s3_episodes)} / {len(episode_items)}")
    return s3_episodes

## construct episode_meta.json
episode_id_to_path = json.load(open(os.path.join(path_to_droid_meta, "episode_id_to_path.json")))
episode_path_to_id = {v: k for k, v in episode_id_to_path.items()}
episode_id_to_task_description = json.load(open(os.path.join(path_to_droid_meta, "droid_language_annotations.json")))
episode_id_to_cam_to_serial = json.load(open(os.path.join(path_to_droid_meta, "camera_serials.json")))

error_episodes = []
episode_missing_path = []   
episode_meta_file = os.path.join(tgt_path, "filtered_episode_meta25.json")
if not os.path.exists(episode_meta_file):
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
                    error_episodes.append((episode_id, "missing from episode_path_to_id"))
                    episode_missing_path.append(rel_path)
                    continue

                try:
                    episode_meta[episode_id] = {
                        "rel_path": rel_path,
                        "task_descriptions": episode_id_to_task_description[episode_id], # NOTE: there are ~385 episodes without task description, figure out why
                        "cam_name_to_serial": {
                            IMAGE2SCENE_CAM_MAP[CAM2IMAGE_MAP[cam_name]]: serial
                            for cam_name, serial in episode_id_to_cam_to_serial[episode_id].items()
                        }
                    }
                except:
                    error_episodes.append((episode_id, "missing task description"))
                    continue
    print(f"Number of episodes with errors: {len(error_episodes)}") # 3348
    print(f"Number of episodes: {len(episode_meta)}") # 56278
    print(f"Number of episodes with missing path: {len(episode_missing_path)}") # 208

    episode_items = list(episode_meta.items())
    episode_items = fileter_episodes_wo_robot_moving_video(episode_items)
    filtered_episode_meta = {k: v for k, v in episode_items}
    with open(episode_meta_file, "w") as f:
        json.dump(filtered_episode_meta, f, indent=4)
    episode_meta = filtered_episode_meta
else:
    episode_meta = json.load(open(episode_meta_file))
episode_items = list(episode_meta.items())

def process_episode(episode_item):
    episode_id, episode_meta_info = episode_item
    robot_moving_video_rel = os.path.join(episode_meta_info['rel_path'], "rgb_sim_rollout_masked")
    robot_moving_video_path = os.path.join(ext_data_path, robot_moving_video_rel) # contains 3 camera folders
    tgt_episode_path = os.path.join(tgt_path, episode_id)

    # skip scenes already processed # comment out because we want to process the scene prior for all episodes
    if os.path.exists(os.path.join(tgt_episode_path, "lowdim", cam_names[0])) and \
        len(os.listdir(os.path.join(tgt_episode_path, "lowdim", cam_names[0])))>0:
        print(f"Episode {episode_id} already processed")
        return True

    if not os.path.exists(robot_moving_video_path) or \
        len(os.listdir(robot_moving_video_path))==0:
        if s3_ext_data_path is None:
            print(f"Missing robot-only render for episode {episode_id}")
            return False
        ok = ensure_s3_prefix_local(
            s3_base_dir=s3_ext_data_path,
            rel_prefix=robot_moving_video_rel,
            local_dir=robot_moving_video_path,
        )
        if not ok:
            print(f"Missing both locally and in s3 for episode {episode_id}")
            return False

    real_observation_video_path = os.path.join(data_path, episode_meta_info['rel_path'], "recordings", "MP4") # contains 3 camera mp4 files named by camera serial
    trajectory_path = os.path.join(data_path, episode_meta_info['rel_path'], "trajectory.h5")
    trajectory_path = trajectory_path if os.path.exists(trajectory_path) else trajectory_path.replace(".h5", "_im128.h5") # NOTE: in rare cases, the trajectory file is named as trajectory_im128.h5
    with h5py.File(trajectory_path, "r") as f:
        trajectory_cartesian_position = f['observation']['robot_state']['cartesian_position'][()]
        trajectory_gripper_position = f['observation']['robot_state']['gripper_position'][()][:, None]

    real_observations = {}
    robot_moving_video = {}
    for cam_name in cam_names:
        for subdir in subdirs:
            os.makedirs(os.path.join(tgt_episode_path, subdir, cam_name), exist_ok=True)
        real_observations[cam_name], _ = load_video(os.path.join(real_observation_video_path, episode_meta_info["cam_name_to_serial"][cam_name] + ".mp4"))
        robot_moving_video[cam_name], _ = load_video(os.path.join(robot_moving_video_path, cam_name))
    min_len = min(min(len(real_observations[cam_name]), len(robot_moving_video[cam_name])) for cam_name in cam_names)
    # min_len = 1
    if min_len == 0:
        print(f"No frames found for episode {episode_id}")
        return False
    for t in range(min_len):
        for cam_name in cam_names:
            # Convert RGB to BGR before saving with OpenCV
            rgb_img = cv2.resize(real_observations[cam_name][t], resize_wh)
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tgt_episode_path, "rgb", cam_name, f"{t:010}.jpg"), bgr_img)

            sim_rgb_img = cv2.resize(robot_moving_video[cam_name][t], resize_wh)
            sim_bgr_img = cv2.cvtColor(sim_rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tgt_episode_path, "robot_moving_sim", cam_name, f"{t:010}.jpg"), sim_bgr_img)

            curr_frame = np.zeros((min_len, 1)) 
            curr_frame[t] = 1 # indicate the current frame
            proprio = {
                "cartesian_position": trajectory_cartesian_position[:min_len],
                "gripper_position": trajectory_gripper_position[:min_len],
                "current_frame": curr_frame,
            }
            np.savez(
                os.path.join(tgt_episode_path, "lowdim", cam_name, f"{t:010}.npz"),
                shape=resize_wh,
                proprio=np.asarray(proprio, dtype=object),
                prompt=list(episode_meta_info['task_descriptions'].values()),
                # prompt_embeddings=np.asarray(embeds, dtype=object),
            )

    return True
    # print(f"processed {min_len} frames for episode {episode_id}")

def write_json(tgt_path):
    episode_list = os.listdir(tgt_path)
    json_dict = {}
    for episode_id in tqdm(episode_list):
        if not os.path.isdir(os.path.join(tgt_path, episode_id)):
            print(f"skipping non-directory {os.path.join(tgt_path, episode_id)}")
            continue
        img_list = sorted(os.listdir(os.path.join(tgt_path, episode_id, "rgb", cam_names[0])))
        json_dict[episode_id] = img_list
    with open(os.path.join(tgt_path, "split_all.json"), "w") as f:
        json.dump(json_dict, f, indent=4)

# # debug mode
# debug = 10 
# if debug > 0:
#     episode_items = episode_items[:debug]

# for episode_item in tqdm(episode_items):
#     # if episode_item[0] != 'AUTOLab+9aed6d7a+2023-12-02-17h-32m-21s':
#     #     continue
#     success = process_episode(episode_item)
#     if not success:
#         error_episodes.append((episode_item[0], "missing robot moving video"))

# real running
episode_items = episode_items[::-1]
with multiprocessing.Pool(processes=args.num_workers) as pool: # multiprocessing.cpu_count()
    for episode_item, success in zip(episode_items, tqdm(pool.imap(process_episode, episode_items), total=len(episode_items))):
        if not success:
            error_episodes.append((episode_item[0], "missing robot moving video"))

write_json(tgt_path)