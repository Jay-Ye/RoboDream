"""
Render robot-only (anchor) videos for DROID episodes by replaying their trajectories in IsaacLab.

This produced the `mask_rgb` (robot-only) modality used to train RoboDream. For every episode with
good camera extrinsics, the recorded end-effector trajectory is replayed in simulation and rendered
from the calibrated DROID camera poses with a black background.

Output layout (consumed by data_processing/01_convert):
    ${ROBODREAM_RENDER_DIR}/<relative_path>/rgb_sim_rollout_masked/{external_cam1,external_cam2,wrist_cam}/%010d.jpg
    ${ROBODREAM_RENDER_DIR}/<relative_path>/rgb_real_rollout_masked/...   (real frames masked by the sim robot; skipped with --skip_real_obs)

Usage (inside the replay venv, from replay/):
    python render_droid_robot_only.py --num_envs 2 [--shard_id 0 --num_shards 8]
"""

import argparse
import json
import os
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation as R

from droid_meta.config import format_intrinsics, CAM2IMAGE_MAP, SCENE_CAM2IMAGE_MAP, DROID_META_DIR, local_dataset_path, replay_img_save_path
from replay.utils import resize_intrinsics, load_episode_data, load_episode_images
from replay.replay_functions import create_replay_context, close_context
from replay.replay import batched_replay


def parse_args():
    parser = argparse.ArgumentParser(description="Render robot-only DROID replays in IsaacLab")
    parser.add_argument("--droid_root", type=str, default=local_dataset_path, help="Raw DROID release root (DROID_ROOT)")
    parser.add_argument("--droid_meta_dir", type=str, default=DROID_META_DIR,
                        help="Folder with episode_id_to_path.json, intrinsics.json, camera_serials.json, pnp_cam2base_multiview.json")
    parser.add_argument("--output_root", type=str, default=replay_img_save_path, help="Where robot-only frames are written")
    parser.add_argument("--video_dir", type=str, default=None, help="Optionally also save a preview mp4 per episode here")
    parser.add_argument("--control_mode", type=str, default="end_effector", choices=["end_effector", "joint_position"])
    parser.add_argument("--num_envs", type=int, default=2, help="Episodes replayed in parallel")
    parser.add_argument("--skip_real_obs", action="store_true", help="Do not load / mask the real DROID frames")
    parser.add_argument("--min_length", type=int, default=45)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args, _ = parser.parse_known_args()  # remaining args are consumed by IsaacLab's AppLauncher
    return args


def load_episodes(meta_dir):
    with open(os.path.join(meta_dir, "pnp_cam2base_multiview.json"), "r") as f:
        cam2base_extrinsics = json.load(f)

    successful_episodes = {}
    for episode_id, episode_metadata in cam2base_extrinsics.items():
        if "failure" in episode_metadata['relative_path']:
            continue
        if len(episode_metadata) == 1:  # camera param is missing
            continue
        successful_episodes[episode_id] = episode_metadata
    print("Number of successful episodes:", len(successful_episodes))

    with open(os.path.join(meta_dir, "intrinsics.json"), "r") as f:
        intrinsics_raw = json.load(f)
    intrinsics = {}
    for episode_id, intrinsics_dict in intrinsics_raw.items():
        # reformat the intrinsics, set default intrinsics if not available
        intrinsics[episode_id] = {cam_name: format_intrinsics(cam_intrinsics['cameraMatrix'], "wrist" in cam_name)
                                  for cam_name, cam_intrinsics in intrinsics_dict.items()}

    with open(os.path.join(meta_dir, "camera_serials.json"), "r") as f:
        camera_serials = json.load(f)
    camera_serial_to_name = {episode_id: {serial: name for name, serial in episode_metadata.items()}
                             for episode_id, episode_metadata in camera_serials.items()}

    # use camera names instead of serials
    episodes = {}
    for episode_id, episode_metadata in successful_episodes.items():
        try:
            serial_to_name = camera_serial_to_name[episode_id]
            episodes[episode_id] = {
                'relative_path': episode_metadata['relative_path'],
                **{CAM2IMAGE_MAP[serial_to_name[s]]: episode_metadata[s] for s in serial_to_name.keys() if 'wrist' not in serial_to_name[s]},
                **{f"{CAM2IMAGE_MAP[serial_to_name[s]]}_intrinsics": intrinsics[episode_id][s] for s in serial_to_name.keys()},
                **{f"{CAM2IMAGE_MAP[serial_to_name[s]]}_to_serial": s for s in serial_to_name.keys()},
            }
        except Exception as e:
            print(f"Error processing episode {episode_id}: {e}")
    return episodes


def get_camera_params(episode_metadata):
    camera_params = {}
    for cam_name, img_name in SCENE_CAM2IMAGE_MAP.items():
        cam_extrinsics = episode_metadata.get(img_name, None)
        if cam_extrinsics is None:
            camera_pose_np = None
        else:
            camera_pose_np = np.eye(4)
            camera_pose_np[:3, :3] = R.from_euler("xyz", cam_extrinsics[3:6]).as_matrix()
            camera_pose_np[:3, 3] = cam_extrinsics[0:3]

        cam_intrinsics = episode_metadata[f"{img_name}_intrinsics"]
        fx, cx, fy, cy = cam_intrinsics["cameraMatrix"]
        intrinsics_np = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]])
        if cam_intrinsics["width"] == 672 and cam_intrinsics["height"] == 376:
            intrinsics_np = resize_intrinsics(intrinsics_np, src_size=(672, 376), dst_size=(1280, 720))

        camera_params[cam_name] = {
            "camera_pose": camera_pose_np,
            "intrinsics": intrinsics_np,
            "camera_serial": episode_metadata[f"{img_name}_to_serial"],
        }
    return camera_params


def main():
    args = parse_args()
    episodes = load_episodes(args.droid_meta_dir)
    episode_ids = sorted(episodes.keys())[args.shard_id::args.num_shards]
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(episode_ids)} episodes")

    ctx = create_replay_context(headless=True, scene=1, num_envs=args.num_envs,
                                env_name="DROID" if args.control_mode == "joint_position" else "DROID_EE")
    batch_store = {
        "camera_params": [],
        "actions": [],
        "real_obs": [],
        "video_paths": [],
        "img_paths": [],
        "episode_ids": [],
    }
    try:
        for episode_id in episode_ids:
            episode_metadata = episodes[episode_id]
            episode_dir = os.path.join(args.droid_root, episode_metadata["relative_path"])
            out_dir = os.path.join(args.output_root, episode_metadata["relative_path"])
            try:
                if not os.path.exists(os.path.join(episode_dir, "trajectory.h5")):
                    print(f"Skipping episode {episode_id} because it does not exist")
                    continue
                if os.path.isdir(os.path.join(out_dir, "rgb_sim_rollout_masked")):
                    continue  # already rendered

                camera_params = get_camera_params(episode_metadata)
                joint_positions, gripper_actions, cartesian_poses = load_episode_data(os.path.join(episode_dir, "trajectory.h5"))

                episode_length = min(len(joint_positions), len(gripper_actions), len(cartesian_poses))
                if episode_length > args.max_length or episode_length < args.min_length:
                    print(f"Skipping episode {episode_id} because it is too long or too short ({episode_length})")
                    continue
                if not args.skip_real_obs:
                    images = load_episode_images(os.path.join(episode_dir, "recordings", "MP4"), camera_params)
                    episode_length = min(episode_length, len(images[list(images.keys())[0]]))
                    for cam in images.keys():
                        images[cam] = images[cam][:episode_length]
                else:
                    images = None

                if args.control_mode == "joint_position":
                    actions = np.concatenate([joint_positions[:episode_length], gripper_actions[:episode_length]], axis=-1)
                else:
                    actions = np.concatenate([cartesian_poses[:episode_length], gripper_actions[:episode_length]], axis=-1)

                batch_store["camera_params"].append(deepcopy(camera_params))
                batch_store["actions"].append(deepcopy(actions))
                batch_store["real_obs"].append(deepcopy(images))
                if args.video_dir is not None:
                    batch_store["video_paths"].append(os.path.join(args.video_dir, f"replay_{episode_id}.mp4"))
                batch_store["img_paths"].append(out_dir)
                batch_store["episode_ids"].append(episode_id)

                if len(batch_store["episode_ids"]) >= args.num_envs:
                    batched_replay(ctx, batch_store, args.control_mode, args.skip_real_obs)
            except Exception as e:
                print(f"Error processing episode {episode_id}: {e}")
                continue
        batched_replay(ctx, batch_store, args.control_mode, args.skip_real_obs)
    finally:
        close_context(ctx)


if __name__ == "__main__":
    main()
