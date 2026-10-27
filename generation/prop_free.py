"""
prop_free.py

Prop-free teleoperation variant of retrieval_rebirth.py.
Instead of retrieving trajectories from DROID by language similarity, it
takes trajectories from locally collected DROID-format episodes
(<data_root>/<dataset>/<episode>/trajectory.h5), e.g. prop-free teleop data
recorded without any objects in the scene:

  <data_root>/cloth_noobject/  — wiping motions
  <data_root>/propfree_pnp/    — pick-and-place motions (shared by the 3 pick-and-place tasks)

The scene prior is always the first frame of each episode's
  recordings/SVO/33790348.mp4
extracted and cached as a .png next to the video, unless
--scene_prior_rgb_path is explicitly set.
"""

import sys
from pathlib import Path
import argparse
import os
import json
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from typing import Dict, List
from dataclasses import dataclass
import cv2
import h5py
from concurrent.futures import ThreadPoolExecutor

# Shared helpers — imported directly to avoid duplication.
# retrieval_rebirth is a script; Python won't execute its
# __main__ block on import. Importing it also puts replay/ on sys.path.
from retrieval_rebirth import (
    split_object_prompts,
    build_object_prompts_per_query,
    get_replay_cache_path,
    load_replay_cache,
    save_replay_cache,
    _save_camera_params,
    _camera_params_match,
    _demo_exists_with_matching_params,
    get_scene_prior_cache_paths,
    load_or_create_scene_prior,
    _handle_server_results,
    load_calibration_camera_params,
    DATA_DIR,
)

from replay.replay_functions import create_replay_context
from replay.replay import batched_replay
from droid_meta.config import DEFAULT_CAMERA_INTRINSICS, DEFAULT_CAMERA_INTRINSICS_WRIST

from retrieval.retriever_new import cartesian_rpy_to_quat

from utils.server_utils import send_to_inference_server, format_data_for_server
from utils.TrajEngine_utils import save_demonstration
from GSamObjectPrior import GroundedSamScenePrior, GSamScenePriorConfig
from ObjectRemoval import ObjectRemover, ObjectRemovalConfig

random.seed(42)

# ---------------------------------------------------------------------------
# Task → dataset mapping
# ---------------------------------------------------------------------------

TASK_TO_DATASET = {
    "prop_free": {
        "wipe the table with the towel": "cloth_noobject",
        "put the marker in the bowl": "propfree_pnp",
        "put the yellow block in the cup": "propfree_pnp",
        "remove the marker from the bowl and put it on the table": "propfree_pnp",
    },
    "real_gen": {
        "wipe the table with the towel": "cloth",
        "put the marker in the bowl": "put_pen",
        "put the yellow block in the cup": "cube",
        "remove the marker from the bowl and put it on the table": "remove_pen",
    },
}


def get_dataset_for_task(task: str, dataset_mode: str) -> str:
    mapping = TASK_TO_DATASET[dataset_mode]
    key = task.strip().lower()
    if key not in mapping:
        raise ValueError(
            f"Unknown task: {task!r}. Known tasks: {list(mapping.keys())}"
        )
    return mapping[key]


# ---------------------------------------------------------------------------
# Local trajectory loading
# ---------------------------------------------------------------------------

@dataclass
class LocalTrajectoryResult:
    sample_path: str    # full path to episode directory
    trajectory: Dict    # dict of numpy arrays


def load_local_trajectory(
    episode_dir: str,
    with_action: bool = False,
    downsample_factor: int = 1,
) -> Dict:
    h5_path = os.path.join(episode_dir, "trajectory.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"trajectory.h5 not found: {h5_path}")

    trajectory: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        trajectory["state_xyzrpy"] = f["observation"]["robot_state"]["cartesian_position"][()]
        trajectory["state_cartesian"] = cartesian_rpy_to_quat(trajectory["state_xyzrpy"])
        trajectory["state_gripper"] = (
            f["observation"]["robot_state"]["gripper_position"][()].reshape(-1, 1)
        )
        if with_action:
            trajectory["action_xyzrpy"] = f["action"]["cartesian_position"][()]
            trajectory["action_cartesian"] = cartesian_rpy_to_quat(trajectory["action_xyzrpy"])
            trajectory["action_gripper"] = f["action"]["gripper_position"][()].reshape(-1, 1)

    if downsample_factor > 1:
        for key in trajectory:
            trajectory[key] = trajectory[key][::downsample_factor]

    return trajectory


def sample_local_trajectories(
    data_root: str,
    dataset: str,
    num_samples: int,
    with_action: bool = False,
    downsample_factor: int = 1,
) -> List[LocalTrajectoryResult]:
    dataset_dir = os.path.join(data_root, dataset)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    episodes = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
        and os.path.exists(os.path.join(dataset_dir, d, "trajectory.h5"))
    ]
    if not episodes:
        raise RuntimeError(f"No valid episodes found in {dataset_dir}")

    # chosen = episodes if (num_samples == -1 or num_samples >= len(episodes)) \
    #     else random.sample(episodes, num_samples)
    chosen = episodes if (num_samples == -1 or num_samples >= len(episodes)) \
        else episodes[:num_samples]

    return [
        LocalTrajectoryResult(
            sample_path=os.path.join(dataset_dir, ep),
            trajectory=load_local_trajectory(
                os.path.join(dataset_dir, ep),
                with_action=with_action,
                downsample_factor=downsample_factor,
            ),
        )
        for ep in chosen
    ]


# ---------------------------------------------------------------------------
# First-frame extraction
# ---------------------------------------------------------------------------

VIDEO_SERIAL = "33790348"


def get_or_extract_first_frame(episode_dir: str) -> str:
    """Extract frame 0 of recordings/SVO/33790348.mp4, cache as _frame0.png."""
    video_path = os.path.join(episode_dir, "recordings", "SVO", f"{VIDEO_SERIAL}.mp4")
    frame_path = os.path.join(episode_dir, "recordings", "SVO", f"{VIDEO_SERIAL}_frame0.png")

    if os.path.exists(frame_path):
        return frame_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Failed to read first frame from: {video_path}")
    cv2.imwrite(frame_path, frame)
    return frame_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if args.scene_prior_rgb_path is not None and not os.path.isfile(args.scene_prior_rgb_path):
        raise ValueError(f"--scene_prior_rgb_path not found: {args.scene_prior_rgb_path}")
    if args.object_prior_rgb_path is not None:
        if len(args.object_prior_rgb_path) != len(args.query):
            raise ValueError(
                f"--object_prior_rgb_path must have the same number of entries as --query "
                f"(got {len(args.object_prior_rgb_path)} paths for {len(args.query)} queries)"
            )
        for path in args.object_prior_rgb_path:
            if not os.path.isfile(path):
                raise ValueError(f"--object_prior_rgb_path not found: {path}")
    if args.visualize and not os.path.isdir(args.droid_rgb_root):
        raise ValueError(f"--droid_rgb_root not found: {args.droid_rgb_root}")

    server_urls = args.server_url or []
    if args.suffix is None:
        server_suffixes = [f"server{i}" for i in range(len(server_urls))]
        broadcast_mode = True
    else:
        if len(args.suffix) == len(server_urls):
            broadcast_mode = True
        elif len(args.suffix) == 1:
            broadcast_mode = False
        else:
            raise ValueError(
                f"--suffix must match --server_url count or be exactly 1 "
                f"(got {len(args.suffix)} suffixes for {len(server_urls)} URLs)"
            )
        server_suffixes = args.suffix
    server_targets = list(zip(server_urls, server_suffixes)) if broadcast_mode \
        else [(url, server_suffixes[0]) for url in server_urls]

    # ---------------------------------------------------------------------------
    # Sample trajectories
    # ---------------------------------------------------------------------------
    trajectories: List[LocalTrajectoryResult] = []
    queries: List[str] = []
    samples_per_query: List[int] = []
    for single_query in args.query:
        samples = sample_local_trajectories(
            data_root=args.data_root,
            dataset=get_dataset_for_task(single_query, args.dataset_mode),
            num_samples=args.num_samples,
            with_action=args.with_action,
            downsample_factor=args.downsample_factor,
        )
        trajectories.extend(samples)
        queries.extend([single_query] * len(samples))
        samples_per_query.append(len(samples))

    actions_replay = [
        np.concatenate([t.trajectory["state_cartesian"], t.trajectory["state_gripper"]], axis=1)
        for t in trajectories
    ]
    actions_retrieval = [
        np.concatenate([t.trajectory["state_xyzrpy"], t.trajectory["state_gripper"]], axis=1)
        for t in trajectories
    ]
    episode_ids = [Path(t.sample_path).name for t in trajectories]
    suffix_str = f"_{args.episode_suffix}" if args.episode_suffix else ""
    output_episode_ids = [f"{eid}{suffix_str}" for eid in episode_ids]

    object_prompts_per_query = build_object_prompts_per_query(args.query, args.object_prompts)
    object_prompts = [
        prompts
        for prompts, n in zip(object_prompts_per_query, samples_per_query)
        for _ in range(n)
    ]
    object_prior_paths = (
        [path for path, n in zip(args.object_prior_rgb_path, samples_per_query) for _ in range(n)]
        if args.object_prior_rgb_path
        else [None] * len(trajectories)
    )

    # ---------------------------------------------------------------------------
    # Camera parameters
    # ---------------------------------------------------------------------------
    camera_param = load_calibration_camera_params(args.cam_calib_path, args.cam_serials)
    camera_params = [camera_param] * len(trajectories)

    # ---------------------------------------------------------------------------
    # Replay + inference loop (mirrors retrieval_rebirth.py)
    # ---------------------------------------------------------------------------
    N_Parallel = min(6, len(trajectories))
    control_mode = "end_effector"
    os.makedirs(args.replay_cache_dir, exist_ok=True)

    total_samples = len(trajectories)
    results = []
    replay_ctx = None

    gsam_prior = GroundedSamScenePrior(GSamScenePriorConfig(input_is_bgr=True, device=args.device))
    remover = ObjectRemover(ObjectRemovalConfig(input_is_bgr=True, device=args.device))

    print(f"\n{'='*60}")
    print(f"Processing {total_samples} samples in batches of {N_Parallel}")
    print(f"{'='*60}")

    for batch_start in range(0, total_samples, N_Parallel):
        batch_end = min(batch_start + N_Parallel, total_samples)

        print(f"\n{'='*60}")
        print(f"Processing batch: samples {batch_start+1} to {batch_end}")
        print(f"{'='*60}")

        batch_actions_replay      = actions_replay[batch_start:batch_end]
        batch_camera_params       = camera_params[batch_start:batch_end]
        batch_episode_ids         = episode_ids[batch_start:batch_end]
        batch_output_episode_ids  = output_episode_ids[batch_start:batch_end]
        batch_trajectories        = trajectories[batch_start:batch_end]

        # --- Replay (with cache) ---
        camera_images_by_episode = {}
        replay_indices = []
        for local_idx, episode_id in enumerate(batch_episode_ids):
            cached = load_replay_cache(args.replay_cache_dir, episode_id)
            desired = batch_camera_params[local_idx]
            if cached is not None and _camera_params_match(cached.get("camera_params"), desired):
                camera_images_by_episode[episode_id] = cached["camera_images"]
            else:
                if cached is not None:
                    print(f"[INFO] Cache for {episode_id} has mismatched camera poses; re-replaying.")
                replay_indices.append(local_idx)

        if replay_indices:
            replay_episode_ids   = [batch_episode_ids[i] for i in replay_indices]
            replay_actions       = [batch_actions_replay[i] for i in replay_indices]
            replay_camera_params = [batch_camera_params[i] for i in replay_indices]
            original_batch_size  = len(replay_episode_ids)

            if original_batch_size < N_Parallel:
                pad_count = N_Parallel - original_batch_size
                for pad_idx in range(pad_count):
                    replay_episode_ids.append(f"{replay_episode_ids[0]}__pad_{batch_start}_{pad_idx}")
                    replay_actions.append(replay_actions[0])
                    replay_camera_params.append(replay_camera_params[0])
                print(f"[INFO] Padded replay batch from {original_batch_size} to {N_Parallel}.")

            batch_store = {
                "episode_ids": replay_episode_ids.copy(),
                "actions": replay_actions,
                "camera_params": replay_camera_params,
                "real_obs": None,
            }
            if replay_ctx is None:
                replay_ctx = create_replay_context(
                    headless=True, scene=1, num_envs=N_Parallel, env_name="DROID_EE",
                )
            outputs = batched_replay(replay_ctx, batch_store, control_mode, skip_real_obs=False)
            replayed_camera_images = outputs.get("camera_images", {})
            for out_idx, episode_id in enumerate(replay_episode_ids[:original_batch_size]):
                camera_images = replayed_camera_images.get(out_idx)
                if camera_images is None:
                    raise RuntimeError(f"Missing replay output for index {out_idx} ({episode_id})")
                camera_images_by_episode[episode_id] = camera_images
                save_replay_cache(
                    args.replay_cache_dir, episode_id,
                    {"camera_images": camera_images,
                     "camera_params": batch_camera_params[replay_indices[out_idx]]},
                )

        batched_camera_images  = [camera_images_by_episode[eid] for eid in batch_episode_ids]
        batch_actions_retrieval  = actions_retrieval[batch_start:batch_end]
        batch_queries            = queries[batch_start:batch_end]
        batch_object_prompts     = object_prompts[batch_start:batch_end]
        batch_object_prior_paths = object_prior_paths[batch_start:batch_end]

        # --- Inference ---
        distributed_executor = (
            ThreadPoolExecutor(max_workers=len(server_targets))
            if (server_targets and not broadcast_mode) else None
        )
        distributed_pending = []

        def send_to_all_servers(data_dict, targets=None):
            effective_targets = targets if targets is not None else server_targets
            if not effective_targets:
                return []
            submitted = []
            with ThreadPoolExecutor(max_workers=len(effective_targets)) as executor:
                for server_url, server_suffix in effective_targets:
                    future = executor.submit(
                        send_to_inference_server,
                        data_dict=data_dict,
                        server_url=server_url,
                        timeout=args.timeout,
                    )
                    submitted.append((server_url, server_suffix, future))
                responses = []
                for server_url, server_suffix, future in submitted:
                    try:
                        response = future.result()
                        success = response.get("status") == "success"
                        responses.append({
                            "sample_id": output_episode_id,
                            "server_url": server_url,
                            "server_suffix": server_suffix,
                            "response": response,
                            "success": success,
                        })
                    except Exception as e:
                        responses.append({
                            "sample_id": output_episode_id,
                            "server_url": server_url,
                            "server_suffix": server_suffix,
                            "error": str(e),
                            "success": False,
                        })
                return responses

        for local_idx, (camera_images, action, prompt, episode_id, prompt_objects, traj, object_prior_path) in enumerate(
            zip(batched_camera_images, batch_actions_retrieval, batch_queries,
                batch_episode_ids, batch_object_prompts, batch_trajectories, batch_object_prior_paths)
        ):
            global_idx = batch_start + local_idx
            output_episode_id = batch_output_episode_ids[local_idx]
            episode_camera_params = batch_camera_params[local_idx]
            task_name = f"{prompt.replace(' ', '_')}_{args.dataset_mode}"

            if server_targets:
                if broadcast_mode:
                    active_targets = [
                        (url, suffix) for url, suffix in server_targets
                        if not _demo_exists_with_matching_params(
                            os.path.join(args.output_dir, suffix, task_name, output_episode_id),
                            episode_camera_params,
                        )
                    ]
                    if not active_targets:
                        print(f"\n--- Skipping {global_idx+1}/{total_samples} ({output_episode_id}): already exists on all servers. ---")
                        continue
                else:
                    server_url, server_suffix = server_targets[global_idx % len(server_targets)]
                    demo_path = os.path.join(args.output_dir, server_suffix, task_name, output_episode_id)
                    if _demo_exists_with_matching_params(demo_path, episode_camera_params):
                        print(f"\n--- Skipping {global_idx+1}/{total_samples} ({output_episode_id}): already exists on {server_suffix}. ---")
                        continue

            print(f"\n--- Processing sample {global_idx+1}/{total_samples} ---")

            # Scene prior: first frame of the episode's video, or the override.
            scene_prior_source_path = (
                args.scene_prior_rgb_path
                if args.scene_prior_rgb_path is not None
                else get_or_extract_first_frame(traj.sample_path)
            )

            scene_prior_rgb, scene_prior_object_rgb = load_or_create_scene_prior(
                scene_prior_path=scene_prior_source_path,
                object_prompts=prompt_objects,
                gsam_prior=gsam_prior,
                remover=remover,
                target_size=(320, 176),
                scene_prior_rgb_override_path=args.scene_prior_rgb_path,
                object_prior_rgb_override_path=object_prior_path,
            )

            data_dict = format_data_for_server(
                scene_prior_rgb=scene_prior_rgb.copy(),
                scene_prior_object_rgb=scene_prior_object_rgb.copy(),
                camera_images=camera_images,
                trajectory=action,
                prompt=prompt,
                sample_id=output_episode_id,
                fps=10.0,
                target_size=(176, 320),
            )

            if server_targets:
                if broadcast_mode:
                    server_results = send_to_all_servers(data_dict, targets=active_targets)
                    _handle_server_results(
                        server_results, results, args, camera_images, action, prompt,
                        scene_prior_rgb, scene_prior_object_rgb, output_episode_id, episode_camera_params,
                        task_name_override=task_name,
                    )
                else:
                    future = distributed_executor.submit(
                        send_to_inference_server,
                        data_dict=data_dict,
                        server_url=server_url,
                        timeout=args.timeout,
                    )
                    distributed_pending.append((
                        future, server_url, server_suffix, output_episode_id, global_idx,
                        camera_images, action, prompt, scene_prior_rgb, scene_prior_object_rgb,
                        episode_camera_params,
                    ))

        if distributed_pending:
            print(f"[INFO] Collecting {len(distributed_pending)} distributed inference results...")
            for (future, server_url, server_suffix, ep_id, _,
                 cam_imgs, act, pmt, sp_rgb, sp_obj_rgb, ep_cam_params) in distributed_pending:
                try:
                    response = future.result()
                    server_result = {
                        "sample_id": ep_id, "server_url": server_url,
                        "server_suffix": server_suffix, "response": response,
                        "success": response.get("status") == "success",
                    }
                except Exception as e:
                    server_result = {
                        "sample_id": ep_id, "server_url": server_url,
                        "server_suffix": server_suffix, "error": str(e), "success": False,
                    }
                _handle_server_results(
                    [server_result], results, args, cam_imgs, act, pmt,
                    sp_rgb, sp_obj_rgb, ep_id, ep_cam_params,
                    task_name_override=task_name,
                )
            distributed_executor.shutdown(wait=False)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Total samples processed: {len(results)}")
    if server_targets:
        successful = sum(1 for r in results if r.get("success", False))
        print(f"Successfully sent to server: {successful}/{len(results)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prop-free trajectory generation from local datasets"
    )
    parser.add_argument("--query", type=str, nargs="+",
                        default=["Put the marker in the bowl"],
                        help="Task query. Must be one of the known prop-free tasks.")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Trajectories to sample per query (-1 = all available)")
    parser.add_argument("--with_action", action="store_true",
                        help="Also load action fields from trajectory.h5")
    parser.add_argument("--downsample_factor", type=int, default=1,
                        help="Downsample the trajectory by this factor")
    parser.add_argument("--data_root", type=str, default=os.path.join(DATA_DIR, "propfree"),
                        help="Root containing the dataset subdirectories")
    parser.add_argument("--dataset_mode", type=str, default="prop_free",
                        choices=list(TASK_TO_DATASET.keys()),
                        help="Dataset mapping to use: 'prop_free' (cloth_noobject/propfree_pnp) "
                             "or 'real_gen' (cloth/put_pen/cube/remove_pen)")
    parser.add_argument("--cam_calib_path", type=str, required=True,
                        help="Path to the camera calibration JSON file (see metadata/calibration_example.json)")
    parser.add_argument("--cam_serials", type=str, nargs=3, default=["33790348", "37998989", "14064085"],
                        help="Camera serials in the calibration file for external_cam1, external_cam2 and wrist_cam")
    parser.add_argument("--object_prompts", type=str, nargs="*",
                        help="Object prompts for GSam")
    parser.add_argument("--scene_prior_rgb_path", type=str, default=None,
                        help="Pre-inpainted scene-prior RGB; skips inpainting for all samples.")
    parser.add_argument("--object_prior_rgb_path", type=str, nargs="*", default=None,
                        help="Pre-computed object-prior RGB(s); one per --query. Skips GSam for that query.")
    parser.add_argument("--server_url", type=str, nargs="+", default=None)
    parser.add_argument("--suffix", type=str, nargs="*", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.environ.get("ROBODREAM_OUTPUT_DIR", "outputs"), "demonstrations"))
    parser.add_argument("--replay_cache_dir", type=str,
                        default=os.path.join(os.environ.get("ROBODREAM_OUTPUT_DIR", "outputs"), "cache", "replay_outputs"))
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualization_output_dir", type=str,
                        default=os.path.join(os.environ.get("ROBODREAM_OUTPUT_DIR", "outputs"), "visualizations"))
    parser.add_argument("--droid_rgb_root", type=str,
                        default=os.path.join(DATA_DIR, "processed", "droid_traj2demo", "real"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--episode_suffix", type=str, default=None,
                        help="Optional suffix appended to saved episode folder names (e.g. 'run1' → 'ep0001_run1')")
    args = parser.parse_args()
    assert args.num_samples > 0 or args.num_samples == -1, \
        "--num_samples must be > 0 or -1 (all)"
    main(args)
