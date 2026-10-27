import numpy as np
import os
import cv2
import mediapy
from pathlib import Path
from droid_meta.config import IMAGE2SCENE_CAM_MAP
from replay.replay_functions import replay_sequence_multiple_cams_return_output


def batched_replay(ctx, batch_store, control_mode, skip_real_obs=False, fps=15, width=1280, height=720):
    """
    Batched replay of episodes.
    
    Args:
        ctx: The replay context created by create_replay_context
        batch_store: Dictionary containing batched data with keys:
            - episode_ids: list of episode IDs (optional)
            - actions: list of action arrays (T, N)
            - camera_params: list of camera parameter dicts
            - real_obs: list of real observation dicts (or None)
            - video_paths: list of output video paths (optional, saves videos if provided)
            - img_paths: list of output image paths (optional, saves images if provided)
        control_mode: Control mode ("joint_position" or "end_effector")
        skip_real_obs: Whether to skip real observations
        fps: Video framerate for saving (default: 15)
        width: Camera resolution width (default: 1280)
        height: Camera resolution height (default: 720)
    
    Returns:
        outputs: Dictionary containing replay outputs (frames, camera_images, success flags, etc.)
    """
    if len(batch_store["actions"]) == 0:
        return []
    
    # Pad joint positions to the maximum length in the batch
    seq_lengths = [arr.shape[0] for arr in batch_store["actions"]]
    max_len = max(seq_lengths)
    feat_dim = batch_store["actions"][0].shape[1]
    B = len(batch_store["actions"]) 
    padded = np.zeros((B, max_len, feat_dim), dtype=batch_store["actions"][0].dtype)
    mask = np.zeros((B, max_len), dtype=np.bool_)  # True for valid timesteps
    for b_idx, seq in enumerate(batch_store["actions"]):
        L = seq.shape[0]
        padded[b_idx, :L] = seq
        mask[b_idx, :L] = True
    
    # Batchify the camera params
    batched_camera_params = {cam: {
        "camera_pose": [],
        "intrinsics": [],
    } for cam in IMAGE2SCENE_CAM_MAP.values()}
    for b_idx, cam_params in enumerate(batch_store["camera_params"]):
        for cam, v in cam_params.items():
            batched_camera_params[cam]["camera_pose"].append(v["camera_pose"])
            batched_camera_params[cam]["intrinsics"].append(v["intrinsics"])
    for k, v in batched_camera_params.items():
        # the camera_pose for each view should either be all None or all not None
        if v["camera_pose"][0] is not None:
            v["camera_pose"] = np.stack(v["camera_pose"])
        else:
            v["camera_pose"] = None
        v["intrinsics"] = np.stack(v["intrinsics"])

    # Determine if we need individual camera images for saving
    save_videos = batch_store.get("video_paths") is not None and len(batch_store["video_paths"]) > 0
    save_images = batch_store.get("img_paths") is not None and len(batch_store["img_paths"]) > 0
    
    # Call replay with batched inputs
    outputs = replay_sequence_multiple_cams_return_output(
        ctx=ctx,
        camera_params=batched_camera_params,
        actions=padded,
        control_mode=control_mode,
        padding_mask=mask,
        real_obs=batch_store["real_obs"] if not skip_real_obs else None,
        include_masked_images=True,
        width=width,
        height=height,
    )

    # Save videos if paths are provided
    if save_videos:
        video_paths = batch_store["video_paths"]
        assert len(video_paths) == B, f"Expected {B} video paths, got {len(video_paths)}"
        for b in range(B):
            if outputs['success'][b] and len(outputs['frames'][b]) > 0:
                output_path = Path(video_paths[b])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                mediapy.write_video(output_path, outputs['frames'][b], fps=fps)
    
    # Save individual camera images if paths are provided
    if save_images:
        img_paths = batch_store["img_paths"]
        assert len(img_paths) == B, f"Expected {B} image paths, got {len(img_paths)}"
        
        cam_names = list(batched_camera_params.keys())
        for b in range(B):
            if not outputs['success'][b]:
                continue
            
            if img_paths[b] is None:
                continue
                
            # Create directories for sim masked images
            sim_masked_path = os.path.join(img_paths[b], "rgb_sim_rollout_masked")
            for cam in cam_names:
                os.makedirs(os.path.join(sim_masked_path, cam), exist_ok=True)
            
            # Save sim camera images
            camera_images = outputs.get('camera_images', {})
            if b in camera_images:
                for cam in cam_names:
                    for t, img in camera_images[b][cam].items():
                        cv2.imwrite(
                            os.path.join(sim_masked_path, cam, f"{t:010d}.jpg"),
                            img
                        )
            
            # Save real masked images if available
            real_masked_images = outputs.get('real_masked_images', {})
            if b in real_masked_images:
                real_masked_path = os.path.join(img_paths[b], "rgb_real_rollout_masked")
                for cam in cam_names:
                    os.makedirs(os.path.join(real_masked_path, cam), exist_ok=True)
                    for t, img in real_masked_images[b][cam].items():
                        cv2.imwrite(
                            os.path.join(real_masked_path, cam, f"{t:010d}.jpg"),
                            img
                        )

    # Clear the batch store (handle None values or non-list types)
    for key in batch_store:
        value = batch_store[key]
        if isinstance(value, list):
            value.clear()
        else:
            batch_store[key] = []
    return outputs
