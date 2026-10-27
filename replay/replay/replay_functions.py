from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import os
import argparse
import numpy as np
import torch
import gymnasium as gym
import mediapy
import cv2
import h5py
import mediapy
from tqdm import tqdm
import json

from replay.utils import world_pose_to_opengl_pose

@dataclass
class ReplayContext:
    """Holds long-lived objects to avoid repeated initialization.

    Attributes
    - app_launcher: IsaacLab AppLauncher instance
    - simulation_app: The running Omniverse app (from app_launcher.app)
    - env_cfg: IsaacLab environment configuration, to be modified per replay
    - device: Device string used by the environment
    - scene: Scene identifier
    """

    app_launcher: object
    simulation_app: object
    env_cfg: object
    device: str
    scene: int


def create_replay_context(
    *,
    headless: bool = True,
    scene: int = 1,
    device: Optional[str] = None,
    num_envs: int = 1,
    env_name: str = "DROID", # "DROID" or "DROID_EE"
) -> ReplayContext:
    """Initialize IsaacLab app and environment configuration once.

    Parameters
    - headless: Whether to run without rendering to a window
    - scene: Scene identifier to pass to the env config
    - device: Optional device override (e.g., "cuda:0"). If None, use default.
    - num_envs: Number of environments to run in parallel
    - env_name: Name of the environment to use ("DROID" or "DROID_EE")
    Returns
    - A `ReplayContext` that should be reused across multiple replays.
    """
    # Import inside function to avoid side effects during module import.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Create IsaacLab app for DROID replays")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import IsaacLab-dependent modules only after the app is launched
    import src.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    resolved_device = device if device is not None else args_cli.device

    env_cfg = parse_env_cfg(
        env_name,
        device=resolved_device,
        num_envs=num_envs,
        use_fabric=True,
    )

    # Persist the scene choice, to be used by replay_sequence per call if needed
    # (The caller can still mutate env_cfg before making the env.)
    _ = scene  # keep for future extension; env is currently built without explicit set_scene

    return ReplayContext(
        app_launcher=app_launcher,
        simulation_app=simulation_app,
        env_cfg=env_cfg,
        device=str(resolved_device),
        scene=scene,
    )

def close_context(ctx: ReplayContext) -> None:
    """Close the Omniverse app held by the context."""
    try:
        ctx.simulation_app.close()
    except Exception:
        # Best-effort shutdown
        pass


def replay_sequence_multiple_cams_return_output(
    *,
    ctx: ReplayContext,
    camera_params,
    actions: np.ndarray,
    control_mode: str = "joint_position",
    padding_mask=None,
    width = 1280,
    height = 720,
    real_obs = None,
    include_masked_images: bool = True,
) -> dict:
    """Replay a joint-position sequence and return the rendered frames.

    This function performs the same replay logic as replay_sequence_multiple_cams_v2
    but returns the output data instead of saving to disk.

    Parameters
    - ctx: A `ReplayContext` created by `create_replay_context`
    - camera_params: Dictionary of camera parameters, with keys as camera names and values as dictionaries containing:
      - camera_pose: (4,4) world transform of the camera as a homogeneous matrix
      - intrinsics: (3,3) camera intrinsic matrix
    - actions: (T, N) numpy array of actions (joint positions or end-effector poses)
    - control_mode: Control mode to use for the environment ("joint_position" or "end_effector")
    - padding_mask: Optional boolean mask indicating valid timesteps
    - width, height: Camera resolution
    - real_obs: Optional dictionary of real observations to use for comparison.
      if provided, include the real observations in the output.
    - include_masked_images: If True, return individual camera images per timestep
    
    Returns
    - Dictionary containing:
      - 'frames': list[list[np.ndarray]] - Combined frames per environment (length B)
      - 'success': list[bool] - Success status per environment
      - 'camera_images': (optional) dict with structure {env_idx: {cam_name: {timestep: np.ndarray}}}
      - 'real_masked_images': (optional) dict with structure {env_idx: {cam_name: {timestep: np.ndarray}}}
    """
    import omni.usd
    import gc

    actions = torch.from_numpy(actions).to(torch.float32).to(ctx.device)
    if actions.ndim == 2:
        actions = actions[None]
    if actions.shape[1] < 1:
        raise ValueError(f"Expected actions to have at least 1 dimension, got shape {actions.shape}")
    B, T, D = actions.shape

    if padding_mask is None:
        padding_mask = torch.ones((B, T), dtype=torch.bool, device=ctx.device)
    else:
        padding_mask = torch.from_numpy(padding_mask).to(torch.bool).to(ctx.device)
        assert padding_mask.shape == (B, T), f"Padding mask shape mismatch: {padding_mask.shape} != {B, T}"

    if real_obs is None:
        real_obs_list = [None] * B
    else:
        real_obs_list = real_obs
        assert len(real_obs_list) == B, f"Expected {B} real observations, got {len(real_obs_list)}"

    # Initialize output structures
    camera_images = {b: {cam: {} for cam in camera_params.keys()} for b in range(B)} if include_masked_images else None
    real_masked_images = {b: {cam: {} for cam in camera_params.keys()} for b in range(B)} if (include_masked_images and any(ro is not None for ro in real_obs_list)) else None

    # build a fresh env for this replay
    if control_mode == "joint_position":
        env_name = "DROID"
    elif control_mode == "end_effector":
        env_name = "DROID_EE"
    else:
        raise ValueError(f"Invalid control mode: {control_mode}")
    env = gym.make(env_name, cfg=ctx.env_cfg)
    try:
        obs, _ = env.reset()
        obs, _ = env.reset()

        # Configure cameras per env
        cam_names = list(camera_params.keys())
        for cam_name in cam_names:
            cam = env.unwrapped.scene[cam_name]
            cam.set_intrinsic_matrices(
                torch.from_numpy(camera_params[cam_name]["intrinsics"]),
                focal_length=2.1,
            )
            if camera_params[cam_name]["camera_pose"] is not None:
                base_positions, body_orientations = env.unwrapped.scene["robot"].data.body_pos_w[:, 0], env.unwrapped.scene["robot"].data.body_quat_w[:, 0]
                camera_params[cam_name]["camera_pose"][..., :3, 3] = camera_params[cam_name]["camera_pose"][..., :3, 3] + base_positions.cpu().numpy()
                positions, orientations, convention = world_pose_to_opengl_pose(torch.from_numpy(camera_params[cam_name]["camera_pose"]))
                cam.set_world_poses(
                    positions=positions,
                    orientations=orientations,
                    env_ids=None,
                    convention=convention,
                )

        # Collect frames while stepping through provided joint positions
        frames_per_env: list[list[np.ndarray]] = [[] for _ in range(B)]

        with torch.no_grad():
            robot = env.unwrapped.scene["robot"]
            for t in tqdm(range(T), desc=f"Replaying episode"):
                if control_mode == "joint_position":
                    # Replace the arm joints while keeping the rest from the env
                    env_joint_positions = robot.data.joint_pos.clone()
                    desired_joint_positions = torch.concatenate([
                        actions[:, t, :-1], # NOTE: exclude the gripper action
                        env_joint_positions[:, 7:],
                    ], dim=-1)
                    desired_joint_positions = torch.where(padding_mask[:, t, None], desired_joint_positions, env_joint_positions)
                    robot.write_joint_position_to_sim(desired_joint_positions)

                    # Apply without a full step by resetting to current scene state
                    obs, _ = env.unwrapped.reset_to(env.unwrapped.scene.get_state(), None)

                    # Build actions for gripper actuation where requested
                    if any(actions[:, t, -1] != 0):
                        action_vec = actions[:, t, :]
                        env.step(action_vec)
                else:
                    env_robot_states = torch.concatenate([
                        obs['policy']['end_effector_pose'],
                        obs['policy']['gripper_pos'],
                    ], dim=-1)
                    action_vec = torch.where(padding_mask[:, t, None], actions[:, t, :], env_robot_states)
                    obs, _, term, trunc, _ = env.step(action_vec)
                    if t == 0:
                        for _ in range(10):
                            obs, _, term, trunc, _ = env.step(action_vec) 

                # Capture frames per env
                for b in range(B):
                    if not padding_mask[b, t]:
                        continue
                    frame_np_multi_cam = []
                    for cam in cam_names:
                        img = obs["policy"][cam].cpu().numpy()[b]
                        if include_masked_images:
                            camera_images[b][cam][t] = img.copy()
                            if real_obs_list[b] is not None:
                                real_img = cv2.resize(real_obs_list[b][cam][t], (width, height), interpolation=cv2.INTER_LINEAR)
                                real_img_masked = real_img * (img.sum(axis=2, keepdims=True) > 0)
                                real_masked_images[b][cam][t] = real_img_masked.copy()
                        frame_np_multi_cam.append(img)
                    frame_np_multi_cam = np.concatenate(frame_np_multi_cam, axis=1)

                    if real_obs_list[b] is not None:
                        frame_np_multi_cam_real = []
                        for cam in cam_names:
                            frame_np_multi_cam_real.append(cv2.resize(real_obs_list[b][cam][t], (width, height), interpolation=cv2.INTER_LINEAR))
                        frame_np_multi_cam_real = np.concatenate(frame_np_multi_cam_real, axis=1)
                        frame_np_multi_cam_real_masked = frame_np_multi_cam_real * (frame_np_multi_cam.sum(axis=2, keepdims=True) > 0)
                        frame_np_multi_cam_real_masked[np.all(frame_np_multi_cam_real_masked == 0, axis=2)] = [255, 255, 255]
                        frame_np_multi_cam[np.all(frame_np_multi_cam == 0, axis=2)] = [255, 255, 255]
                        frame_np_multi_cam = np.concatenate([frame_np_multi_cam_real, frame_np_multi_cam, frame_np_multi_cam_real_masked], axis=0)

                    frames_per_env[b].append(cv2.resize(frame_np_multi_cam, (frame_np_multi_cam.shape[1]//4, frame_np_multi_cam.shape[0]//4), interpolation=cv2.INTER_LINEAR))

        # Build result dictionary
        result = {
            'frames': frames_per_env,
            'success': [True] * B,
        }
        if include_masked_images:
            result['camera_images'] = camera_images
            if real_masked_images is not None:
                result['real_masked_images'] = real_masked_images
        
        return result
    except Exception as e:
        print(e)
        return {
            'frames': [[] for _ in range(B)],
            'success': [False] * B,
        }
    finally:
        env.close()
        del env
        gc.collect()
        omni.usd.get_context().new_stage()


def replay_sequence_multiple_cams_v2(
    *,
    ctx: ReplayContext,
    camera_params,
    actions: np.ndarray,
    control_mode: str = "joint_position",
    output_path,
    padding_mask=None,
    fps = 15,
    width = 1280,
    height = 720,
    real_obs = None,
    img_paths = None,
) -> list[bool]:
    """Replay a joint-position sequence under given camera params and save video.

    This mirrors the behavior in `run_random.py` but reuses `ctx` to avoid
    reinitializing the AppLauncher and base env configuration.

    Parameters
    - ctx: A `ReplayContext` created by `create_replay_context`
    - camera_params: Dictionary of camera parameters, with keys as camera names and values as dictionaries containing:
      - camera_pose: (4,4) world transform of the camera as a homogeneous matrix
      - intrinsics: (3,3) camera intrinsic matrix
    - actions: (T, N) numpy array of actions (joint positions or end-effector poses)
    - control_mode: Control mode to use for the environment ("joint_position" or "end_effector")
    - output_path: Path to the output mp4 file
    - fps: Video framerate
    - width, height: Camera resolution
    - real_obs: Optional dictionary of real observations to use for the replay.
      if provided, mask the real observations with the replay observations.
    - img_path: Optional path to save the replay observations.
    Returns
    - The resolved `Path` to the saved video.
    """
    import omni.usd # , omni.timeline
    import gc

    actions = torch.from_numpy(actions).to(torch.float32).to(ctx.device)
    if actions.ndim == 2:
        actions = actions[None]
    if actions.shape[1] < 1:
        raise ValueError(f"Expected actions to have at least 1 dimension, got shape {actions.shape}")
    B, T, D = actions.shape

    if padding_mask is None:
        padding_mask = torch.ones((B, T), dtype=torch.bool, device=ctx.device)
    else:
        padding_mask = torch.from_numpy(padding_mask).to(torch.bool).to(ctx.device)
        assert padding_mask.shape == (B, T), f"Padding mask shape mismatch: {padding_mask.shape} != {B, T}"

    output_paths = [Path(p) for p in output_path]
    assert len(output_paths) == B, f"Expected {B} output paths, got {len(output_paths)}"

    if img_paths is None:
        img_paths = [None] * B

    if real_obs is None:
        real_obs_list = [None] * B
    else:
        real_obs_list = real_obs
        assert len(real_obs_list) == B, f"Expected {B} real observations, got {len(real_obs_list)}"

    sim_masked_save_paths = [None] * B
    real_masked_save_paths = [None] * B
    for b in range(B):
        if img_paths[b] is not None:
            sim_masked = os.path.join(img_paths[b], "rgb_sim_rollout_masked")
            sim_masked_save_paths[b] = sim_masked
            for cam in camera_params.keys():
                os.makedirs(os.path.join(sim_masked, cam), exist_ok=True)
            if real_obs_list[b] is not None:
                real_masked = os.path.join(img_paths[b], "rgb_real_rollout_masked")
                real_masked_save_paths[b] = real_masked
                for cam in camera_params.keys():
                    os.makedirs(os.path.join(real_masked, cam), exist_ok=True)

    # build a fresh env for this replay
    if control_mode == "joint_position":
        env_name = "DROID"
    elif control_mode == "end_effector":
        env_name = "DROID_EE"
    else:
        raise ValueError(f"Invalid control mode: {control_mode}")
    env = gym.make(env_name, cfg=ctx.env_cfg)
    try:
        obs, _ = env.reset()
        obs, _ = env.reset()

        # Configure cameras per env
        # Assume same set of camera names across envs; take from first
        cam_names = list(camera_params.keys())
        for cam_name in cam_names:
            cam = env.unwrapped.scene[cam_name]
            cam.set_intrinsic_matrices(
                torch.from_numpy(camera_params[cam_name]["intrinsics"]),
                focal_length=2.1,
            )
            if camera_params[cam_name]["camera_pose"] is not None:
                base_positions, body_orientations = env.unwrapped.scene["robot"].data.body_pos_w[:, 0], env.unwrapped.scene["robot"].data.body_quat_w[:, 0]
                # NOTE: the orientations seem to be always in the default openGL convention, so we just do camera_pose + base_positions
                camera_params[cam_name]["camera_pose"][..., :3, 3] = camera_params[cam_name]["camera_pose"][..., :3, 3] + base_positions.cpu().numpy()
                positions, orientations, convention = world_pose_to_opengl_pose(torch.from_numpy(camera_params[cam_name]["camera_pose"]))
                cam.set_world_poses(
                    positions=positions,
                    orientations=orientations,
                    env_ids=None,
                    convention=convention,
                )

        # Collect frames while stepping through provided joint positions
        frames_per_env: list[list[np.ndarray]] = [[] for _ in range(B)]

        with torch.no_grad():
            robot = env.unwrapped.scene["robot"]
            for t in tqdm(range(T), desc=f"Replaying episode"):
                if control_mode == "joint_position":
                    # Replace the arm joints while keeping the rest from the env
                    env_joint_positions = robot.data.joint_pos.clone()
                    desired_joint_positions = torch.concatenate([
                        actions[:, t, :-1], # NOTE: exclude the gripper action
                        env_joint_positions[:, 7:],
                    ], dim=-1)
                    desired_joint_positions = torch.where(padding_mask[:, t, None], desired_joint_positions, env_joint_positions)
                    robot.write_joint_position_to_sim(desired_joint_positions)

                    # Apply without a full step by resetting to current scene state
                    obs, _ = env.unwrapped.reset_to(env.unwrapped.scene.get_state(), None)

                    # Build actions for gripper actuation where requested
                    if any(actions[:, t, -1] != 0):
                        action_vec = actions[:, t, :]
                        env.step(action_vec)
                else:
                    env_robot_states = torch.concatenate([
                        obs['policy']['end_effector_pose'],
                        obs['policy']['gripper_pos'],
                    ], dim=-1)
                    action_vec = torch.where(padding_mask[:, t, None], actions[:, t, :], env_robot_states) # if run out of actions, use the current robot state to keep the robot in the same position
                    obs, _, term, trunc, _ = env.step(action_vec)
                    if t == 0: # make sure the robot is in the same position as the initial state
                        for _ in range(10):
                            obs, _, term, trunc, _ = env.step(action_vec) 

                # Capture frames per env
                for b in range(B):
                    if not padding_mask[b, t]:
                        continue
                    frame_np_multi_cam = []
                    for cam in cam_names:
                        img = obs["policy"][cam].cpu().numpy()[b]
                        if img_paths[b] is not None:
                            cv2.imwrite(os.path.join(sim_masked_save_paths[b], cam, f"{t:010d}.jpg"), img)
                            if real_obs_list[b] is not None:
                                real_img = cv2.resize(real_obs_list[b][cam][t], (width, height), interpolation=cv2.INTER_LINEAR)
                                real_img_masked = real_img * (img.sum(axis=2, keepdims=True) > 0)
                                cv2.imwrite(os.path.join(real_masked_save_paths[b], cam, f"{t:010d}.jpg"), real_img_masked)
                        frame_np_multi_cam.append(img)
                    frame_np_multi_cam = np.concatenate(frame_np_multi_cam, axis=1)

                    if real_obs_list[b] is not None:
                        frame_np_multi_cam_real = []
                        for cam in cam_names:
                            frame_np_multi_cam_real.append(cv2.resize(real_obs_list[b][cam][t], (width, height), interpolation=cv2.INTER_LINEAR))
                        frame_np_multi_cam_real = np.concatenate(frame_np_multi_cam_real, axis=1)
                        frame_np_multi_cam_real_masked = frame_np_multi_cam_real * (frame_np_multi_cam.sum(axis=2, keepdims=True) > 0)
                        frame_np_multi_cam_real_masked[np.all(frame_np_multi_cam_real_masked == 0, axis=2)] = [255, 255, 255]
                        frame_np_multi_cam[np.all(frame_np_multi_cam == 0, axis=2)] = [255, 255, 255]
                        frame_np_multi_cam = np.concatenate([frame_np_multi_cam_real, frame_np_multi_cam, frame_np_multi_cam_real_masked], axis=0)

                    frames_per_env[b].append(cv2.resize(frame_np_multi_cam, (frame_np_multi_cam.shape[1]//4, frame_np_multi_cam.shape[0]//4), interpolation=cv2.INTER_LINEAR))

        # 3) Write videos per env
        results = [False] * B
        for b in range(B):
            output_paths[b].parent.mkdir(parents=True, exist_ok=True)
            mediapy.write_video(output_paths[b], frames_per_env[b], fps=fps)
            results[b] = True
        return results
    except Exception as e:
        print(e)
        return [False] * B
    finally:
        # try:
        #     omni.timeline.get_timeline().stop()
        # except Exception as e:
        #     print(e)
        #     pass
        env.close()
        del env
        gc.collect()
        omni.usd.get_context().new_stage()