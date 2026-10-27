import numpy as np
import h5py
import os
import mediapy
import torch

from replay.math_utils import quat_from_euler_xyz, quat_mul, quat_inv, matrix_from_quat, quat_from_matrix, euler_xyz_from_quat, convert_camera_frame_orientation_convention

def world_pose_to_opengl_pose(world_pose):
    """
    world_pose: (..., 4, 4)
    Returns:
        positions: (..., 3)
        orientations: (..., 4)
        convention: "opengl"
    """
    if world_pose.ndim == 2:
        world_pose = world_pose[None]
    R_matrix = world_pose[..., :3, :3]
    t = world_pose[..., :3, 3]
    quat = quat_from_matrix(R_matrix)
    roll_correction = quat_from_euler_xyz(torch.tensor([-np.pi], dtype=torch.float32), 
                                          torch.tensor([0.0], dtype=torch.float32), 
                                          torch.tensor([0.0], dtype=torch.float32))
    quat = quat_mul(quat, roll_correction.repeat(quat.shape[0], 1))
    return t, quat, "opengl"

def resize_intrinsics(intrinsics, src_size, dst_size):
    fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
    new_fx = fx * dst_size[0] / src_size[0]
    new_fy = fy * dst_size[1] / src_size[1]
    new_cx = cx * dst_size[0] / src_size[0]
    new_cy = cy * dst_size[1] / src_size[1]
    return np.array([[new_fx, 0, new_cx],
                    [0, new_fy, new_cy],
                    [0, 0, 1]])

def _quat_from_rpy(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Convert roll/pitch/yaw (rad) to quaternion (w,x,y,z), using intrinsic XYZ (roll->pitch->yaw)."""
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.stack([qw, qx, qy, qz], axis=-1).astype(np.float32, copy=False)

def cartesian_rpy_to_quat(actions: np.ndarray) -> np.ndarray:
    """
    Convert batched (x, y, z, roll, pitch, yaw) to (x, y, z, qw, qx, qy, qz).
    
    Args:
        actions: Array of shape (N, 6) where each row is [x, y, z, roll, pitch, yaw]
    
    Returns:
        Array of shape (N, 7) where each row is [x, y, z, qw, qx, qy, qz]
    """
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError(f"Expected input shape (N, 6), got {actions.shape}")
    
    # Extract position and orientation
    xyz = actions[:, :3]
    roll, pitch, yaw = actions[:, 3], actions[:, 4], actions[:, 5]
    
    # Convert RPY to quaternion
    quat = _quat_from_rpy(roll, pitch, yaw)
    
    # Concatenate [x, y, z, qw, qx, qy, qz]
    return np.concatenate([xyz, quat], axis=1).astype(np.float32, copy=False)

def load_episode_data(trajectory_path):
    with h5py.File(trajectory_path, "r") as f:
        joint_positions = np.concatenate([
            f["observation"]["robot_state"]["joint_positions"][()],
            # f["observation"]["robot_state"]["gripper_position"][()][:, None],
        ], axis=-1)
        # gripper_actions = np.array(f["action"]["gripper_position"][()][:, None], dtype=np.float32)
        # cartesian_poses = np.array(f["action"]["cartesian_position"][()], dtype=np.float32)

        gripper_actions = np.array(f["observation"]["robot_state"]["gripper_position"][()][:, None], dtype=np.float32)
        cartesian_poses = np.array(f["observation"]["robot_state"]["cartesian_position"][()], dtype=np.float32)

        return joint_positions, gripper_actions, cartesian_rpy_to_quat(cartesian_poses)

def load_episode_images(videos_path, camera_params):
    images = {}
    for cam_name, cam_params in camera_params.items():
        video_path = os.path.join(videos_path, f"{cam_params['camera_serial']}.mp4")
        images[cam_name] = np.array(mediapy.read_video(video_path))
    return images