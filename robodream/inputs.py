# Model inputs shared by training and inference: frame resizing, trajectory tokens, latent entries.

from typing import Optional

import numpy as np
import torch
from PIL import Image

from .config import RoboDreamConfig
from .logistics import build_entries


def resize_image(image, hw: tuple) -> np.ndarray:
    '''
    :param image: PIL image or (H, W, 3) uint8 / float [0, 1] array.
    :param hw: target (H, W).
    :return (H, W, 3) float32 array in [0, 1] (Lanczos, like the training data).
    '''
    if not isinstance(image, Image.Image):
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        image = Image.fromarray(array)
    image = image.convert('RGB')
    if image.size != (hw[1], hw[0]):
        image = image.resize((hw[1], hw[0]), resample=Image.LANCZOS)
    return np.asarray(image).astype(np.float32) / 255.0


def build_trajectory(pose: np.ndarray, frame_start: int, frame_end: int, valid_len: Optional[int],
                     config: RoboDreamConfig = RoboDreamConfig()) -> tuple:
    '''
    Trajectory token input: the WHOLE episode trajectory (not just the clip), with an indicator
    channel marking the frames covered by the current clip.
    :param pose: (N, 7) xyz, rpy, gripper per frame (N <= 512).
    :param frame_start, frame_end: clip frames [frame_start, frame_end) in episode indices.
    :param valid_len: number of valid steps for the mask (training: N); None marks all 512 steps
        valid (what the released inference server did).
    :return (trajectory (512, 8) float32, mask (512, 1) float32).
    '''
    L = config.traj_len
    pose = np.asarray(pose, dtype=np.float32)
    assert pose.ndim == 2 and pose.shape[1] == 7, f'pose must be (N, 7), got {pose.shape}'
    assert pose.shape[0] <= L, f'trajectories longer than {L} steps are not supported ({pose.shape[0]})'
    traj = np.zeros((L, config.traj_channels), dtype=np.float32)
    traj[:pose.shape[0], 0:7] = pose
    traj[frame_start:min(frame_end, L), 7] = 1.0
    mask = np.zeros((L, 1), dtype=np.float32)
    mask[:(L if valid_len is None else valid_len)] = 1.0
    return (traj, mask)


@torch.no_grad()
def encode_entries(vae, batch: dict, cond_frames_raw: int,
                   config: RoboDreamConfig = RoboDreamConfig()) -> dict:
    '''
    Pixel-space batch -> latent entries (see logistics.build_entries).
    :param batch (dict):
        'mask_rgb': (B, V, 3, T, H, W) robot-only renders in [0, 1].
        'rgb': (B, V, 3, T, H, W) real / previously generated frames in [0, 1], or None. Only the
            first cond_frames_raw frames are used as input; during training it is also the target.
        'scene_prior': (B, 3, H, W) or None; 'object_prior': (B, 3, H, W) or None;
        'scene_prior_valid', 'object_prior_valid': (B,) 0/1 flags, or None (= all valid).
        'trajectory': (B, 512, 8); 'trajectory_mask': (B, 512, 1).
    :param cond_frames_raw (int): 0 (generate from priors) or 5 (continue from given frames).
    '''
    device = vae.device
    mask_rgb = batch['mask_rgb'].to(device)
    (B, V, _, T, H, W) = mask_rgb.shape
    cond_frames_latent = (cond_frames_raw - 1) // config.vae_temporal_factor + 1
    Tl = (T - 1) // config.vae_temporal_factor + 1
    (Hl, Wl) = (H // config.vae_spatial_factor, W // config.vae_spatial_factor)

    mask_latents = [vae.encode(mask_rgb[:, v]) for v in range(V)]
    if batch.get('rgb') is not None:
        rgb_latents = [vae.encode(batch['rgb'][:, v].to(device)) for v in range(V)]
    else:
        assert cond_frames_raw == 0, 'conditioning frames require rgb'
        rgb_latents = [torch.zeros((B, config.latent_channels, Tl, Hl, Wl), device=device, dtype=torch.bfloat16)
                       for _ in range(V)]

    # Priors are only used when generating from scratch (no conditioning frames); the object
    # prior is only used together with the scene prior.
    scene_latent = None
    object_latent = None
    if cond_frames_raw == 0 and batch.get('scene_prior') is not None:
        scene_video = batch['scene_prior'].to(device)[:, :, None].repeat(1, 1, T, 1, 1)
        scene_latent = vae.encode(scene_video)
        if batch.get('object_prior') is not None:
            object_latent = vae.encode(batch['object_prior'].to(device)[:, :, None]).flatten(2)  # (B, 16, Hl*Wl)

    trajectory = batch['trajectory'].to(device=device, dtype=torch.bfloat16) * config.action_multiplier

    return build_entries(
        rgb=rgb_latents,
        mask_rgb=mask_latents,
        trajectory=trajectory,
        trajectory_mask=batch['trajectory_mask'].to(device),
        cond_frames_latent=cond_frames_latent,
        scene_prior=scene_latent,
        scene_prior_valid=batch.get('scene_prior_valid'),
        object_prior=object_latent,
        object_prior_valid=batch.get('object_prior_valid'),
    )
