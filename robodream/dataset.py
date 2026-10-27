# DROID training clips read directly from the per-episode Processed layout (no webdataset).

import os
import random
import re
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import RoboDreamConfig
from .inputs import build_trajectory, resize_image

EXTERNAL_CAMS = ('external_cam1', 'external_cam2')
WRIST_CAM = 'wrist_cam'
CLIP_STRIDE = 20  # windows start every 20 frames
CLIP_PATTERN = re.compile(r'(?P<episode>[^/]+)/seq-\d+-(?P<start>\d+)_(?P<end>\d+)(\.tar)?$')


def read_clip_list(path: str) -> list[tuple[str, int]]:
    '''
    Reads a clip list: lines '<episode> <start>' or the released shard lists
    ('.../<episode>/seq-XXXXXX-<start>_<end>.tar', e.g. meta/split_train_station.txt).
    '''
    clips = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = CLIP_PATTERN.search(line)
            if m:
                clips.append((m.group('episode'), int(m.group('start'))))
            else:
                (episode, start) = line.split()
                clips.append((episode, int(start)))
    return clips


def episode_clip_starts(num_frames_episode: int, clip_len: int = 189, stride: int = CLIP_STRIDE) -> list[int]:
    '''
    Window starts used to build the training set: 0, 20, 40, ... while the previous window
    starts before num_frames - 0.6 * clip_len (late windows are padded with the last frame).
    '''
    starts = [0]
    while starts[-1] < num_frames_episode - 0.6 * clip_len:
        starts.append(starts[-1] + stride)
    return starts


class DroidClipDataset(Dataset):
    '''
    One sample = one 189-frame clip of one DROID episode, laid out as produced by
    data_processing/ (stage 01 + released priors):
        <root>/<episode>/rgb/<cam>/%010d.jpg                    real frames (640x360)
        <root>/<episode>/robot_moving_sim/<cam>/%010d.jpg       robot-only renders
        <root>/<episode>/lowdim/<cam>/%010d.npz                 proprio (whole episode) + instructions
        <root>/<episode>/scene_prior_rgb/<cam>/0000000000_omnipaint.jpg
        <root>/<episode>/scene_prior_object_gsam/<cam>/0000000000_gsam.jpg
    View 0 is a random external camera, view 1 is the wrist camera. Frames past the end of the
    episode repeat the last frame.
    '''

    def __init__(self, root: str, clips: list[tuple[str, int]], config: RoboDreamConfig = RoboDreamConfig(),
                 random_view: bool = True, seed: Optional[int] = None):
        '''
        :param clips: (episode, start_frame) pairs (see read_clip_list / episode_clip_starts).
        :param random_view (bool): pick the external camera at random (else external_cam1).
        '''
        self.root = root
        self.clips = clips
        self.config = config
        self.random_view = random_view
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.clips)

    def _frame(self, episode_dir: str, folder: str, cam: str, idx: int) -> np.ndarray:
        path = os.path.join(episode_dir, folder, cam, f'{idx:010d}.jpg')
        return resize_image(Image.open(path), (self.config.height, self.config.width))

    def _video(self, episode_dir: str, folder: str, cam: str, indices: list[int]) -> torch.Tensor:
        frames = np.stack([self._frame(episode_dir, folder, cam, i) for i in indices], axis=0)  # (T, H, W, 3)
        return torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()  # (3, T, H, W)

    def _prior(self, path: str) -> Optional[torch.Tensor]:
        if not os.path.isfile(path):
            return None
        image = resize_image(Image.open(path), (self.config.height, self.config.width))
        if not image.any():
            return None  # all-black fallback images count as missing
        return torch.from_numpy(image).permute(2, 0, 1).contiguous()  # (3, H, W)

    def __getitem__(self, index: int) -> dict:
        (episode, start) = self.clips[index]
        episode_dir = os.path.join(self.root, episode)
        cfg = self.config

        lowdim = np.load(os.path.join(episode_dir, 'lowdim', EXTERNAL_CAMS[0], f'{0:010d}.npz'), allow_pickle=True)
        proprio = lowdim['proprio'].item()
        pose = np.concatenate([proprio['cartesian_position'], proprio['gripper_position']], axis=-1)  # (N, 7)
        prompts = [str(p) for p in np.atleast_1d(lowdim['prompt'])]
        total = len(os.listdir(os.path.join(episode_dir, 'rgb', WRIST_CAM)))
        indices = [min(t, total - 1) for t in range(start, start + cfg.num_frames)]

        ext_cam = self.rng.choice(EXTERNAL_CAMS) if self.random_view else EXTERNAL_CAMS[0]
        cams = (ext_cam, WRIST_CAM)
        rgb = torch.stack([self._video(episode_dir, 'rgb', cam, indices) for cam in cams])
        mask_rgb = torch.stack([self._video(episode_dir, 'robot_moving_sim', cam, indices) for cam in cams])

        # Scene prior of the external view; object prior of the same view, else of the other
        # external camera (both cameras see the same objects).
        scene_prior = self._prior(os.path.join(episode_dir, 'scene_prior_rgb', ext_cam, '0000000000_omnipaint.jpg'))
        object_prior = None
        for cam in (ext_cam,) + tuple(c for c in EXTERNAL_CAMS if c != ext_cam):
            object_prior = self._prior(os.path.join(episode_dir, 'scene_prior_object_gsam', cam, '0000000000_gsam.jpg'))
            if object_prior is not None:
                break

        (trajectory, trajectory_mask) = build_trajectory(
            pose, start, min(start + cfg.num_frames, total), valid_len=pose.shape[0], config=cfg)

        empty = torch.zeros((3, cfg.height, cfg.width))
        return {
            'rgb': rgb,  # (V, 3, T, H, W)
            'mask_rgb': mask_rgb,  # (V, 3, T, H, W)
            'scene_prior': scene_prior if scene_prior is not None else empty,
            'scene_prior_valid': torch.tensor(float(scene_prior is not None)),
            'object_prior': object_prior if object_prior is not None else empty,
            'object_prior_valid': torch.tensor(float(scene_prior is not None and object_prior is not None)),
            'trajectory': torch.from_numpy(trajectory),
            'trajectory_mask': torch.from_numpy(trajectory_mask),
            'prompt': self.rng.choice(prompts),
            'clip_id': f'{episode}/{start}',
        }
