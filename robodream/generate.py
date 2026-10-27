# Inference: robot-only renders + priors + trajectory + instruction -> generated two-view videos of any length.

from typing import Optional

import numpy as np
import torch

from .config import RoboDreamConfig
from .inputs import build_trajectory, encode_entries, resize_image
from .pipe import RoboDreamPipeline, load_dit
from .text import TextEmbedder
from .vae import RoboDreamVAE
from .vendor import log


class RoboDreamGenerator:
    '''
    Loads the DiT, the video tokenizer and the instruction embeddings once, then generates
    videos clip by clip: the first 189-frame clip is generated from the priors, every further
    clip is conditioned on the last 5 generated frames (autoregressive extension).
    '''

    def __init__(self, checkpoint_path: str, tokenizer_path: str, t5_cache_path: Optional[str] = None,
                 default_text_path: Optional[str] = None, t5_path: Optional[str] = None,
                 device: str = 'cuda', num_steps: Optional[int] = None,
                 config: RoboDreamConfig = RoboDreamConfig()):
        self.config = config
        self.device = device
        self.num_steps = num_steps or config.num_steps
        dit = load_dit(checkpoint_path, config).to(device=device, dtype=torch.bfloat16).eval()
        self.pipe = RoboDreamPipeline(dit, config, device=device)
        self.vae = RoboDreamVAE(tokenizer_path, chunk_duration=config.num_frames, device=device)
        self.text = TextEmbedder(t5_cache_path, default_text_path, t5_path, device=device)

    def _prior_tensor(self, image) -> Optional[torch.Tensor]:
        if image is None:
            return None
        image = resize_image(image, (self.config.height, self.config.width))
        if not image.any():
            return None  # all-black prior = no prior
        return torch.from_numpy(image).permute(2, 0, 1)[None]  # (1, 3, H, W)

    @torch.no_grad()
    def generate(self, mask_rgb: np.ndarray, trajectory: np.ndarray, prompt: str,
                 scene_prior: Optional[np.ndarray] = None, object_prior: Optional[np.ndarray] = None,
                 seed: int = 0) -> dict:
        '''
        :param mask_rgb: (V=2, T, H, W, 3) robot-only renders in [0, 1] for (external, wrist) camera.
        :param trajectory: (T, 7) end-effector xyz, rpy, gripper per frame (T <= 512).
        :param prompt (str): task instruction.
        :param scene_prior, object_prior: (H, W, 3) images (uint8 or [0, 1]) of the external view.
        :return dict 'rgb0', 'rgb1': (1, 3, T, H, W) float tensors in [0, 1] (cpu).
        '''
        cfg = self.config
        (V, T) = mask_rgb.shape[0:2]
        assert V == cfg.num_views, f'expected {cfg.num_views} views of robot-only renders, got {V}'
        assert len(trajectory) == T, f'trajectory has {len(trajectory)} steps for {T} frames'
        hw = (cfg.height, cfg.width)
        if mask_rgb.shape[2:4] != hw:
            mask_rgb = np.stack([[resize_image(f, hw) for f in view] for view in mask_rgb])
        mask_rgb = torch.from_numpy(np.ascontiguousarray(mask_rgb, dtype=np.float32)).permute(0, 4, 1, 2, 3)  # (V, 3, T, H, W)

        # Pad to a whole number of clips by repeating the last frame.
        (C, K) = (cfg.num_frames, cfg.num_cond_frames)
        T_pad = T + (-(T - K)) % (C - K) if T > C else C
        if T_pad > T:
            mask_rgb = torch.cat([mask_rgb, mask_rgb[:, :, -1:].repeat(1, 1, T_pad - T, 1, 1)], dim=2)
        num_clips = (T_pad - K) // (C - K)

        scene = self._prior_tensor(scene_prior)
        objects = self._prior_tensor(object_prior) if scene is not None else None
        crossattn_emb = self.text([prompt], device=self.device)

        clips = []
        prev_frames = None
        for i in range(num_clips):
            start = i * (C - K)
            (traj, traj_mask) = build_trajectory(trajectory, start, min(start + C, T), valid_len=None, config=cfg)
            batch = {
                'mask_rgb': mask_rgb[None, :, :, start:start + C],
                'trajectory': torch.from_numpy(traj)[None],
                'trajectory_mask': torch.from_numpy(traj_mask)[None],
            }
            if i == 0:
                cond_frames_raw = 0
                batch['scene_prior'] = scene
                batch['object_prior'] = objects
            else:
                cond_frames_raw = K
                rgb = torch.zeros((1, V, 3, C) + hw)
                rgb[:, :, :, 0:K] = prev_frames
                batch['rgb'] = rgb

            entries = encode_entries(self.vae, batch, cond_frames_raw, cfg)
            out = self.pipe.generate(entries, crossattn_emb, seed=seed, num_steps=self.num_steps)
            videos = torch.stack([self.vae.decode(out['y0_pred_entries'][f'rgb{v}']) for v in range(V)], dim=1)
            # ^ (1, V, 3, C, H, W)
            prev_frames = videos[:, :, :, -K:].float().cpu()
            clips.append(videos.float().cpu() if i == 0 else videos[:, :, :, K:].float().cpu())
            log.info(f'generated clip {i + 1}/{num_clips}')

        video = torch.cat(clips, dim=3)[:, :, :, :T]  # (1, V, 3, T, H, W)
        return {f'rgb{v}': video[:, v] for v in range(V)}
