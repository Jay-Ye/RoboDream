# Cosmos video tokenizer wrapper: load tokenizer.pth, encode/decode RGB videos.

import torch

from .vendor.tokenizer import TokenizerInterface

SIGMA_DATA = 1.0


class RoboDreamVAE:
    '''
    Minimal wrapper around the cosmos-predict2 (Wan2.1) video tokenizer.
    Encode maps pixel videos in [0, 1] to 16-channel latents at (1 + (T-1)/4, H/8, W/8);
    decode maps latents back to pixel videos in [0, 1].
    '''

    def __init__(self, tokenizer_path: str, chunk_duration: int = 189, device: str = 'cuda'):
        '''
        :param tokenizer_path (str): nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth.
        :param chunk_duration (int): one clip (189 pixel frames) is encoded as a single chunk.
        '''
        self.device = device
        self.tensor_kwargs = dict(device=device, dtype=torch.bfloat16)
        self.video_tokenizer = TokenizerInterface(
            chunk_duration=chunk_duration,
            load_mean_std=False,
            vae_pth=tokenizer_path,
        )

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        '''
        :param video: (B, 3, Tp, Hp, Wp) tensor of float in [0, 1], Tp = 1 + 4k.
        :return latent: (B, 16, Tl, Hl, Wl) tensor of bfloat16.
        '''
        assert video.shape[1] == 3 and (video.shape[2] - 1) % 4 == 0, f'bad video shape {tuple(video.shape)}'
        video = video.to(**self.tensor_kwargs, non_blocking=True) * 2.0 - 1.0  # bf16 first, as in training
        return self.video_tokenizer.encode(video) * SIGMA_DATA

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        '''
        :param latent: (B, 16, Tl, Hl, Wl) tensor of float.
        :return video: (B, 3, Tp, Hp, Wp) tensor of float in [0, 1].
        '''
        video = self.video_tokenizer.decode(latent.clone().detach() / SIGMA_DATA)
        return ((video + 1.0) / 2.0).clamp(0.0, 1.0)
