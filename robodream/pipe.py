# RoboDream diffusion pipeline: preconditioning, denoising, sampling, training loss, checkpoint loading.

import math
import re
from typing import Optional

import torch

from .config import RoboDreamConfig
from .logistics import (
    assemble_clean_predictions,
    assemble_network_inputs,
    pack_streams_from_entries,
    unpack_entries_from_streams,
)
from .vendor import log
from .vendor.denoiser_scaling import RectifiedFlowScaling
from .vendor.misc import arch_invariant_rand
from .vendor.rectified_flow_scheduler import RectifiedFlowAB2Scheduler

SAC_WRAP = re.compile(r'\._checkpoint_wrapped_module')  # key segment added by activation checkpointing
GENERATED_STREAMS = ('v0', 'v1')


class RoboDreamPipeline:
    '''
    Rectified-flow sampling and training loss over the packed streams (see logistics.py).
    Only the two video streams are noised / generated; the object-prior (sattn) and trajectory
    (xattn) streams are pure conditioning.
    '''

    def __init__(self, dit: torch.nn.Module, config: RoboDreamConfig = RoboDreamConfig(),
                 device: str = 'cuda', dtype: torch.dtype = torch.bfloat16):
        self.dit = dit
        self.config = config
        self.tensor_kwargs = {'device': device, 'dtype': dtype}
        self.scheduler = RectifiedFlowAB2Scheduler(
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            order=config.scheduler_order,
            t_scaling_factor=config.t_scaling_factor,
        )
        self.scaling = RectifiedFlowScaling(config.sigma_data, config.t_scaling_factor)

    def denoise(self, x0_streams: dict, yt_streams: dict, masks: dict, sigmas: dict,
                crossattn_emb: torch.Tensor, fps: Optional[torch.Tensor] = None) -> tuple:
        '''
        Single denoising step: preconditions, assembles masked network inputs, runs the DiT,
        and converts the network output to clean sample predictions.
        :param x0_streams (dict): all four streams, conditioning content in (B, ...) tensors.
        :param yt_streams (dict): noisy 'v0' / 'v1' streams (float32).
        :param sigmas (dict): 'v0' / 'v1' -> (B, 1, 1, 1, 1) noise levels.
        :param crossattn_emb (Tensor): (B, 512, 1024) T5 embedding.
        :return (xt_assemb_streams, y0_pred_streams).
        '''
        (cs_skip, cs_out, cs_in, cs_noise) = (dict(), dict(), dict(), dict())
        for (k, sigma) in sigmas.items():
            (cs_skip[k], cs_out[k], cs_in[k], cs_noise[k]) = self.scaling(sigma=sigma)

        yt_scaled_streams = {k: cs_in[k] * v for (k, v) in yt_streams.items()}
        xt_assemb_streams = assemble_network_inputs(x0_streams, yt_scaled_streams, masks)

        # The network operates in bfloat16.
        xt_assemb_streams = {k: v.to(**self.tensor_kwargs) for (k, v) in xt_assemb_streams.items()}
        cs_noise = {k: v.to(**self.tensor_kwargs) for (k, v) in cs_noise.items()}

        net_output = self.dit(
            x_in_streams=xt_assemb_streams,
            timesteps=cs_noise,
            crossattn_emb=crossattn_emb.to(**self.tensor_kwargs),
            fps=fps,
        )

        y0_raw_pred_streams = dict()
        for k in net_output.keys():
            y0_raw_pred_streams[k] = cs_skip[k] * yt_streams[k] + cs_out[k] * net_output[k].to(torch.float32)

        y0_pred_streams = assemble_clean_predictions(x0_streams, y0_raw_pred_streams, masks)
        return (xt_assemb_streams, y0_pred_streams)

    @torch.no_grad()
    def generate(self, entries: dict, crossattn_emb: torch.Tensor, seed: int = 0,
                 num_steps: Optional[int] = None, fps: Optional[torch.Tensor] = None) -> dict:
        '''
        Full sampling loop for one batch of latent entries (see logistics.build_entries).
        :return dict with 'y0_pred_entries' (rgb0 / rgb1 latents = generated videos) and streams.
        '''
        num_steps = num_steps or self.config.num_steps
        (x0_streams, masks) = pack_streams_from_entries(entries, self.config)
        B = x0_streams['v0'].shape[0]
        device = x0_streams['v0'].device
        state_shapes = {k: x0_streams[k].shape[1:] for k in GENERATED_STREAMS}

        def y0_pred_fn(yt_streams, sigmas):
            return self.denoise(x0_streams, yt_streams, masks, sigmas, crossattn_emb, fps=fps)

        # Starting noise: every stream draws from the same seed (architecture-invariant numpy RNG).
        sample_streams = {k: arch_invariant_rand((B,) + tuple(state_shapes[k]), torch.float32, device, seed)
                          * self.scheduler.config.sigma_max for k in state_shapes}

        self.scheduler.set_timesteps(num_steps, device=device)
        x0_prev_streams = {k: None for k in state_shapes}
        for (i, _) in enumerate(self.scheduler.timesteps):
            sigma_t = self.scheduler.sigmas[i].to(device, dtype=torch.float32)
            sigma_in = {k: sigma_t.repeat(B, *([1] * len(state_shapes[k]))) for k in state_shapes}
            (_, y0_pred_streams) = y0_pred_fn(sample_streams, sigma_in)
            for k in state_shapes:
                (sample_streams[k], x0_prev_streams[k]) = self.scheduler.step(
                    x0_pred=y0_pred_streams[k], i=i, sample=sample_streams[k], x0_prev=x0_prev_streams[k])

        # Final clean pass at sigma_min.
        sigma_last = self.scheduler.sigmas[-1].to(device, dtype=torch.float32)
        sigma_in = {k: sigma_last.repeat(B, *([1] * len(state_shapes[k]))) for k in state_shapes}
        (_, sample_streams) = y0_pred_fn(sample_streams, sigma_in)
        y0_pred_streams = assemble_clean_predictions(x0_streams, sample_streams, masks)

        return {
            'x0_streams': x0_streams,
            'masks': masks,
            'y0_pred_streams': y0_pred_streams,
            'y0_pred_entries': unpack_entries_from_streams(y0_pred_streams, self.config),
        }

    def sample_sigmas(self, batch_size: int, device) -> torch.Tensor:
        '''
        Training noise levels, one per sample (shared by both video streams):
        exp(N(0, 1)) * sqrt(state_t), replaced with probability high_sigma_ratio by a log-uniform
        draw in [200, 1e5].
        '''
        sigma = torch.exp(torch.randn(batch_size, device=device, dtype=torch.float64))
        sigma = sigma * math.sqrt(self.config.state_t)
        high = torch.rand(batch_size, device=device) < self.config.high_sigma_ratio
        log_uniform = torch.rand(batch_size, device=device, dtype=torch.float64) \
            * (math.log(1e5) - math.log(200.0)) + math.log(200.0)
        sigma = torch.where(high, torch.exp(log_uniform), sigma)
        return sigma.to(torch.float32).view(batch_size, 1, 1, 1, 1)

    def training_loss(self, entries: dict, crossattn_emb: torch.Tensor) -> tuple:
        '''
        One denoising step at a random noise level; EDM-weighted masked MSE on the generated
        rgb latents of both views. As in the training code used for the paper, the per-entry
        means of all five video data entries (rgb0, mask_rgb0, scene_prior_rgb0, rgb1, mask_rgb1)
        are averaged, so only rgb0 / rgb1 contribute (the other three are never supervised).
        :return (loss, info).
        '''
        cfg = self.config
        (x0_streams, masks) = pack_streams_from_entries(entries, cfg)
        B = x0_streams['v0'].shape[0]
        device = x0_streams['v0'].device

        sigma = self.sample_sigmas(B, device)
        sigmas = {k: sigma for k in GENERATED_STREAMS}
        yt_streams = {k: x0_streams[k] + sigmas[k] * torch.randn(x0_streams[k].shape, device=device)
                      for k in GENERATED_STREAMS}

        # Text dropout (classifier-free guidance training): zero the instruction per sample.
        keep = torch.bernoulli((1.0 - cfg.text_dropout) * torch.ones(B, device=device))
        crossattn_emb = crossattn_emb * keep.view(B, 1, 1).to(crossattn_emb.dtype)

        (_, y0_pred_streams) = self.denoise(x0_streams, yt_streams, masks, sigmas, crossattn_emb)

        weight = (sigma ** 2 + cfg.sigma_data ** 2) / (sigma * cfg.sigma_data) ** 2
        losses = {k: (y0_pred_streams[k] - x0_streams[k]) ** 2 * masks['supervise'][k] * weight
                  for k in GENERATED_STREAMS}
        loss_entries = unpack_entries_from_streams(losses, cfg)
        entry_means = [v.mean() * cfg.loss_weights.get(k, 1.0) for (k, v) in loss_entries.items()]
        loss = torch.stack(entry_means).mean() * cfg.loss_scale
        return (loss, {'sigma': sigma.flatten()})


def load_dit(checkpoint_path: str, config: RoboDreamConfig = RoboDreamConfig(), sac_mode: str = 'none'):
    '''
    Builds the DiT on cpu (fp32) and loads a RoboDream checkpoint strictly (net. prefix stripped).
    '''
    from .network import RoboDreamDiT

    dit = RoboDreamDiT(config, sac_mode=sac_mode)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    state = {(k[4:] if k.startswith('net.') else k): v for (k, v) in state.items()}
    # Activation checkpointing wraps blocks (keys gain '_checkpoint_wrapped_module.');
    # checkpoints are always stored in the plain layout.
    plain_to_module = {SAC_WRAP.sub('', mk): mk for mk in dit.state_dict().keys()}
    mapped = {plain_to_module[k]: v for (k, v) in state.items() if k in plain_to_module}
    # Training bookkeeping (_extra_state of fused norms, sample counters) is not needed.
    unexpected = [k for k in state if k not in plain_to_module
                  and not k.endswith('_extra_state') and not k.startswith('accum_')]
    (missing, _) = dit.load_state_dict(mapped, strict=False)
    missing = [SAC_WRAP.sub('', k) for k in missing if not SAC_WRAP.sub('', k).startswith('accum_')]
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint mismatch: missing={sorted(missing)[:8]} '
                           f'unexpected={sorted(unexpected)[:8]} (counts {len(missing)}/{len(unexpected)})')
    log.info(f'Loaded {len(mapped)} tensors from {checkpoint_path}')
    return dit


def save_checkpoint(dit_module: torch.nn.Module, path: str, step: int) -> None:
    '''
    Flat bf16 state dict under the net. prefix (same layout as the released checkpoint).
    '''
    state = {'net.' + SAC_WRAP.sub('', k): v.detach().to('cpu', torch.bfloat16)
             for (k, v) in dit_module.state_dict().items()}
    torch.save({'model': state, 'step': step}, path)
