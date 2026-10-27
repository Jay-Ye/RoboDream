# Entry <-> stream packing: latent entries + masks -> network streams (v0, v1, sattn, xattn).

from typing import Optional

import torch

from .config import RoboDreamConfig

# Every data entry gets an input / output / supervise mask; mask channels (rgb*_input_mask,
# rgb*_output_mask, trajectory_input_mask) carry the mask values as data and always count as
# inputs. Entries that are not given (e.g. no scene prior) stay zero with all masks zero.
MASK_MODES = ('input', 'output', 'supervise')


@torch.no_grad()
def build_entries(
    rgb: list[torch.Tensor],
    mask_rgb: list[torch.Tensor],
    trajectory: torch.Tensor,
    trajectory_mask: torch.Tensor,
    cond_frames_latent: int,
    scene_prior: Optional[torch.Tensor] = None,
    scene_prior_valid: Optional[torch.Tensor] = None,
    object_prior: Optional[torch.Tensor] = None,
    object_prior_valid: Optional[torch.Tensor] = None,
) -> dict:
    '''
    Assemble latent entries and their masks for one batch.
    :param rgb: per view (B, 16, Tl, Hl, Wl) latents; only the first cond_frames_latent frames are
        used as input (the rest is generated), the content is the training target otherwise.
    :param mask_rgb: per view (B, 16, Tl, Hl, Wl) robot-only render latents (always input).
    :param trajectory: (B, 512, 8) trajectory, already multiplied by action_multiplier.
    :param trajectory_mask: (B, 512, 1) validity mask of the trajectory steps.
    :param cond_frames_latent (int): number of conditioning latent frames (0 = generate all).
    :param scene_prior: (B, 16, Tl, Hl, Wl) latent of the static scene prior video (view 0), or None.
    :param object_prior: (B, 16, Hl*Wl) latent of the object prior image, or None.
    :param *_valid: (B,) 0/1 per sample, or None (= all valid when the prior is given).
    :return entries (dict).
    '''
    tensor_kwargs = dict(device=rgb[0].device, dtype=torch.bfloat16)
    (B, _, Tl, Hl, Wl) = rgb[0].shape
    entries = dict()

    for v in range(len(rgb)):
        entries[f'rgb{v}'] = rgb[v].to(**tensor_kwargs)
        input_mask = torch.zeros((B, 1, Tl, Hl, Wl), **tensor_kwargs)
        input_mask[:, :, 0:cond_frames_latent] = 1.0
        entries[f'rgb{v}_input_mask'] = input_mask
        entries[f'rgb{v}_output_mask'] = 1.0 - input_mask
        entries[f'rgb{v}_supervise_mask'] = 1.0 - input_mask

        entries[f'mask_rgb{v}'] = mask_rgb[v].to(**tensor_kwargs)
        entries[f'mask_rgb{v}_input_mask'] = torch.ones((B, 1, Tl, Hl, Wl), **tensor_kwargs)

    if scene_prior is not None:
        valid = torch.ones(B, **tensor_kwargs) if scene_prior_valid is None else scene_prior_valid.to(**tensor_kwargs)
        entries['scene_prior_rgb0'] = scene_prior.to(**tensor_kwargs) * valid[:, None, None, None, None]
        entries['scene_prior_rgb0_input_mask'] = valid[:, None, None, None, None].expand(B, 1, Tl, Hl, Wl).contiguous()

    if object_prior is not None:
        valid = torch.ones(B, **tensor_kwargs) if object_prior_valid is None else object_prior_valid.to(**tensor_kwargs)
        entries['object_prior'] = object_prior.to(**tensor_kwargs) * valid[:, None, None]
        entries['object_prior_input_mask'] = valid[:, None, None].expand(B, 1, object_prior.shape[-1]).contiguous()

    entries['trajectory'] = trajectory.to(**tensor_kwargs)
    entries['trajectory_input_mask'] = trajectory_mask.to(**tensor_kwargs)
    return entries


@torch.no_grad()
def pack_streams_from_entries(entries: dict, config: RoboDreamConfig) -> tuple:
    '''
    :return (streams, masks): streams maps 'v0' / 'v1' to (B, C_v, Tl, Hl, Wl), 'sattn' to
        (B, 16, Hl*Wl) and 'xattn' to (B, 512, 9); masks maps 'input' / 'output' / 'supervise'
        to dicts of the same layout ('sattn' / 'xattn' are input-only).
    '''
    rgb0 = entries['rgb0']
    (B, _, Tl, Hl, Wl) = rgb0.shape
    tensor_kwargs = dict(device=rgb0.device, dtype=rgb0.dtype)

    streams = dict()
    masks = {mode: dict() for mode in MASK_MODES}

    for (v, view_entries) in enumerate(config.video_entries):
        shape = (B, config.view_channels[v], Tl, Hl, Wl)
        streams[f'v{v}'] = torch.zeros(shape, **tensor_kwargs)
        for mode in MASK_MODES:
            masks[mode][f'v{v}'] = torch.zeros(shape, **tensor_kwargs)

        for (k, (c0, c1)) in view_entries.items():
            if k.endswith('_mask'):
                masks['input'][f'v{v}'][:, c0:c1] = 1.0
            if k not in entries:
                continue
            streams[f'v{v}'][:, c0:c1] = entries[k]
            if not k.endswith('_mask'):
                for mode in MASK_MODES:
                    if f'{k}_{mode}_mask' in entries:
                        masks[mode][f'v{v}'][:, c0:c1] = entries[f'{k}_{mode}_mask']
                    elif mode == 'input':
                        masks[mode][f'v{v}'][:, c0:c1] = 1.0

    # Object prior tokens (input only).
    NS = Hl * Wl
    streams['sattn'] = torch.zeros((B, config.object_prior_channels, NS), **tensor_kwargs)
    masks['input']['sattn'] = torch.zeros((B, config.object_prior_channels, NS), **tensor_kwargs)
    if 'object_prior' in entries:
        streams['sattn'][:] = entries['object_prior']
        masks['input']['sattn'][:] = entries['object_prior_input_mask']

    # Trajectory token (input only); the last channel carries the validity mask as data.
    Ct = config.traj_channels
    streams['xattn'] = torch.zeros((B, config.traj_len, Ct + 1), **tensor_kwargs)
    masks['input']['xattn'] = torch.zeros((B, config.traj_len, Ct + 1), **tensor_kwargs)
    masks['input']['xattn'][:, :, Ct:Ct + 1] = 1.0
    streams['xattn'][:, :, 0:Ct] = entries['trajectory']
    streams['xattn'][:, :, Ct:Ct + 1] = entries['trajectory_input_mask']
    masks['input']['xattn'][:, :, 0:Ct] = entries['trajectory_input_mask']

    return (streams, masks)


def unpack_entries_from_streams(streams: dict, config: RoboDreamConfig) -> dict:
    '''
    Slice the video entries (data channels only) back out of the v0 / v1 streams.
    '''
    entries = dict()
    for (v, view_entries) in enumerate(config.video_entries):
        if f'v{v}' not in streams:
            continue
        for (k, (c0, c1)) in view_entries.items():
            if not k.endswith('_mask'):
                entries[k] = streams[f'v{v}'][:, c0:c1]
    return entries


def _masked_combine(a: dict, mask_a: dict, b: dict, mask_b: dict, keys: list) -> dict:
    out = dict()
    for k in sorted(keys):
        if k in a and k in b:
            out[k] = a[k] * mask_a[k] + b[k] * mask_b[k]
        elif k in a:
            out[k] = a[k] * mask_a[k]
        else:
            out[k] = b[k] * mask_b[k]
    return out


@torch.no_grad()
def assemble_network_inputs(x0_streams: dict, yt_streams: dict, masks: dict) -> dict:
    '''
    Network input per stream: conditioning inputs where the input mask is set, (scaled) noisy
    values where the output mask is set. Input-only streams (sattn, xattn) pass through masked.
    '''
    keys = set(x0_streams) | set(yt_streams)
    return _masked_combine(x0_streams, masks['input'], yt_streams, masks['output'], keys)


def assemble_clean_predictions(x0_streams: dict, y0_raw_pred_streams: dict, masks: dict) -> dict:
    '''
    Final prediction per generated stream: non-augmented conditioning inputs where the input mask
    is set, predicted outputs where the output mask is set.
    '''
    keys = set(x0_streams) & set(y0_raw_pred_streams)
    return _masked_combine(x0_streams, masks['input'], y0_raw_pred_streams, masks['output'], keys)
