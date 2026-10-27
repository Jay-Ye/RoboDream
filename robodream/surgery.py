# Initialize the RoboDream DiT from the Cosmos-Predict2-2B-Video2World base checkpoint.

import torch

from .config import RoboDreamConfig
from .network import RoboDreamDiT
from .pipe import SAC_WRAP
from .vendor import log

PATCH = 4  # spatial 2 x 2 x temporal 1: every latent channel spans 4 consecutive projection columns

# Per view: entry -> (source entry in the base view-0 layout, or 'zero', or None = leave as is).
# The base model's 18 input channels are [16 rgb latent, condition mask, padding mask].
IN_INIT = [
    dict(rgb0='rgb0', rgb0_input_mask='rgb0_input_mask', rgb0_output_mask='zero',
         mask_rgb0='rgb0', scene_prior_rgb0='rgb0'),
    dict(rgb1='rgb0', rgb1_input_mask='rgb0_input_mask', rgb1_output_mask='zero', mask_rgb1='rgb0'),
]
OUT_INIT = [dict(rgb0='rgb0'), dict(rgb1='rgb0')]
BASE_IN = dict(rgb0=(0, 16), rgb0_input_mask=(16, 17))
BASE_OUT = dict(rgb0=(0, 16))
BASE_IN_CHANNELS = 18
BASE_OUT_CHANNELS = 16


def init_from_cosmos(base_path: str, config: RoboDreamConfig = RoboDreamConfig(), sac_mode: str = 'none') -> RoboDreamDiT:
    '''
    Builds the DiT and loads the Cosmos base weights: shared blocks are loaded as is; the widened
    input projections copy the base rgb projection for every image-like entry (rgb, robot-only
    render, scene prior), output projections copy the base rgb projection, the view-1 timestep
    embedder copies the base one. Unused output channels keep the constant -0.1 fill used in
    training, everything else (view embedding, object-prior / trajectory embedders) keeps its
    default initialization.
    '''
    dit = RoboDreamDiT(config, sac_mode=sac_mode)
    state = torch.load(base_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    state = {(k[4:] if k.startswith('net.') else k): v for (k, v) in state.items()}
    src_in = state['x_embedder.proj.1.weight']  # (2048, 18 * 4)
    src_out = state['final_layer.linear.weight']  # (16 * 4, 2048)
    assert src_in.shape[1] == BASE_IN_CHANNELS * PATCH and src_out.shape[0] == BASE_OUT_CHANNELS * PATCH

    in_keys = ['x_embedder.proj.1.weight', 'x_embedder_newviews.0.proj.1.weight']
    out_keys = ['final_layer.linear.weight', 'final_layer_newviews.0.linear.weight']
    for (v, entries) in enumerate(config.video_entries):
        C = config.view_channels[v]
        w_in = -0.1 * torch.ones((src_in.shape[0], C * PATCH), dtype=src_in.dtype)
        for (k, src) in IN_INIT[v].items():
            (c0, c1) = entries[k]
            if src == 'zero':
                w_in[:, c0 * PATCH:c1 * PATCH] = 0.0
            else:
                (s0, s1) = BASE_IN[src]
                w_in[:, c0 * PATCH:c1 * PATCH] = src_in[:, s0 * PATCH:s1 * PATCH]
        state[in_keys[v]] = w_in

        # Output columns are position-major: row = channel + p * C.
        w_out = -0.1 * torch.ones((C * PATCH, src_out.shape[1]), dtype=src_out.dtype)
        for (k, src) in OUT_INIT[v].items():
            (c0, c1) = entries[k]
            (s0, s1) = BASE_OUT[src]
            for p in range(PATCH):
                w_out[c0 + p * C:c1 + p * C] = src_out[s0 + p * BASE_OUT_CHANNELS:s1 + p * BASE_OUT_CHANNELS]
        state[out_keys[v]] = w_out

    for name in ['linear_1', 'linear_2']:
        state[f't_embedder_newviews.0.1.{name}.weight'] = state[f't_embedder.1.{name}.weight']

    plain_to_module = {SAC_WRAP.sub('', mk): mk for mk in dit.state_dict().keys()}
    loadable = {plain_to_module[k]: v for (k, v) in state.items() if k in plain_to_module}
    (missing, _) = dit.load_state_dict(loadable, strict=False)
    missing = sorted(SAC_WRAP.sub('', k) for k in missing)
    log.info(f'Initialized from {base_path}; default-initialized: {missing}')
    return dit
