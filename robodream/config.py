# All fixed constants of RoboDream: stream layout, 2B DiT dims, clip geometry, sampling, training recipe.

'''
Frozen configuration of the released RoboDream model.

Two video views are generated jointly: view 0 is an external camera, view 1 is the wrist camera.
Each view is one latent stream; the conditioning inputs live in extra channels of these streams
(robot-only video, scene prior), in extra self-attention tokens (object prior), and in extra
cross-attention tokens (end-effector trajectory), next to the T5 instruction embedding.
'''

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoboDreamConfig:

    # ---- Views and stream layout --------------------------------------------------------------
    # Values are (channel_start, channel_end) inside the latent stream of each view. Mask
    # channels carry the input / output mask values as data and always count as inputs.
    num_views: int = 2
    view0_entries: dict = field(default_factory=lambda: dict(
        rgb0=(0, 16),               # generated video (external camera)
        rgb0_input_mask=(16, 17),
        rgb0_output_mask=(17, 18),
        mask_rgb0=(18, 34),         # robot-only render (motion condition)
        scene_prior_rgb0=(34, 50),  # scene prior image, repeated over time
    ))
    view1_entries: dict = field(default_factory=lambda: dict(
        rgb1=(0, 16),               # generated video (wrist camera)
        rgb1_input_mask=(16, 17),
        rgb1_output_mask=(17, 18),
        mask_rgb1=(18, 34),         # robot-only render (motion condition)
    ))
    view_channels: tuple = (50, 34)

    # Object prior: one latent frame (16 x H_lat x W_lat) given as extra self-attention tokens.
    object_prior_channels: int = 16
    # Trajectory: (traj_len, 8) = xyz, rpy, gripper, clip-frame indicator, plus a validity mask
    # channel, flattened into a single extra cross-attention token.
    traj_len: int = 512
    traj_channels: int = 8
    action_multiplier: float = 2.0  # trajectory values are scaled so they do not drown in noise

    # ---- Frames and resolution ----------------------------------------------------------------
    num_frames: int = 189           # pixel frames per clip (1 + 4k)
    state_t: int = 48               # latent frames per clip
    num_cond_frames: int = 5        # pixel frames carried over between autoregressive clips
    height: int = 176
    width: int = 320
    vae_temporal_factor: int = 4
    vae_spatial_factor: int = 8
    latent_channels: int = 16
    fps: float = 10.0

    # ---- Diffusion sampling (RectifiedFlowAB2Scheduler) ---------------------------------------
    num_steps: int = 35
    sigma_max: float = 80.0
    sigma_min: float = 0.002
    scheduler_order: float = 7.0
    t_scaling_factor: float = 1.0
    sigma_data: float = 1.0

    # ---- Network: 2B DiT dims (cosmos-predict2 2B video2world) --------------------------------
    model_channels: int = 2048
    num_blocks: int = 28
    num_heads: int = 16
    crossattn_channels: int = 1024  # T5-11B hidden size
    text_tokens: int = 512
    view_emb_std: float = 0.06

    # ---- Training recipe ----------------------------------------------------------------------
    cond_frames_weights: dict = field(default_factory=lambda: {0: 0.8, 5: 0.2})
    text_dropout: float = 0.2
    high_sigma_ratio: float = 0.05
    loss_scale: float = 10.0
    loss_weights: dict = field(default_factory=lambda: dict(rgb0=1.0, rgb1=1.0))
    lr: float = 5e-5
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.99)
    max_iter: int = 100000          # data iterations (micro-batches); 50k optimizer steps at grad_accum 2
    grad_accum_iter: int = 2
    warmup_steps: int = 1000        # optimizer steps, lr factor lr_f_start -> 1
    lr_cycle_steps: int = 100000    # optimizer steps of the linear decay to lr_f_min (as configured for the
                                    # released run; it ends at ~53% of the peak after max_iter)
    lr_f_start: float = 0.01
    lr_f_min: float = 0.05

    @property
    def latent_hw(self) -> tuple:
        return (self.height // self.vae_spatial_factor, self.width // self.vae_spatial_factor)

    @property
    def video_entries(self) -> list:
        return [self.view0_entries, self.view1_entries]
