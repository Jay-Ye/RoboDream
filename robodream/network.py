# RoboDream DiT: two-view diffusion transformer with object-prior tokens and a trajectory token.

from typing import Optional

import torch
from einops import rearrange
from torch import nn

from .config import RoboDreamConfig
from .vendor.action_video2world_dit import ActionConditionedMinimalV1LVGDiT, Mlp
from .vendor.text2image_dit import (
    Block,
    FinalLayer,
    PatchEmbed,
    SACConfig,
    TimestepEmbedding,
    Timesteps,
)


class RoboDreamBlock(Block):
    '''
    Transformer block over both view streams plus the object-prior (sattn) tokens:
    per-view AdaLN modulation and MLP, shared self-attention over [view 0, view 1, object prior],
    shared cross-attention onto [T5 instruction, trajectory token].
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rope_dim = kwargs['rope_dim']  # 128
        # Fallback rope for sattn tokens without a spatial grid (unused here: the object prior
        # gets grid rope), kept so the checkpoint loads without key remapping.
        self.sattn_rope = torch.nn.Parameter(torch.zeros(1, 1, 1, self.rope_dim))

    def forward(
        self,
        x_B_T_H_W_D: list[torch.Tensor],  # per view: (B, T, H, W, D) bf16
        emb_B_T_D: list[torch.Tensor],  # per view: (B, T, D) bf16
        crossattn_emb: torch.Tensor,  # (B, NC, D) bf16
        rope_emb_L_1_1_D: list[torch.Tensor],  # per view: (T*H*W, 1, 1, D) f32
        adaln_lora_B_T_3D: list[torch.Tensor],  # per view: (B, T, D*3) bf16
        sattn_emb_in: Optional[torch.Tensor] = None,  # (B, NS, D) bf16
        sattn_t_emb_norm: Optional[torch.Tensor] = None,  # (B, NS, D) bf16
        sattn_adaln_lora: Optional[torch.Tensor] = None,  # (B, NS, D*3) bf16
        sattn_rope_emb_L_1_1_D: Optional[torch.Tensor] = None,  # (NS, 1, 1, D) f32
        xattn_emb: Optional[torch.Tensor] = None,  # (B, NX, D) bf16
        **leftover,
    ) -> tuple[list[torch.Tensor], Optional[torch.Tensor]]:
        '''
        :return (V_x, sattn_out): per view (B, T, H, W, D) bf16, and (B, NS, D) bf16 or None.
        '''
        cat_rope_emb_L_1_1_D = torch.cat(rope_emb_L_1_1_D, dim=0)  # (V*T*H*W, 1, 1, D) f32

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        V_used = len(x_B_T_H_W_D)
        V_x = list(x_B_T_H_W_D)  # shallow copy
        V_normalized_x = []
        V_gate_next = []
        V_shapes = []
        V_seqlens = [0]

        # ================================ SELF-ATTENTION

        for v in range(V_used):
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D[v]) + adaln_lora_B_T_3D[v]
            ).chunk(3, dim=-1)

            shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
            scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
            gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

            normalized_x_B_T_H_W_D = _fn(
                V_x[v],
                self.layer_norm_self_attn,
                scale_self_attn_B_T_1_1_D,
                shift_self_attn_B_T_1_1_D
            )

            normalized_x_B_L_D = rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d")
            (T, H, W) = V_x[v].shape[1:4]

            V_normalized_x.append(normalized_x_B_L_D)
            V_shapes.append((T, H, W))
            V_seqlens.append(V_seqlens[v] + T * H * W)
            V_gate_next.append(gate_self_attn_B_T_1_1_D)

        cat_normalized_x_B_L_D = torch.cat(V_normalized_x, dim=1)  # (B, V*T*H*W, D) bf16
        NV = cat_normalized_x_B_L_D.shape[1]

        if sattn_emb_in is not None:
            sattn_emb = sattn_emb_in
            NS = sattn_emb.shape[1]
            (shift_esa, scale_esa, gate_esa) = (
                self.adaln_modulation_self_attn(sattn_t_emb_norm) + sattn_adaln_lora
            ).chunk(3, dim=-1)
            sattn_emb2 = _fn(sattn_emb, self.layer_norm_self_attn, scale_esa, shift_esa)
            aug_x = torch.cat([cat_normalized_x_B_L_D, sattn_emb2], dim=1)  # (B, NV + NS, D)
            if sattn_rope_emb_L_1_1_D is None:
                sattn_rope = self.sattn_rope.repeat(NS, 1, 1, 1)
            else:
                sattn_rope = sattn_rope_emb_L_1_1_D
            aug_rope = torch.cat([cat_rope_emb_L_1_1_D, sattn_rope], dim=0)
        else:
            aug_x = cat_normalized_x_B_L_D
            aug_rope = cat_rope_emb_L_1_1_D

        aug_res = self.self_attn(aug_x, None, rope_emb=aug_rope)
        cat_result_B_L_D = aug_res[:, 0:NV]

        if sattn_emb_in is not None:
            sattn_emb3 = aug_res[:, NV:NV + NS]
            sattn_emb4 = sattn_emb + gate_esa * sattn_emb3

        # ================================ CROSS-ATTENTION

        for v in range(V_used):
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D[v]) + adaln_lora_B_T_3D[v]
            ).chunk(3, dim=-1)

            shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
            scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
            gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

            result_B_L_D = cat_result_B_L_D[:, V_seqlens[v]:V_seqlens[v + 1]]
            (T, H, W) = V_shapes[v]
            result_B_T_H_W_D = rearrange(result_B_L_D, "b (t h w) d -> b t h w d", t=T, h=H, w=W)

            # this is gate_self_attn_B_T_1_1_D
            V_x[v] = V_x[v] + V_gate_next[v] * result_B_T_H_W_D

            normalized_x_B_T_H_W_D = _fn(
                V_x[v],
                self.layer_norm_cross_attn,
                scale_cross_attn_B_T_1_1_D,
                shift_cross_attn_B_T_1_1_D
            )

            V_normalized_x[v] = rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d")
            V_gate_next[v] = gate_cross_attn_B_T_1_1_D

        cat_normalized_x_B_L_D = torch.cat(V_normalized_x, dim=1)

        if sattn_emb_in is not None:
            (shift_esa, scale_esa, gate_esa) = (
                self.adaln_modulation_cross_attn(sattn_t_emb_norm) + sattn_adaln_lora
            ).chunk(3, dim=-1)
            sattn_emb5 = _fn(sattn_emb4, self.layer_norm_cross_attn, scale_esa, shift_esa)
            aug_x = torch.cat([cat_normalized_x_B_L_D, sattn_emb5], dim=1)
        else:
            aug_x = cat_normalized_x_B_L_D

        if xattn_emb is not None:
            aug_crossattn = torch.cat([crossattn_emb, xattn_emb], dim=1)  # (B, NC + NX, D)
        else:
            aug_crossattn = crossattn_emb

        aug_res = self.cross_attn(aug_x, aug_crossattn, rope_emb=aug_rope)
        cat_result_B_L_D = aug_res[:, 0:NV]

        if sattn_emb_in is not None:
            sattn_emb6 = aug_res[:, NV:NV + NS]
            sattn_emb7 = sattn_emb4 + gate_esa * sattn_emb6

        # ================================ MLP

        for v in range(V_used):
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D[v]) + adaln_lora_B_T_3D[v]
            ).chunk(3, dim=-1)

            shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
            scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
            gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")

            result_B_L_D = cat_result_B_L_D[:, V_seqlens[v]:V_seqlens[v + 1]]
            (T, H, W) = V_shapes[v]
            result_B_T_H_W_D = rearrange(result_B_L_D, "b (t h w) d -> b t h w d", t=T, h=H, w=W)

            # NOTE: residual-gate variant (block_gate_fix=False) the released checkpoint was
            # trained with: gate_mlp gates both the cross-attn residual and the mlp residual
            # (gate_cross_attn is unused).
            V_x[v] = V_x[v] + gate_mlp_B_T_1_1_D * result_B_T_H_W_D

            normalized_x_B_T_H_W_D = _fn(
                V_x[v],
                self.layer_norm_mlp,
                scale_mlp_B_T_1_1_D,
                shift_mlp_B_T_1_1_D
            )

            # mlp = GPT2FeedForward (linear layers only), so per-view application is exact.
            result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
            V_x[v] = V_x[v] + gate_mlp_B_T_1_1_D * result_B_T_H_W_D

        if sattn_emb_in is not None:
            (shift_esa, scale_esa, gate_esa) = (
                self.adaln_modulation_mlp(sattn_t_emb_norm) + sattn_adaln_lora
            ).chunk(3, dim=-1)
            sattn_emb8 = _fn(sattn_emb7, self.layer_norm_mlp, scale_esa, shift_esa)
            sattn_emb9 = self.mlp(sattn_emb8)
            sattn_out = sattn_emb7 + gate_esa * sattn_emb9
        else:
            sattn_out = None

        return (V_x, sattn_out)


class RoboDreamDiT(ActionConditionedMinimalV1LVGDiT):
    '''
    Two-view diffusion transformer (view 0 = external camera, view 1 = wrist camera).
    - Video streams are patch-embedded per view (with an additive view embedding for view 1),
      processed jointly by RoboDreamBlocks, and projected back per view.
    - The object prior latent is patch-embedded into extra self-attention tokens with the same
      spatial rope grid as the first video frame.
    - The flattened trajectory (and its validity mask) is embedded by an MLP into one extra
      cross-attention token, appended to the T5 instruction tokens.
    Module attribute names match the training code, so checkpoints load without key remapping.
    '''

    def __init__(self, config: RoboDreamConfig = RoboDreamConfig(), sac_mode: str = 'none'):
        '''
        :param sac_mode (str): activation checkpointing mode ('none' for inference). Parameters
            and state-dict keys do not depend on it.
        '''
        # Cosmos-Predict2-2B video2world hyperparameters.
        super().__init__(
            max_img_h=240,
            max_img_w=240,
            max_frames=128,
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            concat_padding_mask=True,
            model_channels=config.model_channels,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            atten_backend='minimal_a2a',
            pos_emb_cls='rope3d',
            pos_emb_learnable=True,
            pos_emb_interpolation='crop',
            use_adaln_lora=True,
            adaln_lora_dim=256,
            rope_h_extrapolation_ratio=3.0,
            rope_w_extrapolation_ratio=3.0,
            rope_t_extrapolation_ratio=1.0,
            extra_per_block_abs_pos_emb=False,
            rope_enable_fps_modulation=False,
            sac_config=SACConfig(mode=sac_mode),
            # view-0 in/out projections cover its full stream layout (masks included):
            in_channels_override=config.view_channels[0],
            out_channels_override=config.view_channels[0],
            block_cls=RoboDreamBlock,
            block_gate_fix=False,
            action_dim=0,  # no adaln action conditioning
        )
        # ^ this initializes x_embedder, pos_embedder, t_embedder, blocks, final_layer,
        # and t_embedding_norm (all for view 0 / shared).

        self.config = config
        D = self.model_channels

        # View 1 ("newviews" = all views beyond the pretrained view 0).
        self.x_embedder_newviews = nn.ModuleList([
            PatchEmbed(
                spatial_patch_size=self.patch_spatial,
                temporal_patch_size=self.patch_temporal,
                in_channels=config.view_channels[1],
                out_channels=D,
            )
        ])
        self.view_embs_newviews = nn.ParameterList([
            nn.Parameter(torch.randn(1, D) * config.view_emb_std)
        ])
        self.t_embedder_newviews = nn.ModuleList([
            nn.Sequential(
                Timesteps(D),
                TimestepEmbedding(D, D, use_adaln_lora=self.use_adaln_lora),
            )
        ])
        self.final_layer_newviews = nn.ModuleList([
            FinalLayer(
                hidden_size=D,
                spatial_patch_size=self.patch_spatial,
                temporal_patch_size=self.patch_temporal,
                out_channels=config.view_channels[1],
                use_adaln_lora=self.use_adaln_lora,
                adaln_lora_dim=self.adaln_lora_dim,
            )
        ])

        # Object prior -> self-attention tokens, with its own timestep embedder.
        self.sattn_inproj = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=config.object_prior_channels,
            out_channels=D,
        )
        self.t_embedder_sattn = nn.Sequential(
            Timesteps(D),
            TimestepEmbedding(D, D, use_adaln_lora=self.use_adaln_lora),
        )

        # Trajectory (+ validity mask), flattened -> one cross-attention token.
        C = config.crossattn_channels
        self.xattn_embedders = nn.ModuleDict({
            'trajectory': Mlp(
                in_features=config.traj_len * (config.traj_channels + 1),
                hidden_features=C * 4,
                out_features=C,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                drop=0,
            ),
        })
        self.xattn_type_embeddings = nn.ParameterDict({
            'trajectory': nn.Parameter(torch.zeros(1, 1, C)),
        })

    def embed_trajectory(self, xattn_in: torch.Tensor) -> Optional[torch.Tensor]:
        '''
        :param xattn_in: (B, traj_len, traj_channels + 1) trajectory (already masked) + mask.
        :return: (B, 1, C) bf16 token, or None when the trajectory is all zeros (not given).
        '''
        traj = xattn_in[..., :self.config.traj_channels]
        if torch.allclose(traj, torch.zeros_like(traj), atol=1e-8):
            return None
        flat = rearrange(xattn_in, "b t d -> b 1 (t d)")
        return self.xattn_embedders['trajectory'](flat) + self.xattn_type_embeddings['trajectory']

    def forward(
        self,
        x_in_streams: dict[str, torch.Tensor],
        timesteps: dict[str, torch.Tensor],
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        **leftover,
    ) -> dict[str, torch.Tensor]:
        '''
        :param x_in_streams: 'v0' (B, 50, T, H, W), 'v1' (B, 34, T, H, W),
            'sattn' (B, 16, H*W) object-prior latent, 'xattn' (B, 512, 9) trajectory + mask.
        :param timesteps: maps 'v0' / 'v1' to (B, 1, 1 or T, 1, 1) noise level tensors.
        :param crossattn_emb: (B, 512, 1024) T5 instruction embedding.
        :param fps: (B) tensor; inert (rope fps modulation is off).
        :return y_out_streams: maps 'v0' / 'v1' to (B, C_v, T, H, W) tensors.
        '''
        my_x_embedders = [self.x_embedder, *self.x_embedder_newviews]
        my_final_layers = [self.final_layer, *self.final_layer_newviews]
        my_t_embedders = [self.t_embedder, *self.t_embedder_newviews]
        V = self.config.num_views
        (lat_H, lat_W) = x_in_streams['v0'].shape[-2:]

        # Per-view input projection (+ view embedding), rope, and timestep embeddings.
        x_proj_all = []
        rope_emb_all = []
        t_embedding_all = []
        t_emb_norm_all = []
        adaln_lora_all = []

        for v in range(V):
            x_proj_v = my_x_embedders[v](x_in_streams[f'v{v}'])  # (B, T, H, W, D) bf16
            if v >= 1:
                x_proj_v = x_proj_v + self.view_embs_newviews[v - 1]
            timesteps_v = timesteps[f'v{v}'][:, 0, :, 0, 0]  # (B, 1 or T)
            if timesteps_v.shape[1] == 1:
                timesteps_v = timesteps_v.repeat(1, x_proj_v.shape[1])

            rope_emb_v = self.pos_embedder(x_proj_v, fps=fps)  # (T*H*W, 1, 1, D) f32
            (t_embedding_v, adaln_lora_v) = my_t_embedders[v](timesteps_v)
            t_emb_norm_v = self.t_embedding_norm(t_embedding_v)

            x_proj_all.append(x_proj_v)
            rope_emb_all.append(rope_emb_v)
            t_embedding_all.append(t_embedding_v)
            t_emb_norm_all.append(t_emb_norm_v)
            adaln_lora_all.append(adaln_lora_v)

        # Object prior tokens: one latent frame on the video grid, with grid rope.
        sattn_in = rearrange(x_in_streams['sattn'], "b d (t h w) -> b d t h w", t=1, h=lat_H, w=lat_W)
        sattn_emb = self.sattn_inproj(sattn_in)  # (B, 1, H', W', D)
        sattn_rope_emb = self.pos_embedder(sattn_emb, fps=fps)
        sattn_emb = rearrange(sattn_emb, "b t h w d -> b (t h w) d")
        NS = sattn_emb.shape[1]
        timesteps_sattn = timesteps['v0'][:, 0, :, 0, 0]
        if timesteps_sattn.shape[1] == 1:
            timesteps_sattn = timesteps_sattn.repeat(1, NS)
        (sattn_t_embedding, sattn_adaln_lora) = self.t_embedder_sattn(timesteps_sattn)
        sattn_t_emb_norm = self.t_embedding_norm(sattn_t_embedding)

        xattn_emb = self.embed_trajectory(x_in_streams['xattn'])

        x_block_all = x_proj_all
        for block in self.blocks:
            (x_block_all, sattn_out) = block(
                x_B_T_H_W_D=x_block_all,
                emb_B_T_D=t_emb_norm_all,
                crossattn_emb=crossattn_emb,
                rope_emb_L_1_1_D=rope_emb_all,
                adaln_lora_B_T_3D=adaln_lora_all,
                sattn_emb_in=sattn_emb,
                sattn_t_emb_norm=sattn_t_emb_norm,
                sattn_adaln_lora=sattn_adaln_lora,
                sattn_rope_emb_L_1_1_D=sattn_rope_emb,
                xattn_emb=xattn_emb,
            )
            sattn_emb = sattn_out

        y_out_streams = dict()
        for v in range(V):
            y_proj_v = my_final_layers[v](x_block_all[v], t_embedding_all[v], adaln_lora_B_T_3D=adaln_lora_all[v])
            y_out_streams[f'v{v}'] = self.unpatchify(y_proj_v)  # (B, C_v, T, H, W)

        return y_out_streams
