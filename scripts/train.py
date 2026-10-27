# RoboDream training on DROID clips (torch DDP or FSDP2, bf16 autocast, fp32 master weights).
'''
Parallelism: FSDP2 (parameters, gradients and optimizer state sharded across GPUs; ~56 GB peak per
GPU on 2 x 80 GB, less with more GPUs). Plain DDP (--ddp) needs more than 80 GB per GPU at 189 frames.

Recipe of the released model: rectified-flow denoising with per-sample noise level
sigma = exp(N(0, 1)) * sqrt(48) (5% replaced by a log-uniform draw in [200, 1e5]), shared by both
views; the first 5 clip frames are given as input with probability 0.2 (no priors then) and the
clip is generated from the priors otherwise; instruction dropout 0.2; EDM-weighted MSE on the
generated latents x 10; AdamW (lr 5e-5, betas 0.9 / 0.99, weight decay 0.1), 1000 warmup steps from
1% then linear decay towards 5%; batch 1 per GPU, 2 accumulation steps, 16 GPUs, 100k iterations
(= data batches per GPU; 50k optimizer steps).

Examples (inside the container, after `source shell/export.sh`):
    # fine-tune / resume from the released checkpoint
    torchrun --nproc_per_node=8 scripts/train.py --init ${ROBODREAM_CKPT_DIR}/robodream/model.pt
    # train from the Cosmos-Predict2-2B-Video2World base model (as in the paper)
    torchrun --nproc_per_node=8 scripts/train.py --init_cosmos \
        ${ROBODREAM_CKPT_DIR}/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-10fps.pt
'''

import argparse
import contextlib
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    ckpt_dir = os.environ.get('ROBODREAM_CKPT_DIR', 'checkpoints')
    data_dir = os.environ.get('ROBODREAM_DATA_DIR', 'datasets')
    out_dir = os.environ.get('ROBODREAM_OUTPUT_DIR', 'outputs')
    parser = argparse.ArgumentParser(description='Train RoboDream', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', default=f'{data_dir}/processed/droid_traj2demo/real',
                        help='Processed DROID episodes (see data_processing/README.md)')
    parser.add_argument('--clips', default=f'{data_dir}/hf/data/meta/split_train_station.txt',
                        help='Clip list: released shard list or "<episode> <start>" lines')
    parser.add_argument('--init', default='', help='Start from a RoboDream checkpoint')
    parser.add_argument('--init_cosmos', default='', help='Start from the Cosmos-Predict2-2B-Video2World base checkpoint')
    parser.add_argument('--tokenizer', default=f'{ckpt_dir}/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth')
    parser.add_argument('--t5_cache', default=f'{data_dir}/droid_t5_text_embeddings.pkl.gz')
    parser.add_argument('--out', default=f'{out_dir}/robodream_train')
    parser.add_argument('--max_iter', type=int, default=None, help='Data iterations per GPU (default: config)')
    parser.add_argument('--lr', type=float, default=None, help='Peak learning rate (default: config)')
    parser.add_argument('--batch_size', type=int, default=1, help='Clips per GPU per micro-step')
    parser.add_argument('--grad_accum', type=int, default=None, help='Micro-steps per optimizer step (default: config)')
    parser.add_argument('--ddp', action='store_true', help='Replicate the model with DDP instead of sharding with FSDP2')
    parser.add_argument('--sac_mode', default='mm_only', help="Activation checkpointing: 'mm_only', 'predict2_2b_720' or 'none'")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=1000, help='Iterations between checkpoints')
    parser.add_argument('--log_every', type=int, default=10, help='Iterations between log lines')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases (WANDB_API_KEY / WANDB_PROJECT)')
    return parser.parse_args()


def setup_distributed():
    import torch
    import torch.distributed as dist
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', device_id=torch.device('cuda', local_rank))
        return (dist.get_rank(), dist.get_world_size(), local_rank)
    torch.cuda.set_device(0)
    return (0, 1, 0)


def lr_lambda_factory(warmup_steps, cycle_length, f_start, f_min):
    '''
    Linear warmup from f_start to 1, then linear decay that reaches f_min at cycle_length.
    '''
    def lr_lambda(step):
        if step < warmup_steps:
            return f_start + (1.0 - f_start) * step / max(1, warmup_steps)
        return f_min + (1.0 - f_min) * max(0.0, cycle_length - step) / max(1, cycle_length - warmup_steps)
    return lr_lambda


def collate(items):
    import torch
    batch = {k: torch.stack([it[k] for it in items]) for k in items[0] if k not in ('prompt', 'clip_id')}
    batch['prompt'] = [it['prompt'] for it in items]
    batch['clip_id'] = [it['clip_id'] for it in items]
    return batch


class StateHolder:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler

    from robodream.config import RoboDreamConfig
    from robodream.dataset import DroidClipDataset, read_clip_list
    from robodream.inputs import encode_entries
    from robodream.pipe import RoboDreamPipeline, load_dit, save_checkpoint
    from robodream.text import TextEmbedder
    from robodream.vae import RoboDreamVAE

    cfg = RoboDreamConfig()
    max_iter = args.max_iter or cfg.max_iter
    grad_accum = args.grad_accum or cfg.grad_accum_iter
    (rank, world_size, local_rank) = setup_distributed()
    device = torch.device('cuda', local_rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    is_main = rank == 0

    # ---- model
    assert bool(args.init) != bool(args.init_cosmos), 'pass exactly one of --init / --init_cosmos'
    if args.init:
        dit = load_dit(args.init, cfg, sac_mode=args.sac_mode)
    else:
        from robodream.surgery import init_from_cosmos
        dit = init_from_cosmos(args.init_cosmos, cfg, sac_mode=args.sac_mode)
    dit = dit.to(device=device, dtype=torch.float32).train()

    if world_size > 1 and not args.ddp:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for block in dit.blocks:
            fully_shard(block, mp_policy=mp)
        fully_shard(dit, mp_policy=mp)
        dit_train = dit
    elif world_size > 1:  # --ddp
        from torch.nn.parallel import DistributedDataParallel
        # Some modules see no gradient in a given step (e.g. the trajectory MLP when no
        # trajectory is given), so DDP must not wait for them.
        dit_train = DistributedDataParallel(dit, device_ids=[local_rank], find_unused_parameters=True)
    else:
        dit_train = dit
    pipe = RoboDreamPipeline(dit_train, cfg, device=str(device))
    vae = RoboDreamVAE(args.tokenizer, chunk_duration=cfg.num_frames, device=str(device))
    text = TextEmbedder(args.t5_cache, device=str(device))

    # ---- data
    dataset = DroidClipDataset(args.data_root, read_clip_list(args.clips), cfg, random_view=True, seed=args.seed + rank)

    def worker_init_fn(worker_id):
        seed = args.seed + 1000 * rank + worker_id
        dataset.rng.seed(seed)
        np.random.seed(seed)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed,
                                 drop_last=True) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
                        num_workers=args.num_workers, collate_fn=collate, drop_last=True, pin_memory=True,
                        worker_init_fn=worker_init_fn, persistent_workers=args.num_workers > 0)
    if is_main:
        os.makedirs(args.out, exist_ok=True)
        print(f'{len(dataset)} clips, world_size={world_size}, batch/gpu={args.batch_size}, '
              f'grad_accum={grad_accum}, max_iter={max_iter}, {"ddp" if args.ddp else "fsdp"}')

    # ---- optimization
    optimizer = torch.optim.AdamW(dit.parameters(), lr=args.lr or cfg.lr, betas=cfg.betas,
                                  weight_decay=cfg.weight_decay, eps=1e-8, fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(cfg.warmup_steps, cfg.lr_cycle_steps, cfg.lr_f_start, cfg.lr_f_min))
    cond_choices = list(cfg.cond_frames_weights.keys())
    cond_weights = list(cfg.cond_frames_weights.values())
    wandb_run = None
    if args.wandb and is_main:
        import wandb
        wandb_run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'robodream'), config=vars(args))

    log_path = os.path.join(args.out, 'train_log.jsonl')
    (iteration, epoch) = (0, 0)
    t_last = time.time()
    while iteration < max_iter:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            cond_frames_raw = random.choices(cond_choices, weights=cond_weights)[0]
            with torch.no_grad():
                entries = encode_entries(vae, batch, cond_frames_raw, cfg)
                crossattn_emb = text(batch['prompt'], device=device)
            sync = (iteration + 1) % grad_accum == 0
            no_sync = world_size > 1 and args.ddp and not sync
            with (dit_train.no_sync() if no_sync else contextlib.nullcontext()):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    (loss, info) = pipe.training_loss(entries, crossattn_emb)
                (loss / grad_accum).backward()
            iteration += 1
            if sync:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if is_main and (iteration % args.log_every == 0 or iteration == 1):
                now = time.time()
                record = {'iteration': iteration, 'loss': float(loss.item()), 'lr': float(scheduler.get_last_lr()[0]),
                          'sigma': float(info['sigma'][0]), 'cond_frames': cond_frames_raw,
                          'sec_per_iter': (now - t_last) / (args.log_every if iteration > 1 else 1),
                          'peak_vram_gb': round(torch.cuda.max_memory_allocated(device) / 1e9, 2)}
                print(json.dumps(record))
                with open(log_path, 'a') as f:
                    f.write(json.dumps(record) + '\n')
                if wandb_run is not None:
                    wandb_run.log(record, step=iteration)
                t_last = now

            if iteration % args.save_every == 0 or iteration == max_iter:
                state_source = dit
                if world_size > 1 and not args.ddp:
                    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
                    state_source = StateHolder(get_model_state_dict(
                        dit, options=StateDictOptions(full_state_dict=True, cpu_offload=True)))
                if is_main:
                    path = os.path.join(args.out, f'model_iter{iteration:06d}.pt')
                    save_checkpoint(state_source, path, iteration)
                    print(f'saved {path}')
            if iteration >= max_iter:
                break
        epoch += 1

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
