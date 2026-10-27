#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Cache T5 text embeddings for all Droid language instructions.

Creates a dict:
  { "<language instruction str>": np.ndarray(shape=(seq_len, 1024), dtype=float16) }

The embeddings come from `robodream.text.T5Encoder.encode(...)` (T5-11B last hidden state) and are trimmed
to the true sequence length (derived from the attention mask) to save space.

Example:
  python data_processing/06_text_embeddings/cache_text_embedding.py \
    --annotations_json ${ROBODREAM_DATA_DIR}/droid_meta/droid_language_annotations.json \
    --text_encoder_path ${ROBODREAM_CKPT_DIR}/google-t5/t5-11b \
    --output ${ROBODREAM_DATA_DIR}/droid_t5_text_embeddings.pkl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import subprocess
from typing import Dict, Iterable, List, Optional

import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from robodream.text import T5Encoder
from robodream.vendor import log

try:
    import tqdm as _tqdm  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    _tqdm = None


def _tqdm_wrap(it, *, total: Optional[int] = None, disable: bool = False):
    if disable or _tqdm is None:
        return it
    return _tqdm.tqdm(it, total=total)


def _iter_instructions(annotations: dict) -> Iterable[str]:
    for _, anno_per_episode in annotations.items():
        if not isinstance(anno_per_episode, dict):
            continue
        for anno in anno_per_episode.values():
            if not isinstance(anno, str):
                continue
            anno = anno.strip()
            if not anno:
                continue
            yield anno


def _batcher(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _save_pickle(obj, path: str, compress: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if compress == "gzip":
        with gzip.open(path, "wb", compresslevel=6) as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif compress == "none":
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Unknown compress mode: {compress}")


def _aws_s3_cp(local_path: str, s3_prefix: str) -> None:
    # If prefix is a directory-like S3 URI, append filename.
    if s3_prefix.startswith("s3://") and s3_prefix.endswith("/"):
        s3_dst = s3_prefix + os.path.basename(local_path)
    else:
        s3_dst = s3_prefix
    log.info(f"Uploading {local_path} -> {s3_dst}")
    subprocess.run(["aws", "s3", "cp", local_path, s3_dst], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache T5 embeddings for Droid language instructions.")
    parser.add_argument(
        "--annotations_json",
        type=str,
        default=os.path.join(os.environ.get("ROBODREAM_DATA_DIR", "datasets"), "droid_meta", "droid_language_annotations.json"),
        help="Path to droid_language_annotations.json",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=os.path.join(os.environ.get("ROBODREAM_CKPT_DIR", "checkpoints"), "google-t5", "t5-11b"),
        help="S3 or local path to the cached T5 model folder.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for encoding (e.g., cuda or cpu).")
    parser.add_argument("--max_length", type=int, default=512, help="Tokenizer max_length.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for text encoding.")
    parser.add_argument(
        "--total_splits",
        type=int,
        default=1,
        help="Split the unique instructions into total_splits disjoint parts and only run one part.",
    )
    parser.add_argument(
        "--split_idx",
        type=int,
        default=0,
        help="Which split to run in [0, total_splits). Only this split will be encoded.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.environ.get("ROBODREAM_DATA_DIR", "datasets"), "droid_t5_text_embeddings.pkl.gz"),
        help="Output cache file path (.pkl or .pkl.gz recommended).",
    )
    parser.add_argument(
        "--compress",
        type=str,
        choices=["none", "gzip"],
        default="gzip",
        help="Compression for the output pickle.",
    )
    parser.add_argument(
        "--upload_s3",
        type=str,
        default="",
        help="If set, upload output file to this S3 prefix (or full S3 URI).",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Force HF loading from local cache_dir only (recommended when using S3-synced cache).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device
    if str(device).startswith("cuda") and torch.cuda.is_available():
        # Respect CUDA_VISIBLE_DEVICES; user can pin to a specific GPU by setting it.
        torch.cuda.set_device(0)
        device = "cuda:0"

    if not os.path.exists(args.annotations_json):
        raise FileNotFoundError(f"annotations_json not found: {args.annotations_json}")

    log.info(f"Loading annotations: {args.annotations_json}")
    with open(args.annotations_json, "r") as f:
        annotations = json.load(f)

    instructions = list(_iter_instructions(annotations))
    uniq = sorted(set(instructions))
    log.info(f"Found {len(instructions)} total (non-empty) instructions, {len(uniq)} unique.")

    total_splits = int(args.total_splits)
    split_idx = int(args.split_idx)
    if total_splits <= 0:
        raise ValueError(f"--total_splits must be >= 1, got {total_splits}")
    if not (0 <= split_idx < total_splits):
        raise ValueError(f"--split_idx must be in [0, total_splits), got split_idx={split_idx}, total_splits={total_splits}")

    log.info(f"text_encoder_path={args.text_encoder_path}")
    encoder = T5Encoder(args.text_encoder_path, device=device)

    max_length = int(args.max_length)
    bs = int(args.batch_size)

    # Split work across multiple independent jobs (no distributed / multi-GPU code).
    # We use a strided split for determinism and simple load balancing.
    uniq_split = uniq[split_idx::total_splits]

    cache: Dict[str, np.ndarray] = {}
    total_batches = (len(uniq_split) + bs - 1) // bs if bs > 0 else 0
    log.info(
        f"Encoding split {split_idx}/{total_splits} with {len(uniq_split)} unique instructions "
        f"(batch_size={bs}, max_length={max_length})..."
    )

    for batch in _tqdm_wrap(_batcher(uniq_split, bs), total=total_batches, disable=False):
        encoded_text, lengths = encoder.encode(batch, max_length=max_length)
        # encoded_text: (B, max_length, C) on device, zero beyond each length.
        lengths = lengths.numpy().tolist()
        encoded_np = encoded_text.detach().cpu().numpy().astype(np.float16)
        for i, instr in enumerate(batch):
            L = int(lengths[i])
            cache[instr] = encoded_np[i, :L]  # (L, 1024)

    # If running multiple splits, write split-specific output to avoid clobbering.
    out_path = args.output
    if total_splits > 1:
        base, ext = os.path.splitext(out_path)
        if ext == ".gz" and base.endswith(".pkl"):
            base2, _ = os.path.splitext(base)  # base2 + ".pkl" + ".gz"
            out_path = f"{base2}.part{split_idx}of{total_splits}.pkl{ext}"
        else:
            out_path = f"{base}.part{split_idx}of{total_splits}{ext}"

    log.info(f"Built cache dict with {len(cache)} entries. Saving: {out_path}")
    _save_pickle(cache, out_path, compress=args.compress)
    log.info(f"Saved: {out_path}")

    if args.upload_s3:
        _aws_s3_cp(out_path, args.upload_s3)
        log.info("Upload complete.")


if __name__ == "__main__":
    # Avoid autograd overhead
    torch.set_grad_enabled(False)
    main()

