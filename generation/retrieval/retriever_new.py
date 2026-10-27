#!/usr/bin/env python3
# Originally developed in our VLA-0-TRL fork; this file is standalone and contains no VLA-0 code.
"""Prompt retriever using cached T5 token embeddings and CosmosT5TextEncoder.

This script loads a cached prompt embedding file (pickle or pickle.gz), builds
prompt-level vectors by mean-pooling token embeddings, and retrieves top-K
prompts by cosine similarity for a text query.
"""

import argparse
import gzip
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch


# Make generation/ importable for `utils.text_encoder`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = os.environ.get("ROBODREAM_DATA_DIR", "datasets")
CKPT_DIR = os.environ.get("ROBODREAM_CKPT_DIR", "checkpoints")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.text_encoder import CosmosT5TextEncoder  # noqa: E402


@dataclass
class RetrievalResult:
    """Container for prompt retrieval result."""

    sample_path: str
    description: str
    similarity: float
    trajectory: Dict
    idx: int


def _quat_from_rpy(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Convert roll/pitch/yaw (rad) to quaternion (w,x,y,z), intrinsic XYZ."""
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.stack([qw, qx, qy, qz], axis=-1).astype(np.float32, copy=False)


def cartesian_rpy_to_quat(actions: np.ndarray) -> np.ndarray:
    """Convert (N, 6) [x,y,z,roll,pitch,yaw] to (N, 7) [x,y,z,qw,qx,qy,qz]."""
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError(f"Expected input shape (N, 6), got {actions.shape}")

    xyz = actions[:, :3]
    roll, pitch, yaw = actions[:, 3], actions[:, 4], actions[:, 5]
    quat = _quat_from_rpy(roll, pitch, yaw)
    return np.concatenate([xyz, quat], axis=1).astype(np.float32, copy=False)


class CachedPromptRetriever:
    """Retrieve top-K prompts from cached T5 token embeddings."""

    def __init__(
        self,
        text_embedding_path: str,
        language_annotations_path: str,
        trajectory_root: str = os.path.join(DATA_DIR, "trajengine"),
        model_name: str = "google-t5/t5-11b",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_cache_dir: str = None,
        local_files_only: bool = False,
        max_length: int = 512,
        cache_dtype: torch.dtype = torch.float16,
    ):
        self.text_embedding_path = text_embedding_path
        self.language_annotations_path = language_annotations_path
        self.trajectory_root = trajectory_root
        self.device = device
        self.max_length = max_length
        self.cache_dtype = cache_dtype
        self._model_name = model_name
        self._model_cache_dir = model_cache_dir
        self._local_files_only = local_files_only
        self.text_encoder = None

        print(f"Loading prompt embedding cache from {text_embedding_path}")
        self.prompt_to_vec, self.prompt_to_length = self._load_prompt_cache()

        print(f"Loading language annotations from {language_annotations_path}")
        self.metadata, self.embeddings = self._build_metadata_and_embeddings()

        self.num_descriptions = len(self.metadata)
        self.embedding_dim = int(self.embeddings.shape[1])

        # Lazily initialize query encoder only when cache miss happens.
        print(
            f"Retriever ready: {self.num_descriptions} descriptions, embedding_dim={self.embedding_dim}"
        )

    def _load_prompt_cache(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        opener = gzip.open if str(self.text_embedding_path).endswith(".gz") else open
        with opener(self.text_embedding_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected dict[str, np.ndarray] in {self.text_embedding_path}, got {type(data)}"
            )

        prompt_to_vec: Dict[str, np.ndarray] = {}
        prompt_to_length: Dict[str, int] = {}

        for prompt, emb in data.items():
            if not isinstance(prompt, str) or emb is None:
                continue

            arr = self._parse_embedding(prompt, emb)
            if arr is None:
                continue

            # Truncate to match query encoding max length.
            length = int(min(arr.shape[0], self.max_length))
            arr = arr[:length]
            vec = arr.mean(axis=0, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm <= 0:
                continue

            prompt_to_vec[prompt] = (vec / norm).astype(np.float32, copy=False)
            prompt_to_length[prompt] = length

        if not prompt_to_vec:
            raise ValueError(
                f"No valid prompt embeddings found in {self.text_embedding_path}"
            )

        return prompt_to_vec, prompt_to_length

    def _build_metadata_and_embeddings(self) -> Tuple[List[Dict[str, object]], np.ndarray]:
        with open(self.language_annotations_path, "r") as f:
            annotations = json.load(f)

        if not isinstance(annotations, dict):
            raise ValueError(
                f"Expected dict in {self.language_annotations_path}, got {type(annotations)}"
            )

        metadata: List[Dict[str, object]] = []
        vectors: List[np.ndarray] = []
        filtered_missing_episode = 0
        missing_episode_ids = set()

        for episode_id, instruction_dict in annotations.items():
            if not isinstance(episode_id, str) or not isinstance(instruction_dict, dict):
                continue

            sample_path = os.path.join(self.trajectory_root, episode_id)
            traj_h5_path = os.path.join(sample_path, "trajectory.h5")
            if not os.path.isdir(sample_path) or not os.path.isfile(traj_h5_path):
                filtered_missing_episode += 1
                missing_episode_ids.add(episode_id)
                continue

            for _, prompt in instruction_dict.items():
                if not isinstance(prompt, str):
                    continue
                if prompt not in self.prompt_to_vec:
                    continue

                metadata.append(
                    {
                        "sample_path": sample_path,
                        "episode_id": episode_id,
                        "description": prompt,
                        "length": self.prompt_to_length[prompt],
                    }
                )
                vectors.append(self.prompt_to_vec[prompt])

        print(
            f"Filtered out {filtered_missing_episode} episode_ids missing directory or "
            f"trajectory.h5 under {self.trajectory_root} (unique: {len(missing_episode_ids)})"
        )

        if not vectors:
            raise ValueError(
                "No overlap between language annotations and cached prompt embeddings."
            )

        return metadata, np.stack(vectors, axis=0)

    @staticmethod
    def _parse_embedding(prompt: str, emb) -> Optional[np.ndarray]:
        """Convert one cached item to np.ndarray with shape (L, 1024)."""
        if isinstance(emb, dict):
            # Support already-processed cache dict entries:
            # {"t5_text_embeddings": Tensor/ndarray, "length": int}
            emb = emb.get("t5_text_embeddings", None)
            if emb is None:
                return None

        if isinstance(emb, torch.Tensor):
            arr = emb.detach().cpu().numpy()
        elif isinstance(emb, list):
            arr = np.asarray(emb)
        else:
            arr = np.asarray(emb)

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Unexpected embedding shape {arr.shape} for prompt={prompt!r}")
        if arr.shape[1] != 1024:
            raise ValueError(
                f"Expected embedding dim 1024, got {arr.shape[1]} for prompt={prompt!r}"
            )
        return arr

    def _init_text_encoder_if_needed(self) -> None:
        if self.text_encoder is not None:
            return
        print("Cache miss for query embedding. Initializing T5 text encoder...")
        self.text_encoder = CosmosT5TextEncoder(
            model_name=self._model_name,
            device=self.device,
            cache_dir=self._model_cache_dir,
            local_files_only=self._local_files_only,
        )

    def _get_cached_query_embedding(self, query: str) -> Optional[np.ndarray]:
        # Try exact match first, then a stripped fallback.
        if query in self.prompt_to_vec:
            return self.prompt_to_vec[query]
        stripped = query.strip()
        if stripped in self.prompt_to_vec:
            return self.prompt_to_vec[stripped]
        return None

    @torch.inference_mode()
    def _encode_query(self, query: str) -> np.ndarray:
        cached_query_vec = self._get_cached_query_embedding(query)
        if cached_query_vec is not None:
            print("Using cached embedding for query (no online T5 encoding).")
            return cached_query_vec.astype(np.float32, copy=False)

        self._init_text_encoder_if_needed()
        encoded, mask = self.text_encoder.encode_prompts(
            query, max_length=self.max_length, return_mask=True
        )
        # encoded: (1, max_len, 1024), mask: (1, max_len)
        encoded = encoded[0].to(dtype=torch.float32)
        mask = mask[0].to(dtype=torch.float32).unsqueeze(-1)

        valid_count = int(mask.sum().item())
        if valid_count <= 0:
            raise ValueError("Query tokenization produced zero valid tokens.")

        vec = (encoded * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1.0)
        vec = torch.nn.functional.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze(0)
        return vec.cpu().numpy().astype(np.float32, copy=False)

    def _load_trajectory(
        self,
        sample_path: str,
        with_action: bool = False,
        downsample_factor: int = 1,
    ) -> Dict:
        """Load trajectory data from trajectory.h5, matching retriever.py format."""
        h5_path = os.path.join(sample_path, "trajectory.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"trajectory.h5 not found for sample_path={sample_path} (expected {h5_path})"
            )

        trajectory: Dict[str, np.ndarray] = {}
        with h5py.File(h5_path, "r") as f:
            trajectory["state_xyzrpy"] = f["observation"]["robot_state"]["cartesian_position"][()]
            trajectory["state_cartesian"] = cartesian_rpy_to_quat(trajectory["state_xyzrpy"])
            trajectory["state_gripper"] = (
                f["observation"]["robot_state"]["gripper_position"][()].reshape(-1, 1)
            )

            if with_action:
                trajectory["action_xyzrpy"] = f["action"]["cartesian_position"][()]
                trajectory["action_cartesian"] = cartesian_rpy_to_quat(
                    trajectory["action_xyzrpy"]
                )
                trajectory["action_gripper"] = f["action"]["gripper_position"][()].reshape(-1, 1)

        if downsample_factor > 1:
            for key in trajectory:
                trajectory[key] = trajectory[key][::downsample_factor]

        return trajectory

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        confidence_threshold: float = -1.0,
        with_action: bool = False,
        downsample_factor: int = 1,
        return_unique_samples: bool = False,
    ) -> List[RetrievalResult]:
        if top_k == 0:
            return []
        if top_k < -1:
            raise ValueError(f"top_k must be -1 or >= 1, got {top_k}")

        query_vec = self._encode_query(query)
        similarities = self.embeddings @ query_vec  # cosine due to l2-normalization

        if confidence_threshold > -1.0:
            valid_idx = np.where(similarities >= confidence_threshold)[0]
            similarities = similarities[valid_idx]
            base_indices = valid_idx
        else:
            base_indices = np.arange(similarities.shape[0])

        if similarities.size == 0:
            return []

        sorted_indices = np.argsort(-similarities)
        if return_unique_samples:
            unique_local_indices: List[int] = []
            seen_samples = set()
            for local_idx in sorted_indices:
                global_idx = int(base_indices[local_idx])
                sample_id = self.metadata[global_idx]["sample_path"]
                if sample_id in seen_samples:
                    continue
                seen_samples.add(sample_id)
                unique_local_indices.append(int(local_idx))
                if top_k != -1 and len(unique_local_indices) >= top_k:
                    break
            order = np.asarray(unique_local_indices, dtype=np.int64)
        else:
            if top_k == -1:
                order = sorted_indices
            else:
                top_k = min(int(top_k), int(similarities.size))
                order = sorted_indices[:top_k]

        results: List[RetrievalResult] = []
        for local_idx in order:
            global_idx = int(base_indices[local_idx])
            meta = self.metadata[global_idx]
            trajectory = self._load_trajectory(
                str(meta["sample_path"]),
                with_action=with_action,
                downsample_factor=downsample_factor,
            )
            results.append(
                RetrievalResult(
                    sample_path=str(meta["sample_path"]),
                    description=str(meta["description"]),
                    similarity=float(similarities[local_idx]),
                    trajectory=trajectory,
                    idx=global_idx,
                )
            )
        return results

    def get_stats(self) -> Dict[str, object]:
        return {
            "text_embedding_path": self.text_embedding_path,
            "language_annotations_path": self.language_annotations_path,
            "trajectory_root": self.trajectory_root,
            "num_descriptions": self.num_descriptions,
            "embedding_dim": self.embedding_dim,
            "max_length": self.max_length,
            "device": self.device,
            "cache_dtype": str(self.cache_dtype),
            "text_encoder_initialized": self.text_encoder is not None,
        }


def _str_to_dtype(dtype_name: str) -> torch.dtype:
    name = str(dtype_name).lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[name]


def interactive_retrieval(retriever: CachedPromptRetriever, args) -> None:
    """Interactive prompt retrieval mode."""
    print("\n" + "=" * 80)
    print("Interactive Prompt Retrieval")
    print("=" * 80)
    print("Enter queries to retrieve similar prompts.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 80 + "\n")

    while True:
        query = input("Query: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not query:
            continue

        results = retriever.retrieve(
            query=query,
            top_k=args.top_k,
            confidence_threshold=args.confidence_threshold,
            with_action=args.with_action,
            downsample_factor=args.downsample_factor,
            return_unique_samples=args.unique_samples,
        )

        if not results:
            print("\nNo results found.\n")
            continue

        print(f"\nTop {len(results)} results:")
        print("-" * 80)
        for rank, result in enumerate(results, start=1):
            print(f"Rank {rank}")
            print(f"  Similarity: {result.similarity:.6f}")
            print(f"  Description: {result.description}")
            print(f"  Sample: {result.sample_path}")
            print(f"  Retrieval idx: {result.idx}")
            print("-" * 80)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve top-K prompts from cached T5 embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--text_embedding_path",
        type=str,
        default=os.path.join(DATA_DIR, "droid_t5_text_embeddings.pkl.gz"),
        help="Path to prompt embedding cache (.pkl or .pkl.gz).",
    )
    parser.add_argument(
        "--language_annotations_path",
        type=str,
        default=os.path.join(DATA_DIR, "droid_meta", "droid_language_annotations.json"),
        help="Path to droid language annotation json mapping episode_id -> prompts.",
    )
    parser.add_argument(
        "--trajectory_root",
        type=str,
        default=os.path.join(DATA_DIR, "trajengine"),
        help="Root directory where each episode is stored at trajectory_root/episode_id.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Text query prompt for single-query mode.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of results. -1 means all.")
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=-1.0,
        help="Optional cosine similarity threshold. -1 disables filtering.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google-t5/t5-11b",
        help="Model name used by CosmosT5TextEncoder fallback loading.",
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=os.path.join(CKPT_DIR, "google-t5", "t5-11b"),
        help="Local directory for tokenizer/model files.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only load model/tokenizer from local files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for query encoding.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token length used for both cached and query embeddings.",
    )
    parser.add_argument(
        "--cache_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32", "half"],
        help="Dtype label for stats/reporting consistency.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode.",
    )
    parser.add_argument(
        "--with_action",
        action="store_true",
        help="Include actions in trajectory output.",
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=1,
        help="Downsample trajectory by this factor.",
    )
    parser.add_argument(
        "--unique_samples",
        action="store_true",
        help="Return only unique sample paths (one per episode).",
    )

    args = parser.parse_args()
    cache_dtype = _str_to_dtype(args.cache_dtype)

    retriever = CachedPromptRetriever(
        text_embedding_path=args.text_embedding_path,
        language_annotations_path=args.language_annotations_path,
        trajectory_root=args.trajectory_root,
        model_name=args.model_name,
        device=args.device,
        model_cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
        max_length=args.max_length,
        cache_dtype=cache_dtype,
    )

    print("\n" + "=" * 80)
    print("Retriever Statistics")
    print("=" * 80)
    for k, v in retriever.get_stats().items():
        print(f"{k}: {v}")

    if args.interactive:
        interactive_retrieval(retriever, args)
        return

    if args.query is None:
        parser.error("--query is required in non-interactive mode")

    print("\n" + "=" * 80)
    print(f"Query: {args.query}")
    print("=" * 80)

    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        confidence_threshold=args.confidence_threshold,
        with_action=args.with_action,
        downsample_factor=args.downsample_factor,
        return_unique_samples=args.unique_samples,
    )

    if not results:
        print("No results found.")
        return

    print(f"Top {len(results)} results:")
    print("-" * 80)
    for rank, result in enumerate(results, start=1):
        print(f"Rank {rank}")
        print(f"  Similarity: {result.similarity:.6f}")
        print(f"  Description: {result.description}")
        print(f"  Sample: {result.sample_path}")
        print(f"  Retrieval idx: {result.idx}")
        print(f"  Trajectory keys: {list(result.trajectory.keys())}")
        print("-" * 80)


if __name__ == "__main__":
    main()
