#!/usr/bin/env python
"""
Multi-node + multi-GPU batch object removal runner for OmniPaint.

Coordination happens via a shared filesystem (lock files + status logs).
Run from the OmniPaint checkout (see data_processing/04_scene_prior/setup_omnipaint.sh):
cd ${OMNIPAINT_DIR} && python main_omnipaint_multi_node.py \
    --raw_dir ${ROBODREAM_DATA_DIR}/first_frames_processed/raw \
    --mask_dir ${ROBODREAM_DATA_DIR}/first_frames_processed/object_masks \
    --output_dir ${ROBODREAM_DATA_DIR}/first_frames_processed/edited \
    --claim-mode chunk --chunk-size 1400 --chunks-per-node 100
"""

import os
import argparse
import json
import queue
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple
import hashlib
import math

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from PIL import Image, ImageOps
from diffusers.pipelines import FluxPipeline

from src.condition import Condition
from src.generate import generate, seed_everything
from src.embedding_loader import load_npz_embeddings

mp.set_start_method("spawn", force=True)

# Keep these in sync with `data_processing/03_object_prior/segment_objects_of_interest_gsam_dataset.py`
_FIRST_FRAMES_ROOT = os.path.join(os.environ.get("ROBODREAM_DATA_DIR", "datasets"), "first_frames_processed")
RAW_DIR = os.path.join(_FIRST_FRAMES_ROOT, "raw")
MASK_DIR = os.path.join(_FIRST_FRAMES_ROOT, "object_masks")
OUTPUT_DIR = os.path.join(_FIRST_FRAMES_ROOT, "edited")
DEBUG_DIR = "visualizaitons_edit"

MAX_LENGTH = 1024
DEFAULT_LORA = "weights/omnipaint_remove.safetensors"
DEFAULT_EMBED_PATH = "demo_assets/embeddings/remove.npz"
DEBUG_FREQ = 80

# Configuration
NUM_GPUS_TO_USE = int(os.environ.get("NUM_GPUS_TO_USE", torch.cuda.device_count()))
TASK_QUEUE_SIZE = int(os.environ.get("TASK_QUEUE_SIZE", 100))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pair_id(raw_path: Path, mask_path: Path) -> str:
    return f"{raw_path.name}__{mask_path.name}"


def _lock_path(lock_dir: Path, pair_id: str) -> Path:
    return lock_dir / f"{pair_id}.lock"


def _try_acquire_lock(lock_file: Path, payload: dict) -> bool:
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        return True
    except FileExistsError:
        return False


def _is_lock_stale(lock_file: Path, timeout_secs: float) -> bool:
    try:
        age = time.time() - lock_file.stat().st_mtime
        return age > timeout_secs
    except FileNotFoundError:
        return False


def _maybe_reclaim_stale_lock(lock_file: Path, timeout_secs: float) -> bool:
    if not lock_file.exists():
        return False
    if not _is_lock_stale(lock_file, timeout_secs):
        return False
    try:
        lock_file.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _touch(path: Path) -> None:
    try:
        os.utime(path, None)
    except Exception:
        pass


def _default_local_lock_path(node_id: str, lora_weights: str, embed_path: str) -> str:
    tmp = os.environ.get("TMPDIR") or "/tmp"
    name = hashlib.sha1(f"{node_id}|{lora_weights}|{embed_path}".encode("utf-8")).hexdigest()[:12]
    return str(Path(tmp) / f"omnipaint_model_init_{name}.lock")


def _stable_pair_hash_int(pair_id: str) -> int:
    return int(hashlib.sha1(pair_id.encode("utf-8")).hexdigest(), 16)


def _chunk_lock_path(chunk_lock_dir: Path, chunk_idx: int) -> Path:
    return chunk_lock_dir / f"chunk_{chunk_idx:05d}.lock"


def _chunk_done_path(chunk_done_dir: Path, chunk_idx: int) -> Path:
    return chunk_done_dir / f"chunk_{chunk_idx:05d}.done"


def _count_outputs_fast(output_dir: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    try:
        n = 0
        for p in output_dir.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                n += 1
        return n
    except Exception:
        return -1


def _count_dir_glob(dir_path: Path, pattern: str, *, exclude_names: set[str] | None = None) -> int:
    exclude_names = exclude_names or set()
    try:
        return sum(1 for p in dir_path.glob(pattern) if p.name not in exclude_names)
    except Exception:
        return -1


def _list_images(directory: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for name in os.listdir(directory):
        _, ext = os.path.splitext(name)
        if ext.lower() in exts:
            files.append(name)
    return files


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _resize_with_aspect_ratio(img: Image.Image, max_length: int = MAX_LENGTH) -> Image.Image:
    w, h = img.size
    if max(w, h) > max_length:
        scale = max_length / max(w, h)
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def _get_device(pipe: FluxPipeline):
    return getattr(
        pipe,
        "_execution_device",
        getattr(pipe, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    )


def _load_model(device: str, lora_weights: str) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    ).to(device)
    pipe.load_lora_weights(lora_weights, adapter_name="removal")
    pipe.set_adapters(["removal"])
    return pipe


def _load_static_embeddings(pipe: FluxPipeline, embed_path: str):
    device = _get_device(pipe)
    dtype = pipe.transformer.dtype
    return load_npz_embeddings(embed_path, device=device, dtype=dtype)


def _build_pairs(raw_dir: str, mask_dir: str) -> List[Tuple[str, str, str]]:
    raw_files = _list_images(raw_dir)
    mask_files = _list_images(mask_dir)
    raw_map = {_stem(name): name for name in raw_files}
    mask_map = {_stem(name): name for name in mask_files}

    pairs = []
    for stem in tqdm(sorted(set(raw_map.keys()) & set(mask_map.keys())), desc="Building pairs"):
        raw_path = os.path.join(raw_dir, raw_map[stem])
        mask_path = os.path.join(mask_dir, mask_map[stem])
        output_name = raw_map[stem]
        pairs.append((raw_path, mask_path, output_name))
    return pairs


def _build_composite(raw_img: Image.Image, mask_img: Image.Image) -> Image.Image:
    mask = mask_img.convert("L")
    inverted_mask = ImageOps.invert(mask)
    black_background = Image.new("RGB", raw_img.size, (0, 0, 0))
    composite_img = Image.composite(raw_img, black_background, inverted_mask)
    return composite_img


def _save_debug_strip(raw_img: Image.Image, mask_img: Image.Image, output_img: Image.Image, out_path: str) -> None:
    images = [
        raw_img.convert("RGB"),
        mask_img.convert("RGB"),
        output_img.convert("RGB"),
    ]
    total_w = sum(img.width for img in images)
    target_h = images[0].height
    strip = Image.new("RGB", (total_w, target_h))
    x_offset = 0
    for img in images:
        if img.size != (images[0].width, images[0].height):
            img = img.resize(images[0].size, Image.Resampling.NEAREST)
        strip.paste(img, (x_offset, 0))
        x_offset += img.width
    strip.save(out_path)


def _expand_prompt_embeds(
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prompt_embeds.shape[0] == batch_size:
        return prompt_embeds, pooled_prompt_embeds
    if prompt_embeds.shape[0] == 1:
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1)
        return prompt_embeds, pooled_prompt_embeds
    raise ValueError(
        f"prompt_embeds batch mismatch: got {prompt_embeds.shape[0]} for batch_size={batch_size}"
    )


def _process_batch(
    pipe: FluxPipeline,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    batch_items: List[Tuple[str, str, str]],
    output_dir: str,
    debug_dir: str,
    debug_freq: int,
    seed: int,
    num_steps: int,
    infer_count: int,
) -> int:
    raw_images: List[Image.Image] = []
    mask_images: List[Image.Image] = []
    composite_images: List[Image.Image] = []
    output_paths: List[str] = []

    for raw_path, mask_path, output_name in batch_items:
        raw_img = Image.open(raw_path).convert("RGB")
        raw_img = _resize_with_aspect_ratio(raw_img)
        mask_img = Image.open(mask_path).convert("L").resize(
            raw_img.size, Image.Resampling.LANCZOS
        )
        composite_img = _build_composite(raw_img, mask_img)

        raw_images.append(raw_img)
        mask_images.append(mask_img)
        composite_images.append(composite_img)
        output_paths.append(os.path.join(output_dir, output_name))

    batch_size = len(batch_items)
    prompt_embeds_b, pooled_prompt_embeds_b = _expand_prompt_embeds(
        prompt_embeds, pooled_prompt_embeds, batch_size
    )
    condition = Condition("removal", condition=composite_images)

    seed_everything(seed)
    width = (raw_images[0].size[0] // 8) * 8
    height = (raw_images[0].size[1] // 8) * 8

    result = generate(
        pipe,
        conditions=[condition],
        width=width,
        height=height,
        num_inference_steps=num_steps,
        prompt=None,
        prompt_embeds=prompt_embeds_b,
        pooled_prompt_embeds=pooled_prompt_embeds_b,
        text_ids=text_ids,
    ).images

    for idx, output_img in enumerate(result):
        output_img = output_img.resize(raw_images[idx].size, Image.Resampling.LANCZOS)
        output_img.save(output_paths[idx])
        infer_count += 1
        if debug_freq > 0 and infer_count % debug_freq == 0:
            debug_path = os.path.join(debug_dir, os.path.basename(output_paths[idx]))
            _save_debug_strip(raw_images[idx], mask_images[idx], output_img, debug_path)
    return infer_count


class OmniPaintGPUWorker:
    def __init__(
        self,
        gpu_id: int,
        task_queue,
        result_queue,
        stop_event,
        *,
        lora_weights: str,
        embed_path: str,
        debug_dir: str,
        debug_freq: int,
        batch_size: int,
        num_steps: int,
        model_init_stagger_secs: float = 0.0,
        model_init_lock_path: str | None = None,
        ready_queue=None,
    ):
        self.gpu_id = int(gpu_id)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.lora_weights = lora_weights
        self.embed_path = embed_path
        self.debug_dir = debug_dir
        self.debug_freq = int(debug_freq)
        self.batch_size = int(batch_size)
        self.num_steps = int(num_steps)
        self.model_init_stagger_secs = float(model_init_stagger_secs or 0.0)
        self.model_init_lock_path = model_init_lock_path
        self.ready_queue = ready_queue

        self.pipe = None
        self.prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.text_ids = None
        self.infer_count = 0

    def _init_model(self):
        if self.model_init_stagger_secs > 0:
            time.sleep(self.gpu_id * self.model_init_stagger_secs)
        if self.model_init_lock_path:
            try:
                fd = os.open(self.model_init_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                # Wait for another worker to finish initialization.
                while os.path.exists(self.model_init_lock_path):
                    time.sleep(1.0)

        device = f"cuda:{self.gpu_id}"
        self.pipe = _load_model(device, self.lora_weights)
        self.prompt_embeds, self.pooled_prompt_embeds, self.text_ids = _load_static_embeddings(
            self.pipe, self.embed_path
        )

        if self.model_init_lock_path and os.path.exists(self.model_init_lock_path):
            try:
                os.unlink(self.model_init_lock_path)
            except Exception:
                pass

        if self.ready_queue is not None:
            try:
                self.ready_queue.put({"event": "model_ready", "gpu_id": self.gpu_id})
            except Exception:
                pass

    def run(self):
        torch.cuda.set_device(self.gpu_id)
        self._init_model()

        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1)
            except queue.Empty:
                continue

            if task is None:
                break

            task_id = task.get("task_id")
            batch_items = task.get("batch_items") or []
            output_dir = task.get("output_dir")
            seed = int(task.get("seed", 0))

            try:
                self.infer_count = _process_batch(
                    pipe=self.pipe,
                    prompt_embeds=self.prompt_embeds,
                    pooled_prompt_embeds=self.pooled_prompt_embeds,
                    text_ids=self.text_ids,
                    batch_items=batch_items,
                    output_dir=output_dir,
                    debug_dir=self.debug_dir,
                    debug_freq=self.debug_freq,
                    seed=seed,
                    num_steps=self.num_steps,
                    infer_count=self.infer_count,
                )
                self.result_queue.put(
                    {
                        "task_id": task_id,
                        "success": True,
                        "num_images": len(batch_items),
                        "gpu_id": self.gpu_id,
                    }
                )
            except Exception as exc:
                self.result_queue.put(
                    {
                        "task_id": task_id,
                        "success": False,
                        "error": str(exc),
                        "num_images": len(batch_items),
                        "gpu_id": self.gpu_id,
                    }
                )


def _omnipaint_worker_process(
    gpu_id,
    task_queue,
    result_queue,
    stop_event,
    *,
    lora_weights: str,
    embed_path: str,
    debug_dir: str,
    debug_freq: int,
    batch_size: int,
    num_steps: int,
    model_init_stagger_secs: float = 0.0,
    model_init_lock_path: str | None = None,
    ready_queue=None,
):
    worker = OmniPaintGPUWorker(
        gpu_id=gpu_id,
        task_queue=task_queue,
        result_queue=result_queue,
        stop_event=stop_event,
        lora_weights=lora_weights,
        embed_path=embed_path,
        debug_dir=debug_dir,
        debug_freq=debug_freq,
        batch_size=batch_size,
        num_steps=num_steps,
        model_init_stagger_secs=model_init_stagger_secs,
        model_init_lock_path=model_init_lock_path,
        ready_queue=ready_queue,
    )
    worker.run()


class MultiGPUOmniPaintManager:
    def __init__(
        self,
        *,
        num_gpus: int,
        task_queue_size: int,
        lora_weights: str,
        embed_path: str,
        debug_dir: str,
        debug_freq: int,
        batch_size: int,
        num_steps: int,
        model_init_stagger_secs: float = 0.0,
        model_init_lock_path: str | None = None,
        wait_for_model_init: bool = True,
        model_init_timeout_secs: float = 1800.0,
    ):
        self.num_gpus = int(num_gpus)
        self.task_queue = mp.Queue(maxsize=task_queue_size)
        self.result_queue = mp.Queue()
        self.stop_event = mp.Event()
        self.worker_processes = []
        self.task_counter = 0
        self.ready_queue = mp.Queue()

        self.lora_weights = lora_weights
        self.embed_path = embed_path
        self.debug_dir = debug_dir
        self.debug_freq = int(debug_freq)
        self.batch_size = int(batch_size)
        self.num_steps = int(num_steps)
        self.model_init_stagger_secs = float(model_init_stagger_secs or 0.0)
        self.model_init_lock_path = model_init_lock_path
        self.wait_for_model_init = bool(wait_for_model_init)
        self.model_init_timeout_secs = float(model_init_timeout_secs or 0.0)

        print(
            f"Initializing Multi-GPU OmniPaint Manager with {self.num_gpus} GPUs, queue size {task_queue_size}"
        )

    def start_workers(self):
        for gpu_id in range(self.num_gpus):
            process = mp.Process(
                target=_omnipaint_worker_process,
                args=(
                    gpu_id,
                    self.task_queue,
                    self.result_queue,
                    self.stop_event,
                ),
                kwargs={
                    "lora_weights": self.lora_weights,
                    "embed_path": self.embed_path,
                    "debug_dir": self.debug_dir,
                    "debug_freq": self.debug_freq,
                    "batch_size": self.batch_size,
                    "num_steps": self.num_steps,
                    "model_init_stagger_secs": self.model_init_stagger_secs,
                    "model_init_lock_path": self.model_init_lock_path,
                    "ready_queue": self.ready_queue,
                },
            )
            process.start()
            self.worker_processes.append(process)

        print(f"All {self.num_gpus} GPU workers have started")

        if self.wait_for_model_init:
            ready = 0
            deadline = time.time() + (self.model_init_timeout_secs if self.model_init_timeout_secs > 0 else 0.0)
            while ready < self.num_gpus:
                if self.model_init_timeout_secs > 0 and time.time() > deadline:
                    print(
                        f"[warn] Timed out waiting for model init: ready={ready}/{self.num_gpus} "
                        f"(timeout={self.model_init_timeout_secs}s). Continuing."
                    )
                    break
                try:
                    msg = self.ready_queue.get(timeout=5)
                    if isinstance(msg, dict) and msg.get("event") == "model_ready":
                        ready += 1
                except queue.Empty:
                    continue

    def submit_task_async(self, batch_items: List[Tuple[str, str, str]], output_dir: str, seed: int) -> str:
        task_id = f"task_{self.task_counter}_{time.time()}"
        self.task_counter += 1
        task = {
            "task_id": task_id,
            "batch_items": batch_items,
            "output_dir": output_dir,
            "seed": int(seed),
        }
        self.task_queue.put(task)
        return task_id

    def stop(self):
        print("Stopping Multi-GPU OmniPaint Manager...")
        self.stop_event.set()

        for _ in range(self.num_gpus):
            try:
                self.task_queue.put(None, timeout=1)
            except queue.Full:
                pass

        for process in self.worker_processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        print("Multi-GPU OmniPaint Manager has stopped")


def process_pairs_multi_node(
    raw_dir: str,
    mask_dir: str,
    output_dir: str,
    debug_dir: str,
    *,
    debug_freq: int = DEBUG_FREQ,
    seed: int = 42,
    num_steps: int = 28,
    batch_size: int = 4,
    skip_existing: bool = True,
    retry_failed: bool = False,
    node_id: str = "node",
    lock_timeout_secs: float = 3600.0,
    lock_heartbeat_secs: float = 60.0,
    rescan_interval_secs: float = 60.0,
    model_init_stagger_secs: float = 0.0,
    serialize_model_init: bool = False,
    model_init_lock_path: str | None = None,
    lora_weights: str = DEFAULT_LORA,
    embed_path: str = DEFAULT_EMBED_PATH,
    claim_mode: str = "chunk",
    chunk_size: int = 1400,
    chunks_per_node: int = 1,
    chunk_lock_timeout_secs: float = 1200.0,
    exit_after_secs: float = 0.0,
    exit_grace_secs: float = 60.0,
):
    pairs_all = _build_pairs(raw_dir, mask_dir)
    global_total_pairs = len(pairs_all)
    print(f"Found {global_total_pairs} raw/mask pairs (global dataset)")
    print(
        "Run config:"
        f" node_id={node_id}"
        f" output_dir={output_dir}"
        f" claim_mode={claim_mode}"
        f" chunk_size={chunk_size}"
        f" chunks_per_node={chunks_per_node}"
        f" skip_existing={skip_existing}"
        f" retry_failed={retry_failed}"
        f" lock_timeout_secs={lock_timeout_secs}"
        f" chunk_lock_timeout_secs={chunk_lock_timeout_secs}"
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)

    lock_dir = _safe_mkdir(output_path / "_locks")
    chunk_lock_dir = _safe_mkdir(output_path / "_chunk_locks")
    chunk_done_dir = _safe_mkdir(output_path / "_chunk_done")
    status_dir = _safe_mkdir(output_path / "_status")
    failed_dir = _safe_mkdir(output_path / "_failed_markers")

    status_jsonl = status_dir / "status.jsonl"
    failures_jsonl = status_dir / "failures.jsonl"

    processed = 0
    failed = 0
    skipped_existing_count = 0
    skipped_failed_count = 0
    skipped_locked_count = 0
    reclaimed_stale_locks = 0
    reclaimed_stale_chunk_locks = 0

    inflight_limit = max(1, min(TASK_QUEUE_SIZE, NUM_GPUS_TO_USE * 4))
    last_rescan = 0.0

    def _should_skip(raw_path: Path, mask_path: Path, output_name: str) -> tuple[bool, str]:
        out_file = output_path / output_name
        if skip_existing and out_file.exists():
            return True, "existing_output"
        marker = failed_dir / f"{_pair_id(raw_path, mask_path)}.json"
        if (not retry_failed) and marker.exists():
            return True, "failed_marker"
        return False, ""

    if claim_mode not in {"chunk", "per_pair"}:
        raise ValueError(f"Unknown claim_mode: {claim_mode}")
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be >0, got {chunk_size}")
    if int(chunks_per_node) <= 0:
        raise ValueError(f"chunks_per_node must be >0, got {chunks_per_node}")

    total_chunks = int(math.ceil(global_total_pairs / float(chunk_size)))
    print(f"Total chunks: {total_chunks}")
    if total_chunks <= 0:
        print("Nothing to do (no pairs).")
        return

    def _claim_one_chunk() -> tuple[int, Path] | None:
        nonlocal reclaimed_stale_chunk_locks
        start_idx = _stable_pair_hash_int(node_id) % total_chunks
        for off in range(total_chunks):
            idx = int((start_idx + off) % total_chunks)
            done_path = _chunk_done_path(chunk_done_dir, idx)
            if done_path.exists():
                continue
            lf = _chunk_lock_path(chunk_lock_dir, idx)
            if lf.exists():
                if _maybe_reclaim_stale_lock(lf, float(chunk_lock_timeout_secs)):
                    reclaimed_stale_chunk_locks += 1
                else:
                    continue
            payload = {
                "event": "chunk_claimed",
                "time": _utc_now_iso(),
                "chunk_idx": idx,
                "chunk_size": int(chunk_size),
                "node_id": node_id,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
            }
            if _try_acquire_lock(lf, payload):
                _touch(lf)
                return idx, lf
        return None

    print("Initializing GPU manager (local)...")
    if serialize_model_init and not model_init_lock_path:
        model_init_lock_path = _default_local_lock_path(
            node_id=node_id, lora_weights=lora_weights, embed_path=embed_path
        )

    gpu_manager = MultiGPUOmniPaintManager(
        num_gpus=NUM_GPUS_TO_USE,
        task_queue_size=TASK_QUEUE_SIZE,
        lora_weights=lora_weights,
        embed_path=embed_path,
        debug_dir=str(debug_path),
        debug_freq=int(debug_freq),
        batch_size=int(batch_size),
        num_steps=int(num_steps),
        model_init_stagger_secs=float(model_init_stagger_secs or 0.0),
        model_init_lock_path=(model_init_lock_path if serialize_model_init else None),
        wait_for_model_init=True,
    )
    gpu_manager.start_workers()

    task_meta: dict[str, dict] = {}

    try:
        chunks_done_local = 0

        while chunks_done_local < int(chunks_per_node):
            claimed_chunk = None
            if claim_mode == "chunk":
                claimed_chunk = _claim_one_chunk()
                if claimed_chunk is None:
                    print("No available chunks to claim (all done or locked).")
                    break
                chunk_idx, chunk_lock_file = claimed_chunk
                chunk_start = int(chunk_idx * int(chunk_size))
                chunk_end = int(min(global_total_pairs, (chunk_idx + 1) * int(chunk_size)))
                chunk_pairs_all = pairs_all[chunk_start:chunk_end]
                _append_jsonl(
                    status_jsonl,
                    {
                        "event": "chunk_claimed",
                        "time": _utc_now_iso(),
                        "node_id": node_id,
                        "chunk_idx": chunk_idx,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                        "chunk_size": int(chunk_size),
                    },
                )
            else:
                chunk_idx = None
                chunk_lock_file = None
                chunk_pairs_all = pairs_all

            print(f"Building worklist for chunk {chunk_idx} (pairs={len(chunk_pairs_all)})...")
            work_pairs: List[Tuple[str, str, str]] = []
            for raw_path_s, mask_path_s, output_name in chunk_pairs_all:
                raw_path = Path(raw_path_s)
                mask_path = Path(mask_path_s)
                skip, _ = _should_skip(raw_path, mask_path, output_name)
                if skip:
                    continue
                work_pairs.append((raw_path_s, mask_path_s, output_name))

            if not work_pairs:
                if claim_mode == "chunk" and chunk_idx is not None:
                    try:
                        _chunk_done_path(chunk_done_dir, int(chunk_idx)).write_text(_utc_now_iso())
                    except Exception:
                        pass
                    if chunk_lock_file is not None:
                        try:
                            chunk_lock_file.unlink(missing_ok=True)
                        except Exception:
                            pass
                    chunks_done_local += 1
                    print(f"Chunk {chunk_idx} has no remaining work; marked done.")
                    continue
                else:
                    print("Nothing to do (everything done or marked failed).")
                    break

            deadline_ts = (time.time() + float(exit_after_secs)) if float(exit_after_secs) > 0 else None
            last_chunk_hb = 0.0

            with tqdm(total=len(work_pairs), desc=f"Editing chunk {chunk_idx} (local)", unit="image") as pbar:
                submit_idx = 0
                stop_submitting = False
                while True:
                    now = time.time()

                    if now - last_rescan >= float(rescan_interval_secs):
                        last_rescan = now
                        if claim_mode == "chunk":
                            global_in_progress = _count_dir_glob(chunk_lock_dir, "*.lock")
                            global_done_chunks = _count_dir_glob(chunk_done_dir, "*.done")
                        else:
                            global_in_progress = _count_dir_glob(lock_dir, "*.lock")
                            global_done_chunks = -1
                        global_failed = _count_dir_glob(failed_dir, "*.json")
                        global_done = _count_outputs_fast(output_path)
                        global_remaining = (
                            (global_total_pairs - global_done - global_failed)
                            if (global_done >= 0 and global_failed >= 0)
                            else -1
                        )
                        _append_jsonl(
                            status_jsonl,
                            {
                                "event": "global_rescan",
                                "time": _utc_now_iso(),
                                "node_id": node_id,
                                "global_total_pairs": global_total_pairs,
                                "global_done_outputs": global_done,
                                "global_failed_markers": global_failed,
                                "global_in_progress_locks": global_in_progress,
                                "global_done_chunks": global_done_chunks,
                                "global_remaining_est": global_remaining,
                                "processed_local": processed,
                                "failed_local": failed,
                                "inflight_local": len(task_meta),
                                "skipped_existing_local": skipped_existing_count,
                                "skipped_failed_marker_local": skipped_failed_count,
                                "skipped_locked_local": skipped_locked_count,
                                "reclaimed_stale_locks_local": reclaimed_stale_locks,
                                "reclaimed_stale_chunk_locks_local": reclaimed_stale_chunk_locks,
                            },
                        )

                    if (deadline_ts is not None) and (not stop_submitting):
                        if now >= (deadline_ts - float(exit_grace_secs)):
                            stop_submitting = True

                    if claim_mode == "chunk" and chunk_lock_file is not None and float(lock_heartbeat_secs) > 0:
                        if now - last_chunk_hb >= float(lock_heartbeat_secs):
                            _touch(chunk_lock_file)
                            last_chunk_hb = now

                    if (claim_mode == "per_pair") and float(lock_heartbeat_secs) > 0 and task_meta:
                        for _tid, m in task_meta.items():
                            last_hb = float(m.get("last_heartbeat_ts", 0.0))
                            if now - last_hb >= float(lock_heartbeat_secs):
                                lf_s = (m.get("lock_file") or "").strip()
                                if lf_s:
                                    _touch(Path(lf_s))
                                    m["last_heartbeat_ts"] = now

                    if len(task_meta) < inflight_limit:
                        while (not stop_submitting) and (submit_idx < len(work_pairs)) and (
                            len(task_meta) < inflight_limit
                        ):
                            batch_items = work_pairs[submit_idx : submit_idx + int(batch_size)]
                            submit_idx += int(batch_size)

                            batch_filtered: List[Tuple[str, str, str]] = []
                            for raw_path_s, mask_path_s, output_name in batch_items:
                                raw_path = Path(raw_path_s)
                                mask_path = Path(mask_path_s)
                                skip, reason = _should_skip(raw_path, mask_path, output_name)
                                if skip:
                                    if reason == "existing_output":
                                        skipped_existing_count += 1
                                    elif reason == "failed_marker":
                                        skipped_failed_count += 1
                                    continue
                                batch_filtered.append((raw_path_s, mask_path_s, output_name))

                            if not batch_filtered:
                                continue

                            if claim_mode == "per_pair":
                                for raw_path_s, mask_path_s, output_name in batch_filtered:
                                    pair_id = _pair_id(Path(raw_path_s), Path(mask_path_s))
                                    lock_file = _lock_path(lock_dir, pair_id)
                                    if lock_file.exists():
                                        if _maybe_reclaim_stale_lock(lock_file, lock_timeout_secs):
                                            reclaimed_stale_locks += 1
                                        else:
                                            skipped_locked_count += 1
                                            continue
                                    claim_payload = {
                                        "event": "claimed",
                                        "time": _utc_now_iso(),
                                        "pair_id": pair_id,
                                        "raw_path": str(raw_path_s),
                                        "mask_path": str(mask_path_s),
                                        "node_id": node_id,
                                        "hostname": socket.gethostname(),
                                        "pid": os.getpid(),
                                        "chunk_idx": chunk_idx,
                                    }
                                    if not _try_acquire_lock(lock_file, claim_payload):
                                        skipped_locked_count += 1
                                        continue
                                    _touch(lock_file)
                                    task_id = gpu_manager.submit_task_async(
                                        batch_items=[(raw_path_s, mask_path_s, output_name)],
                                        output_dir=str(output_path),
                                        seed=int(seed) + (_stable_pair_hash_int(pair_id) % 10_000_000),
                                    )
                                    task_meta[task_id] = {
                                        "batch_items": [(raw_path_s, mask_path_s, output_name)],
                                        "lock_file": str(lock_file),
                                        "last_heartbeat_ts": time.time(),
                                        "chunk_idx": chunk_idx,
                                    }
                                    _append_jsonl(
                                        status_jsonl,
                                        {**claim_payload, "task_id": task_id},
                                    )
                                continue

                            # Chunk claim mode: submit batch to local GPU workers.
                            seed_offset = _stable_pair_hash_int(batch_filtered[0][2]) % 10_000_000
                            task_id = gpu_manager.submit_task_async(
                                batch_items=batch_filtered,
                                output_dir=str(output_path),
                                seed=int(seed) + int(seed_offset),
                            )
                            task_meta[task_id] = {
                                "batch_items": batch_filtered,
                                "lock_file": None,
                                "last_heartbeat_ts": time.time(),
                                "chunk_idx": chunk_idx,
                            }
                            _append_jsonl(
                                status_jsonl,
                                {
                                    "event": "claimed_in_chunk",
                                    "time": _utc_now_iso(),
                                    "node_id": node_id,
                                    "task_id": task_id,
                                    "chunk_idx": chunk_idx,
                                    "batch_size": len(batch_filtered),
                                },
                            )

                    got = False
                    try:
                        result = gpu_manager.result_queue.get(timeout=5)
                        got = True
                    except queue.Empty:
                        got = False

                    if got:
                        task_id = result.get("task_id")
                        meta = task_meta.pop(task_id, None) or {}
                        batch_items = meta.get("batch_items") or []
                        lock_file = Path(meta["lock_file"]) if meta.get("lock_file") else None

                        if result.get("success"):
                            processed += int(result.get("num_images") or len(batch_items))
                            _append_jsonl(
                                status_jsonl,
                                {
                                    "event": "completed",
                                    "time": _utc_now_iso(),
                                    "node_id": node_id,
                                    "task_id": task_id,
                                    "batch_items": batch_items,
                                    "gpu_id": result.get("gpu_id"),
                                    "chunk_idx": meta.get("chunk_idx"),
                                },
                            )
                        else:
                            failed += int(result.get("num_images") or len(batch_items))
                            err = (result.get("error") or "Unknown error")
                            failure_record = {
                                "event": "failed",
                                "time": _utc_now_iso(),
                                "node_id": node_id,
                                "task_id": task_id,
                                "batch_items": batch_items,
                                "error": err,
                                "gpu_id": result.get("gpu_id"),
                                "chunk_idx": meta.get("chunk_idx"),
                            }
                            _append_jsonl(failures_jsonl, failure_record)
                            _append_jsonl(status_jsonl, failure_record)

                            for raw_path_s, mask_path_s, _output_name in batch_items:
                                pair_id = _pair_id(Path(raw_path_s), Path(mask_path_s))
                                marker = failed_dir / f"{pair_id}.json"
                                try:
                                    with open(marker, "w") as f:
                                        json.dump(failure_record, f, indent=2)
                                except Exception:
                                    pass

                        pbar.update(int(result.get("num_images") or len(batch_items)))

                        if lock_file is not None:
                            try:
                                lock_file.unlink(missing_ok=True)
                            except Exception:
                                pass

                    if len(task_meta) == 0:
                        if stop_submitting:
                            break
                        if submit_idx >= len(work_pairs):
                            break
                        time.sleep(1.0)

            completed_whole_chunk = (submit_idx >= len(work_pairs)) and (not stop_submitting)
            if claim_mode == "chunk" and chunk_idx is not None:
                if completed_whole_chunk:
                    try:
                        _chunk_done_path(chunk_done_dir, int(chunk_idx)).write_text(_utc_now_iso())
                    except Exception:
                        pass
                    _append_jsonl(
                        status_jsonl,
                        {"event": "chunk_done", "time": _utc_now_iso(), "node_id": node_id, "chunk_idx": chunk_idx},
                    )
                else:
                    _append_jsonl(
                        status_jsonl,
                        {
                            "event": "chunk_partial_exit",
                            "time": _utc_now_iso(),
                            "node_id": node_id,
                            "chunk_idx": chunk_idx,
                        },
                    )
                if chunk_lock_file is not None:
                    try:
                        chunk_lock_file.unlink(missing_ok=True)
                    except Exception:
                        pass

            chunks_done_local += 1
            if chunks_done_local >= int(chunks_per_node):
                break

        print("\nMulti-node run finished (local).")
        print(f"  ✓ Processed (local): {processed}")
        print(f"  ✗ Failed (local): {failed}")
        print(f"  Chunks completed/claimed (local): {chunks_done_local}")
        print(f"  Skipped existing (local): {skipped_existing_count}")
        print(f"  Skipped failed markers (local): {skipped_failed_count}")
        print(f"  Skipped locked (local): {skipped_locked_count}")
        print(f"  Reclaimed stale pair locks (local): {reclaimed_stale_locks}")
        print(f"  Reclaimed stale chunk locks (local): {reclaimed_stale_chunk_locks}")

    finally:
        gpu_manager.stop()


def main():
    parser = argparse.ArgumentParser(description="OmniPaint multi-node batch object removal")
    parser.add_argument("--raw_dir", type=str, default=RAW_DIR, help="Input images directory")
    parser.add_argument("--mask_dir", type=str, default=MASK_DIR, help="Mask images directory")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--debug_dir", type=str, default=DEBUG_DIR, help="Debug visualization directory")
    parser.add_argument("--debug_freq", type=int, default=DEBUG_FREQ, help="Save debug strip every N images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--lora_weights", type=str, default=DEFAULT_LORA, help="Path to LoRA weights")
    parser.add_argument("--embed_path", type=str, default=DEFAULT_EMBED_PATH, help="Path to NPZ embeddings")

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do NOT skip images whose outputs already exist in output_dir (default: skip existing).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry pairs with failure markers (default: skip failed pairs).",
    )
    parser.add_argument(
        "--lock-timeout-secs",
        type=float,
        default=600.0,
        help="Reclaim locks older than this many seconds (handles node crashes). Default: 600.",
    )
    parser.add_argument(
        "--lock-heartbeat-secs",
        type=float,
        default=60.0,
        help="How often to touch lock mtimes for in-flight tasks. Default: 60.",
    )
    parser.add_argument(
        "--node-id",
        type=str,
        default=None,
        help="Node identifier for logs/locks. Default: <hostname>:<pid>.",
    )
    parser.add_argument(
        "--rescan-interval-secs",
        type=float,
        default=60.0,
        help="How often to log global lock/failed counts. Default: 60.",
    )
    parser.add_argument(
        "--model-init-stagger-secs",
        type=float,
        default=0.0,
        help="Stagger model initialization per GPU worker by (gpu_id * stagger) seconds. Default: 0.",
    )
    parser.add_argument(
        "--serialize-model-init",
        action="store_true",
        help="Serialize model initialization within a node (one worker loads ckpt at a time).",
    )
    parser.add_argument(
        "--model-init-lock-path",
        type=str,
        default=None,
        help="Optional explicit lock path for --serialize-model-init.",
    )

    parser.add_argument(
        "--claim-mode",
        type=str,
        default="chunk",
        choices=["chunk", "per_pair"],
        help='Work assignment strategy. "chunk" claims one fixed-size chunk lock per node; "per_pair" uses per-pair lock files (legacy, heavier I/O).',
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1400,
        help="Number of image pairs per claimed chunk. Default: 1400.",
    )
    parser.add_argument(
        "--chunks-per-node",
        type=int,
        default=1,
        help="How many chunks this process should claim+run before exiting. Default: 1.",
    )
    parser.add_argument(
        "--chunk-lock-timeout-secs",
        type=float,
        default=1200.0,
        help="Reclaim chunk locks older than this many seconds. Default: 1200.",
    )
    parser.add_argument(
        "--exit-after-secs",
        type=float,
        default=0.0,
        help="If >0, stop submitting new work near this walltime and drain in-flight tasks, then exit. Default: 0.",
    )
    parser.add_argument(
        "--exit-grace-secs",
        type=float,
        default=60.0,
        help="Grace period before --exit-after-secs during which we stop submitting new work and drain. Default: 60.",
    )

    args = parser.parse_args()
    if args.node_id is None:
        args.node_id = f"{socket.gethostname()}:{os.getpid()}"

    print(
        "Parsed args:"
        f" raw_dir={args.raw_dir}"
        f" mask_dir={args.mask_dir}"
        f" output_dir={args.output_dir}"
        f" node_id={args.node_id}"
        f" claim_mode={args.claim_mode}"
        f" chunk_size={args.chunk_size}"
        f" chunks_per_node={args.chunks_per_node}"
        f" skip_existing={not args.no_skip_existing}"
        f" retry_failed={args.retry_failed}"
    )

    process_pairs_multi_node(
        raw_dir=args.raw_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        debug_dir=args.debug_dir,
        debug_freq=args.debug_freq,
        seed=args.seed,
        num_steps=args.steps,
        batch_size=args.batch_size,
        skip_existing=(not args.no_skip_existing),
        retry_failed=args.retry_failed,
        node_id=args.node_id,
        lock_timeout_secs=args.lock_timeout_secs,
        lock_heartbeat_secs=args.lock_heartbeat_secs,
        rescan_interval_secs=args.rescan_interval_secs,
        model_init_stagger_secs=args.model_init_stagger_secs,
        serialize_model_init=args.serialize_model_init,
        model_init_lock_path=args.model_init_lock_path,
        lora_weights=args.lora_weights,
        embed_path=args.embed_path,
        claim_mode=args.claim_mode,
        chunk_size=args.chunk_size,
        chunks_per_node=args.chunks_per_node,
        chunk_lock_timeout_secs=args.chunk_lock_timeout_secs,
        exit_after_secs=args.exit_after_secs,
        exit_grace_secs=args.exit_grace_secs,
    )


if __name__ == "__main__":
    main()
