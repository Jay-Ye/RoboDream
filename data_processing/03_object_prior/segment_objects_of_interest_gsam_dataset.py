import argparse
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pickle

import cv2
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline


@dataclass
class Config:
    cam_names: List[str]
    debug: bool
    debug_dir: str
    debug_save_freq: int
    episode_meta_file: str
    processed_data_path: str
    first_frame_data_path: str
    local_data_path: str
    prompt_file: str
    select_best: bool
    negative_labels: List[str]
    negative_overlap_threshold: float
    duplicate_iou_threshold: float
    gsam_threshold: float
    detector_id: str
    segmenter_id: str


@dataclass
class EpisodeItem:
    episode_id: str
    episode_meta_info: Dict[str, Any]
    objects_of_interest: List[str]
    index: int


class EpisodeDataset(Dataset):
    def __init__(
        self,
        episode_meta: Dict[str, Any],
        prompt_dict: Dict[str, List[str]],
        episode_filter: Optional[Iterable[str]] = None,
    ) -> None:
        allowed = set(episode_filter) if episode_filter else None
        self._items: List[EpisodeItem] = []
        for idx, (episode_id, episode_meta_info) in enumerate(episode_meta.items()):
            if allowed is not None and episode_id not in allowed:
                continue
            objects = prompt_dict.get(episode_id, [])
            objects = [f"a {obj}." for obj in objects]
            self._items.append(
                EpisodeItem(
                    episode_id=episode_id,
                    episode_meta_info=episode_meta_info,
                    objects_of_interest=objects,
                    index=idx,
                )
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> EpisodeItem:
        return self._items[index]


class ProcessEpisodeDataset(Dataset):
    def __init__(self, base_dataset: EpisodeDataset) -> None:
        self._base = base_dataset

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> List[Tuple[str, str]]:
        item = self._base[index]
        return _process_episode(item)


object_detector = None
sam_segmenter = None
sam_processor = None
worker_gpu_id = None
worker_cuda_device = None
models_loaded = False
CONFIG: Optional[Config] = None


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict[str, Any]) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GSAM segmentation with dataset-style loading.")
    data_dir = os.environ.get("ROBODREAM_DATA_DIR", "datasets")
    processed_dir = os.path.join(data_dir, "processed", "droid_traj2demo", "real")
    parser.add_argument("--episode-meta-file", default=os.path.join(processed_dir, "filtered_episode_meta25.json"))
    parser.add_argument("--processed-data-path", default=processed_dir + "/")
    parser.add_argument("--first-frame-data-path", default=os.path.join(data_dir, "first_frames_processed") + "/")
    parser.add_argument("--local-data-path", default=os.environ.get("DROID_ROOT", "droid") + "/")
    parser.add_argument("--prompt-file", default=os.path.join(data_dir, "object_segmentation_prompts.json"))
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--debug-dir", default="visualizations_gsam_new")
    parser.add_argument("--debug-save-freq", type=int, default=40)
    parser.add_argument("--select-best", action="store_true")
    parser.add_argument("--negative-labels-json", default='["a table.", "a bed.", "a sink.", "a sofa.", "floor.", "a shelf.", "a cabinet.", "countertop."]')
    parser.add_argument("--negative-overlap-threshold", type=float, default=0.9)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.98)
    parser.add_argument("--gsam-threshold", type=float, default=0.3)
    parser.add_argument("--detector-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--segmenter-id", default="facebook/sam-vit-huge")
    parser.add_argument("--cam-names", default="external_cam1,external_cam2")
    parser.add_argument("--episode-filter-json", default="")
    parser.add_argument("--use-dataloader", action="store_true")
    parser.add_argument("--dataloader-workers", type=int, default=0)
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> Tuple[Config, List[str]]:
    negative_labels = json.loads(args.negative_labels_json) if args.negative_labels_json else []
    episode_filter = json.loads(args.episode_filter_json) if args.episode_filter_json else []
    return Config(
        cam_names=[name.strip() for name in args.cam_names.split(",") if name.strip()],
        debug=args.debug,
        debug_dir=args.debug_dir,
        debug_save_freq=args.debug_save_freq,
        episode_meta_file=args.episode_meta_file,
        processed_data_path=args.processed_data_path,
        first_frame_data_path=args.first_frame_data_path,
        local_data_path=args.local_data_path,
        prompt_file=args.prompt_file,
        select_best=args.select_best,
        negative_labels=negative_labels,
        negative_overlap_threshold=args.negative_overlap_threshold,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        gsam_threshold=args.gsam_threshold,
        detector_id=args.detector_id,
        segmenter_id=args.segmenter_id,
    ), episode_filter


def _set_seed() -> None:
    random.seed(42)


def get_first_frame_of_video(video_path: str) -> np.ndarray:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    vr.seek(0)
    return vr[0].asnumpy()


def post_process_cropped_image(cropped_image_list: List[Optional[np.ndarray]], img_size: Tuple[int, int] = (640, 360)) -> np.ndarray:
    width, height = img_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if len(cropped_image_list) == 0:
        return canvas
    valid_images = [img for img in cropped_image_list if img is not None and img.size > 0]
    if len(valid_images) == 0:
        return canvas

    num_images = len(valid_images)
    grid_cols = int(np.ceil(np.sqrt(num_images)))
    grid_rows = int(np.ceil(num_images / grid_cols))
    cell_width = width // grid_cols
    cell_height = height // grid_rows
    target_size = int(min(cell_width, cell_height) * 0.8)

    processed_images = []
    for img in valid_images:
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        angle = random.uniform(-45, 45)
        h_r, w_r = resized_img.shape[:2]
        center = (w_r // 2, h_r // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = abs(rotation_matrix[0, 0]), abs(rotation_matrix[0, 1])
        new_w_rot = int(h_r * sin + w_r * cos)
        new_h_rot = int(h_r * cos + w_r * sin)
        rotation_matrix[0, 2] += (new_w_rot - w_r) / 2
        rotation_matrix[1, 2] += (new_h_rot - h_r) / 2
        rotated_img = cv2.warpAffine(
            resized_img,
            rotation_matrix,
            (new_w_rot, new_h_rot),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        gray = cv2.cvtColor(rotated_img, cv2.COLOR_RGB2GRAY) if len(rotated_img.shape) == 3 else rotated_img
        coords = cv2.findNonZero(gray)
        if coords is not None:
            x, y, w_crop, h_crop = cv2.boundingRect(coords)
            recropped_img = rotated_img[y:y + h_crop, x:x + w_crop]
            processed_images.append(recropped_img)
        else:
            processed_images.append(rotated_img)

    random.shuffle(processed_images)
    for idx, img in enumerate(processed_images):
        row = idx // grid_cols
        col = idx % grid_cols
        cell_center_x = col * cell_width + cell_width // 2
        cell_center_y = row * cell_height + cell_height // 2
        h_img, w_img = img.shape[:2]
        x = max(0, min(width - w_img, cell_center_x - w_img // 2))
        y = max(0, min(height - h_img, cell_center_y - h_img // 2))
        x_end = min(x + w_img, width)
        y_end = min(y + h_img, height)
        img_crop = img[:y_end - y, :x_end - x]
        canvas[y:y_end, x:x_end] = img_crop
    return canvas


def _normalize_label(label: str) -> str:
    return label.strip().rstrip(".").lower()


def _get_boxes(results: List[DetectionResult]) -> List[List[List[float]]]:
    return [[result.box.xyxy for result in results]]


def _refine_masks(masks: torch.BoolTensor) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    return list(masks.numpy().astype(np.uint8))


def _get_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if worker_cuda_device is not None:
        return worker_cuda_device
    if worker_gpu_id is not None:
        return f"cuda:{worker_gpu_id}"
    return "cuda"


def _get_pipeline_device() -> int:
    if not torch.cuda.is_available():
        return -1
    if worker_gpu_id is not None:
        return int(worker_gpu_id)
    return 0


def _detect(image: Image.Image, labels: List[str], threshold: float) -> List[DetectionResult]:
    labels = [label if label.endswith(".") else f"{label}." for label in labels]
    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    return [DetectionResult.from_dict(result) for result in results]


def _segment(image: Image.Image, detection_results: List[DetectionResult]) -> List[DetectionResult]:
    boxes = _get_boxes(detection_results)
    inputs = sam_processor(images=image, input_boxes=boxes, return_tensors="pt").to(_get_device())
    outputs = sam_segmenter(**inputs)
    masks = sam_processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes,
    )[0]
    masks = _refine_masks(masks)
    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask
    return detection_results


def _grounded_segmentation(image: Image.Image, labels: List[str], threshold: float) -> List[DetectionResult]:
    detections = _detect(image, labels, threshold)
    if not detections:
        return []
    return _segment(image, detections)


def _select_best_detections(
    detections: List[DetectionResult],
    labels: List[str],
) -> Dict[str, DetectionResult]:
    best_by_label: Dict[str, DetectionResult] = {}
    for detection in detections:
        key = _normalize_label(detection.label)
        if key not in best_by_label or detection.score > best_by_label[key].score:
            best_by_label[key] = detection
    return {label: best_by_label[_normalize_label(label)] for label in labels if _normalize_label(label) in best_by_label}


def _crop_with_mask(image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    if mask is None:
        return None
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    cropped = image[y:y + h, x:x + w].copy()
    cropped_mask = mask[y:y + h, x:x + w].astype(bool)
    if cropped_mask.ndim == 2:
        cropped[~cropped_mask] = 0
    return cropped


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def _is_mostly_inside(
    mask: np.ndarray,
    mask_score: float,
    negative_detections: List[DetectionResult],
    threshold: float,
) -> bool:
    if mask is None:
        return False
    mask_bool = mask.astype(bool)
    mask_area = mask_bool.sum()
    if mask_area == 0:
        return False
    for neg_det in negative_detections:
        neg_mask = neg_det.mask
        if neg_mask is None:
            continue
        overlap = np.logical_and(mask_bool, neg_mask.astype(bool)).sum()
        if (overlap / mask_area) >= threshold and mask_score < neg_det.score:
            return True
    return False


def _filter_masks_by_negative(
    detections: List[DetectionResult],
    threshold: float,
    negative_detections: List[DetectionResult],
) -> List[DetectionResult]:
    if not negative_detections:
        return detections
    kept: List[DetectionResult] = []
    for detection in detections:
        mask = detection.mask
        if mask is None:
            continue
        if _is_mostly_inside(mask, detection.score, negative_detections, threshold):
            continue
        kept.append(detection)
    return kept


def _dedupe_masks(
    masks: List[Optional[np.ndarray]],
    labels: List[str],
    iou_threshold: float,
) -> Tuple[List[Optional[np.ndarray]], List[str]]:
    kept_masks: List[Optional[np.ndarray]] = []
    kept_labels: List[str] = []
    for mask, label in zip(masks, labels):
        if mask is None:
            continue
        is_duplicate = False
        for kept in kept_masks:
            if _mask_iou(mask, kept) >= iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept_masks.append(mask)
            kept_labels.append(label)
    return kept_masks, kept_labels


def _get_models() -> Tuple[Any, Any, Any]:
    global object_detector, sam_segmenter, sam_processor, models_loaded
    if not models_loaded:
        print("Initializing Grounded-SAM...")
        device = _get_device()
        object_detector = pipeline(
            model=CONFIG.detector_id,
            task="zero-shot-object-detection",
            device=_get_pipeline_device(),
        )
        sam_segmenter = AutoModelForMaskGeneration.from_pretrained(CONFIG.segmenter_id).to(device)
        sam_processor = AutoProcessor.from_pretrained(CONFIG.segmenter_id, use_fast=True)
        models_loaded = True
    return object_detector, sam_segmenter, sam_processor


def _get_visible_gpu_ids() -> List[str]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        return [s.strip() for s in cuda_visible.split(",") if s.strip() != ""]
    try:
        count = torch.cuda.device_count()
        return [str(i) for i in range(count)]
    except Exception:
        return []


def _init_worker(gpu_id_queue, config: Config) -> None:
    global worker_gpu_id, worker_cuda_device, CONFIG
    worker_gpu_id = None
    worker_cuda_device = None
    CONFIG = config
    gpu_id = None
    if hasattr(gpu_id_queue, "get_nowait"):
        try:
            gpu_id = gpu_id_queue.get_nowait()
        except Exception:
            gpu_id = None
    else:
        gpu_id = gpu_id_queue
    if gpu_id is not None:
        try:
            gpu_id_int = int(gpu_id)
        except Exception:
            gpu_id_int = None
        if gpu_id_int is not None:
            try:
                torch.cuda.set_device(gpu_id_int)
            except Exception:
                pass
            worker_gpu_id = gpu_id_int
            worker_cuda_device = f"cuda:{gpu_id_int}"
        else:
            worker_gpu_id = gpu_id
    _get_models()


def _init_dataloader_worker(worker_id: int, config: Config, gpu_ids: List[str]) -> None:
    if gpu_ids:
        gpu_id = gpu_ids[worker_id % len(gpu_ids)]
    else:
        gpu_id = 0
    _init_worker(gpu_id, config)


def _process_episode(item: EpisodeItem) -> List[Tuple[str, str]]:
    local_failed: List[Tuple[str, str]] = []
    episode_id = item.episode_id
    episode_index = item.index
    episode_meta_info = item.episode_meta_info
    objects_of_interest = item.objects_of_interest
    episode_rel_path = episode_meta_info["rel_path"]
    task_descriptions = list(episode_meta_info["task_descriptions"].values())
    task_descriptions = " / ".join(task_descriptions)
    cam_name_to_serial = episode_meta_info["cam_name_to_serial"]

    tgt_dir = os.path.join(CONFIG.processed_data_path, episode_id, "scene_prior_object_gsam")

    try:
        for cam_name in CONFIG.cam_names:
            tgt_cam_dir = os.path.join(tgt_dir, cam_name)
            if not CONFIG.debug:
                os.makedirs(tgt_cam_dir, exist_ok=True)
                if os.path.exists(os.path.join(tgt_cam_dir, "0000000000_gsam.jpg")):
                    print(f"Scene prior for episode {episode_id} and cam {cam_name} already exists")
                    continue

            mp4_path = os.path.join(
                CONFIG.local_data_path,
                episode_rel_path,
                "recordings",
                "MP4",
                f"{cam_name_to_serial[cam_name]}.mp4",
            )
            if not os.path.exists(mp4_path):
                local_failed.append((episode_id, cam_name))
                print(f"Failed to find video for episode {episode_id} and cam {cam_name}")
                continue

            first_frame = get_first_frame_of_video(mp4_path)

            mask_list: List[Optional[np.ndarray]] = []
            mask_labels: List[str] = []
            if len(objects_of_interest) > 0:
                image_pil = Image.fromarray(first_frame)
                all_labels = objects_of_interest + CONFIG.negative_labels
                detections = _grounded_segmentation(
                    image=image_pil,
                    labels=all_labels,
                    threshold=CONFIG.gsam_threshold,
                )
                positive_label_set = {_normalize_label(label) for label in objects_of_interest}
                negative_label_set = {_normalize_label(label) for label in CONFIG.negative_labels}
                positive_detections = []
                negative_detections = []
                for detection in detections:
                    normalized = _normalize_label(detection.label)
                    if normalized in negative_label_set:
                        negative_detections.append(detection)
                    elif normalized in positive_label_set:
                        positive_detections.append(detection)
                for det in positive_detections:
                    if det.mask is not None and det.mask.dtype != np.uint8:
                        det.mask = det.mask.astype(np.uint8)
                for det in negative_detections:
                    if det.mask is not None and det.mask.dtype != np.uint8:
                        det.mask = det.mask.astype(np.uint8)
                if negative_detections:
                    positive_detections = _filter_masks_by_negative(
                        positive_detections,
                        CONFIG.negative_overlap_threshold,
                        negative_detections,
                    )
                if CONFIG.select_best:
                    best_by_label = _select_best_detections(positive_detections, objects_of_interest)
                    for object_of_interest in objects_of_interest:
                        detection = best_by_label.get(object_of_interest)
                        if detection is None:
                            continue
                        mask_list.append(detection.mask)
                        mask_labels.append(object_of_interest)
                else:
                    for detection in positive_detections:
                        mask_list.append(detection.mask)
                        mask_labels.append(detection.label)

                mask_list, mask_labels = _dedupe_masks(
                    mask_list,
                    mask_labels,
                    CONFIG.duplicate_iou_threshold,
                )

            cropped_image_list = [
                _crop_with_mask(first_frame, mask) if mask is not None else None
                for mask in mask_list
            ]
            scene_prior_object_img = post_process_cropped_image(cropped_image_list, img_size=(640, 360))
            if len(scene_prior_object_img.shape) == 3 and scene_prior_object_img.shape[2] == 3:
                scene_prior_object_img_bgr = cv2.cvtColor(scene_prior_object_img, cv2.COLOR_RGB2BGR)
            else:
                scene_prior_object_img_bgr = scene_prior_object_img

            if not CONFIG.debug:
                cv2.imwrite(
                    os.path.join(CONFIG.first_frame_data_path, "raw", f"{episode_id}_{cam_name}.jpg"),
                    cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB),
                )
                if len(mask_list) > 0:
                    merged_mask = np.sum(mask_list, axis=0) > 0
                    merged_mask = merged_mask.astype(np.uint8) * 255
                else:
                    merged_mask = np.zeros(first_frame.shape[:2], dtype=np.uint8)
                cv2.imwrite(
                    os.path.join(CONFIG.first_frame_data_path, "object_masks", f"{episode_id}_{cam_name}.png"),
                    merged_mask,
                )
                cv2.imwrite(os.path.join(tgt_cam_dir, "0000000000_gsam.jpg"), scene_prior_object_img_bgr)
                masks_payload = {
                    "mask_list": mask_list,
                    "mask_labels": mask_labels,
                    "cropped_image_list": cropped_image_list,
                }
                with open(os.path.join(tgt_cam_dir, "masks_payload.pkl"), "wb") as f:
                    pickle.dump(masks_payload, f)

            should_save_debug = CONFIG.debug or (
                not CONFIG.debug and CONFIG.debug_save_freq > 0 and (episode_index % CONFIG.debug_save_freq == 0)
            )
            if should_save_debug:
                if len(mask_list) > 0:
                    joint_mask = np.sum(mask_list, axis=0)
                    joint_mask = joint_mask > 0
                    masked_image = first_frame.copy()
                    masked_image[~joint_mask] = 0
                    target_height, target_width = scene_prior_object_img.shape[:2]
                    first_frame_resized = cv2.resize(first_frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                    masked_image_resized = cv2.resize(masked_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                    concatenated_image = np.concatenate(
                        [first_frame_resized, masked_image_resized, scene_prior_object_img],
                        axis=1,
                    )
                    text_height = 80
                    text_area = np.ones((text_height, concatenated_image.shape[1], 3), dtype=np.uint8) * 255
                    task_text = f"Task: {task_descriptions}"
                    max_chars_per_line = 120
                    if len(task_text) > max_chars_per_line:
                        task_text = task_text[:max_chars_per_line - 3] + "..."
                    cv2.putText(text_area, task_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    objects_text = f"Objects: {', '.join(mask_labels)}"
                    if len(objects_text) > max_chars_per_line:
                        objects_text = objects_text[:max_chars_per_line - 3] + "..."
                    cv2.putText(text_area, objects_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    final_debug_image = np.concatenate([text_area, concatenated_image], axis=0)
                    if len(final_debug_image.shape) == 3 and final_debug_image.shape[2] == 3:
                        final_debug_image_bgr = cv2.cvtColor(final_debug_image, cv2.COLOR_RGB2BGR)
                    else:
                        final_debug_image_bgr = final_debug_image
                    cv2.imwrite(os.path.join(CONFIG.debug_dir, f"{episode_id}_{cam_name}.jpg"), final_debug_image_bgr)
                else:
                    print(f"Warning: No masks found for episode {episode_id} and cam {cam_name}")
    except Exception as e:
        print(f"Error processing episode {episode_id}: {e}")
        local_failed.append((episode_id, str(e)))
    return local_failed


def main() -> None:
    _set_seed()
    args = _parse_args()
    config, episode_filter = _build_config(args)

    if config.debug or config.debug_save_freq > 0:
        os.makedirs(config.debug_dir, exist_ok=True)
    os.makedirs(os.path.join(config.first_frame_data_path, "raw"), exist_ok=True)
    os.makedirs(os.path.join(config.first_frame_data_path, "object_masks"), exist_ok=True)

    with open(config.episode_meta_file, "r") as f:
        episode_meta = json.load(f)
    with open(config.prompt_file, "r") as f:
        prompt_dict = json.load(f)

    dataset = EpisodeDataset(episode_meta, prompt_dict, episode_filter or None)

    print("Starting to segment objects of interest...")
    failed_episodes: List[Tuple[str, str]] = []
    gpu_ids = _get_visible_gpu_ids()
    max_workers = max(1, len(gpu_ids))
    loader: Iterable[EpisodeItem]
    if args.use_dataloader:
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.dataloader_workers,
            collate_fn=lambda batch: batch[0],
        )
    else:
        loader = dataset

    if gpu_ids:
        print(f"Using {len(gpu_ids)} GPUs")
        ctx = mp.get_context("spawn")
        with ctx.Manager() as manager:
            gpu_queue = manager.Queue()
            for gpu_id in gpu_ids:
                gpu_queue.put(gpu_id)
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(gpu_queue, config),
            ) as executor:
                futures = set()
                progress = tqdm(total=len(dataset))
                for item in loader:
                    futures.add(executor.submit(_process_episode, item))
                    if len(futures) >= max_workers:
                        done, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            failed_episodes.extend(future.result())
                            progress.update(1)
                for future in as_completed(futures):
                    failed_episodes.extend(future.result())
                    progress.update(1)
                progress.close()
    else:
        _init_worker(0, config)
        for item in tqdm(loader, total=len(dataset)):
            failed_episodes.extend(_process_episode(item))

    with open(os.path.join(config.processed_data_path, "failed_episodes.txt"), "w") as f:
        for episode_id, cam_name in failed_episodes:
            f.write(f"{episode_id} {cam_name}\n")


if __name__ == "__main__":
    main()
