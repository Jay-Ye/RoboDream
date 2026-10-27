from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline


@dataclass
class GSamScenePriorConfig:
    """
    Configuration for Grounded-SAM (GroundingDINO + SAM) scene-prior generation.
    """

    detector_id: str = "IDEA-Research/grounding-dino-base"
    segmenter_id: str = "facebook/sam-vit-huge"
    gsam_threshold: float = 0.3

    negative_labels: Tuple[str, ...] = (
        "a table.",
        "a bed.",
        "a sink.",
        "a sofa.",
        "floor.",
        "a shelf.",
        "a cabinet.",
        "countertop.",
    )
    negative_overlap_threshold: float = 0.9
    duplicate_iou_threshold: float = 0.98
    select_best: bool = False
    output_size: Tuple[int, int] = (640, 360)
    bg_color: Optional[Tuple[int, int, int]] = None

    # If input image is BGR (OpenCV), set this to True (we convert internally to RGB for PIL/transformers)
    input_is_bgr: bool = False

    # Device selection (single source of truth):
    # - "cpu"
    # - "cuda" (defaults to GPU 0)
    # - "cuda:1" (select GPU 1)
    # If None: auto-detect (cuda if available else cpu).
    device: Optional[str] = None


@dataclass
class ScenePriorResult:
    """
    Result of running Grounded-SAM on an image.
    """

    # Individual binary masks (0/1 uint8), one per kept detection
    mask_list: List[np.ndarray]
    mask_labels: List[str]

    # Collage of cropped masked objects (BGR uint8, suitable for cv2.imwrite)
    scene_prior_object_img_bgr: np.ndarray

    # Union of all masks (0/255 uint8, suitable for cv2.imwrite)
    merged_mask: np.ndarray

    # (Optional) debug payload; can be useful for downstream tasks
    cropped_image_list: List[Optional[np.ndarray]]


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
            score=float(detection_dict["score"]),
            label=str(detection_dict["label"]),
            box=BoundingBox(
                xmin=int(detection_dict["box"]["xmin"]),
                ymin=int(detection_dict["box"]["ymin"]),
                xmax=int(detection_dict["box"]["xmax"]),
                ymax=int(detection_dict["box"]["ymax"]),
            ),
        )


def _normalize_label(label: str) -> str:
    return label.strip().rstrip(".").lower()


def _get_boxes(results: List[DetectionResult]) -> List[List[List[float]]]:
    # SAM processor expects shape [batch, num_boxes, 4]
    return [[result.box.xyxy for result in results]]


def _refine_masks(masks: torch.BoolTensor) -> List[np.ndarray]:
    # Convert to a simple 2D binary mask per detection, matching the original scripts' behavior.
    masks = masks.detach().cpu().float()
    masks = masks.permute(0, 2, 3, 1)  # [N, H, W, C]
    masks = masks.mean(axis=-1)  # [N, H, W]
    masks = (masks > 0).to(torch.uint8)  # [N, H, W] in {0,1}
    return list(masks.numpy().astype(np.uint8))


def _crop_with_mask(image_rgb: np.ndarray, mask01: np.ndarray) -> Optional[np.ndarray]:
    if mask01 is None:
        return None
    coords = cv2.findNonZero(mask01.astype(np.uint8))
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    cropped = image_rgb[y : y + h, x : x + w].copy()
    cropped_mask = mask01[y : y + h, x : x + w].astype(bool)
    if cropped_mask.ndim == 2:
        cropped[~cropped_mask] = 0
    return cropped


def post_process_cropped_image(
    cropped_image_list: Sequence[Optional[np.ndarray]],
    img_size: Tuple[int, int] = (640, 360),
    bg_color: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Arrange a list of cropped RGB images into a grid collage (with random rotations), on a black canvas.
    """

    width, height = img_size
    canvas = np.full((height, width, 3), bg_color if bg_color is not None else (0, 0, 0), dtype=np.uint8)
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

    # Deterministic randomness should be controlled by caller (set random.seed) if desired.
    import random

    processed_images: List[np.ndarray] = []
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

        gray = cv2.cvtColor(rotated_img, cv2.COLOR_RGB2GRAY) if rotated_img.ndim == 3 else rotated_img
        coords = cv2.findNonZero(gray)
        if coords is not None:
            x, y, w_crop, h_crop = cv2.boundingRect(coords)
            recropped_img = rotated_img[y : y + h_crop, x : x + w_crop]
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
        img_crop = img[: y_end - y, : x_end - x]
        canvas[y:y_end, x:x_end] = img_crop

    return canvas


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def _dedupe_masks(
    masks: List[Optional[np.ndarray]],
    labels: List[str],
    iou_threshold: float,
) -> Tuple[List[np.ndarray], List[str]]:
    kept_masks: List[np.ndarray] = []
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


def _is_mostly_inside(
    mask01: np.ndarray,
    mask_score: float,
    negative_detections: List[DetectionResult],
    threshold: float,
) -> bool:
    if mask01 is None:
        return False
    mask_bool = mask01.astype(bool)
    mask_area = int(mask_bool.sum())
    if mask_area == 0:
        return False
    for neg_det in negative_detections:
        neg_mask = neg_det.mask
        if neg_mask is None:
            continue
        overlap = int(np.logical_and(mask_bool, neg_mask.astype(bool)).sum())
        if (overlap / mask_area) >= threshold and (neg_det.score > mask_score or neg_det.score > 0.6):
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


class GroundedSamScenePrior:
    def __init__(self, config: Optional[GSamScenePriorConfig] = None) -> None:
        self.config = config or GSamScenePriorConfig()
        self._object_detector = None
        self._sam_segmenter = None
        self._sam_processor = None

    def _get_torch_device(self) -> str:
        """
        Returns a torch device string, e.g. "cpu", "cuda", "cuda:1".
        """
        if self.config.device is not None:
            return str(self.config.device)
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_pipeline_device(self) -> int:
        """
        Returns a transformers pipeline device:
        - CPU: -1
        - CUDA: integer GPU index
        """
        device = self._get_torch_device()
        if device == "cpu":
            return -1
        if device == "cuda":
            return 0
        if device.startswith("cuda:"):
            try:
                return int(device.split(":", 1)[1])
            except Exception:
                return 0
        # Fallback: if something like "mps" is provided, treat as CPU for pipeline.
        return -1

    def _ensure_models(self) -> None:
        if self._object_detector is not None:
            return
        torch_device = self._get_torch_device()
        pipe_device = self._get_pipeline_device()

        self._object_detector = pipeline(
            model=self.config.detector_id,
            task="zero-shot-object-detection",
            device=pipe_device,
        )
        self._sam_segmenter = AutoModelForMaskGeneration.from_pretrained(self.config.segmenter_id).to(torch_device)
        self._sam_processor = AutoProcessor.from_pretrained(self.config.segmenter_id, use_fast=True)
        self._sam_segmenter.eval()

    def _detect(self, image: Image.Image, labels: List[str]) -> List[DetectionResult]:
        labels = [label if label.endswith(".") else f"{label}." for label in labels]
        results = self._object_detector(image, candidate_labels=labels, threshold=self.config.gsam_threshold)
        return [DetectionResult.from_dict(result) for result in results]

    @torch.inference_mode()
    def _segment(self, image: Image.Image, detection_results: List[DetectionResult]) -> List[DetectionResult]:
        boxes = _get_boxes(detection_results)
        inputs = self._sam_processor(
            images=image,
            input_boxes=boxes,
            return_tensors="pt",
        ).to(self._get_torch_device())
        outputs = self._sam_segmenter(**inputs)
        masks = self._sam_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes,
        )[0]
        masks = _refine_masks(masks)
        for det, mask in zip(detection_results, masks):
            det.mask = mask.astype(np.uint8) if mask is not None else None
        return detection_results

    def _grounded_segmentation(self, image: Image.Image, labels: List[str]) -> List[DetectionResult]:
        detections = self._detect(image, labels)
        if not detections:
            return []
        return self._segment(image, detections)

    def __call__(self, image: np.ndarray, objects_of_interest: Sequence[str]) -> ScenePriorResult:
        self._ensure_models()

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape={image.shape}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if self.config.input_is_bgr else image
        objects = [obj if obj.endswith(".") else f"{obj}." for obj in objects_of_interest]

        mask_list: List[np.ndarray] = []
        mask_labels: List[str] = []
        cropped_image_list: List[Optional[np.ndarray]] = []

        if len(objects) > 0:
            image_pil = Image.fromarray(image_rgb)
            all_labels = list(objects) + list(self.config.negative_labels)
            detections = self._grounded_segmentation(image=image_pil, labels=all_labels)

            pos_set = {_normalize_label(label) for label in objects}
            neg_set = {_normalize_label(label) for label in self.config.negative_labels}
            positive_detections: List[DetectionResult] = []
            negative_detections: List[DetectionResult] = []
            for det in detections:
                key = _normalize_label(det.label)
                if key in neg_set:
                    negative_detections.append(det)
                elif key in pos_set:
                    positive_detections.append(det)

            if negative_detections:
                positive_detections = _filter_masks_by_negative(
                    positive_detections,
                    self.config.negative_overlap_threshold,
                    negative_detections,
                )

            if self.config.select_best:
                best_by_label: Dict[str, DetectionResult] = {}
                for det in positive_detections:
                    key = _normalize_label(det.label)
                    if key not in best_by_label or det.score > best_by_label[key].score:
                        best_by_label[key] = det
                for obj in objects:
                    det = best_by_label.get(_normalize_label(obj))
                    if det is None or det.mask is None:
                        continue
                    mask_list.append(det.mask.astype(np.uint8))
                    mask_labels.append(obj)
            else:
                for det in positive_detections:
                    if det.mask is None:
                        continue
                    mask_list.append(det.mask.astype(np.uint8))
                    mask_labels.append(det.label if det.label.endswith(".") else f"{det.label}.")

            mask_list, mask_labels = _dedupe_masks(mask_list, mask_labels, self.config.duplicate_iou_threshold)
            cropped_image_list = [_crop_with_mask(image_rgb, m) for m in mask_list]

        scene_prior_object_img = post_process_cropped_image(cropped_image_list, img_size=self.config.output_size, bg_color=self.config.bg_color)
        scene_prior_object_img_bgr = cv2.cvtColor(scene_prior_object_img, cv2.COLOR_RGB2BGR)

        if len(mask_list) > 0:
            merged_mask = (np.sum(np.stack(mask_list, axis=0), axis=0) > 0).astype(np.uint8) * 255
        else:
            merged_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

        return ScenePriorResult(
            mask_list=mask_list,
            mask_labels=mask_labels,
            scene_prior_object_img_bgr=scene_prior_object_img_bgr,
            merged_mask=merged_mask,
            cropped_image_list=cropped_image_list,
        )

