import os
import numpy as np
from PIL import Image
import json


def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0:
        raise ValueError("Image has invalid height.")
    if height == target_height:
        return image
    scale = target_height / float(height)
    target_width = int(round(width * scale))
    resized = Image.fromarray(image).resize(
        (target_width, target_height), resample=Image.BILINEAR
    )
    return np.asarray(resized)


def _read_image_sequence_as_video(image_dir: str) -> np.ndarray:
    # Lazy import so this file is safe even when mediapy isn't installed unless used.
    import mediapy as media

    if not os.path.isdir(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")

    image_files = sorted(
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not image_files:
        raise ValueError(f"No images found in directory: {image_dir}")

    frames = []
    for image_name in image_files:
        frame = media.read_image(os.path.join(image_dir, image_name))
        if frame is None:
            raise ValueError(
                f"Failed to read image frame: {os.path.join(image_dir, image_name)}"
            )
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        frames.append(frame)
    return np.stack(frames, axis=0)


def _resize_video_to_height(video: np.ndarray, target_height: int) -> np.ndarray:
    if video.shape[1] == target_height:
        return video
    return np.stack([_resize_to_height(frame, target_height) for frame in video], axis=0)


def _pad_video_to_length(video: np.ndarray, target_len: int) -> np.ndarray:
    if len(video) >= target_len:
        return video
    pad = np.repeat(video[-1:, :, :, :], target_len - len(video), axis=0)
    return np.concatenate([video, pad], axis=0)


def _pad_video_width(video: np.ndarray, target_width: int) -> np.ndarray:
    current_width = video.shape[2]
    if current_width == target_width:
        return video
    pad_width = target_width - current_width
    if pad_width < 0:
        raise ValueError("target_width must be >= current width.")
    pad = np.zeros((video.shape[0], video.shape[1], pad_width, video.shape[3]), dtype=video.dtype)
    return np.concatenate([video, pad], axis=2)


def _extract_episode_id_from_demo_folder_name(folder_name: str) -> str:
    # Folder format example:
    # Put_the_marker_in_the_cup_16_IRIS+ef107c48+2023-03-09-19h-48m-45s
    # In this repo we also use:
    # <prompt>_<global_idx>_<episode_id>
    if "_" not in folder_name:
        raise ValueError(f"Unexpected demo folder name format: {folder_name}")
    return folder_name.split("_")[-1]


def _get_episode_id_from_demo_folder(demo_folder_path: str) -> str:
    meta_path = os.path.join(demo_folder_path, "meta_data.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            episode_id = meta.get("episode_id")
            if isinstance(episode_id, str) and episode_id:
                return episode_id
        except Exception:
            # Fall back to parsing the folder name.
            pass
    return _extract_episode_id_from_demo_folder_name(os.path.basename(demo_folder_path))


def write_demo_visualization_mp4(
    demo_folder_path: str,
    output_path: str,
    droid_rgb_root: str,
    overwrite: bool = False,
) -> None:
    """
    Create a 2x3 visualization mp4 for a single demonstration folder.

    Layout matches `unit_test/visualization.py`:
      Row 1: scene_prior_rgb | predicted video (wrist) | real wrist_cam
      Row 2: scene_prior_object_rgb | predicted video (external_cam1) | real external_cam2
    """
    import mediapy as media

    if (not overwrite) and os.path.exists(output_path):
        return

    episode_id = _get_episode_id_from_demo_folder(demo_folder_path)

    image_path = os.path.join(demo_folder_path, "scene_prior_rgb.png")
    image_object_path = os.path.join(demo_folder_path, "scene_prior_object_rgb.png")
    video1_path = os.path.join(demo_folder_path, "recordings", "SVO", "14064085.mp4")
    video2_path = os.path.join(demo_folder_path, "recordings", "SVO", "33790348.mp4")
    external_cam2_dir = os.path.join(droid_rgb_root, episode_id, "rgb", "external_cam2")
    wrist_cam_dir = os.path.join(droid_rgb_root, episode_id, "rgb", "wrist_cam")

    image = media.read_image(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image_object = media.read_image(image_object_path)
    if image_object is None:
        raise ValueError(f"Failed to read image: {image_object_path}")

    try:
        video1, meta1 = media.read_video(video1_path, return_metadata=True)
        fps = getattr(meta1, "fps", None) or 30.0
    except TypeError:
        video1 = media.read_video(video1_path)
        fps = 30.0

    try:
        video2 = media.read_video(video2_path)
    except TypeError:
        video2, _ = media.read_video(video2_path, return_metadata=True)

    external_cam2_video = _read_image_sequence_as_video(external_cam2_dir)
    wrist_cam_video = _read_image_sequence_as_video(wrist_cam_dir)

    if (
        len(video1) == 0
        or len(video2) == 0
        or len(external_cam2_video) == 0
        or len(wrist_cam_video) == 0
    ):
        raise ValueError("One of the videos appears to be empty.")

    target_height = video1.shape[1]
    if target_height <= 0:
        raise ValueError(f"Invalid height from video: {video1_path}")

    if image.shape[-1] == 4:
        image = image[:, :, :3]
    if image_object.shape[-1] == 4:
        image_object = image_object[:, :, :3]

    image = _resize_to_height(image, target_height)
    image_object = _resize_to_height(image_object, target_height)

    video1 = _resize_video_to_height(video1, target_height)
    video2 = _resize_video_to_height(video2, target_height)
    external_cam2_video = _resize_video_to_height(external_cam2_video, target_height)
    wrist_cam_video = _resize_video_to_height(wrist_cam_video, target_height)

    max_len = max(len(video1), len(video2), len(external_cam2_video), len(wrist_cam_video))
    video1 = _pad_video_to_length(video1, max_len)
    video2 = _pad_video_to_length(video2, max_len)
    external_cam2_video = _pad_video_to_length(external_cam2_video, max_len)
    wrist_cam_video = _pad_video_to_length(wrist_cam_video, max_len)

    image_stack = np.repeat(image[None, :, :, :], max_len, axis=0)
    image_object_stack = np.repeat(image_object[None, :, :, :], max_len, axis=0)

    col1_width = max(image_stack.shape[2], image_object_stack.shape[2])
    col2_width = max(video1.shape[2], video2.shape[2])
    col3_width = max(wrist_cam_video.shape[2], external_cam2_video.shape[2])

    image_stack = _pad_video_width(image_stack, col1_width)
    image_object_stack = _pad_video_width(image_object_stack, col1_width)
    video1 = _pad_video_width(video1, col2_width)
    video2 = _pad_video_width(video2, col2_width)
    wrist_cam_video = _pad_video_width(wrist_cam_video, col3_width)
    external_cam2_video = _pad_video_width(external_cam2_video, col3_width)

    row1 = np.concatenate([image_stack, video1, wrist_cam_video], axis=2)
    row2 = np.concatenate([image_object_stack, video2, external_cam2_video], axis=2)
    concat = np.concatenate([row1, row2], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    media.write_video(output_path, concat, fps=fps)

