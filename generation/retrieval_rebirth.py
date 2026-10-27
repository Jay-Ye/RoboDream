import datetime
import pickle
import sys
from pathlib import Path
import argparse
import os
import json
import numpy as np
import glob
import random
from scipy.spatial.transform import Rotation as R
from typing import Dict, Any
from PIL import Image
import re
import cv2
from concurrent.futures import ThreadPoolExecutor
# Make `replay/` (IsaacLab replay) and `generation/` importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "replay"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay.replay_functions import create_replay_context
from replay.replay import batched_replay
from replay.utils import resize_intrinsics
from droid_meta.config import SCENE_CAM2IMAGE_MAP, DEFAULT_CAMERA_INTRINSICS, DEFAULT_CAMERA_INTRINSICS_WRIST, DROID_META_DIR

from retrieval.retriever_new import CachedPromptRetriever as TrajectoryRetriever

from utils.server_utils import send_to_inference_server, format_data_for_server
from utils.TrajEngine_utils import save_demonstration
from utils.visualization_utils import write_demo_visualization_mp4
from GSamObjectPrior import GroundedSamScenePrior, GSamScenePriorConfig
from ObjectRemoval import ObjectRemover, ObjectRemovalConfig

DATA_DIR = os.environ.get("ROBODREAM_DATA_DIR", "datasets")
CKPT_DIR = os.environ.get("ROBODREAM_CKPT_DIR", "checkpoints")


def load_calibration_camera_params(cam_calib_path: str, cam_serials: list) -> Dict[str, Dict[str, Any]]:
    """Camera params from a DROID calibration file ({"<serial>_left": {"pose": [x,y,z,rx,ry,rz]}, ...}).

    `cam_serials` are the serials of (external_cam1, external_cam2, wrist_cam). Intrinsics use the
    DROID defaults; the wrist camera pose is left to the simulator (it moves with the gripper).
    """
    with open(cam_calib_path, "r") as f:
        camera_calib_info = json.load(f)
    camera_param = {}
    for cam_name, serial in zip(["external_cam1", "external_cam2", "wrist_cam"], cam_serials):
        cam_extrinsics = camera_calib_info[f"{serial}_left"]['pose']
        camera_pose_np = np.eye(4)
        camera_pose_np[:3, :3] = R.from_euler("xyz", cam_extrinsics[3:6]).as_matrix()
        camera_pose_np[:3, 3] = cam_extrinsics[0:3]
        intrinsics_np = DEFAULT_CAMERA_INTRINSICS_WRIST if "wrist" in cam_name else DEFAULT_CAMERA_INTRINSICS
        camera_param[cam_name] = {
            "camera_pose": camera_pose_np if "wrist" not in cam_name else None, # should not pass in camera pose for wrist cam
            "intrinsics": intrinsics_np,
        }
    return camera_param

def save_data_to_cache(
    data_dict: Dict[str, Any],
    cache_dir: str,
) -> str:
    """
    Save formatted data to cache file.
    
    Args:
        data_dict: Formatted data from format_data_for_server()
        cache_dir: Directory to save cache files
        
    Returns:
        cache_path: Path to saved cache file
    """
    os.makedirs(cache_dir, exist_ok=True)
    if isinstance(data_dict['sample_id'], list):
        data_dict['sample_id'] = data_dict['sample_id'][0]
    sample_id = data_dict['sample_id'].replace('/', '_')
    cache_path = os.path.join(cache_dir, f"{sample_id}_cached_inputs.pkl")
    
    with open(cache_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"[INFO] Saved cached data to: {cache_path}")
    return cache_path

def get_image_from_folder(image_folder: str) -> list:
    img_extensions = ["jpg", "JPG", "png", "PNG", "jpeg", "JPEG"]
    image_paths = []
    for ext in img_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, f"*.{ext}")))
    # Exclude cached outputs created by scene prior helpers
    filtered_paths = []
    for path in image_paths:
        stem = Path(path).stem
        if "_edited" in stem or "_object_prior" in stem:
            continue
        filtered_paths.append(path)
    return filtered_paths

def split_object_prompts(prompt_string: str) -> list:
    if "|" in prompt_string or ";" in prompt_string:
        parts = re.split(r"[|;]", prompt_string)
        return [p.strip() for p in parts if p.strip()]
    if "," in prompt_string:
        parts = prompt_string.split(",")
        return [p.strip() for p in parts if p.strip()]
    return [prompt_string.strip()] if prompt_string.strip() else []

def build_object_prompts_per_query(queries: list, object_prompts: list | None) -> list:
    if not object_prompts:
        return [[] for _ in queries]
    if len(queries) == 1:
        return [object_prompts]
    if len(object_prompts) == len(queries):
        return [split_object_prompts(p) for p in object_prompts]
    print("[WARNING] --object_prompts length does not match --query. "
          "Applying the same prompts to all queries.")
    return [object_prompts for _ in queries]

def get_replay_cache_path(cache_dir: str, episode_id: str) -> str:
    safe_episode_id = episode_id.replace("/", "_")
    return os.path.join(cache_dir, f"{safe_episode_id}_replay.pkl")

def load_replay_cache(cache_dir: str, episode_id: str) -> Dict[str, Any] | None:
    cache_path = get_replay_cache_path(cache_dir, episode_id)
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as f:
        return pickle.load(f)

_CACHE_IMAGE_SIZE = (320, 176)  # cv2 dsize (width, height); numpy shape → (176, 320, 3)


def save_replay_cache(cache_dir: str, episode_id: str, data: Dict[str, Any]) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = get_replay_cache_path(cache_dir, episode_id)
    # Downscale camera images before pickling. Replay produces 1280×720 frames but
    # format_data_for_server resizes to 176×320 anyway, so full-res storage wastes
    # 20-60× disk space. Old cache files (full-res) still load and work correctly.
    if "camera_images" in data:
        downscaled = {
            cam: {t: cv2.resize(img, _CACHE_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                  for t, img in ts_dict.items()}
            for cam, ts_dict in data["camera_images"].items()
        }
        data = {**data, "camera_images": downscaled}
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    return cache_path


def _save_camera_params(demo_folder_path: str, camera_params: Dict) -> None:
    """Persist camera params alongside a saved demonstration."""
    os.makedirs(demo_folder_path, exist_ok=True)
    with open(os.path.join(demo_folder_path, "camera_params.pkl"), "wb") as f:
        pickle.dump(camera_params, f)


def _demo_exists_with_matching_params(demo_folder_path: str, camera_params: Dict) -> bool:
    """Return True if the demo folder exists and was rendered with matching camera poses."""
    if not os.path.isdir(demo_folder_path):
        return False
    params_path = os.path.join(demo_folder_path, "camera_params.pkl")
    if not os.path.exists(params_path):
        return False
    with open(params_path, "rb") as f:
        cached_params = pickle.load(f)
    return _camera_params_match(cached_params, camera_params)


def _camera_params_match(cached_params: Dict | None, desired_params: Dict) -> bool:
    """Return True if the cached camera poses match the desired ones.

    Only the extrinsic pose (4×4 matrix) is compared; a None pose (wrist cam)
    matches another None pose. Missing or mismatched entries are treated as a
    cache miss so the replay runs again at the correct viewpoint.
    """
    if cached_params is None:
        return False
    for cam_name, desired in desired_params.items():
        cached = cached_params.get(cam_name)
        if cached is None:
            return False
        desired_pose = desired.get("camera_pose")
        cached_pose = cached.get("camera_pose")
        if desired_pose is None and cached_pose is None:
            continue
        if desired_pose is None or cached_pose is None:
            return False
        if not np.allclose(desired_pose, cached_pose, atol=1e-6):
            return False
    return True

_ASPECT_CROP_TOLERANCE = 0.1  # skip center-crop if aspect ratio mismatch is within this fraction


def _center_crop_to_aspect_pil(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Center-crop a PIL image to match (target_w / target_h) aspect ratio.

    Cropping is skipped when the relative mismatch is within _ASPECT_CROP_TOLERANCE.
    """
    w, h = img.size
    target_aspect = target_w / target_h
    current_aspect = w / h
    if abs(current_aspect - target_aspect) / target_aspect <= _ASPECT_CROP_TOLERANCE:
        return img
    if current_aspect > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_aspect)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    return img


def _center_crop_to_aspect_bgr(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center-crop a BGR (H×W×C) numpy array to match (target_w / target_h) aspect ratio.

    Cropping is skipped when the relative mismatch is within _ASPECT_CROP_TOLERANCE.
    """
    h, w = img.shape[:2]
    target_aspect = target_w / target_h
    current_aspect = w / h
    if abs(current_aspect - target_aspect) / target_aspect <= _ASPECT_CROP_TOLERANCE:
        return img
    if current_aspect > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        img = img[:, left:left + new_w]
    else:
        new_h = int(w / target_aspect)
        top = (h - new_h) // 2
        img = img[top:top + new_h, :]
    return img


def get_scene_prior_cache_paths(scene_prior_path: str) -> tuple[str, str]:
    base_path = Path(scene_prior_path)
    stem = base_path.stem
    edited_path = base_path.with_name(f"{stem}_edited.png")
    object_path = base_path.with_name(f"{stem}_object_prior.png")
    return str(edited_path), str(object_path)

def load_or_create_scene_prior(
    scene_prior_path: str,
    object_prompts: list,
    gsam_prior: GroundedSamScenePrior,
    remover: ObjectRemover,
    target_size: tuple[int, int],
    scene_prior_rgb_override_path: str | None = None,
    object_prior_rgb_override_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    edited_path, object_path = get_scene_prior_cache_paths(scene_prior_path)

    # If the user provides a scene-prior RGB image, we always use that image for
    # the scene prior (no inpainting). The object prior stays coupled to
    # `scene_prior_path` and its cached `_object_prior.png`.
    use_override_scene_prior = scene_prior_rgb_override_path is not None
    use_override_object_prior = object_prior_rgb_override_path is not None

    if (not use_override_scene_prior) and (not use_override_object_prior) and os.path.exists(edited_path) and os.path.exists(object_path):
        print(f"[INFO] Loaded cached scene prior from: {edited_path} and {object_path}")
        edited_rgb = Image.open(edited_path).convert("RGB")
        object_bgr = cv2.imread(object_path)
        if object_bgr is None:
            raise RuntimeError(f"Failed to read cached object prior: {object_path}")
        edited_rgb = np.array(
            _center_crop_to_aspect_pil(edited_rgb, target_size[0], target_size[1]).resize(target_size, Image.Resampling.LANCZOS)
        )
        object_bgr = cv2.resize(_center_crop_to_aspect_bgr(object_bgr, target_size[0], target_size[1]), target_size, interpolation=cv2.INTER_AREA)
        object_rgb = cv2.cvtColor(object_bgr, cv2.COLOR_BGR2RGB)
        return edited_rgb, object_rgb

    # Both priors are explicitly provided — no source image, GSam, or inpainting needed.
    if use_override_scene_prior and use_override_object_prior:
        override_pil = Image.open(scene_prior_rgb_override_path).convert("RGB")
        edited_rgb = np.array(
            _center_crop_to_aspect_pil(override_pil, target_size[0], target_size[1]).resize(target_size, Image.Resampling.LANCZOS)
        )
        object_bgr = cv2.imread(object_prior_rgb_override_path)
        if object_bgr is None:
            raise RuntimeError(f"Failed to read object prior override: {object_prior_rgb_override_path}")
        object_bgr = cv2.resize(_center_crop_to_aspect_bgr(object_bgr, target_size[0], target_size[1]), target_size, interpolation=cv2.INTER_AREA)
        object_rgb = cv2.cvtColor(object_bgr, cv2.COLOR_BGR2RGB)
        return edited_rgb, object_rgb

    img_bgr = cv2.imread(scene_prior_path)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read scene prior image: {scene_prior_path}")

    # Load / compute object prior (cached as BGR).
    merged_mask = None
    if use_override_object_prior:
        print(f"[INFO] Using object prior override: {object_prior_rgb_override_path}")
        scene_prior_object_img_bgr = cv2.imread(object_prior_rgb_override_path)
        if scene_prior_object_img_bgr is None:
            raise RuntimeError(f"Failed to read object prior override: {object_prior_rgb_override_path}")
    else:
        scene_prior_object_img_bgr = None
        if os.path.exists(object_path):
            scene_prior_object_img_bgr = cv2.imread(object_path)
            if scene_prior_object_img_bgr is None:
                raise RuntimeError(f"Failed to read cached object prior: {object_path}")

        if (scene_prior_object_img_bgr is None) and object_prompts:
            print(f"[INFO] Creating object prior from: {scene_prior_path} with prompts: {object_prompts}")
            print("[INFO] Running GSam ...")
            out = gsam_prior(img_bgr, object_prompts)
            scene_prior_object_img_bgr = out.scene_prior_object_img_bgr
            merged_mask = out.merged_mask
            cv2.imwrite(object_path, scene_prior_object_img_bgr)
        elif scene_prior_object_img_bgr is None:
            # No object prompts: object prior defaults to the input image.
            scene_prior_object_img_bgr = img_bgr
            cv2.imwrite(object_path, scene_prior_object_img_bgr)

    # Load / compute scene prior RGB.
    if use_override_scene_prior:
        override_pil = Image.open(scene_prior_rgb_override_path).convert("RGB")
        edited_rgb = np.array(
            _center_crop_to_aspect_pil(override_pil, target_size[0], target_size[1]).resize(target_size, Image.Resampling.LANCZOS)
        )
    else:
        if os.path.exists(edited_path):
            edited_rgb = np.array(Image.open(edited_path).convert("RGB"))
        else:
            if object_prompts:
                # If we loaded object prior from cache, we don't have a mask. Re-run GSam
                # to obtain the merged mask for inpainting.
                if merged_mask is None:
                    print(f"[INFO] Creating scene prior from: {scene_prior_path} with prompts: {object_prompts}")
                    print("[INFO] Running GSam (for mask) ...")
                    out = gsam_prior(img_bgr, object_prompts)
                    merged_mask = out.merged_mask
                print("[INFO] Running ObjectRemoval ...")
                edited_pil = remover.remove(img_bgr, merged_mask)
                edited_rgb = np.array(edited_pil.convert("RGB"))
            else:
                edited_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(edited_rgb).save(edited_path)
        edited_rgb = np.array(
            _center_crop_to_aspect_pil(Image.fromarray(edited_rgb), target_size[0], target_size[1]).resize(target_size, Image.Resampling.LANCZOS)
        )

    scene_prior_object_img_bgr = cv2.resize(_center_crop_to_aspect_bgr(scene_prior_object_img_bgr, target_size[0], target_size[1]), target_size, interpolation=cv2.INTER_AREA)
    scene_prior_object_img_rgb = cv2.cvtColor(scene_prior_object_img_bgr, cv2.COLOR_BGR2RGB)
    return edited_rgb, scene_prior_object_img_rgb
    
def _handle_server_results(server_results, results, args, camera_images, action, prompt,
                            scene_prior_rgb, scene_prior_object_rgb, episode_id, camera_params,
                            task_name_override=None):
    for server_result in server_results:
        results.append(server_result)
        if not server_result["success"]:
            err = server_result.get("error", "<unknown>")
            print(f"[ERROR] Failed to process sample {episode_id} on {server_result['server_url']}: {err}")
            continue
        task_name = task_name_override if task_name_override is not None else prompt.replace(" ", "_")
        server_output_dir = os.path.join(args.output_dir, server_result["server_suffix"])
        output_demonstration_path = os.path.join(server_output_dir, task_name, episode_id)
        response = server_result["response"]
        save_demonstration(output_demonstration_path, response["videos"], camera_images, action, prompt,
                           scene_prior_rgb, scene_prior_object_rgb, episode_id)
        _save_camera_params(output_demonstration_path, camera_params)
        if args.visualize:
            try:
                vis_suffix_dir = os.path.join(args.visualization_output_dir, server_result["server_suffix"])
                vis_out_path = os.path.join(vis_suffix_dir, task_name, f"{episode_id}.mp4")
                write_demo_visualization_mp4(
                    demo_folder_path=output_demonstration_path,
                    output_path=vis_out_path,
                    droid_rgb_root=args.droid_rgb_root,
                    overwrite=True,
                )
                print(f"[INFO] Wrote visualization: {vis_out_path}")
            except Exception as e:
                print(f"[WARNING] Visualization failed for {output_demonstration_path}: {e}")


def main(args):
    if args.scene_prior_rgb_path is not None:
        if not os.path.isfile(args.scene_prior_rgb_path):
            raise ValueError(f"--scene_prior_rgb_path not found: {args.scene_prior_rgb_path}")

    if args.object_prior_rgb_path is not None:
        if len(args.object_prior_rgb_path) != len(args.query):
            raise ValueError(
                f"--object_prior_rgb_path must have the same number of entries as --query "
                f"(got {len(args.object_prior_rgb_path)} paths for {len(args.query)} queries)"
            )
        for path in args.object_prior_rgb_path:
            if not os.path.isfile(path):
                raise ValueError(f"--object_prior_rgb_path not found: {path}")

    if args.visualize:
        if not os.path.isdir(args.droid_rgb_root):
            raise ValueError(f"--droid_rgb_root not found: {args.droid_rgb_root}")

    server_urls = args.server_url or []
    if args.suffix is None:
        server_suffixes = [f"server{i}" for i in range(len(server_urls))]
        broadcast_mode = True
    else:
        if len(args.suffix) == len(server_urls):
            # Broadcast mode: same request sent to all servers (e.g. comparing models).
            broadcast_mode = True
        elif len(args.suffix) == 1:
            # Distributed mode: requests are round-robin'd across servers in parallel.
            broadcast_mode = False
        else:
            raise ValueError(
                f"--suffix must have the same number of items as --server_url "
                f"(got {len(args.suffix)} suffixes for {len(server_urls)} URLs), "
                f"or exactly 1 suffix to distribute requests across servers."
            )
        server_suffixes = args.suffix
    if broadcast_mode:
        server_targets = list(zip(server_urls, server_suffixes))
    else:
        # Distributed: all server URLs share the single suffix.
        server_targets = [(url, server_suffixes[0]) for url in server_urls]

    def send_to_all_servers(data_dict: Dict[str, Any], targets=None) -> list[Dict[str, Any]]:
        effective_targets = targets if targets is not None else server_targets
        if not effective_targets:
            return []
        submitted = []
        with ThreadPoolExecutor(max_workers=len(effective_targets)) as executor:
            for server_url, server_suffix in effective_targets:
                future = executor.submit(
                    send_to_inference_server,
                    data_dict=data_dict,
                    server_url=server_url,
                    timeout=args.timeout,
                )
                submitted.append((server_url, server_suffix, future))

            responses = []
            for server_url, server_suffix, future in submitted:
                try:
                    response = future.result()
                    success = response.get("status") == "success"
                    responses.append({
                        "sample_id": episode_id,
                        "server_url": server_url,
                        "server_suffix": server_suffix,
                        "response": response,
                        "success": success,
                    })
                except Exception as e:
                    responses.append({
                        "sample_id": episode_id,
                        "server_url": server_url,
                        "server_suffix": server_suffix,
                        "error": str(e),
                        "success": False,
                    })
            return responses

    # initialize trajectory generator
    if args.generation_mode == "retrieval":
        generator = TrajectoryRetriever(
            text_embedding_path=os.environ.get(
                "TEXT_EMBEDDING_CACHE_PATH",
                os.path.join(DATA_DIR, "droid_t5_text_embeddings.pkl.gz"),
            ),
            language_annotations_path=os.environ.get(
                "DROID_LANGUAGE_ANNOTATIONS_PATH",
                os.path.join(DATA_DIR, "droid_meta", "droid_language_annotations.json"),
            ),
            trajectory_root=os.environ.get(
                "DROID_TRAJECTORY_ROOT",
                os.path.join(DATA_DIR, "trajengine"),
            ),
            model_name="google-t5/t5-11b",
            device=args.device,
            model_cache_dir=os.environ.get("T5_MODEL_CACHE_DIR", os.path.join(CKPT_DIR, "google-t5", "t5-11b")),
            local_files_only=bool(int(os.environ.get("LOCAL_FILES_ONLY", "0"))),
        )
        trajectories = [
            generator.retrieve(
                query=single_query,
                top_k=args.num_samples,
                confidence_threshold=args.confidence_threshold,
                with_action=args.with_action,
                downsample_factor=args.downsample_factor,
                return_unique_samples=True,
            )
            for single_query in args.query
        ]
    elif args.generation_mode == "generation":
        raise NotImplementedError("Generation mode not implemented yet")
    else:
        raise ValueError(f"Invalid generation mode: {args.generation_mode}")

    # Save retrieval metadata (per-episode similarity, description, length, etc.)
    retrieval_meta_path = os.path.join(args.output_dir, "retrieval_metadata.json")
    os.makedirs(args.output_dir, exist_ok=True)
    retrieval_meta = {}
    if os.path.exists(retrieval_meta_path):
        with open(retrieval_meta_path, "r") as _f:
            retrieval_meta = json.load(_f)
    retrieved_at = datetime.datetime.now().isoformat()
    for single_query, traj_list in zip(args.query, trajectories):
        for rank, traj in enumerate(traj_list, start=1):
            ep_id = traj.sample_path.split("/")[-1]
            key = f"{single_query}|{ep_id}"
            retrieval_meta[key] = {
                "query": single_query,
                "episode_id": ep_id,
                "description": traj.description,
                "similarity": round(float(traj.similarity), 6),
                "trajectory_length": int(len(traj.trajectory["state_xyzrpy"])),
                "sample_path": traj.sample_path,
                "rank": rank,
                "retrieved_at": retrieved_at,
            }
    with open(retrieval_meta_path, "w") as _f:
        json.dump(retrieval_meta, _f, indent=2)
    print(f"[INFO] Saved retrieval metadata ({len(retrieval_meta)} total entries) to {retrieval_meta_path}")

    # flatten trajectories and queries
    trajectories = [trajectory for traj_per_query in trajectories for trajectory in traj_per_query]
    queries = [single_query for single_query in args.query for _ in range(args.num_samples)]
    actions_replay = [
        np.concatenate([trajectory.trajectory["state_cartesian"], trajectory.trajectory["state_gripper"]], axis=1) 
        for trajectory in trajectories
    ] # the replay script expects the actions to be in the format of [x, y, z, qw, qx, qy, qz, gripper]
    actions_retrieval = [
        np.concatenate([trajectory.trajectory["state_xyzrpy"], trajectory.trajectory["state_gripper"]], axis=1)
        for trajectory in trajectories
    ] # the video generation model expects the actions to be in the format of [x, y, z, roll, pitch, yaw, gripper]

    episode_ids = [trajectory.sample_path.split("/")[-1] for trajectory in trajectories]
    object_prompts_per_query = build_object_prompts_per_query(args.query, args.object_prompts)
    object_prompts = [
        prompts_per_query for prompts_per_query in object_prompts_per_query for _ in range(args.num_samples)
    ]
    object_prior_paths = (
        [path for path in args.object_prior_rgb_path for _ in range(args.num_samples)]
        if args.object_prior_rgb_path
        else [None] * len(trajectories)
    )
    if args.cam_calib_path is None:
        #########################################################
        # no calibration given: render from each retrieved episode's original DROID camera poses
        with open(os.path.join(DROID_META_DIR, "reformatted_episodes_with_good_extrinsics.json"), "r") as f:
            reformatted_episodes_with_good_extrinsics = json.load(f)
        metadatas = [
            reformatted_episodes_with_good_extrinsics[episode_id] for episode_id in episode_ids
        ]
        camera_params = []
        for metadata in metadatas:
            camera_param = {}
            for cam_name, img_name in SCENE_CAM2IMAGE_MAP.items():
                cam_extrinsics = metadata.get(img_name, None)
                if cam_extrinsics is None:
                    camera_pose_np = None
                else:
                    pos = cam_extrinsics[0:3]
                    rot = R.from_euler("xyz", cam_extrinsics[3:6]).as_matrix()
                    camera_pose_np = np.eye(4)
                    camera_pose_np[:3, :3] = rot
                    camera_pose_np[:3, 3] = pos
                cam_intrinsics = metadata[f"{img_name}_intrinsics"]
                fx, cx, fy, cy = cam_intrinsics["cameraMatrix"]
                intrinsics_np = np.array([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0, 0, 1]])
                if cam_intrinsics["width"] == 672 and cam_intrinsics["height"] == 376:
                    intrinsics_np = resize_intrinsics(intrinsics_np, src_size=(672, 376), dst_size=(1280, 720))
                camera_param[cam_name] = {
                    "camera_pose": camera_pose_np,
                    "intrinsics": intrinsics_np,
                    "camera_serial": metadata[f"{img_name}_to_serial"],
                }
            camera_params.append(camera_param)
        #########################################################
    else:
        camera_param = load_calibration_camera_params(args.cam_calib_path, args.cam_serials)
        camera_params = [camera_param] * len(trajectories)

    N_Parallel = min(6, len(trajectories))  # Maximum number of samples to replay at once
    control_mode = "end_effector" # "joint_position" or "end_effector"
    replay_cache_dir = args.replay_cache_dir
    os.makedirs(replay_cache_dir, exist_ok=True)
    
    # Process samples in batches
    total_samples = len(trajectories)
    results = []
    
    print(f"\n{'='*60}")
    print(f"Processing {total_samples} samples in batches of {N_Parallel}")
    print(f"{'='*60}")
    
    # Create one replay context once for the max batch size.
    replay_ctx = None

    # Initialize scene prior helpers once
    gsam_prior = GroundedSamScenePrior(GSamScenePriorConfig(input_is_bgr=True, device=args.device))
    remover = ObjectRemover(ObjectRemovalConfig(input_is_bgr=True, device=args.device))
    
    for batch_start in range(0, total_samples, N_Parallel):
        batch_end = min(batch_start + N_Parallel, total_samples)
        batch_size = batch_end - batch_start
        
        print(f"\n{'='*60}")
        print(f"Processing batch: samples {batch_start+1} to {batch_end} (batch size: {batch_size})")
        print(f"{'='*60}")
        
        # Create batch store for current batch
        batch_actions_replay = actions_replay[batch_start:batch_end]
        batch_camera_params = camera_params[batch_start:batch_end]
        batch_episode_ids = episode_ids[batch_start:batch_end]

        camera_images_by_episode = {}
        replay_indices = []
        for local_idx, episode_id in enumerate(batch_episode_ids):
            cached = load_replay_cache(replay_cache_dir, episode_id)
            desired_camera_params = batch_camera_params[local_idx]
            if cached is not None and _camera_params_match(cached.get("camera_params"), desired_camera_params):
                camera_images_by_episode[episode_id] = cached["camera_images"]
            else:
                if cached is not None:
                    print(f"[INFO] Cache for {episode_id} has mismatched camera poses; re-replaying.")
                replay_indices.append(local_idx)

        if replay_indices:
            replay_episode_ids = [batch_episode_ids[i] for i in replay_indices]
            replay_actions = [batch_actions_replay[i] for i in replay_indices]
            replay_camera_params = [batch_camera_params[i] for i in replay_indices]
            replay_batch_size = len(replay_episode_ids)
            original_replay_batch_size = replay_batch_size

            # Always run replay with N_Parallel environments by padding with duplicates.
            if replay_batch_size < N_Parallel:
                pad_count = N_Parallel - replay_batch_size
                pad_episode_id = replay_episode_ids[0]
                pad_action = replay_actions[0]
                pad_camera_param = replay_camera_params[0]
                for pad_idx in range(pad_count):
                    replay_episode_ids.append(f"{pad_episode_id}__pad_{batch_start}_{pad_idx}")
                    replay_actions.append(pad_action)
                    replay_camera_params.append(pad_camera_param)
                print(f"[INFO] Padded replay batch from {replay_batch_size} to {N_Parallel} using duplicated samples.")

            # Note that everything in batch_store will be cleared after the replay is completed
            batch_store = {
                "episode_ids": replay_episode_ids.copy(),
                "actions": replay_actions,
                "camera_params": replay_camera_params,
                "real_obs": None,
                # "video_paths": [f"runs/{episode_id}.mp4" for episode_id in replay_episode_ids], # path to the video files of the replayed episodes
                # "img_paths": [f"runs/{episode_id}" for episode_id in replay_episode_ids], # path to the image files of the replayed episodes
            }

            if replay_ctx is None:
                replay_ctx = create_replay_context(
                    headless=True,
                    scene=1,
                    num_envs=N_Parallel,
                    env_name="DROID" if control_mode == "joint_position" else "DROID_EE",
                )
            outputs = batched_replay(
                replay_ctx,
                batch_store,
                control_mode,
                skip_real_obs=False,
            )
            replayed_camera_images = outputs.get("camera_images", {})
            for output_idx, episode_id in enumerate(replay_episode_ids[:original_replay_batch_size]):
                camera_images = replayed_camera_images.get(output_idx)
                if camera_images is None:
                    raise RuntimeError(f"Missing replay output for batch index {output_idx} ({episode_id})")
                camera_images_by_episode[episode_id] = camera_images
                save_replay_cache(
                    replay_cache_dir,
                    episode_id,
                    {
                        "camera_images": camera_images,
                        "camera_params": batch_camera_params[replay_indices[output_idx]],
                    },
                )

        batched_camera_images = [camera_images_by_episode[eid] for eid in batch_episode_ids]
        print(f"Replay completed for batch. Processing {len(batched_camera_images)} samples...")
        
        # Process each sample in current batch
        batch_actions_retrieval = actions_retrieval[batch_start:batch_end]
        batch_queries = queries[batch_start:batch_end]
        batch_object_prompts = object_prompts[batch_start:batch_end]
        batch_object_prior_paths = object_prior_paths[batch_start:batch_end]

        # In distributed mode accumulate (future, metadata) across the batch so all
        # network calls are in-flight simultaneously before we block on results.
        distributed_executor = (
            ThreadPoolExecutor(max_workers=len(server_targets))
            if (server_targets and not broadcast_mode) else None
        )
        distributed_pending = []  # (future, server_url, server_suffix, episode_id, global_idx, camera_images, action, prompt, scene_prior_rgb, scene_prior_object_rgb)

        for local_idx, (camera_images, action, prompt, episode_id, prompt_objects, object_prior_path) in enumerate(
            zip(batched_camera_images, batch_actions_retrieval, batch_queries, batch_episode_ids,
                batch_object_prompts, batch_object_prior_paths)
        ):
            global_idx = batch_start + local_idx
            episode_camera_params = batch_camera_params[local_idx]
            task_name = prompt.replace(" ", "_")

            # Early skip: check which servers still need this demo before doing any
            # expensive work (scene prior generation, data formatting, network I/O).
            if server_targets:
                if broadcast_mode:
                    active_targets = [
                        (url, suffix) for url, suffix in server_targets
                        if not _demo_exists_with_matching_params(
                            os.path.join(args.output_dir, suffix, task_name, episode_id),
                            episode_camera_params,
                        )
                    ]
                    if not active_targets:
                        print(f"\n--- Skipping sample {global_idx+1}/{total_samples} ({episode_id}): demo already exists on all servers. ---")
                        continue
                else:
                    server_url, server_suffix = server_targets[global_idx % len(server_targets)]
                    demo_path = os.path.join(args.output_dir, server_suffix, task_name, episode_id)
                    if _demo_exists_with_matching_params(demo_path, episode_camera_params):
                        print(f"\n--- Skipping sample {global_idx+1}/{total_samples} ({episode_id}): demo already exists on {server_suffix}. ---")
                        continue

            print(f"\n--- Processing sample {global_idx+1}/{total_samples} ---")

            # Choose a *source* image for object-prior segmentation (and for inpainting
            # unless the user overrides the final scene-prior RGB).
            scene_prior_rgb_paths = []
            if args.scene_prior_dir:
                scene_prior_rgb_folder = os.path.join(
                    args.scene_prior_dir, prompt.replace(" ", "_").lower()
                )
                if os.path.isdir(scene_prior_rgb_folder):
                    scene_prior_rgb_paths = get_image_from_folder(scene_prior_rgb_folder)

            if len(scene_prior_rgb_paths) > 0:
                scene_prior_source_path = random.choice(scene_prior_rgb_paths)
            elif args.scene_prior_rgb_path is not None and object_prior_path is not None:
                scene_prior_source_path = args.scene_prior_rgb_path  # both priors given explicitly; source image unused
            else:
                raise ValueError(
                    f"No scene prior source image for '{prompt}'. Put photos under "
                    f"<--scene_prior_dir>/{prompt.replace(' ', '_').lower()}/ or pass "
                    f"--scene_prior_rgb_path and --object_prior_rgb_path."
                )

            scene_prior_rgb, scene_prior_object_rgb = load_or_create_scene_prior(
                scene_prior_path=scene_prior_source_path,
                object_prompts=prompt_objects,
                gsam_prior=gsam_prior,
                remover=remover,
                target_size=(320, 176),
                scene_prior_rgb_override_path=args.scene_prior_rgb_path,
                object_prior_rgb_override_path=object_prior_path,
            )
            
            # Format data for server
            data_dict = format_data_for_server(
                scene_prior_rgb=scene_prior_rgb.copy(),
                scene_prior_object_rgb=scene_prior_object_rgb.copy(),
                camera_images=camera_images,
                trajectory=action,
                prompt=prompt,
                sample_id=episode_id,
                fps=10.0,
                target_size=(176, 320),  # Standard size for the model
            )
            # Send to server if URL provided
            if server_targets:
                if broadcast_mode:
                    # active_targets was already computed in the early-skip block above.
                    server_results = send_to_all_servers(data_dict, targets=active_targets)
                    _handle_server_results(server_results, results, args, camera_images, action, prompt,
                                           scene_prior_rgb, scene_prior_object_rgb, episode_id, episode_camera_params)
                else:
                    # server_url / server_suffix already resolved in the early-skip block.
                    future = distributed_executor.submit(
                        send_to_inference_server,
                        data_dict=data_dict,
                        server_url=server_url,
                        timeout=args.timeout,
                    )
                    distributed_pending.append((
                        future, server_url, server_suffix, episode_id, global_idx,
                        camera_images, action, prompt, scene_prior_rgb, scene_prior_object_rgb,
                        episode_camera_params,
                    ))

        # Collect distributed results (all futures were submitted; now wait for them).
        if distributed_pending:
            print(f"[INFO] Collecting {len(distributed_pending)} distributed inference results...")
            for (future, server_url, server_suffix, ep_id, g_idx,
                 cam_imgs, act, pmt, sp_rgb, sp_obj_rgb, ep_camera_params) in distributed_pending:
                try:
                    response = future.result()
                    success = response.get("status") == "success"
                    server_result = {
                        "sample_id": ep_id,
                        "server_url": server_url,
                        "server_suffix": server_suffix,
                        "response": response,
                        "success": success,
                    }
                except Exception as e:
                    server_result = {
                        "sample_id": ep_id,
                        "server_url": server_url,
                        "server_suffix": server_suffix,
                        "error": str(e),
                        "success": False,
                    }
                _handle_server_results([server_result], results, args, cam_imgs, act, pmt,
                                       sp_rgb, sp_obj_rgb, ep_id, ep_camera_params)
            distributed_executor.shutdown(wait=False)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples processed: {len(results)}")
    
    if server_targets:
        successful = sum(1 for r in results if r.get('success', False))
        print(f"Successfully sent to server: {successful}/{len(results)}")
    
    # if args.save_cache:
    #     print(f"Cached data saved to: {args.cache_dir}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate trajectories and send to inference server"
    )
    
    # Trajectory generation arguments
    parser.add_argument("--generation_mode", type=str, default="retrieval", 
                       choices=["retrieval", "generation"],
                       help="Mode for trajectory generation")
    parser.add_argument("--query", type=str, nargs='+', default=["pick up the cup"],
                       help="Task query/prompt")
    parser.add_argument("--num_samples", type=int, default=-1,
                       help="Number of trajectory samples to generate")
    parser.add_argument("--confidence_threshold", type=float, default=0.8,
                       help="Confidence threshold for trajectory retrieval")
    parser.add_argument("--with_action", action="store_true", 
                       help="Generate both states and actions")
    parser.add_argument("--downsample_factor", type=int, default=1, 
                       help="Downsample the trajectory by this factor")
    parser.add_argument("--cam_calib_path", type=str, default=None,
                       help="Path to the camera calibration file (see metadata/calibration_example.json). "
                            "If omitted, each retrieved episode is rendered from its original DROID camera poses")
    parser.add_argument("--cam_serials", type=str, nargs=3, default=["33790348", "37998989", "14064085"],
                       help="Camera serials in the calibration file for external_cam1, external_cam2 and wrist_cam")
    parser.add_argument("--scene_prior_dir", type=str, default=None,
                       help="Directory to load source images for scene/object priors from")
    parser.add_argument("--scene_prior_rgb_path", type=str, default=None,
                       help="Optional: path to an already-inpainted scene-prior RGB image. "
                            "When set, we skip inpainting and load this image as `scene_prior_rgb` "
                            "for all samples, while still computing `scene_prior_object_rgb` from the "
                            "selected source image and `--object_prompts`.")
    parser.add_argument("--object_prior_rgb_path", type=str, nargs="*", default=None,
                       help="Optional: path(s) to already-computed object-prior RGB image(s) "
                            "(e.g. *_object_prior.png). Supply one path per --query entry. "
                            "When set, GSam segmentation is skipped for the corresponding query.")
    parser.add_argument("--object_prompts", type=str, nargs="*",
                       help="Object prompts for GSam (e.g., 'a marker.' 'a bowl.' "
                            "or per-query comma-separated strings)")

    # Server arguments
    parser.add_argument("--server_url", type=str, nargs="+", default=None,
                       help="One or more inference server URLs (e.g., http://localhost:5000). "
                            "If not provided, requests are not sent.")
    parser.add_argument("--suffix", type=str, nargs="*", default=None,
                       help="Suffix per server URL used in output path naming. "
                            "Must match the number of --server_url values.")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Server request timeout in seconds")
    
    # Cache arguments
    # parser.add_argument("--save_cache", action="store_true",
    #                    help="Save formatted data to cache files")
    # parser.add_argument("--cache_dir", type=str, default="cache/inference_inputs",
    #                    help="Directory to save cached inputs")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str,
                       default=os.path.join(os.environ.get("ROBODREAM_OUTPUT_DIR", "outputs"), "demonstrations"),
                       help="Directory to save demonstrations")
    parser.add_argument("--visualize", action="store_true",
                       help="Write a concatenated visualization mp4 after each successful inference")
    parser.add_argument("--visualization_output_dir", type=str,
                       default=os.path.join(os.environ.get("ROBODREAM_OUTPUT_DIR", "outputs"), "visualizations"),
                       help="Directory to write visualization mp4s (mirrors server suffix subdirs)")
    parser.add_argument("--droid_rgb_root", type=str,
                       default=os.path.join(DATA_DIR, "processed", "droid_traj2demo", "real"),
                       help="Root directory containing per-episode rgb folders used by the visualizer")
    parser.add_argument("--replay_cache_dir", type=str,
                       default=os.path.join(os.environ.get("ROBODREAM_OUTPUT_DIR", "outputs"), "cache", "replay_outputs"),
                       help="Directory to save cached replay outputs")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run scene prior helper on")
    args = parser.parse_args()
    
    assert args.num_samples > 0 or args.confidence_threshold > 0, "Either --num_samples or --confidence_threshold must be provided, num_samples will be prioritized over confidence_threshold"
    
    main(args)