from typing import Dict, Any
import base64
import numpy as np
from PIL import Image
import requests


def encode_array(array: np.ndarray) -> Dict[str, Any]:
    """numpy array -> JSON-safe dict (raw bytes in base64, no pickle)."""
    array = np.ascontiguousarray(array)
    return {'dtype': array.dtype.str, 'shape': list(array.shape),
            'data': base64.b64encode(array.tobytes()).decode('ascii')}


def decode_array(value: Dict[str, Any]) -> np.ndarray:
    """Inverse of encode_array (e.g. for the videos in the server response)."""
    data = base64.b64decode(value['data'])
    return np.frombuffer(data, dtype=np.dtype(value['dtype'])).reshape(value['shape']).copy()


def _to_json(value):
    if isinstance(value, np.ndarray):
        return encode_array(value)
    if isinstance(value, dict):
        return {k: _to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

def format_data_for_server(
    scene_prior_rgb: np.ndarray,
    scene_prior_object_rgb: np.ndarray | None,
    camera_images: Dict[str, Dict[int, np.ndarray]],
    trajectory: np.ndarray,
    prompt: str,
    sample_id: str,
    fps: float = 10.0,
    target_size: tuple = (176, 320),
) -> Dict[str, Any]:
    """
    Format simulation output data for inference server.
    
    Args:
        scene_prior_rgb: (H, W, 3) array of scene prior RGB image
        scene_prior_object_rgb: (H, W, 3) array of scene prior object RGB image
        camera_images: Dict mapping camera names to {timestep: image_array}
        trajectory: (T, 7) array with [xyz(3) + rpy(3) + gripper(1)]
        prompt: Task instruction text
        sample_id: Unique identifier for this sample
        fps: Frame rate
        target_size: (H, W) for resizing images
        
    Returns:
        data_dict: Formatted data ready to send to server
    """
    
    # Map camera names to indices (external_cam1 -> 0, external_cam2 -> 1, wrist_cam -> 2)
    camera_name_to_idx = {
        "external_cam1": 0,
        "external_cam2": 1, 
        "wrist_cam": 2,
    }
    
    # Get number of frames from trajectory
    num_frames = len(trajectory)
    H, W = target_size
    
    # Initialize mask_rgb_per_camera: List[List[np.ndarray]]
    # We'll use the actual camera images as masks (common in this setup)
    mask_rgb_per_camera = [[] for _ in range(3)]  # 3 cameras
    
    # Process each camera
    for cam_name, timestep_dict in camera_images.items():
        if cam_name not in camera_name_to_idx:
            print(f"Warning: Unknown camera {cam_name}, skipping")
            continue
            
        cam_idx = camera_name_to_idx[cam_name]
        
        # Sort timesteps and process frames in order
        for t in sorted(timestep_dict.keys()):
            img = timestep_dict[t]  # (H', W', 3) uint8
            
            # Resize to target size
            img_pil = Image.fromarray(img)
            img_resized = img_pil.resize((W, H), Image.Resampling.LANCZOS)
            img_np = np.array(img_resized).astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            mask_rgb_per_camera[cam_idx].append(img_np)
    
    # Pad any missing frames to match trajectory length
    for cam_idx in range(3):
        while len(mask_rgb_per_camera[cam_idx]) < num_frames:
            # Pad with zeros if frames are missing
            mask_rgb_per_camera[cam_idx].append(np.zeros((H, W, 3), dtype=np.float32))
    
    scene_prior_rgb = (scene_prior_rgb / 255.0).astype(np.float32)  # Normalize to [0, 1]
    if scene_prior_object_rgb is not None:
        scene_prior_object_rgb = (scene_prior_object_rgb / 255.0).astype(np.float32)
    
    # Format trajectory
    trajectory_formatted = trajectory.astype(np.float32)
    
    # Prepare data dict for server
    data_dict = {
        'sample_id': [sample_id],
        'prompt': [prompt],
        'num_frames': num_frames,
        'image_size': np.array([[H, W, H, W]], dtype=np.float32),
        'mask_rgb_per_camera': mask_rgb_per_camera,
        'scene_prior_rgb': scene_prior_rgb,
        'scene_prior_object_rgb': scene_prior_object_rgb,
        'trajectory': trajectory_formatted, # TODO: we seems need to pad the trajectory to 512, 7
        'fps': np.array([fps], dtype=np.float32),
        # Optional parameters
        'cond_frames_raw': 0,  # No conditioning frames
        'seed': 0,
        'used_cams': [0, 2],  # external_cam1 (0) and wrist cam (2)
    }
    
    return data_dict

def send_to_inference_server(
    data_dict: Dict[str, Any],
    server_url: str,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Send formatted data to inference server.
    
    Args:
        data_dict: Formatted data from format_data_for_server()
        server_url: Server URL (e.g., 'http://localhost:5000')
        timeout: Request timeout in seconds
        
    Returns:
        response_dict: Server response with predicted videos
    """
    endpoint = f"{server_url}/predict"
    
    print(f"\n[INFO] Sending request to {endpoint}")
    print(f"  Sample ID: {data_dict['sample_id']}")
    print(f"  Prompt: {data_dict['prompt']}")
    print(f"  Trajectory shape: {data_dict['trajectory'].shape}")
    print(f"  Num cameras: {len(data_dict['mask_rgb_per_camera'])}")
    
    try:
        # Arrays travel as raw bytes inside JSON (see encode_array); frames are sent as one uint8
        # array per camera (they come from uint8 images, so this is lossless and 4x smaller).
        payload = dict(data_dict)
        payload['mask_rgb_per_camera'] = [
            (np.clip(np.stack(frames), 0.0, 1.0) * 255.0).round().astype(np.uint8)
            for frames in data_dict['mask_rgb_per_camera']
        ]
        response = requests.post(
            endpoint,
            json=_to_json(payload),
            timeout=timeout,
        )
        
        response.raise_for_status()
        result = response.json()
        
        print(f"[SUCCESS] Server responded: {result.get('status', 'unknown')}")
        if 'video_shapes' in result:
            print(f"  Video shapes: {result['video_shapes']}")
        
        return result
        
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Could not connect to server at {server_url}")
        print("  Make sure the server is running with:")
        print("  python scripts/server.py --gpu 0 --port 5000")
        raise
    except requests.exceptions.Timeout:
        print(f"[ERROR] Request timed out after {timeout}s")
        raise
    except Exception as e:
        print(f"[ERROR] Request failed: {str(e)}")
        raise