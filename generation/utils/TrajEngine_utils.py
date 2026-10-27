import os
import imageio
import numpy as np
from PIL import Image
import json
import cv2

from utils.server_utils import decode_array

def save_demonstration(output_demonstration_path, response_videos, camera_images, action, prompt, scene_prior_rgb, scene_prior_object_rgb, episode_id):
    os.makedirs(output_demonstration_path, exist_ok=True)
    assert len(response_videos) == 2, "Only 2 videos are supported for now"

    view_names = ['33790348', '14064085'] #  external_cam1 and wrist_cam

    for i, (view_name, video_info) in enumerate(response_videos.items()):
        video_np = decode_array(video_info) # (1, c, T, H, W), batch is always 1
        out_path = os.path.join(output_demonstration_path, 'recordings', 'SVO')
        os.makedirs(out_path, exist_ok=True)
        # Process batch index 0, since batch=1
        b, c, T, H, W = video_np.shape
        assert b == 1, f"Batch size expected to be 1, but got {b}"
        frames = []
        for t in range(T):
            frame = video_np[0, :, t, :, :]  # (c, H, W)
            frame = np.transpose(frame, (1, 2, 0))  # (H, W, c)
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    frame = frame.clip(0, 255).astype(np.uint8)
            frames.append(frame)
        video_path = os.path.join(out_path, f"{view_names[i]}.mp4")
        imageio.mimsave(video_path, frames, fps=10)

    # Save replay camera images as mp4s in two versions:
    #   robot_only/       — original black background (sent to server)
    #   robot_only_white/ — white background (for paper visualization)
    # Background pixels are pure black (0,0,0) due to the clipping range cutting
    # off the scene at 2 m, so replacement is safe and exact.
    if camera_images is not None:
        robot_only_path = os.path.join(output_demonstration_path, 'robot_only')
        robot_only_white_path = os.path.join(output_demonstration_path, 'robot_only_white')
        os.makedirs(robot_only_path, exist_ok=True)
        os.makedirs(robot_only_white_path, exist_ok=True)
        for cam_name, ts_dict in camera_images.items():
            frames = []
            frames_white = []
            for t in sorted(ts_dict.keys()):
                img_bgr = ts_dict[t]
                frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                img_white = img_bgr.copy()
                img_white[np.all(img_white == 0, axis=2)] = [255, 255, 255]
                frames_white.append(cv2.cvtColor(img_white, cv2.COLOR_BGR2RGB))
            if frames:
                imageio.mimsave(os.path.join(robot_only_path, f"{cam_name}.mp4"), frames, fps=10)
                imageio.mimsave(os.path.join(robot_only_white_path, f"{cam_name}.mp4"), frames_white, fps=10)

    Image.fromarray(scene_prior_rgb).save(os.path.join(output_demonstration_path, 'scene_prior_rgb.png'))
    Image.fromarray(scene_prior_object_rgb).save(os.path.join(output_demonstration_path, 'scene_prior_object_rgb.png'))

    meta_data = {
        'prompt': prompt,
        'episode_id': episode_id,
    }
    with open(os.path.join(output_demonstration_path, 'meta_data.json'), 'w') as f:
        json.dump(meta_data, f, indent=4)