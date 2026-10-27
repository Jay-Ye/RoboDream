# Smoke-test a running inference server with a cached request (a pickle produced by
# format_data_for_server, e.g. saved with retrieval_rebirth.save_data_to_cache).
#   python generation/tools/test_server.py --cache_path <sample>_cached_inputs.pkl [--server_url http://127.0.0.1:5000]
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.server_utils import send_to_inference_server, decode_array
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--cache_path", required=True)
parser.add_argument("--server_url", default="http://127.0.0.1:5000")
parser.add_argument("--out_dir", default="saved_videos")
args = parser.parse_args()

data_dict = pickle.load(open(args.cache_path, "rb"))

response = send_to_inference_server(
    data_dict=data_dict,
    server_url=args.server_url,
    timeout=300,
)

if response.get('status') == 'success':
    import os
    import imageio

    videos = {}
    for view_name, video_info in response['videos'].items():
        video_np = decode_array(video_info)
        videos[view_name] = video_np  # video_np is a np.ndarray, shape=(b, c, T, H, W)
        print(f"Video {view_name} shape: {video_np.shape}")

        # Save the video
        # video_np is (b, c, T, H, W). For most use cases, b=1.
        b, c, T, H, W = video_np.shape
        for idx in range(b):
            out_dir = args.out_dir
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{view_name}_batch{idx}.mp4")
            # Assume c==3, convert (T, H, W, 3) and convert [0,1] or [0,255] float to uint8
            frames = []
            for t in range(T):
                frame = video_np[idx, :, t, :, :]  # (c, H, W)
                frame = np.transpose(frame, (1, 2, 0))  # (H, W, c)
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        frame = frame.clip(0, 255).astype(np.uint8)
                frames.append(frame)
            imageio.mimsave(out_path, frames, fps=10)
            print(f"Saved video to {out_path}")