# HTTP inference server: loads RoboDream once and serves generation requests from the generation/ clients.
'''
Usage (inside the container, after `source shell/export.sh`):
    python scripts/server.py --port 5000 --gpu 0

Endpoints:
    GET  /health    model status
    POST /predict   JSON request (arrays as {'dtype', 'shape', 'data': base64 of the raw bytes}),
                    see generation/utils/server_utils.py:format_data_for_server / send_to_inference_server

Request fields:
    mask_rgb_per_camera     [external_cam1, external_cam2, wrist_cam], each a (T, 176, 320, 3) uint8 array
                            (or a list of float frames in [0, 1])
    scene_prior_rgb         (H, W, 3) scene prior of the external camera in [0, 1]
    scene_prior_object_rgb  (optional) (H, W, 3) object prior canvas in [0, 1]
    prompt                  str or [str]
    trajectory              (T, 7) end-effector xyz, rpy, gripper (T <= 512)
    sample_id               str or [str]
    seed, used_cams         optional (default 0 and [0, 2] = external_cam1 + wrist camera)
Response: {'status', 'sample_id', 'videos': {'rgb0' | 'rgb1': array}} with (1, 3, T, H, W) float32
arrays in [0, 1] encoded the same way; with --save_videos the mp4s are written instead.
'''

import argparse
import base64
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request

app = Flask(__name__)
generator = None
args = None


def parse_args():
    ckpt_dir = os.environ.get('ROBODREAM_CKPT_DIR', 'checkpoints')
    data_dir = os.environ.get('ROBODREAM_DATA_DIR', 'datasets')
    parser = argparse.ArgumentParser(description='RoboDream inference server',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt', default=f'{ckpt_dir}/robodream/model.pt', help='RoboDream DiT checkpoint')
    parser.add_argument('--tokenizer', default=f'{ckpt_dir}/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth')
    parser.add_argument('--t5_cache', default=f'{data_dir}/droid_t5_text_embeddings.pkl.gz',
                        help='Cached T5-11B embeddings of DROID instructions')
    parser.add_argument('--default_text', default=f'{ckpt_dir}/robodream/default_text_emb.pt',
                        help='Embedding used for instructions outside the cache (when --t5 is not set)')
    parser.add_argument('--t5', default='', help='Optional T5-11B folder to encode new instructions on the fly')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', default='127.0.0.1', help='Use 0.0.0.0 only on a trusted network')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_steps', type=int, default=35)
    parser.add_argument('--output_dir', default=os.path.join(os.environ.get('ROBODREAM_OUTPUT_DIR', 'outputs'), 'server'))
    parser.add_argument('--save_videos', action='store_true', help='Write mp4s instead of returning arrays')
    return parser.parse_args()


def decode_array(value: dict) -> np.ndarray:
    data = base64.b64decode(value['data'])
    return np.frombuffer(data, dtype=np.dtype(value['dtype'])).reshape(value['shape']).copy()


def encode_array(array: np.ndarray) -> dict:
    array = np.ascontiguousarray(array)
    return {'dtype': array.dtype.str, 'shape': list(array.shape), 'data': base64.b64encode(array.tobytes()).decode('ascii')}


def decode_json(value):
    if isinstance(value, dict) and {'dtype', 'shape', 'data'} <= set(value):
        return decode_array(value)
    if isinstance(value, dict):
        return {k: decode_json(v) for (k, v) in value.items()}
    if isinstance(value, list):
        return [decode_json(v) for v in value]
    return value


def to_unit(array: np.ndarray) -> np.ndarray:
    return array.astype(np.float32) / 255.0 if array.dtype == np.uint8 else array.astype(np.float32)


def first(value):
    return value[0] if isinstance(value, (list, tuple)) else value


def run_request(data: dict) -> dict:
    import torch
    from robodream.vendor import log

    missing = [k for k in ['mask_rgb_per_camera', 'scene_prior_rgb', 'prompt', 'trajectory', 'sample_id'] if k not in data]
    if missing:
        raise ValueError(f'Missing required fields: {missing}')
    used_cams = [int(c) for c in data.get('used_cams', [0, 2])]
    masks = data['mask_rgb_per_camera']
    mask_rgb = np.stack([to_unit(np.stack(masks[cam])) for cam in used_cams])  # (2, T, H, W, 3) in [0, 1]
    prompt = str(first(data['prompt']))
    sample_id = str(first(data['sample_id']))
    seed = int(data.get('seed', 0))
    if int(data.get('cond_frames_raw', 0)) != 0:
        raise ValueError('cond_frames_raw must be 0 (the first clip is always generated from the priors)')
    log.info(f'request {sample_id}: {mask_rgb.shape[1]} frames, prompt={prompt!r}, seed={seed}')

    videos = generator.generate(
        mask_rgb=mask_rgb,
        trajectory=np.asarray(data['trajectory'], dtype=np.float32),
        prompt=prompt,
        scene_prior=to_unit(np.asarray(data['scene_prior_rgb'])),
        object_prior=None if data.get('scene_prior_object_rgb') is None else to_unit(np.asarray(data['scene_prior_object_rgb'])),
        seed=seed,
    )

    if args.save_videos:
        import imageio
        out_dir = os.path.join(args.output_dir, os.path.basename(sample_id.replace('/', '_')))
        os.makedirs(out_dir, exist_ok=True)
        for (k, v) in videos.items():
            frames = (v[0].permute(1, 2, 3, 0).numpy() * 255.0).round().astype(np.uint8)
            imageio.mimwrite(os.path.join(out_dir, f'{k}.mp4'), frames, fps=10, quality=8)
        return {'status': 'success', 'sample_id': sample_id, 'output_dir': out_dir,
                'video_shapes': {k: list(v.shape) for (k, v) in videos.items()}}

    return {'status': 'success', 'sample_id': sample_id,
            'videos': {k: encode_array(v.to(torch.float32).numpy()) for (k, v) in videos.items()}}


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': generator is not None,
                    'gpu_id': args.gpu if args else None, 'device': generator.device if generator else None})


@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': f'Expected application/json, got {request.content_type}'}), 400
    try:
        return jsonify(run_request(decode_json(request.get_json())))
    except Exception as e:
        return jsonify({'error': f'{e}\n{traceback.format_exc()}'}), 500


def main():
    global generator, args
    args = parse_args()
    import torch
    from robodream.generate import RoboDreamGenerator

    torch.cuda.set_device(args.gpu)
    generator = RoboDreamGenerator(
        checkpoint_path=args.ckpt,
        tokenizer_path=args.tokenizer,
        t5_cache_path=args.t5_cache if os.path.isfile(args.t5_cache) else None,
        default_text_path=args.default_text if os.path.isfile(args.default_text) else None,
        t5_path=args.t5 or None,
        device=f'cuda:{args.gpu}',
        num_steps=args.num_steps,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == '__main__':
    main()
