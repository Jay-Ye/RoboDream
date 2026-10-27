# Demo cases shipped in demos/ and the video layouts of the project page figures.

import json
import os
from typing import Optional

import cv2
import imageio.v3 as iio
import numpy as np
from PIL import Image

H, W = 176, 320
LABEL_H = 44
GAP = 6
COLORS = {  # RGB
    'white': (255, 255, 255), 'gray': (150, 150, 150), 'red': (230, 30, 40), 'blue': (60, 110, 240),
    'green': (40, 180, 70), 'orange': (245, 160, 0),
}


def load_manifest(demo_dir: str) -> dict:
    with open(os.path.join(demo_dir, 'manifest.json')) as f:
        return json.load(f)


def read_video(path: str) -> np.ndarray:
    '''
    :return (T, H, W, 3) uint8 array.
    '''
    return np.asarray(iio.imread(path))


def write_video(path: str, frames: np.ndarray, fps: int = 10) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    frames = np.asarray(frames)
    (h, w) = frames.shape[1:3]
    frames = np.pad(frames, ((0, 0), (0, -h % 8), (0, -w % 8), (0, 0)))  # pad instead of letting the encoder resize
    iio.imwrite(path, frames, fps=fps, codec='libx264', quality=8, macro_block_size=8)


def read_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert('RGB').resize((W, H), Image.Resampling.LANCZOS))


def load_case(demo_dir: str, manifest: dict, name: str) -> dict:
    '''
    :return dict with the generator inputs of one demo case:
        mask_rgb (2, T, H, W, 3) float32 in [0, 1] (external, wrist), trajectory (T, 7), prompt,
        scene_prior / object_prior (H, W, 3) uint8, and the raw uint8 renders for display.
    '''
    case = manifest['cases'][name]
    render_dir = os.path.join(demo_dir, 'renders', case['render'])
    external = read_video(os.path.join(render_dir, 'external.mp4'))
    wrist = read_video(os.path.join(render_dir, 'wrist.mp4'))
    trajectory = np.load(os.path.join(render_dir, 'trajectory.npy'))
    T = len(trajectory)
    (external, wrist) = (external[:T], wrist[:T])
    assert len(external) == len(wrist) == T, f'{name}: {len(external)}, {len(wrist)} frames for {T} trajectory steps'
    return {
        'name': name,
        'mode': case['mode'],
        'label': case.get('label', ''),
        'prompt': case['prompt'],
        'mask_rgb': np.stack([external, wrist]).astype(np.float32) / 255.0,
        'render_external': external,
        'render_wrist': wrist,
        'trajectory': trajectory,
        'scene_prior': read_image(os.path.join(demo_dir, case['scene_prior'])),
        'object_prior': read_image(os.path.join(demo_dir, case['object_prior'])),
    }


def to_uint8(video) -> np.ndarray:
    '''
    (1, 3, T, H, W) float tensor / array in [0, 1] -> (T, H, W, 3) uint8.
    '''
    if hasattr(video, 'detach'):
        video = video.detach().float().cpu().numpy()
    return (np.clip(video[0].transpose(1, 2, 3, 0), 0.0, 1.0) * 255.0).round().astype(np.uint8)


def run_case(generator, case: dict, seed: int = 0) -> dict:
    '''
    :return dict 'external', 'wrist': (T, H, W, 3) uint8 generated videos.
    '''
    out = generator.generate(mask_rgb=case['mask_rgb'], trajectory=case['trajectory'], prompt=case['prompt'],
                             scene_prior=case['scene_prior'], object_prior=case['object_prior'], seed=seed)
    return {'external': to_uint8(out['rgb0']), 'wrist': to_uint8(out['rgb1'])}


def save_result(out_dir: str, name: str, result: dict) -> None:
    for (view, frames) in result.items():
        write_video(os.path.join(out_dir, name, f'{view}.mp4'), frames)


def load_result(out_dir: str, name: str) -> Optional[dict]:
    paths = {v: os.path.join(out_dir, name, f'{v}.mp4') for v in ['external', 'wrist']}
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    return {v: read_video(p) for (v, p) in paths.items()}


# ----------------------------------------------------------------------------------------------
# Layout helpers. A tile is a (T, h, w, 3) uint8 video; still images are repeated over time.
# ----------------------------------------------------------------------------------------------

def _as_video(x: np.ndarray, T: int) -> np.ndarray:
    if x.ndim == 3:
        return np.repeat(x[None], T, axis=0)
    idx = np.minimum(np.arange(T), len(x) - 1)  # hold the last frame of shorter videos
    return x[idx]


def stack(tiles: list, T: int, axis: int) -> np.ndarray:
    '''
    Stack videos / images vertically (axis=1) or horizontally (axis=2).
    '''
    return np.concatenate([_as_video(t, T) for t in tiles], axis=axis)


def border(video: np.ndarray, color: str, width: int = 4) -> np.ndarray:
    out = video.copy()
    c = np.array(COLORS[color], dtype=np.uint8)
    out[:, :width] = c
    out[:, -width:] = c
    out[:, :, :width] = c
    out[:, :, -width:] = c
    return out


def pad(video: np.ndarray, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0) -> np.ndarray:
    return np.pad(video, ((0, 0), (top, bottom), (left, right), (0, 0)))


def label(width: int, text: str, color: str = 'white', height: int = LABEL_H, scale: float = 0.8) -> np.ndarray:
    '''
    :return (height, width, 3) uint8 image with centered text on black.
    '''
    img = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    org = ((width - tw) // 2, (height + th) // 2)
    cv2.putText(img, text, org, font, scale, COLORS[color], 1, cv2.LINE_AA)
    return img


def columns(cols: list, T: int) -> np.ndarray:
    '''
    :param cols: list of (video, label text, label color); videos must share the same height.
    :return video with a label row under the columns and a gap between them.
    '''
    tiles = []
    for (i, (video, text, color)) in enumerate(cols):
        video = _as_video(video, T)
        col = np.concatenate([video, _as_video(label(video.shape[2], text, color), T)], axis=1)
        tiles.append(pad(col, left=GAP if i > 0 else 0))
    return np.concatenate(tiles, axis=2)


def two_views(external: np.ndarray, wrist: np.ndarray, T: int) -> np.ndarray:
    '''
    Wrist camera on top of the external camera, as on the project page.
    '''
    return stack([wrist, external], T, axis=1)


def retrieval_figure(cases: list, results: dict, references: dict, T: Optional[int] = None) -> np.ndarray:
    '''
    One row per task: input priors | generated | retrieved DROID episode.
    '''
    T = T or max(len(results[c['name']]['external']) for c in cases)
    rows = []
    for c in cases:
        r = results[c['name']]
        priors = stack([c['scene_prior'], c['object_prior']], T, axis=1)
        generated = border(two_views(r['external'], r['wrist'], T), 'red')
        retrieved = _as_video(references[c['name']], T)
        rows.append(stack([priors, pad(generated, left=GAP, right=GAP), retrieved], T, axis=2))
        rows.append(np.zeros((T, GAP) + rows[-1].shape[2:], dtype=np.uint8))
    body = np.concatenate(rows[:-1], axis=1)
    w = body.shape[2]
    head = np.concatenate([label(W, 'Input priors'), label(W + 2 * GAP, 'Generated', 'red'), label(w - 2 * W - 2 * GAP, 'Retrieved')], axis=1)
    return np.concatenate([body, _as_video(head, T)], axis=1)


def prop_free_figure(teleop: np.ndarray, cases: list, results: dict, T: Optional[int] = None) -> np.ndarray:
    '''
    Prop-free teleoperation recording | demonstrations generated from it with different object priors.
    '''
    T = T or len(teleop)
    top = [_as_video(teleop, T)]
    bottom = [np.zeros((H, W, 3), np.uint8)]
    for c in cases:
        top.append(pad(two_views(results[c['name']]['external'], results[c['name']]['wrist'], T), left=GAP))
        bottom.append(np.pad(c['object_prior'], ((0, 0), (GAP, 0), (0, 0))))
    body = stack(top, T, axis=2)
    priors = pad(np.concatenate(bottom, axis=1)[None], top=GAP)[0]
    head = np.concatenate([label(W, 'Prop-free teleop'), label(body.shape[2] - W, 'Generated demonstrations', 'red')], axis=1)
    return np.concatenate([body, _as_video(priors, T), _as_video(head, T)], axis=1)


def compositional_figure(base: dict, variants: list, results: dict, T: Optional[int] = None) -> np.ndarray:
    '''
    Same trajectory, different priors. Top: robot-only render and generated videos;
    bottom: the priors that change in each column.
    :param variants: list of (case, label, color, prior key shown below).
    '''
    T = T or len(base['render_external'])
    top = [two_views(base['render_external'], base['render_wrist'], T)]
    bottom = [base['scene_prior']]
    names = [('Input priors', 'white')]
    for (case, text, color, prior_key) in variants:
        r = results[case['name']]
        top.append(border(two_views(r['external'], r['wrist'], T), color))
        bottom.append(border(case[prior_key][None], color)[0])
        names.append((text, color))
    cols = []
    for (t, b, (text, color)) in zip(top, bottom, names):
        cols.append((np.concatenate([_as_video(t, T), _as_video(pad(b[None], top=GAP)[0], T)], axis=1), text, color))
    return columns(cols, T)
