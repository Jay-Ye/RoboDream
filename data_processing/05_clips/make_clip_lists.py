# Write the training / validation clip lists ('<episode> <start_frame>' per line) for scripts/train.py.
# Clips are 189-frame windows starting every 20 frames (late windows repeat the last frame), the
# GuptaLab station is held out for validation. The lists used for the released checkpoint are in the
# priors repo (meta/split_train_station.txt, meta/split_val_station_first_clip.txt); scripts/train.py
# reads those directly, so this script is only needed for a different episode set.
#   python data_processing/05_clips/make_clip_lists.py
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from robodream.dataset import WRIST_CAM, episode_clip_starts


def main():
    data_dir = os.environ.get('ROBODREAM_DATA_DIR', 'datasets')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', default=f'{data_dir}/processed/droid_traj2demo/real', help='Processed episodes')
    parser.add_argument('--episodes', default=None,
                        help='Episode split json {episode: [frames]} (default: <root>/split_all_max512.json); '
                             'all folders if missing')
    parser.add_argument('--out_dir', default=f'{data_dir}/clips')
    parser.add_argument('--val_station', default='GuptaLab', help='Lab / station held out for validation')
    args = parser.parse_args()

    split_path = args.episodes or os.path.join(args.root, 'split_all_max512.json')
    if os.path.isfile(split_path):
        with open(split_path) as f:
            split = json.load(f)
        episodes = list(split) if isinstance(split, (dict, list)) else []  # {episode: [frames]} or [episodes]
    else:
        episodes = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    episodes = sorted({os.path.basename(str(e).rstrip('/')) for e in episodes})

    train, val, val_first = [], [], []
    for ep in episodes:
        frames_dir = os.path.join(args.root, ep, 'rgb', WRIST_CAM)
        if not os.path.isdir(frames_dir):
            continue
        starts = episode_clip_starts(len(os.listdir(frames_dir)))
        rows = [f'{ep} {s}' for s in starts]
        if ep.startswith(args.val_station):
            val += rows
            val_first.append(rows[0])
        else:
            train += rows
    os.makedirs(args.out_dir, exist_ok=True)
    for (name, rows) in [('clips_train.txt', train), ('clips_val.txt', val), ('clips_val_first.txt', val_first)]:
        with open(os.path.join(args.out_dir, name), 'w') as f:
            f.write(''.join(r + '\n' for r in rows))
        print(f'{name}: {len(rows)} clips')


if __name__ == '__main__':
    main()
