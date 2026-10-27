import argparse
import json
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Keep episodes with 1..512 frames (split_all.json -> split_all_max512.json).")
parser.add_argument("--processed_dir", default=os.path.join(os.environ.get("ROBODREAM_DATA_DIR", "datasets"), "processed", "droid_traj2demo", "real"))
args = parser.parse_args()

path_to_processed_droid_meta = os.path.join(args.processed_dir, 'split_all.json')
filtered_path_to_save = os.path.join(args.processed_dir, 'split_all_max512.json') # max 512 frames
with open(path_to_processed_droid_meta, 'r') as file:
    processed_droid_meta = json.load(file)

episode_lengths = []
for episode_id, episode_meta in processed_droid_meta.items():
    episode_lengths.append(len(episode_meta))

filtered_processed_droid_meta = {}
for episode_id, episode_meta in processed_droid_meta.items():
    if len(episode_meta) <= 512 and len(episode_meta) > 0:
        filtered_processed_droid_meta[episode_id] = episode_meta
with open(filtered_path_to_save, 'w') as file:
    json.dump(filtered_processed_droid_meta, file, indent=4)

print(f"Number of total episodes: {len(processed_droid_meta)}")
print(f"Number of filtered episodes: {len(filtered_processed_droid_meta)}")

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(episode_lengths, bins=50, edgecolor='black')
plt.xlabel('Episode Length (# of frames)')
plt.ylabel('Frequency')
plt.title('Distribution of Episode Lengths')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(args.processed_dir, 'episode_lengths.png'))