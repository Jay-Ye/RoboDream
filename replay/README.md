# Replay: robot-only rendering in IsaacLab

This folder replays DROID end-effector trajectories in IsaacLab (Franka + Robotiq 2F-85) and renders the
robot alone, on a black background, from calibrated camera poses. These **robot-only (anchor) videos**
are the motion condition of RoboDream:

- **Training data.** `render_droid_robot_only.py` renders every DROID episode that has good extrinsics. Its output becomes the `mask_rgb` modality.
- **Deployment.** `generation/retrieval_rebirth.py` and `generation/prop_free.py` re-render retrieved or teleoperated trajectories from a *new* camera, and pair them with new scene and object priors.

The code is adapted from [sim-evals](https://github.com/arhanjain/sim-evals): the environments in
`src/environments/` and the simulation assets. A camera clipping range of 2 m removes the scene, so
only the robot is visible (`src/environments/droid_environment.py`).

## Setup

Run this inside the RoboDream container, from the repository root:

```bash
source replay/setup_replay_env.sh   # uv venv with isaaclab 2.2 / isaacsim 5.0 (py3.11), assets, apt deps
```

The script builds an isolated `replay/.venv`, so it does not conflict with the training environment.
Simulation assets are downloaded to `replay/assets/` from the public sim-evals bucket.

## Metadata

| File | Source | Used by |
|---|---|---|
| `episode_id_to_path.json`, `intrinsics.json`, `camera_serials.json`, `droid_language_annotations.json` | public DROID metadata ([KarlP/droid](https://huggingface.co/KarlP/droid)) | rendering, retrieval |
| `pnp_cam2base_multiview.json` (26 MB) | per-episode refined camera-to-base extrinsics (RoboDream HF dataset) | `render_droid_robot_only.py` |
| `reformatted_episodes_with_good_extrinsics.json` (78 MB) | the same extrinsics re-keyed by camera name, plus intrinsics (RoboDream HF dataset) | `generation/retrieval_rebirth.py` when no `--cam_calib_path` is given |

Put all of these files in `${ROBODREAM_DATA_DIR}/droid_meta/`. You can override that location with `DROID_META_DIR`.

## Render robot-only videos for DROID

```bash
source shell/export.sh && source replay/.venv/bin/activate
cd replay
python render_droid_robot_only.py --num_envs 2                         # all episodes
python render_droid_robot_only.py --num_envs 2 --shard_id 0 --num_shards 8   # one shard of 8 (e.g. one per GPU)
```

The script works as follows:
- **Input.** It reads the raw DROID release (`${DROID_ROOT}/<relative_path>/trajectory.h5` and `recordings/MP4/<serial>.mp4`).
- **Skipped episodes.** Episodes without extrinsics, failures, and episodes shorter than 45 or longer than 512 steps are skipped.
- **Resuming.** Episodes that were already rendered are skipped, so it is safe to resume.

Output layout (`ROBODREAM_RENDER_DIR`, default `${ROBODREAM_DATA_DIR}/droid_renders`):

```
<ROBODREAM_RENDER_DIR>/<relative_path>/
    rgb_sim_rollout_masked/{external_cam1,external_cam2,wrist_cam}/%010d.jpg   # robot-only frames, 1280x720
    rgb_real_rollout_masked/{external_cam1,external_cam2,wrist_cam}/%010d.jpg  # real frames masked by the sim robot (diagnostic; skip with --skip_real_obs)
```

`<relative_path>` is the episode's DROID path (`<lab>/success/<date>/<timestamp>`). The data-processing
conversion step reads `rgb_sim_rollout_masked/` and writes it as `robot_moving_sim/<cam>/%010d.jpg` in the
per-episode processed layout.

Use `--video_dir` to also write one preview mp4 per episode.

## Python API

```python
from replay.replay_functions import create_replay_context, close_context
from replay.replay import batched_replay

ctx = create_replay_context(headless=True, scene=1, num_envs=N, env_name="DROID_EE")
outputs = batched_replay(ctx, batch_store, control_mode="end_effector")   # outputs["camera_images"][b][cam][t] -> BGR image
close_context(ctx)
```

`batch_store["actions"]` takes trajectories as `[x, y, z, qw, qx, qy, qz, gripper]`.
`batch_store["camera_params"][b][cam]` takes `{"camera_pose": 4x4 or None, "intrinsics": 3x3}`; pass `None` for the wrist camera, since it follows the gripper.
