# Generation: retrieval & rebirth and prop-free teleoperation

Two client scripts turn trajectories into RoboDream demonstrations:

| Script | Trajectory source |
|---|---|
| `retrieval_rebirth.py` | trajectories retrieved from DROID by T5 similarity to a task instruction |
| `prop_free.py` | your own DROID-format teleoperation episodes, recorded *without* objects |

Both scripts follow the same steps:
1. **Render the robot.** The trajectory is replayed in IsaacLab (`replay/`) from your calibrated camera, which gives the robot-only video.
2. **Build the priors.** The scene prior (background) and object prior (object canvas) are built from a photo of your scene. Grounded-SAM segments the objects and OmniPaint removes them. You can also pass both priors directly.
3. **Generate.** Everything is sent to one or more RoboDream inference servers, and the generated multi-view demonstrations are saved.

All commands run inside the RoboDream container from the repository root.

## 1. Setup

```bash
source shell/export.sh
source replay/setup_replay_env.sh          # IsaacLab venv (replay/.venv)
bash generation/setup_generation_env.sh    # + T5 retrieval, Grounded-SAM, OmniPaint (needs HF_TOKEN for FLUX.1-dev)
bash shell/download_checkpoints.sh  # Cosmos base, T5-11B, RoboDream weights
```

Data used here, with defaults under `${ROBODREAM_DATA_DIR}`. Each default can be overridden with the environment variable in brackets:

| Path | Content |
|---|---|
| `droid_t5_text_embeddings.pkl.gz` [`TEXT_EMBEDDING_CACHE_PATH`] | T5-11B embeddings of all DROID instructions (HF dataset) |
| `droid_meta/droid_language_annotations.json` [`DROID_LANGUAGE_ANNOTATIONS_PATH`] | DROID language annotations |
| `droid_meta/reformatted_episodes_with_good_extrinsics.json` | DROID camera poses (only used without `--cam_calib_path`) |
| `trajengine/<episode_id>/trajectory.h5` [`DROID_TRAJECTORY_ROOT`] | retrieval pool, built with `retrieval/build_trajectory_root.py` |
| `scene_prior_images/` [`PRIOR_DIR`] | photos and priors of our evaluation scene (HF dataset) |
| `propfree/{propfree_pnp,cloth_noobject}/` [`PROPFREE_ROOT`] | our prop-free teleoperation episodes (HF dataset) |

To build the retrieval pool from raw DROID:

```bash
python generation/retrieval/build_trajectory_root.py --droid_root ${DROID_ROOT} --output_root ${ROBODREAM_DATA_DIR}/trajengine
```

## 2. Launch inference servers

Each server holds one copy of the model on one GPU. You can run them in a second shell in the same container:

```bash
bash shell/launch_inference_servers.sh "0 1 2 3"    # GPUs 0-3 -> http://localhost:5000-5003
curl http://localhost:5000/health
```

Clients take `--server_url` (one or more). `--suffix` controls how requests are spread across servers:
- **One `--suffix`.** Requests are distributed round-robin across the servers.
- **One `--suffix` per URL.** Every request is broadcast to every server, which is useful for comparing checkpoints.

## 3. Retrieval & rebirth

```bash
bash generation/scripts/run_retrieval_rebirth.sh       # paper setting (marker tasks, 400 retrieved trajectories each)
```

Or call the script directly:

```bash
python generation/retrieval_rebirth.py \
    --query "Put the yellow block in the cup" \
    --object_prompts "a yellow block.;a cup." \
    --num_samples 100 --confidence_threshold 0.7 \
    --scene_prior_dir ${ROBODREAM_DATA_DIR}/scene_prior_images \
    --cam_calib_path generation/metadata/calibration_example.json \
    --server_url http://localhost:5000 http://localhost:5001 --suffix robodream --visualize
```

Arguments:
- `--scene_prior_dir`: for each query, a photo is drawn at random from `<scene_prior_dir>/<query_lowercase_with_underscores>/`. Grounded-SAM segments the `--object_prompts` objects to build the object prior, and OmniPaint removes them to build the scene prior. Both priors are cached next to the photo (`*_object_prior.png`, `*_edited.png`).
- `--scene_prior_rgb_path` / `--object_prior_rgb_path`: use ready-made priors instead (one object prior per query).
- `--cam_calib_path`: a DROID-style calibration file (`{"<serial>_left": {"pose": [x, y, z, rx, ry, rz]}, ...}`, see `metadata/calibration_example.json`). Use `--cam_serials` to choose which serials are external_cam1, external_cam2 and wrist_cam. Without this file, each trajectory is rendered from its original DROID camera.
- `--replay_cache_dir`: replays are cached here. They are only re-rendered when the camera pose changes.

## 4. Prop-free teleoperation

Teleoperate the task motions in an empty workspace with the standard DROID stack. Then run:

```bash
bash generation/scripts/run_prop_free.sh
```

Input data is the raw DROID episode format, grouped per motion pool:

```
${PROPFREE_ROOT}/<dataset>/<episode, e.g. Fri_Feb_27_18:51:37_2026>/
    trajectory.h5                       # observation/robot_state/{cartesian_position, gripper_position}
    recordings/SVO/<serial>.mp4         # optional; frame 0 of 33790348.mp4 is the default scene prior
```

`TASK_TO_DATASET` in `prop_free.py` maps each task instruction to a `<dataset>`. The three pick-and-place tasks share one pool. `--dataset_mode real_gen` instead regenerates the observations of real demonstrations collected *with* objects (the "Real w/ Gen Obs" baseline).

## 5. Novel views

Calibrate a new camera pose, capture a scene prior from that viewpoint, and run:

```bash
bash generation/scripts/run_novel_view.sh <new_calibration.json> <scene_prior_from_new_view.png>
```

## Output layout

Each generated demonstration is written to `<output_dir>/<suffix>/<Task_Name>/<episode_id>/` (default `${ROBODREAM_OUTPUT_DIR}/demonstrations`):

```
recordings/SVO/33790348.mp4     # generated third-person view (176x320, 10 fps)
recordings/SVO/14064085.mp4     # generated wrist view
robot_only/<cam>.mp4            # rendered robot-only condition (black background)
robot_only_white/<cam>.mp4      # same, white background (for figures)
scene_prior_rgb.png             # scene prior used
scene_prior_object_rgb.png      # object prior used
meta_data.json                  # {"prompt", "episode_id"}
camera_params.pkl               # camera poses used for rendering
```

Retrieval writes `<output_dir>/retrieval_metadata.json`, which records the similarity, rank and source episode of every retrieved trajectory. To train a policy on generated demonstrations, pair each generated video with its original actions: for retrieved DROID episodes look them up through `episode_id`, for prop-free episodes read them from the teleoperated `trajectory.h5`.

## Priors from your own photos

```bash
python generation/tools/extract_priors.py <photo_or_folder> --object_prompts "a sponge." "a bowl." --prior_type both
python generation/tools/gsam_object_prior.py <photo> "a mouse."               # object prior + mask only
python generation/tools/object_removal.py <photo> merged_mask.png             # scene prior from a mask
python generation/tools/test_server.py --cache_path <sample>_cached_inputs.pkl  # server smoke test
```

Grounding prompts follow the Grounding-DINO convention: lower case, ending with a period. Separate multiple objects with `;`.
