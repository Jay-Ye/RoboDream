# RoboDream data processing

This folder builds the RoboDream training set from raw [DROID](https://droid-dataset.github.io/) in seven stages:
raw DROID → per-episode *Processed* layout → object list → object prior → scene prior → clip lists → T5 text embeddings.
The trainer (`scripts/train.py`) reads 189-frame clips directly from the Processed layout.

Every command runs **inside the RoboDream container** from `/workspace`, after:

```bash
bash shell/run_docker.sh           # on the host (reads .env)
source shell/export.sh             # inside the container
```

`export.sh` sets `ROBODREAM_DATA_DIR`, `ROBODREAM_CKPT_DIR` and `DROID_ROOT`. Every script also accepts CLI overrides; run any script with `--help` to see them.

> **Skip ahead.** We publish the intermediate products on Hugging Face (see [What we release](#what-we-release)). With those you can skip stages 02–04 (the paid VLM calls and the GPU-heavy prior extraction) and only re-pack DROID locally.

## Folder layout under `${ROBODREAM_DATA_DIR}`

```
droid_meta/                      # DROID metadata (huggingface.co/KarlP/droid)
droid_renders/<rel_path>/rgb_sim_rollout_masked/{external_cam1,external_cam2,wrist_cam}/%010d.jpg   # robot-only renders (replay/)
processed/droid_traj2demo/real/
    filtered_episode_meta25.json # episodes with renders + language + camera serials
    split_all.json, split_all_max512.json
    <episode_id>/
        rgb/<cam>/%010d.jpg                      # real frames, 640x360
        robot_moving_sim/<cam>/%010d.jpg         # robot-only render (mask_rgb)
        lowdim/<cam>/%010d.npz                   # proprio (cartesian_position, gripper_position, current_frame) + prompts
        scene_prior_rgb/<cam>/0000000000_omnipaint.jpg           # scene prior (stage 04)
        scene_prior_object_gsam/<cam>/0000000000_gsam.jpg        # object prior canvas (stage 03)
first_frames_processed/{raw,object_masks,edited}/<episode_id>_<cam>.{jpg,png}
object_segmentation_prompts.json # episode_id -> task-relevant objects (stage 02)
clips/clips_{train,val,val_first}.txt  # '<episode_id> <start_frame>' per line (stage 05)
droid_t5_text_embeddings.pkl.gz  # T5-11B embeddings of all DROID instructions
```

Priors are extracted only for the two external cameras. The wrist camera gets a black fallback image.

## Stages

### 00 — DROID metadata and raw episodes

```bash
huggingface-cli download KarlP/droid --repo-type dataset --local-dir ${ROBODREAM_DATA_DIR}/droid_meta
pip install gsutil   # or use the Google Cloud SDK
python data_processing/00_droid/download_droid_raw.py          # -> ${DROID_ROOT}, about 40k episodes, no .svo
python data_processing/00_droid/fix_episode_id_to_path.py      # patch episode_id_to_path.json for renamed folders
```

`download_droid_raw.py` keeps only episodes with good camera extrinsics (IoU > 0.75, reprojection error < 5). The resulting list is checked in as `assets/episodes_with_good_extrinsics.json`.

### 01 — Robot-only renders and conversion to the Processed layout

1. Render robot-only videos by replaying every episode in IsaacLab. See [`replay/README.md`](../replay/README.md). The output goes to `${ROBODREAM_DATA_DIR}/droid_renders` (override with `ROBODREAM_RENDER_DIR` or `--render_root`).
2. Convert and filter:

```bash
python data_processing/01_convert/convert_droid_traj2demo.py --num_workers 64
python data_processing/01_convert/filter_droid_traj2demo.py    # keep 1..512-frame episodes -> split_all_max512.json
```

An episode is kept when it has a language annotation, camera serials and robot-only renders. `--render_root` may also be an `s3://` prefix; renders are then synced locally on demand.

### 02 — Task-object identification (GPT-5-nano)

```bash
pip install openai tiktoken
python data_processing/02_object_identification/identify_objects.py   # uses OPENAI_API_KEY from .env
```

For each episode the script sends the first frame of both external cameras plus all instruction paraphrases to the OpenAI Batch API. It gets back at most 3 task-relevant objects per episode.

- Before submitting, it prints a token and cost estimate and asks for confirmation. Pass `--yes` to skip the prompt.
- If you re-run it, only episodes with empty answers are retried.
- Our output for all episodes is in `assets/object_segmentation_prompts.json`. Copy it to `${ROBODREAM_DATA_DIR}/object_segmentation_prompts.json` to skip this stage.

### 03 — Object prior (Grounded-SAM)

```bash
python data_processing/03_object_prior/segment_objects_of_interest_gsam_dataset.py
```

- Models: Grounding-DINO (`IDEA-Research/grounding-dino-base`) and SAM ViT-H (`facebook/sam-vit-huge`).
- Detection: threshold 0.3. Detections that overlap negative labels (table, sink, …) by more than 0.9 are dropped. Duplicates are merged at IoU 0.98.
- Canvas: the segmented crops are randomly scaled, rotated (±45°) and shuffled onto a blank 640×360 image. The result is written to `scene_prior_object_gsam/`.
- The first frames and binary object masks are also written to `first_frames_processed/{raw,object_masks}` for stage 04.
- The script uses all visible GPUs.

### 04 — Scene prior (OmniPaint object removal)

```bash
bash data_processing/04_scene_prior/setup_omnipaint.sh        # clone + patch + install OmniPaint (pinned commit)
cd data_processing/04_scene_prior/OmniPaint
python main_omnipaint_multi_node.py --claim-mode chunk --chunk-size 1400 --chunks-per-node 100
python construct_scene_prior_folder.py                         # edited/<ep>_<cam>.jpg -> <ep>/scene_prior_rgb/<cam>/0000000000_omnipaint.jpg
cd /workspace
```

- OmniPaint is built on **FLUX.1-dev**, which is gated on Hugging Face. Accept its license and set `HF_TOKEN` in `.env` before running.
- OmniPaint pins older `transformers` and `diffusers` versions. Run this stage in a separate container session, and restart the container (the image is unchanged) before training.
- Multi-node runs coordinate through lock files on the shared `${ROBODREAM_DATA_DIR}`. Launch the same command on each node.

### 05 — Clip lists

```bash
python data_processing/05_clips/make_clip_lists.py   # -> ${ROBODREAM_DATA_DIR}/clips/clips_{train,val,val_first}.txt
```

- **Clips:** each episode is cut into 189-frame windows starting every 20 frames (while the previous window starts before `num_frames - 0.6 * 189`); windows that run past the end repeat the last frame. Frames are resized to 176×320 (Lanczos) on the fly by `robodream/dataset.py`.
- **Splits:** the GuptaLab station is held out for validation; `clips_val_first.txt` keeps only the first window of each validation episode.
- The exact clip lists of the released checkpoint are `meta/split_train_station.txt` / `meta/split_val_station_first_clip.txt` in the priors repo; `scripts/train.py --clips` accepts them as they are.

### 06 — T5 text embeddings

```bash
bash data_processing/06_text_embeddings/run_cache_text_embedding_splits.sh   # 8 GPUs; TOTAL_SPLITS=<n> to change
```

This encodes every DROID instruction with T5-11B (`${ROBODREAM_CKPT_DIR}/google-t5/t5-11b`) and writes `${ROBODREAM_DATA_DIR}/droid_t5_text_embeddings.pkl.gz` (about 3 GB). The file is used both by training (`text_embedding_path`) and by trajectory retrieval in `generation/`.

## Compute notes

These are rough figures for the full DROID subset (about 40k episodes):

| Stage | Cost |
|---|---|
| 01 render | IsaacLab, about 1 GPU-day per 1–2k episodes |
| 03 / 04 priors | A few GPU-days each; OmniPaint (FLUX) is the slowest |
| 02 GPT-5-nano | Small cost via the Batch API |

## Building the training set from released artifacts

This path skips rendering (01.1), VLM calls (02) and prior extraction (03–04). You still need raw DROID (stage 00) for the real frames and trajectories.

```bash
HF=${ROBODREAM_DATA_DIR}/hf
huggingface-cli download Jay-Ye/RoboDream-Data --repo-type dataset --local-dir $HF/data

# robot-only renders -> ${ROBODREAM_RENDER_DIR} (layout written by replay/)
mkdir -p ${ROBODREAM_RENDER_DIR}
for t in $HF/data/renders/robot_only_*.tar; do tar -xf $t -C ${ROBODREAM_RENDER_DIR}; done

# stage 01: real frames + trajectories from DROID, renders from above
python data_processing/01_convert/convert_droid_traj2demo.py --num_workers 64
python data_processing/01_convert/filter_droid_traj2demo.py

# stages 03-04 replaced by the released priors (adds scene_prior_rgb/ and scene_prior_object_gsam/ per episode)
P=${ROBODREAM_DATA_DIR}/processed/droid_traj2demo/real
for t in $HF/data/shards/episode_priors_*.tar; do tar -xf $t -C $P; done
cp $HF/data/meta/droid_t5_text_embeddings.pkl.gz ${ROBODREAM_DATA_DIR}/   # stage 06

# train on the released clip list (no further packing needed)
bash shell/train.sh 8 --clips $HF/data/meta/split_train_station.txt
```

## What we release

These are published on Hugging Face in [`Jay-Ye/RoboDream-Data`](https://huggingface.co/datasets/Jay-Ye/RoboDream-Data):

| Artifact | Lets you skip |
|---|---|
| `object_segmentation_prompts.json` (also in `assets/`) | Stage 02 |
| Per-episode priors: `scene_prior_rgb/*_omnipaint.jpg`, `scene_prior_object_gsam/*_gsam.jpg`, `first_frames_processed/object_masks` | Stages 03–04 |
| `droid_t5_text_embeddings.pkl.gz` | Stage 06 |
| Clip lists of the released checkpoint (`meta/split_*.txt`) | Stage 05 |

Real DROID frames and trajectories are not re-hosted; download them from the official DROID release (stage 00).
