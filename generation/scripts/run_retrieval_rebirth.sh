#!/usr/bin/env bash
# Retrieval & rebirth (paper setting): retrieve DROID trajectories for each task, replay them from our
# calibrated camera viewpoint, and send them to RoboDream servers together with our scene / object priors.
# Run from the repository root inside the container, with the servers already running
# (bash shell/launch_inference_servers.sh "0 1 2 3 4 5 6").
set -euo pipefail
source shell/export.sh
source replay/.venv/bin/activate

PRIOR_DIR=${PRIOR_DIR:-${ROBODREAM_DATA_DIR}/scene_prior_images}
CALIB=${CALIB:-generation/metadata/calibration_example.json}
NUM_SERVERS=${NUM_SERVERS:-7}
SERVERS=$(for i in $(seq 0 $((NUM_SERVERS - 1))); do printf "http://localhost:%d " $((5000 + i)); done)

python generation/retrieval_rebirth.py \
    --generation_mode retrieval \
    --query \
        "Put the marker in the bowl" \
        "Remove the marker from the bowl and put it on the table" \
    --object_prompts \
        "a marker.;a bowl." \
        "a marker.;a bowl." \
    --num_samples 400 \
    --confidence_threshold 0.7 \
    --scene_prior_dir "${PRIOR_DIR}" \
    --scene_prior_rgb_path "${PRIOR_DIR}/scene_prior_rgb/frame_00053_0301.jpg" \
    --object_prior_rgb_path \
        "${PRIOR_DIR}/put_the_marker_in_the_bowl/put_marker_in_bowl_frame_0275_object_prior.png" \
        "${PRIOR_DIR}/put_the_marker_in_the_bowl/put_marker_in_bowl_frame_0275_object_prior.png" \
    --server_url ${SERVERS} \
    --suffix robodream \
    --timeout 300 \
    --cam_calib_path "${CALIB}" \
    --device cuda:0 \
    --visualize

# Other paper tasks: object priors are built on the fly from the photos in ${PRIOR_DIR}/<task_name>/
# (Grounded-SAM + OmniPaint, cached next to each photo as *_object_prior.png / *_edited.png).
# python generation/retrieval_rebirth.py \
#     --query "Put the yellow block in the cup" "Wipe the table with the towel" \
#     --object_prompts "a yellow block.;a cup." "a towel." \
#     --num_samples 400 --confidence_threshold 0.7 \
#     --scene_prior_dir "${PRIOR_DIR}" \
#     --scene_prior_rgb_path "${PRIOR_DIR}/scene_prior_rgb/frame_00053_0301.jpg" \
#     --server_url ${SERVERS} --suffix robodream --cam_calib_path "${CALIB}" --device cuda:0
