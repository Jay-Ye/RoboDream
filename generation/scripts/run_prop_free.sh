#!/usr/bin/env bash
# Prop-free teleoperation (paper setting): trajectories teleoperated in an empty workspace are replayed and
# RoboDream paints the task objects in. One shared pool of pick-and-place motions is used for all
# pick-and-place tasks (see TASK_TO_DATASET in generation/prop_free.py).
# Expected data: ${PROPFREE_ROOT}/{propfree_pnp,cloth_noobject}/<episode>/trajectory.h5 (DROID format).
set -euo pipefail
source shell/export.sh
source replay/.venv/bin/activate

PRIOR_DIR=${PRIOR_DIR:-${ROBODREAM_DATA_DIR}/scene_prior_images}
PROPFREE_ROOT=${PROPFREE_ROOT:-${ROBODREAM_DATA_DIR}/propfree}
CALIB=${CALIB:-generation/metadata/calibration_example.json}
NUM_SERVERS=${NUM_SERVERS:-7}
SERVERS=$(for i in $(seq 0 $((NUM_SERVERS - 1))); do printf "http://localhost:%d " $((5000 + i)); done)

python generation/prop_free.py \
    --query \
        "Put the marker in the bowl" \
        "Remove the marker from the bowl and put it on the table" \
    --object_prompts \
        "a marker.;a bowl." \
        "a marker.;a bowl." \
    --num_samples -1 \
    --data_root "${PROPFREE_ROOT}" \
    --scene_prior_rgb_path "${PRIOR_DIR}/scene_prior_rgb/frame_00053_0301.jpg" \
    --object_prior_rgb_path \
        "${PRIOR_DIR}/put_the_marker_in_the_bowl/put_marker_in_bowl_frame_0275_object_prior.png" \
        "${PRIOR_DIR}/put_the_marker_in_the_bowl/put_marker_in_bowl_frame_0275_object_prior.png" \
    --server_url ${SERVERS} \
    --suffix robodream \
    --timeout 300 \
    --cam_calib_path "${CALIB}" \
    --device cuda:0 \
    --dataset_mode prop_free
