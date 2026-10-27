#!/usr/bin/env bash
# Novel-view generation: re-render the same motion from a newly calibrated camera and pair it with a
# scene prior captured from that camera (paper Fig. 6). Usage:
#   bash generation/scripts/run_novel_view.sh <calibration.json> <scene_prior_from_new_view.png>
set -euo pipefail
source shell/export.sh
source replay/.venv/bin/activate

CALIB=${1:?calibration json of the new viewpoint}
SCENE_PRIOR=${2:?scene prior image captured from the new viewpoint}
PRIOR_DIR=${PRIOR_DIR:-${ROBODREAM_DATA_DIR}/scene_prior_images}
PROPFREE_ROOT=${PROPFREE_ROOT:-${ROBODREAM_DATA_DIR}/propfree}
# In our novel-view calibration files the two external cameras are swapped w.r.t. the default order.
CAM_SERIALS=${CAM_SERIALS:-"37998989 33790348 14064085"}

python generation/prop_free.py \
    --query "Put the yellow block in the cup" \
    --object_prompts "a yellow block.;a cup." \
    --num_samples 2 \
    --data_root "${PROPFREE_ROOT}" \
    --scene_prior_rgb_path "${SCENE_PRIOR}" \
    --object_prior_rgb_path "${PRIOR_DIR}/phone_camera/cube_object_prior.png" \
    --server_url http://localhost:5000 \
    --suffix robodream_novel_view \
    --timeout 300 \
    --cam_calib_path "${CALIB}" \
    --cam_serials ${CAM_SERIALS} \
    --device cuda:0 \
    --dataset_mode prop_free \
    --episode_suffix "new_view_$(basename "${CALIB}" .json)"
