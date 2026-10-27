#!/usr/bin/env bash
# Source inside the container before training / inference:  source shell/export.sh
export ROBODREAM_ROOT=${ROBODREAM_ROOT:-/workspace}
export ROBODREAM_CKPT_DIR=${ROBODREAM_CKPT_DIR:-${ROBODREAM_ROOT}/checkpoints}
export ROBODREAM_DATA_DIR=${ROBODREAM_DATA_DIR:-${ROBODREAM_ROOT}/datasets}
export ROBODREAM_OUTPUT_DIR=${ROBODREAM_OUTPUT_DIR:-${ROBODREAM_ROOT}/outputs}
export ROBODREAM_RENDER_DIR=${ROBODREAM_RENDER_DIR:-${ROBODREAM_DATA_DIR}/droid_renders}
export DROID_ROOT=${DROID_ROOT:-${ROBODREAM_ROOT}/droid}
export HF_HOME=${HF_HOME:-${ROBODREAM_CKPT_DIR}/hf_cache}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${ROBODREAM_ROOT}:${PYTHONPATH:-}
git config --global --add safe.directory "${ROBODREAM_ROOT}" 2>/dev/null || true
