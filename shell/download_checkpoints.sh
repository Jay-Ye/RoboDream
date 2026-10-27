#!/usr/bin/env bash
# Download base models and RoboDream weights into ${ROBODREAM_CKPT_DIR}. Run inside the container.
#   bash shell/download_checkpoints.sh            # base models + RoboDream
#   bash shell/download_checkpoints.sh --no-t5    # skip T5-11B (only needed to embed new prompts)
set -euo pipefail
source shell/export.sh
HF_ORG=${ROBODREAM_HF_ORG:-Jay-Ye}

huggingface-cli download nvidia/Cosmos-Predict2-2B-Video2World \
    --local-dir "${ROBODREAM_CKPT_DIR}/nvidia/Cosmos-Predict2-2B-Video2World"
if [[ "${1:-}" != "--no-t5" ]]; then
    huggingface-cli download google-t5/t5-11b --local-dir "${ROBODREAM_CKPT_DIR}/google-t5/t5-11b"
fi
huggingface-cli download "${HF_ORG}/RoboDream" --local-dir "${ROBODREAM_CKPT_DIR}/robodream"
