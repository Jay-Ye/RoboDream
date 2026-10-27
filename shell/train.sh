#!/usr/bin/env bash
# Train RoboDream. Run inside the container from /workspace.
#   bash shell/train.sh [NUM_GPUS] [extra scripts/train.py args...]
# Default: train from the Cosmos-Predict2-2B base model on the released clip list.
# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR (16 GPUs were used for the paper).
set -euo pipefail
source shell/export.sh
NUM_GPUS=${1:-8}; shift || true
INIT=(--init_cosmos "${ROBODREAM_CKPT_DIR}/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-10fps.pt")
[[ " $* " == *" --init "* || " $* " == *" --init_cosmos "* ]] && INIT=()

torchrun --nnodes="${NNODES:-1}" --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT:-22001}" scripts/train.py "${INIT[@]}" "$@"
