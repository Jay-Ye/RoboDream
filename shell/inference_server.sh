#!/usr/bin/env bash
# Launch one RoboDream inference server (one GPU per server).
#   bash shell/inference_server.sh [GPU_ID] [PORT] [extra scripts/server.py args...]
set -euo pipefail
source shell/export.sh
GPU="${1:-0}"; PORT="${2:-5000}"; shift 2 || true
CUDA_VISIBLE_DEVICES="${GPU}" python scripts/server.py --gpu 0 --port "${PORT}" "$@"
