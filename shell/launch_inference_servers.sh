#!/usr/bin/env bash
# Launch one inference server per GPU on consecutive ports (5000, 5001, ...).
#   bash shell/launch_inference_servers.sh "0 1 2 3" [extra scripts/server.py args...]
set -euo pipefail
GPUS=(${1:-0}); shift || true
source shell/export.sh
mkdir -p "${ROBODREAM_OUTPUT_DIR}/server_logs"
for i in "${!GPUS[@]}"; do
    port=$((5000 + i))
    log="${ROBODREAM_OUTPUT_DIR}/server_logs/server_${port}.log"
    nohup bash shell/inference_server.sh "${GPUS[$i]}" "$port" "$@" > "$log" 2>&1 &
    echo "GPU ${GPUS[$i]} -> http://localhost:${port} (log: $log)"
done
