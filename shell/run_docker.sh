#!/usr/bin/env bash
# Launch the RoboDream container. Run from the repository root after `cp .env.example .env`.
#   bash shell/run_docker.sh            # interactive shell
#   bash shell/run_docker.sh <cmd...>   # run a command
set -euo pipefail
REPO_DIR=$(pwd)
[ -f "${REPO_DIR}/.env" ] || { echo "Missing .env (cp .env.example .env)"; exit 1; }
set -a; source "${REPO_DIR}/.env"; set +a

mkdir -p "${CKPT_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}"
MOUNTS=(-v "${REPO_DIR}:/workspace"
        -v "${CKPT_DIR}:/workspace/checkpoints"
        -v "${DATA_DIR}:/workspace/datasets"
        -v "${OUTPUT_DIR}:/workspace/outputs")
[ -n "${DROID_DIR:-}" ] && MOUNTS+=(-v "${DROID_DIR}:/workspace/droid")

# X11 forwarding is optional (only useful for IsaacLab GUI debugging).
X11=()
[ -n "${DISPLAY:-}" ] && X11=(-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix)

TTY=(-it); [ $# -gt 0 ] && TTY=()
docker run "${TTY[@]}" --rm --network host --shm-size=64g --gpus all --runtime=nvidia \
    --name "${CONTAINER_NAME:-robodream}" \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
    -e HF_TOKEN -e WANDB_API_KEY -e WANDB_ENTITY -e WANDB_PROJECT -e OPENAI_API_KEY \
    "${X11[@]}" "${MOUNTS[@]}" \
    "${DOCKER_IMAGE:-robodream:latest}" "$@"
