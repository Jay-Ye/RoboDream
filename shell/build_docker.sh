#!/usr/bin/env bash
# Build the RoboDream image. Run from the repository root.
set -euo pipefail
[ -f .env ] && set -a && source .env && set +a
docker build -t "${DOCKER_IMAGE:-robodream:latest}" -f docker/Dockerfile .
