#!/usr/bin/env bash
# Set up OmniPaint (object removal) for scene-prior extraction. Run inside the container from /workspace.
#   bash data_processing/04_scene_prior/setup_omnipaint.sh
# Clones OmniPaint at the commit used in the paper, applies our small patch, adds our batch runner and the
# `ObjectRemoval` wrapper package (also used at generation time), downloads OmniPaint weights, and installs it.
# NOTE: OmniPaint builds on FLUX.1-dev, which is gated on Hugging Face: accept its license and set HF_TOKEN in .env.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNIPAINT_DIR="${OMNIPAINT_DIR:-${HERE}/OmniPaint}"
OMNIPAINT_COMMIT=e893e35a5850bce34181ea37f00a9730352c5657

if [ ! -d "${OMNIPAINT_DIR}/.git" ]; then
    git clone https://github.com/yeates/OmniPaint "${OMNIPAINT_DIR}"
fi
cd "${OMNIPAINT_DIR}"
git checkout -q "${OMNIPAINT_COMMIT}"
git apply --check "${HERE}/omnipaint.patch" 2>/dev/null && git apply "${HERE}/omnipaint.patch" || echo "omnipaint.patch already applied"

cp -r "${HERE}/overlay/." "${OMNIPAINT_DIR}/"
cp "${HERE}/main_omnipaint_multi_node.py" "${HERE}/construct_scene_prior_folder.py" "${OMNIPAINT_DIR}/"

bash scripts/setup.sh
pip install "huggingface-hub>=0.23.0,<1.0"
pip install -e "${OMNIPAINT_DIR}"
echo "OmniPaint ready at ${OMNIPAINT_DIR}"
