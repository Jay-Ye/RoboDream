#!/usr/bin/env bash
# Install the retrieval / prior-extraction dependencies into the replay venv.
# Run inside the RoboDream container from the repository root, after `source replay/setup_replay_env.sh`:
#   bash generation/setup_generation_env.sh
set -euo pipefail

source replay/.venv/bin/activate

# T5 retrieval + trajectory loading
uv pip install transformers h5py numpy tqdm sentencepiece requests

# Grounded-SAM object prior (models are pulled from the HF Hub on first use)
uv pip install -e generation/priors/segment_object

# OmniPaint object removal (scene prior). Requires access to the gated FLUX.1-dev model (set HF_TOKEN).
OMNIPAINT_DIR=generation/priors/OmniPaint
OMNIPAINT_COMMIT=e893e35a5850bce34181ea37f00a9730352c5657
if [ ! -d "${OMNIPAINT_DIR}/.git" ]; then
    git clone https://github.com/yeates/OmniPaint "${OMNIPAINT_DIR}"
    git -C "${OMNIPAINT_DIR}" checkout "${OMNIPAINT_COMMIT}"
fi
cp -r generation/priors/omnipaint_overlay/ObjectRemoval "${OMNIPAINT_DIR}/"
cp generation/priors/omnipaint_overlay/pyproject.toml generation/priors/omnipaint_overlay/README_ObjectRemoval.md "${OMNIPAINT_DIR}/"

mkdir -p "${OMNIPAINT_DIR}/weights" "${OMNIPAINT_DIR}/demo_assets/embeddings"
[ -f "${OMNIPAINT_DIR}/weights/omnipaint_remove.safetensors" ] || \
    wget -O "${OMNIPAINT_DIR}/weights/omnipaint_remove.safetensors" https://huggingface.co/yeates/OmniPaint/resolve/main/weights/omnipaint_remove.safetensors
[ -f "${OMNIPAINT_DIR}/demo_assets/embeddings/remove.npz" ] || \
    wget -O "${OMNIPAINT_DIR}/demo_assets/embeddings/remove.npz" https://huggingface.co/yeates/OmniPaint/resolve/main/embeddings/remove.npz
uv pip install -e "${OMNIPAINT_DIR}"

echo "Generation environment ready (venv: replay/.venv)."
