#!/usr/bin/env bash
# Set up the isolated IsaacLab (uv) environment used for trajectory replay / robot-only rendering.
# Run inside the RoboDream container from the repository root:
#   source replay/setup_replay_env.sh
set -e

pushd replay > /dev/null

# 1. uv venv with IsaacLab 2.2 / Isaac Sim 5.0 (Python 3.11)
command -v uv &> /dev/null || pip install uv
# Keep uv's Python inside the repository so the venv survives container restarts; relink a venv
# whose interpreter was lost with an earlier container (installed packages are kept).
export UV_PYTHON_INSTALL_DIR="$PWD/.uv/python"
if [ -f ".venv/bin/activate" ] && ! .venv/bin/python -c "" &> /dev/null; then
    uv venv --allow-existing --python 3.11
fi
if [ ! -f ".venv/bin/activate" ]; then
    uv venv
    source .venv/bin/activate
    # `flatdict` imports `pkg_resources` during build: preinstall an old setuptools and disable build isolation.
    uv pip install -U "pip" "wheel" "setuptools==68.2.2" "hatchling" "wheel_stub"
    uv sync --no-build-isolation
else
    source .venv/bin/activate
fi

# 2. Simulation assets (Franka + Robotiq USD, scenes) from the public sim-evals release
if [ ! -d "assets" ]; then
    curl -O https://storage.googleapis.com/openpi-assets-simeval/env_assets/simple_example_v2/assets.zip
    unzip -q assets.zip && rm assets.zip
fi

# 3. System libraries needed by Isaac Sim rendering
apt-get update && apt-get install -y \
    libxt6 libglu1-mesa libxrandr2 libxinerama1 libxcursor1 libxi6 \
    libxkbcommon0 libxkbcommon-x11-0 mesa-utils vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

popd > /dev/null
echo "Replay environment ready (venv: replay/.venv)."
