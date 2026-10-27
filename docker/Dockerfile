# RoboDream container (training, inference server, data processing; trajectory replay builds its own
# IsaacLab venv with uv, see replay/README.md).
# Build from the repository root:  bash shell/build_docker.sh
FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN apt-get update && apt-get install -y git git-lfs tree ffmpeg wget unzip libglib2.0-0 freeglut3-dev \
    libvulkan1 libgl1 libegl1 libxt6 libxrandr2 libxinerama1 libxcursor1 libxi6 && rm -rf /var/lib/apt/lists/*
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

COPY docker/requirements-docker.txt /requirements.txt
RUN PIP_CONSTRAINT= pip install --no-cache-dir -r /requirements.txt

# uv creates the isolated IsaacLab environment for trajectory replay
RUN pip install --no-cache-dir uv

RUN mkdir -p /workspace
WORKDIR /workspace
CMD ["/bin/bash"]
