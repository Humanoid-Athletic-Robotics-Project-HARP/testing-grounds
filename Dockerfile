FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# NVIDIA env
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV OMNI_KIT_ACCEPT_EULA=Y
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:1

# System packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git curl vim wget ffmpeg unzip \
    libgl1-mesa-glx libglib2.0-0 \
    libvulkan1 libglu1-mesa libxt6 \
    xvfb libegl1-mesa-dev libosmesa6-dev \
    tigervnc-standalone-server novnc websockify \
    xfce4 xfce4-terminal \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/x-terminal-emulator x-terminal-emulator /usr/bin/xfce4-terminal 50 \
    && update-alternatives --set x-terminal-emulator /usr/bin/xfce4-terminal

RUN mkdir -p /root/.vnc && \
    printf '#!/bin/bash\nunset SESSION_MANAGER\nunset DBUS_SESSION_BUS_ADDRESS\nexec startxfce4\n' > /root/.vnc/xstartup && \
    chmod +x /root/.vnc/xstartup

RUN wget -O /tmp/code.deb "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64" \
    && apt-get update \
    && apt-get install -y xdg-utils bubblewrap socat \
    && apt install -y /tmp/code.deb \
    && rm /tmp/code.deb

# Miniforge/conda
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

WORKDIR /workspace

# Isaac Lab
RUN python3 -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com "isaaclab==2.3.2.post1"
RUN python3 -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com "isaaclab[isaacsim]==2.3.2.post1"
RUN python3 -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com "isaaclab[all]==2.3.2.post1" \
    wandb tensorboard mujoco scipy psutil

# GVHMR + GMR
RUN git clone --depth 1 https://github.com/zju3dv/GVHMR.git /workspace/GVHMR \
    && git clone --depth 1 https://github.com/YanjieZe/GMR.git /workspace/GMR

# gvhmr conda env
RUN /opt/conda/bin/conda create -y -n gvhmr python=3.10 \
    && /opt/conda/bin/conda run --no-capture-output -n gvhmr pip install --no-cache-dir setuptools wheel numpy \
    && /opt/conda/bin/conda run --no-capture-output -n gvhmr pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

RUN /opt/conda/bin/conda run --no-capture-output -n gvhmr pip install --no-cache-dir --no-build-isolation chumpy \
    && /opt/conda/bin/conda run --no-capture-output -n gvhmr pip install --no-cache-dir -r /workspace/GVHMR/requirements.txt

RUN /opt/conda/bin/conda run --no-capture-output -n gvhmr pip install --no-cache-dir -e /workspace/GVHMR \
    && /opt/conda/bin/conda run --no-capture-output -n gvhmr pip install --no-cache-dir -e /workspace/GMR \
    && /opt/conda/bin/conda run --no-capture-output -n gvhmr pip install --no-cache-dir gdown \
    && /opt/conda/bin/conda clean -afy

# Body model symlinks
RUN mkdir -p /workspace/body_models/smplx /workspace/body_models/smpl \
    && mkdir -p /workspace/GVHMR/inputs/checkpoints/body_models \
    && ln -sf /workspace/body_models/smplx /workspace/GVHMR/inputs/checkpoints/body_models/smplx \
    && ln -sf /workspace/body_models/smpl /workspace/GVHMR/inputs/checkpoints/body_models/smpl \
    && mkdir -p /workspace/GMR/assets/body_models \
    && ln -sf /workspace/body_models/smplx /workspace/GMR/assets/body_models/smplx \
    && mkdir -p /workspace/GVHMR/inputs/checkpoints/gvhmr \
    /workspace/GVHMR/inputs/checkpoints/hmr2 \
    /workspace/GVHMR/inputs/checkpoints/vitpose \
    /workspace/GVHMR/inputs/checkpoints/yolo \
    /workspace/GVHMR/inputs/checkpoints/dpvo \
    /workspace/GVHMR/outputs

RUN pip install --no-cache-dir gdown

# Isaac Lab extensions
# RUN gdown --folder "https://drive.google.com/drive/folders/1-CbYxSwqHiHPgpHU2i1dDchH42Jn5EBS" \
RUN gdown "https://drive.google.com/uc?id=1eer7lKgbLyCwK2B6JIKlcyfpGLAf6oTx" -O /tmp/isaaclab_extensions.zip \
    && unzip /tmp/isaaclab_extensions.zip -d /workspace/ \
    && rm /tmp/isaaclab_extensions.zip \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools toml \
    && python3 -m pip install --no-cache-dir -e /workspace/isaaclab_extensions/isaaclab_assets \
    && python3 -m pip install --no-cache-dir -e /workspace/isaaclab_extensions/isaaclab_tasks \
    && python3 -m pip install --no-cache-dir -e "/workspace/isaaclab_extensions/isaaclab_rl[rsl-rl]" \
    && python3 -m pip install --no-cache-dir -e /workspace/isaaclab_extensions/isaaclab_mimic

# HIM collaborators
# https://drive.google.com/file/d/18bY9W4mDD16Y08-dDWoCBveJIzltAuEJ/view?usp=drive_link
# RUN gdown --folder "https://drive.google.com/drive/folders/19A1LaP5VewDfnoBr3dI4lYrCEwvUR5w8" \
RUN gdown "https://drive.google.com/uc?id=18bY9W4mDD16Y08-dDWoCBveJIzltAuEJ" -O /tmp/him-collaborators.zip \
    && unzip /tmp/him-collaborators.zip -d /workspace/ \
    && rm /tmp/him-collaborators.zip \
    && python3 -m pip install --no-cache-dir -e /workspace/him-collaborators/booster_assets \
    && python3 -m pip install --no-cache-dir -e /workspace/him-collaborators/booster_train/source/booster_train

# Pretrained checkpoint patch
RUN echo 'from isaaclab_rl.utils.pretrained_checkpoint import *' > \
    /usr/local/lib/python3.11/dist-packages/isaaclab/source/isaaclab/isaaclab/utils/pretrained_checkpoint.py

# Scripts
RUN mkdir -p /workspace/motions/input \
    && gdown "https://drive.google.com/uc?id=1xv8PV7Xf_3ItbSIoKAWDY98Am7uKkDn6" -O /workspace/setup_weights.sh \
    && gdown "https://drive.google.com/uc?id=1_8-LBxFsZ2b5P5hFOoZp9d9zqOyAeaj2" -O /workspace/mp4_to_csv.sh \
    && chmod +x /workspace/setup_weights.sh /workspace/mp4_to_csv.sh

EXPOSE 6006 8888 5999

CMD ["bash"]
