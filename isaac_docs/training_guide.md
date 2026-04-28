# K1 Training Guide

End-to-end guide for training a Booster K1 dance/motion policy using the `humanoid-dev` Docker image. Covers everything from building the Docker image locally, deploying to RunPod, converting MP4 videos to robot motions, training with Isaac Lab, monitoring progress, and exporting the final model.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Building the Docker Image](#building-the-docker-image)
3. [Deploying to RunPod](#deploying-to-runpod)
4. [Running Locally with GPU](#running-locally-with-gpu)
5. [Option A: Start from an MP4 Video](#option-a-start-from-an-mp4-video-recommended)
6. [Option B: Start from a Pre-Made CSV](#option-b-start-from-a-pre-made-csv)
7. [Step 1: Copy CSV into booster_assets](#step-1-copy-csv-into-booster_assets)
8. [Step 2: Convert CSV to NPZ](#step-2-convert-csv-to-npz)
9. [Step 3: Append Standing Frames](#step-3-append-standing-frames)
10. [Step 4: Create the Training Task Config](#step-4-create-the-training-task-config)
11. [Step 5: Train](#step-5-train)
12. [Monitoring Training Progress](#monitoring-training-progress)
13. [Step 6: Export the Trained Model](#step-6-export-the-trained-model)
14. [Docker Image Maintenance](#docker-image-maintenance)
15. [Troubleshooting](#troubleshooting)
16. [Quick Reference](#quick-reference)

---

## Architecture Overview

The Docker image is split into two layers for fast rebuilds:

| Layer | Image | Contents | Rebuild time |
|-------|-------|----------|--------------|
| Base | `humanoid-dev:base` | CUDA 12.4, Python 3.11, Isaac Lab 2.3.2, Isaac Sim, wandb, tensorboard, mujoco | ~30 min (rare) |
| Thin | `humanoid-dev:latest` | GVHMR, GMR (Python 3.10 conda env), Isaac Lab extensions, booster_train, booster_assets, wrapper scripts | ~10-15 min first build, seconds if cached |

**Key components inside the container:**

| Component | Python | Purpose |
|-----------|--------|---------|
| Isaac Lab + Isaac Sim | 3.11 (system) | Physics simulation, RL training environment |
| booster_train | 3.11 (system) | K1 training scripts, task configs, reward functions |
| booster_assets | 3.11 (system) | K1 robot model, motion files (CSV/NPZ) |
| GVHMR | 3.10 (conda `gvhmr`) | Extracts 3D human pose from monocular video |
| GMR | 3.10 (conda `gvhmr`) | Retargets human motion to K1 robot joints |
| wandb + TensorBoard | 3.11 (system) | Training monitoring and visualization |

**Container filesystem layout:**

```
/workspace/
├── GVHMR/                        # Human pose extraction
├── GMR/                          # Motion retargeting to K1
├── body_models/                  # SMPLX/SMPL body models (upload manually)
│   ├── smplx/
│   └── smpl/
├── him-collaborators/
│   ├── booster_assets/           # Robot model + motion files
│   │   └── motions/K1/           # Put your CSV/NPZ files here
│   └── booster_train/
│       ├── scripts/              # csv_to_npz.py, train.py, play.py, etc.
│       └── source/booster_train/ # Task configs, rewards, environments
├── isaaclab_extensions/          # Isaac Lab task/asset/RL extensions
├── motions/input/                # Drop zone for MP4s and generated CSVs
├── mp4_to_csv.sh                 # MP4 → K1 CSV pipeline script
└── setup_weights.sh              # Model weight download helper
```

---

## Building the Docker Image

### Prerequisites

- Docker Desktop with WSL2 backend (Windows) or Docker Engine (Linux)
- NVIDIA Container Toolkit (`nvidia-docker`) for GPU access
- ~50 GB disk space for the full image

### Step 1: Build the base image (rare, only when Isaac Lab version changes)

```bash
cd C:\Users\kylel\Humanoid
docker build -t humanoid-dev:base -f Dockerfile.base .
```

This downloads Isaac Lab + Isaac Sim (~8 GB) and takes ~30 minutes. You only need to do this once unless `requirements.txt` or system packages change.

### Step 2: Build the thin image (fast after first build)

```bash
docker build -t humanoid-dev:latest -f Dockerfile .
```

First build takes ~10-15 minutes (downloads GVHMR/GMR dependencies, PyTorch for conda env). Subsequent builds are fast (~seconds) because Docker caches the heavy layers.

### Layer caching strategy

The Dockerfile is ordered so that the most stable layers are at the top:

1. System packages (cached unless Dockerfile changes)
2. Miniforge + conda env creation (cached unless Dockerfile changes)
3. GVHMR/GMR git clone + pip install (cached unless Dockerfile changes)
4. Body model directories + symlinks (cached)
5. Isaac Lab extensions COPY + install (rebuilds if extension source changes)
6. HIM collaborators COPY + install (rebuilds if booster_train/assets change)
7. Wrapper scripts COPY (rebuilds if mp4_to_csv.sh or setup_weights.sh change)

Changing a wrapper script only rebuilds step 7 (seconds). Changing booster_train code rebuilds steps 6-7 (~1-2 minutes). The expensive conda/GVHMR layers stay cached.

---

## Deploying to RunPod

### Step 1: Push the image to a registry

```bash
# Tag for Docker Hub (replace with your username)
docker tag humanoid-dev:latest yourusername/humanoid-dev:latest

# Push
docker push yourusername/humanoid-dev:latest
```

For large images, consider using a registry closer to RunPod's data centers (e.g., Docker Hub, GitHub Container Registry, or a private registry).

### Step 2: Create a RunPod template

1. Go to [runpod.io](https://runpod.io) → **Templates** → **New Template**
2. Fill in:
   - **Template Name:** `humanoid-dev`
   - **Container Image:** `yourusername/humanoid-dev:latest`
   - **Container Disk:** `50 GB` (minimum, more if you'll store many training runs)
   - **Volume Disk:** `50 GB` (optional, for persistent storage across pod restarts)
   - **Expose HTTP Ports:** `6006` (TensorBoard), `8888` (Jupyter, optional)
   - **Expose TCP Ports:** `22` (SSH)
   - **Docker Command:** leave empty (defaults to bash)
   - **Environment Variables:**
     - `OMNI_KIT_ACCEPT_EULA` = `Y`
     - `WANDB_API_KEY` = `your-key-here` (get from [wandb.ai/authorize](https://wandb.ai/authorize))
     - `PUBLIC_KEY` = `your-ssh-public-key` (for SSH access)

### Step 3: Launch a pod

1. Go to **Pods** → **Deploy**
2. Select your template
3. Choose a GPU:

| GPU | VRAM | Recommended `--num_envs` | ~Time for 50k iterations |
|-----|------|--------------------------|--------------------------|
| RTX 4090 | 24 GB | 4096 | ~6-10 hours |
| A100 | 40-80 GB | 4096 | ~3-6 hours |
| H100 | 80 GB | 4096-8192 | ~2-3 hours |

4. Click **Deploy**

### Step 4: Connect via SSH

Once the pod is running, RunPod provides SSH connection details:

```bash
ssh root@<POD_IP> -p <SSH_PORT> -i ~/.ssh/id_ed25519
```

### Persistent storage tips

- Mount a **network volume** to `/workspace/logs` to persist training checkpoints across pod restarts
- Model weights (SMPLX, GVHMR checkpoints) should be stored on the network volume to avoid re-downloading
- Use `rsync` or `scp` to transfer files between pods

### Transferring files with `runpodctl` (Recommended)

RunPod's SSH proxy does not support SCP/SFTP, and the exposed TCP ports can take several minutes to come online after pod startup (you'll see "Connection refused" errors). **`runpodctl`** is a peer-to-peer transfer tool that works immediately and doesn't depend on SSH.

**Install `runpodctl` on your local machine:**

Windows (PowerShell):
```powershell
Invoke-WebRequest -Uri "https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.exe" -OutFile "runpodctl.exe"
```

macOS/Linux:
```bash
curl -fsSL https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-$(uname -s | tr '[:upper:]' '[:lower:]')-amd64 -o runpodctl
chmod +x runpodctl
```

**Upload a file to the pod:**

```bash
# On your local machine
runpodctl send my_video.mp4

# It prints a receive code like: 4571-fiber-office-oberon-9
# On the pod (web terminal or SSH)
cd /workspace/motions/input
runpodctl receive 4571-fiber-office-oberon-9
```

**Download a file from the pod:**

```bash
# On the pod
runpodctl send /workspace/motions/input/my_dance_preview.mp4

# On your local machine
runpodctl receive <CODE>
```

`runpodctl` is pre-installed on RunPod pods. Transfers are direct peer-to-peer, so speed depends on your internet connection (typically 1-5 MB/s).

---

## Running Locally with GPU

If you have an NVIDIA GPU locally (e.g., RTX 4070, 3090), you can run the full pipeline on your machine.

### Requirements

- Docker Desktop with WSL2 backend
- NVIDIA Container Toolkit installed
- NVIDIA GPU drivers installed

### Start the container

```bash
docker run -it --gpus all \
  -e OMNI_KIT_ACCEPT_EULA=Y \
  -p 6006:6006 \
  humanoid-dev:latest
```

This gives you an interactive shell inside the container with GPU access. Port 6006 is forwarded for TensorBoard.

### Verify GPU access

```bash
nvidia-smi
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

### Getting files into the container

```bash
# From another terminal on your host machine:
docker cp my_video.mp4 <container_id>:/workspace/motions/input/
docker cp my_dance.csv <container_id>:/workspace/motions/input/

# Or use docker run with a volume mount:
docker run -it --gpus all \
  -e OMNI_KIT_ACCEPT_EULA=Y \
  -v C:\Users\kylel\my_motions:/workspace/motions/input \
  -p 6006:6006 \
  humanoid-dev:latest
```

---

## Option A: Start from an MP4 Video (Recommended)

The fastest path: upload an MP4 video of a person performing the motion, and the pipeline automatically extracts the human pose (GVHMR) and retargets it to the K1 robot (GMR).

### First-Time Setup: Download Model Weights

Run this once when your pod first starts:

```bash
bash /workspace/setup_weights.sh
```

This downloads GVHMR checkpoints (~2 GB from Google Drive). If the automatic download fails (Google Drive rate limits), the script prints manual download instructions.

You also need SMPLX/SMPL body models which require free registration:

| Model | Registration | Required Files | Destination |
|-------|-------------|----------------|-------------|
| SMPLX | [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) | `SMPLX_NEUTRAL.npz`, `SMPLX_NEUTRAL.pkl` | `/workspace/body_models/smplx/` |
| SMPL | [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) | `SMPL_NEUTRAL.pkl` | `/workspace/body_models/smpl/` |

Upload them to the container:

```bash
# From your local machine
scp -P <SSH_PORT> SMPLX_NEUTRAL.npz root@<POD_IP>:/workspace/body_models/smplx/
scp -P <SSH_PORT> SMPLX_NEUTRAL.pkl root@<POD_IP>:/workspace/body_models/smplx/
scp -P <SSH_PORT> SMPL_NEUTRAL.pkl  root@<POD_IP>:/workspace/body_models/smpl/
```

Run `setup_weights.sh` again to verify everything shows `[OK]`.

### Upload Your MP4

```bash
# Option 1: runpodctl (recommended, see "Transferring files" section above)
# On your local machine:
runpodctl send dance_video.mp4
# On the pod:
cd /workspace/motions/input && runpodctl receive <CODE>

# Option 2: SCP (requires exposed TCP port)
scp -P <SSH_PORT> dance_video.mp4 root@<POD_IP>:/workspace/motions/input/

# Option 3: Download from a URL inside the pod
wget -O /workspace/motions/input/dance_video.mp4 "https://your-link-here"
```

**Tips for best results:**
- Static camera works best (the pipeline uses `-s` flag by default)
- Good lighting, single person clearly visible
- Trim the video to just the relevant motion section
- 10-60 second clips work well

### Verify Your Video Format

**Always verify the file is an actual video before processing.** Two common pitfalls:

1. **iPhone / QuickTime videos** are often MOV/HEVC format, which GVHMR's pyav cannot read. The pipeline auto-converts these to H.264, but if you're running steps manually you'll need to convert first.
2. **Cloud downloads (Google Drive, Dropbox, etc.)** sometimes save the HTML login/confirmation page instead of the actual file, especially with expired or auth-gated links.

Check your file on the pod:

```bash
file /workspace/motions/input/my_video.mp4
```

| `file` output | Meaning | Action |
|---------------|---------|--------|
| `ISO Media, MP4 Base Media v1` | Valid MP4 (H.264) | Good to go |
| `ISO Media, Apple QuickTime movie` | MOV/HEVC format | Convert with ffmpeg (see below) |
| `HTML document, Unicode text` | Not a video -- web page saved as .mp4 | Re-download the actual file |

If your video is MOV/HEVC, convert it:

```bash
ffmpeg -i /workspace/motions/input/my_video.mp4 \
  -c:v libx264 -preset fast -crf 23 \
  /workspace/motions/input/my_video_h264.mp4
```

Then use `my_video_h264.mp4` as input. The `mp4_to_csv.sh` script handles this conversion automatically, but if you run GVHMR manually you must convert first.

### Run the MP4-to-CSV Pipeline

```bash
bash /workspace/mp4_to_csv.sh /workspace/motions/input/dance_video.mp4 my_dance
```

This runs 4 steps automatically:

| Step | Tool | What it does |
|------|------|-------------|
| 1 | GVHMR | Extracts 3D human body pose (SMPLX) from the video |
| 2 | GMR | Retargets the human motion to K1 robot joints (22 DOF) |
| 3 | GMR | Generates a preview video of the K1 robot performing the motion |
| 4 | Python | Converts GMR output to HIM-compatible K1 CSV format |

Output files:
- `/workspace/motions/input/my_dance.csv` -- K1 motion CSV (29 columns)
- `/workspace/motions/input/my_dance_preview.mp4` -- Visual preview of the robot

### Visual Check: Review Before Training

**Always review the preview video before starting a multi-hour training run.** Download it:

```bash
# Option 1: runpodctl (recommended, works immediately)
# On the pod:
runpodctl send /workspace/motions/input/my_dance_preview.mp4
# On your local machine:
runpodctl receive <CODE>

# Option 2: SCP (requires exposed TCP port to be active)
scp -P <SSH_PORT> root@<POD_IP>:/workspace/motions/input/my_dance_preview.mp4 .
```

Things to check in the preview:
- Does the robot perform the expected motion?
- Are there any limbs clipping through the body?
- Is the motion speed reasonable?
- Are there frames where the robot falls or enters unnatural poses?

If the motion looks wrong, try:
- A different source video (better lighting, clearer person, static camera)
- Trim the video to the relevant section before processing
- A shorter clip (less chance of tracking errors accumulating)

To skip preview generation (faster, for headless environments):

```bash
bash /workspace/mp4_to_csv.sh /workspace/motions/input/dance_video.mp4 my_dance --skip-preview
```

If the preview looks good, continue to [Step 1](#step-1-copy-csv-into-booster_assets).

---

## Option B: Start from a Pre-Made CSV

If you already have a K1-format CSV (from another retargeting tool, motion capture, etc.), skip the MP4 pipeline above.

### CSV Format

Your input CSV must follow the K1 motion format. No header row. One row per frame. Each row = 29 comma-separated numbers.

| Columns | Data | Units |
|---------|------|-------|
| 1-3 | Base position: `x, y, z` | meters |
| 4-7 | Base orientation quaternion: `qx, qy, qz, qw` | scalar-last |
| 8-29 | Joint positions (22 joints, order below) | radians |

**K1 joint order (columns 8-29):**

| Col | Joint | Col | Joint |
|-----|-------|-----|-------|
| 8 | AAHead_yaw | 19 | Left_Ankle_Roll |
| 9 | Head_pitch | 20 | Right_Hip_Pitch |
| 10 | ALeft_Shoulder_Pitch | 21 | Right_Hip_Roll |
| 11 | Left_Shoulder_Roll | 22 | Right_Hip_Yaw |
| 12 | Left_Elbow_Pitch | 23 | Right_Knee_Pitch |
| 13 | Left_Elbow_Yaw | 24 | Right_Ankle_Pitch |
| 14 | ARight_Shoulder_Pitch | 25 | Right_Ankle_Roll |
| 15 | Right_Shoulder_Roll | 26-29 | (unused, pad with 0) |
| 16 | Right_Elbow_Pitch | | |
| 17 | Right_Elbow_Yaw | | |
| 18 | Left_Hip_Pitch | | |

Full order: `AAHead_yaw, Head_pitch, ALeft_Shoulder_Pitch, Left_Shoulder_Roll, Left_Elbow_Pitch, Left_Elbow_Yaw, ARight_Shoulder_Pitch, Right_Shoulder_Roll, Right_Elbow_Pitch, Right_Elbow_Yaw, Left_Hip_Pitch, Left_Hip_Roll, Left_Hip_Yaw, Left_Knee_Pitch, Left_Ankle_Pitch, Left_Ankle_Roll, Right_Hip_Pitch, Right_Hip_Roll, Right_Hip_Yaw, Right_Knee_Pitch, Right_Ankle_Pitch, Right_Ankle_Roll`

### Get Your CSV onto the Pod

```bash
# SCP from local machine
scp -P <SSH_PORT> my_dance.csv root@<POD_IP>:/workspace/motions/input/

# Or download from URL
wget -O /workspace/motions/input/my_dance.csv "https://your-link-here"

# Or use docker cp for local testing
docker cp my_dance.csv <container_id>:/workspace/motions/input/
```

Then continue to [Step 1](#step-1-copy-csv-into-booster_assets).

---

## Step 1: Copy CSV into booster_assets

Whether you generated the CSV from an MP4 (Option A) or have a pre-made CSV (Option B), copy it into the assets directory:

```bash
DANCE=my_dance

cp /workspace/motions/input/${DANCE}.csv \
   /workspace/him-collaborators/booster_assets/motions/K1/${DANCE}.csv
```

---

## Step 2: Convert CSV to NPZ

The trainer uses NPZ files (replayed through Isaac Sim to capture full body state), not raw CSV.

```bash
cd /workspace/him-collaborators/booster_train

python3 scripts/csv_to_npz.py \
  --headless \
  --input_file ../booster_assets/motions/K1/${DANCE}.csv \
  --input_fps 30 \
  --output_fps 50 \
  --output_name ../booster_assets/motions/K1/${DANCE}.npz
```

| Flag | Description |
|------|-------------|
| `--headless` | Required for running without a display |
| `--input_fps` | FPS of your CSV (30 for MP4 pipeline output, check your source otherwise) |
| `--output_fps` | Always use 50 (standard for all training tasks) |
| `--frame_range START END` | Optional. Clip bad frames (1-indexed, inclusive) |

The script will print the number of frames processed. If it fails, check that the CSV path is correct and the format matches the expected 29-column layout.

---

## Step 3: Append Standing Frames

Adds 3 seconds of standing at the end of the motion so the policy learns to hold a stable final pose.

```bash
python3 scripts/append_standing_frames.py \
  --input ../booster_assets/motions/K1/${DANCE}.npz \
  --output ../booster_assets/motions/K1/${DANCE}_extended.npz \
  --seconds 3.0
```

---

## Step 4: Create the Training Task Config

The training system uses Gymnasium task registrations. Each motion needs a task config that points to its NPZ file.

### Quick start: Use the existing `custom_dance` template

If you just want to train quickly without creating a new task, edit the existing template to point at your NPZ file:

```bash
TASK_DIR=source/booster_train/booster_train/tasks/manager_based/beyond_mimic/robots/k1
```

Edit `${TASK_DIR}/custom_dance/env_cfg.py`, line 27:

```python
MOTION_FILE = f"{BOOSTER_ASSETS_DIR}/motions/K1/my_dance_extended.npz"
```

Then use task ID `Booster-K1-Custom_Dance_A-v0` for training. Skip to [Step 5](#step-5-train).

### Creating a new task (for organizing multiple motions)

Copy the existing `custom_dance` template:

```bash
cp -r ${TASK_DIR}/custom_dance ${TASK_DIR}/${DANCE}
```

Edit the files in your new directory:

**`env_cfg.py`** -- Change the motion file path:

```python
MOTION_FILE = f"{BOOSTER_ASSETS_DIR}/motions/K1/{DANCE}_extended.npz"
```

**`ppo_cfg.py`** -- Change the experiment name:

```python
@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 50000
    experiment_name = "k1_my_dance"
```

**`__init__.py`** -- Update the Gym task IDs. Replace `Custom_Dance` with your dance name throughout. The task IDs follow the pattern `Booster-K1-<Name>_<Variant>-v0`. The template has 4 variants (A through D) with different reward configurations:

| Variant | Description |
|---------|-------------|
| A | Baseline rewards (dance_004 style) |
| B | Anchor boost (stronger root tracking) |
| C | Anchor + velocity boost |
| D | Full locomotion + relaxed terminations |

Start with variant A. Try B or D if A doesn't produce good results.

**Verify registration:**

```bash
python3 scripts/list_envs.py | grep -i ${DANCE}
```

Your new task should appear in the output.

---

## Step 5: Train

```bash
cd /workspace/him-collaborators/booster_train

python3 scripts/rsl_rl/train.py \
  --task Booster-K1-Custom_Dance_A-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
  --seed 1 \
  --max_iterations 50000 \
  --run_name ${DANCE}_s1 \
  --logger wandb \
  --log_project_name him-training \
  --video \
  --video_length 200 \
  --video_interval 5000
```

### Training arguments reference

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | (required) | Gymnasium task ID |
| `--headless` | false | Run without display (required on pods) |
| `--device` | `cuda:0` | GPU device |
| `--num_envs` | 4096 | Number of parallel environments |
| `--seed` | None | Random seed for reproducibility |
| `--max_iterations` | 50000 | Total training iterations |
| `--run_name` | None | Suffix for the log directory name |
| `--logger` | None | `wandb`, `tensorboard`, or `neptune` |
| `--log_project_name` | None | wandb/neptune project name |
| `--video` | false | Record video clips during training |
| `--video_length` | 200 | Length of each video clip (sim steps) |
| `--video_interval` | 2000 | Record a clip every N steps |
| `--resume` | false | Resume from a previous checkpoint |
| `--load_run` | None | Run directory name to resume from |
| `--distributed` | false | Multi-GPU training |

### wandb setup (do this before first training run)

```bash
# Option 1: Environment variable (recommended for RunPod)
export WANDB_API_KEY="your-key-here"

# Option 2: Interactive login (prompts for API key)
wandb login
```

Get your API key at [wandb.ai/authorize](https://wandb.ai/authorize) (40+ characters). On RunPod, set `WANDB_API_KEY` as a pod environment variable in the template so it persists across restarts.

**If you see `WANDB_API_KEY invalid: API key must have 40+ characters`:** Your RunPod template has a placeholder or incorrect key. Fix it:

```bash
unset WANDB_API_KEY
export WANDB_API_KEY="your-actual-40-char-key-from-wandb-authorize"
```

To skip wandb entirely (e.g., for a quick test), remove `--logger wandb` and `--log_project_name` from the train command. Training will still log to the console.

### GPU sizing

| GPU | VRAM | Recommended `--num_envs` | ~Time for 50k iterations |
|-----|------|--------------------------|--------------------------|
| RTX 4070 | 12 GB | 512-1024 | ~12-18 hours |
| RTX 4090 | 24 GB | 4096 | ~6-10 hours |
| A100 | 40-80 GB | 4096 | ~3-6 hours |
| H100 | 80 GB | 4096-8192 | ~2-3 hours |

If you get CUDA OOM errors, halve `--num_envs`.

**Tuning `--num_envs` for your GPU:** After training starts, check GPU utilization with `nvidia-smi` on the pod. If GPU utilization is low (< 50%) and VRAM usage is well under the limit, you can stop and restart with more envs. More envs = more samples per iteration = faster wall-clock convergence. For example, an RTX 4090 at 2048 envs uses only ~54% VRAM and ~17% GPU -- doubling to 4096 significantly improves throughput.

### What training produces

Training creates a log directory at:

```
/workspace/him-collaborators/booster_train/logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/
```

Contents:
- `model_<iteration>.pt` -- Checkpoint files (saved periodically)
- `params/env.yaml` -- Environment config snapshot
- `params/agent.yaml` -- Agent config snapshot
- `videos/train/*.mp4` -- Video clips (if `--video` was used)

### Resuming interrupted training

If training is interrupted (pod preempted, connection lost, etc.):

```bash
python3 scripts/rsl_rl/train.py \
  --task Booster-K1-Custom_Dance_A-v0 \
  --headless --device cuda:0 --num_envs 4096 --seed 1 \
  --max_iterations 50000 --run_name ${DANCE}_s1 \
  --logger wandb --log_project_name him-training \
  --resume --load_run <TIMESTAMP>_${DANCE}_s1
```

The `--resume` flag loads the latest checkpoint from the specified run directory. Training continues from where it left off, and wandb will append to the existing run.

---

## Monitoring Training Progress

### wandb Dashboard (Recommended)

Once training starts with `--logger wandb`, open [wandb.ai](https://wandb.ai) and navigate to your project (`him-training`).

**Key panels to watch:**

| Panel | What it shows | What to look for |
|-------|--------------|------------------|
| `Mean reward` | Total mean reward per iteration | Should climb over time; if flat after 5k iterations, something is wrong |
| `Mean episode length` | How long the robot stays alive | Should increase as policy improves |
| `Episode_Reward/*` | Per-component reward breakdown | Individual rewards trending upward |
| `Loss/value_function` | Critic loss | Should decrease and stabilize |
| `Loss/surrogate` | PPO surrogate loss | Should oscillate but trend downward |
| `System` | GPU memory, utilization, CPU | Useful for tuning `--num_envs` |

**Key reward components:**

| Reward | What it measures |
|--------|-----------------|
| `motion_trunk_pos` / `motion_trunk_ori` | How well the torso tracks the reference motion |
| `motion_hand_pos` / `motion_foot_pos` | End-effector tracking accuracy |
| `motion_body_lin_vel` / `motion_body_ang_vel` | Velocity tracking |
| `action_rate_l2` | Smoothness of actions (negative; closer to 0 = smoother) |
| `undesired_contacts` | Penalty for bad contacts (closer to 0 = better) |
| `joint_limit` | Joint limit violation penalty |

### Video Clips

When training with `--video --video_length 200 --video_interval 5000`:

- A clip is recorded every 5000 training iterations
- Clips show the robot executing the current policy
- **On wandb:** Clips appear in the **Media** tab. Scrub through them to see how the policy evolves.
- **Local files:** Saved under `logs/rsl_rl/<experiment>/<timestamp>_<run>/videos/train/`

Download clips from RunPod:

```bash
scp -P <SSH_PORT> "root@<POD_IP>:/workspace/him-collaborators/booster_train/logs/rsl_rl/k1_custom_dance/*/videos/train/*.mp4" ./training_videos/
```

### TensorBoard (Alternative)

If you prefer TensorBoard or can't use wandb:

```bash
# Start training with tensorboard logger
python3 scripts/rsl_rl/train.py \
  --task Booster-K1-Custom_Dance_A-v0 \
  --headless --device cuda:0 --num_envs 4096 --seed 1 \
  --max_iterations 50000 --run_name ${DANCE}_s1 \
  --logger tensorboard
```

Then in a second terminal (or tmux pane):

```bash
tensorboard --logdir /workspace/him-collaborators/booster_train/logs/rsl_rl --bind_all --port 6006
```

Access it at `http://<POD_IP>:6006` (port 6006 is exposed in the Docker image).

### Terminal Output

While training is running, the console prints a summary after each iteration:

```
Learning iteration 500/50000
               Mean reward: -8.42
       Mean episode length: 142.3
       ...
       Total timesteps: 2048000
        Iteration time: 2.15s
          Time elapsed: 00:17:55
                   ETA: 17:42:05
```

Key indicators:
- **Mean reward** should trend upward over thousands of iterations
- **Mean episode length** should increase (robot staying upright longer)
- **ETA** gives you a wall-clock estimate of remaining time

### Quick sanity check at 10k iterations

You can evaluate mid-training without stopping it. In a separate terminal:

```bash
python3 scripts/rsl_rl/play.py \
  --task Booster-K1-Custom_Dance_A-v0-Play \
  --headless \
  --num_envs 1 \
  --video --video_length 500 \
  --checkpoint logs/rsl_rl/k1_custom_dance/<TIMESTAMP>_${DANCE}_s1/model_10000.pt
```

This exports the model and records a video you can review.

---

## Step 6: Export the Trained Model

After training finishes (or at any good checkpoint):

```bash
cd /workspace/him-collaborators/booster_train

python3 scripts/rsl_rl/play.py \
  --task Booster-K1-Custom_Dance_A-v0-Play \
  --headless \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/k1_custom_dance/<TIMESTAMP>_${DANCE}_s1/model_50000.pt
```

This exports two files under `logs/.../exported/`:

| File | Format | Use |
|------|--------|-----|
| `*.pt` | TorchScript | Deployment on the physical robot |
| `*.onnx` | ONNX | Cross-platform inference |

### Download the exported model

```bash
scp -P <SSH_PORT> \
  "root@<POD_IP>:/workspace/him-collaborators/booster_train/logs/rsl_rl/k1_custom_dance/*/exported/*.pt" \
  ./

scp -P <SSH_PORT> \
  "root@<POD_IP>:/workspace/him-collaborators/booster_train/logs/rsl_rl/k1_custom_dance/*/exported/*.onnx" \
  ./
```

---

## Docker Image Maintenance

### When to rebuild `humanoid-dev:base`

Only rebuild when:
- Isaac Lab version changes in `requirements.txt`
- System packages change in `Dockerfile.base`
- CUDA version needs updating

```bash
docker build -t humanoid-dev:base -f Dockerfile.base .
```

Then rebuild the thin image:

```bash
docker build -t humanoid-dev:latest -f Dockerfile .
```

### When to rebuild `humanoid-dev:latest`

Rebuild when:
- booster_train or booster_assets code changes
- Isaac Lab extensions change
- GVHMR/GMR setup changes
- Wrapper scripts (`mp4_to_csv.sh`, `setup_weights.sh`) change

```bash
docker build -t humanoid-dev:latest -f Dockerfile .
```

Docker will cache unchanged layers and only rebuild what changed.

### `.dockerignore`

The `.dockerignore` excludes large/unnecessary files from the build context:
- `logs/`, `wandb/`, `tb_logs/` (training outputs)
- `*.pt`, `*.pth`, `*.ckpt`, `*.onnx` (model weights)
- `.git`, `__pycache__`, `*.pyc`

If you add new large files to the workspace, add them to `.dockerignore` to keep build context transfers fast.

### `.gitattributes`

The `.gitattributes` file ensures shell scripts keep Unix line endings (LF) on Windows:

```
*.sh text eol=lf
```

This prevents `\r` errors when running scripts inside the Linux container.

---

## Troubleshooting

### "Do you accept the EULA?"

The `OMNI_KIT_ACCEPT_EULA=Y` environment variable should handle this automatically. If it still prompts:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

On RunPod, set this as a pod environment variable in the template.

### CUDA out of memory

Reduce `--num_envs` (try halving it). Memory requirements scale linearly with environment count.

| `--num_envs` | Approximate VRAM |
|--------------|------------------|
| 512 | ~4-6 GB |
| 1024 | ~6-10 GB |
| 2048 | ~10-20 GB |
| 4096 | ~20-40 GB |

### csv_to_npz.py fails

- **"Invalid file path"** -- Check the CSV is in the right location: `ls /workspace/him-collaborators/booster_assets/motions/K1/`
- **Wrong number of columns** -- Verify your CSV has exactly 29 columns per row, no header
- **NaN values** -- Check for corrupted frames in your source video/CSV

### Training reward stays flat

- Run for at least 5000 iterations before judging
- Try a different task variant (B or D instead of A)
- Check that the NPZ was generated correctly (`csv_to_npz.py` should print frame count)
- Verify the motion file path in `env_cfg.py` is correct

### wandb not logging

- Make sure you ran `wandb login` or set `WANDB_API_KEY` before starting training
- Verify `--logger wandb` and `--log_project_name him-training` are in the train command
- Check internet connectivity from the pod

### Task not found

```bash
python3 scripts/list_envs.py | grep -i your_task_name
```

If missing, verify:
1. The `__init__.py` in your task directory has `gym.register()` calls
2. The task directory is under `booster_train/tasks/manager_based/beyond_mimic/robots/k1/`
3. You rebuilt/reinstalled booster_train after adding the new task: `pip install -e source/booster_train`

### Vulkan / Graphics errors

Isaac Sim requires Vulkan for rendering. On headless environments, the `--headless` flag should handle this. If you see `ERROR_INCOMPATIBLE_DRIVER`:
- Make sure `--gpus all` is passed to `docker run`
- Verify NVIDIA drivers are installed on the host
- Try `--device cuda:0` explicitly

### GVHMR / GMR errors

- **"GVHMR model weights not found"** -- Run `bash /workspace/setup_weights.sh`
- **"SMPLX body models not found"** -- Upload SMPLX/SMPL models (requires registration)
- **GVHMR produces no output** -- Video must contain a clearly visible person; try better lighting or a different video
- **Preview video generation fails** -- Try setting `MUJOCO_GL=osmesa` before running the pipeline; the CSV will still be generated even if preview fails

- **"pyav can not handle the given uri"** -- The input file is not a valid video. Run `file /workspace/motions/input/your_video.mp4` to check. If it says "HTML document", the download saved a web page instead of the video -- re-download the actual file. If it says "Apple QuickTime" or "HEVC", convert to H.264 first: `ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 23 output.mp4`. The `mp4_to_csv.sh` script handles both cases automatically.

- **"X11: The DISPLAY environment variable is missing" / "could not initialize GLFW"** -- The GMR viewer requires a display, but the pod is headless. Wrap the command with `xvfb-run -a` and set `MUJOCO_GL=egl`:
  ```bash
  MUJOCO_GL=egl xvfb-run -a /opt/conda/bin/conda run --no-capture-output -n gvhmr python scripts/gvhmr_to_robot.py ...
  ```
  The `mp4_to_csv.sh` script already includes this fix. This only matters when running GMR scripts manually.

- **"RuntimeError: Sizes of tensors must match except in dimension 1"** or **"einsum(): subscript 1 has size 20 ... does not broadcast with previously seen size 26"** in `smplx/body_models.py` -- This is a version incompatibility between GMR's code and the installed `smplx` package. The Dockerfile patches GMR's `smpl.py` automatically. If you're on an older container without the patch, apply it manually:
  ```bash
  cd /workspace/GMR
  # Fix 1: Remove unnecessary betas padding
  sed -i "s/betas = np.pad(smpl_params_global\['betas'\]\[0\], (0,6))/betas = smpl_params_global['betas'][0].numpy()/" \
      general_motion_retargeting/utils/smpl.py
  # Fix 2: Pass expression parameter with correct batch size
  sed -i 's/# expression=torch.zeros(num_frames, 10).float(),/expression=torch.zeros(num_frames, 10).float(),/g' \
      general_motion_retargeting/utils/smpl.py
  ```

### wandb errors

- **"WANDB_API_KEY invalid: API key must have 40+ characters, has 36"** -- The RunPod template has a placeholder API key. Run `unset WANDB_API_KEY`, then set the real key from [wandb.ai/authorize](https://wandb.ai/authorize). See [wandb setup](#wandb-setup-do-this-before-first-training-run).

### File transfer errors

- **SCP "Connection refused"** -- RunPod's exposed TCP ports can take several minutes to initialize after pod startup. Use [`runpodctl`](#transferring-files-with-runpodctl-recommended) instead, which works immediately via peer-to-peer transfer. Alternatively, wait a few minutes and retry SCP.

### Google Drive rate limit on weight downloads

The `setup_weights.sh` script downloads from Google Drive which has rate limits. If it fails:
1. Wait a few hours and retry
2. Or manually download from the Google Drive link printed by the script
3. Upload the files to the container via `scp`

---

## Quick Reference

### From MP4 (full pipeline)

```bash
export DANCE=my_dance

# 0. (First time only) Download model weights
bash /workspace/setup_weights.sh

# 1. Upload video to pod, then convert MP4 -> K1 CSV
bash /workspace/mp4_to_csv.sh /workspace/motions/input/${DANCE}.mp4 ${DANCE}
# Review the preview: /workspace/motions/input/${DANCE}_preview.mp4

# 2. Copy CSV into assets
cp /workspace/motions/input/${DANCE}.csv \
   /workspace/him-collaborators/booster_assets/motions/K1/

# 3. Convert to NPZ
cd /workspace/him-collaborators/booster_train
python3 scripts/csv_to_npz.py --headless \
  --input_file ../booster_assets/motions/K1/${DANCE}.csv \
  --input_fps 30 --output_fps 50 \
  --output_name ../booster_assets/motions/K1/${DANCE}.npz

# 4. Append standing frames
python3 scripts/append_standing_frames.py \
  --input ../booster_assets/motions/K1/${DANCE}.npz \
  --output ../booster_assets/motions/K1/${DANCE}_extended.npz \
  --seconds 3.0

# 5. (Optional) Edit env_cfg.py to point at your NPZ
#    Or use the default custom_dance task if you updated its MOTION_FILE

# 6. Train
python3 scripts/rsl_rl/train.py \
  --task Booster-K1-Custom_Dance_A-v0 \
  --headless --device cuda:0 --num_envs 4096 --seed 1 \
  --max_iterations 50000 --run_name ${DANCE}_s1 \
  --logger wandb --log_project_name him-training \
  --video --video_length 200 --video_interval 5000

# 7. Export (replace <TIMESTAMP> with actual directory name)
python3 scripts/rsl_rl/play.py \
  --task Booster-K1-Custom_Dance_A-v0-Play \
  --headless --num_envs 1 \
  --checkpoint logs/rsl_rl/k1_custom_dance/<TIMESTAMP>_${DANCE}_s1/model_50000.pt
```

### From pre-made CSV (skip MP4 step)

```bash
export DANCE=my_dance

# 1. Upload CSV to pod
scp -P <SSH_PORT> ${DANCE}.csv root@<POD_IP>:/workspace/motions/input/

# 2. Copy into assets
cp /workspace/motions/input/${DANCE}.csv \
   /workspace/him-collaborators/booster_assets/motions/K1/

# 3-7: Same as steps 3-7 above
```

### Useful commands

```bash
# Check what tasks are registered
python3 scripts/list_envs.py

# Check GPU status
nvidia-smi

# List training runs
ls logs/rsl_rl/

# Watch training logs in real time (from another terminal)
tail -f logs/rsl_rl/k1_custom_dance/*/train.log

# Download all videos from a training run
scp -P <SSH_PORT> -r "root@<POD_IP>:/workspace/him-collaborators/booster_train/logs/rsl_rl/k1_custom_dance/*/videos/" ./

# Check model weight setup status
bash /workspace/setup_weights.sh
```
