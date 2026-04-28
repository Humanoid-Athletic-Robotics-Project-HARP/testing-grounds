# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
import sys

from teacher_policy_cfg import TeacherPolicyCfg

from isaaclab.app import AppLauncher

# local imports
from utils import get_player_args  # isort: skip


def _is_placeholder_motion_path(path: str | None) -> bool:
    if not path:
        return True
    norm = path.replace("\\", "/").lower()
    return "path/to" in norm or "your/motion" in norm


def _apply_saved_motion_and_validate_checkpoint(args_cli) -> None:
    """Use motion path from the training run's env_config when CLI omits or leaves a placeholder."""
    teacher_cfg = TeacherPolicyCfg.from_argparse_args(args_cli)
    run_dir = teacher_cfg.runner.resume_path
    ckpt_name = teacher_cfg.runner.checkpoint
    if not run_dir or not ckpt_name:
        print("[ERROR] Set --teacher_policy.resume_path and --teacher_policy.checkpoint.", file=sys.stderr)
        sys.exit(1)
    ckpt_path = os.path.join(run_dir, ckpt_name)
    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    if not _is_placeholder_motion_path(args_cli.reference_motion_path):
        return
    env_json = os.path.join(run_dir, "env_config.json")
    if not os.path.isfile(env_json):
        print(
            f"[ERROR] reference_motion_path missing/placeholder and no env_config.json at {env_json}",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(env_json, encoding="utf-8") as fh:
        env_cfg = json.load(fh)
    ref = env_cfg.get("reference_motion_manager") or {}
    mp = ref.get("motion_path")
    if not mp:
        print("[ERROR] env_config.json has no reference_motion_manager.motion_path", file=sys.stderr)
        sys.exit(1)
    args_cli.reference_motion_path = mp
    print(f"[INFO] Using reference_motion_path from env_config.json: {mp}")


# add argparse arguments
parser = get_player_args(description="Plays motion tracking policy in Isaac Lab.")
parser.add_argument(
    "--no_randomize",
    action="store_true",
    help=(
        "Deterministic evaluation (TEST mode): motions are not sampled like training, and each reset replays "
        "from the start of the clip (t=0). Default play matches training: TRAIN mode with random motion phase on reset."
    ),
)
parser.add_argument(
    "--gui",
    action="store_true",
    help=(
        "Use full Omniverse viewport (isaaclab.python.kit). "
        "On many Isaac Sim 4.5 + Python 3.11 installs the GUI Kit fails extension resolution; omit this flag (default)."
    ),
)
parser.add_argument(
    "--no_video",
    action="store_true",
    help="When running headless (default), do not write MP4s to RUN_DIR/videos/play (faster, no visuals).",
)
parser.add_argument(
    "--video",
    action="store_true",
    help="Also record RGB clips (implies --enable_cameras). Default headless mode records automatically unless --no_video.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=500,
    help="Env steps per recorded clip (passed to gymnasium.RecordVideo).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=500,
    help="Trigger a new clip when the global env step count is divisible by this value.",
)
parser.add_argument(
    "--debug_spawn",
    action="store_true",
    help=(
        "After the first reset, print env0 root pose, projected gravity, and reference root quat from the motion "
        "manager. Use this to tell apart an upside-down spawn (bad FK / motion alignment) from a policy collapse."
    ),
)

# append RSL-RL cli arguments
TeacherPolicyCfg.add_args_to_parser(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_apply_saved_motion_and_validate_checkpoint(args_cli)

# Default: headless Kit (matches train_teacher_policy.py) + MP4s — GUI Kit often breaks (py3.11 vs cp310 extensions).
if args_cli.gui:
    args_cli.headless = False
    wants_record = args_cli.video
else:
    args_cli.headless = True
    wants_record = not args_cli.no_video
    args_cli.video = wants_record

if wants_record:
    args_cli.enable_cameras = True
else:
    args_cli.enable_cameras = False

if not args_cli.gui:
    print(
        "[INFO] Play is headless (same Kit profile as typical training). "
        "Videos: RUN_DIR/videos/play/ — use --gui for a live viewport if isaaclab.python.kit starts on your machine."
    )

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from players import DemoPlayer


def main():
    randomize = not args_cli.no_randomize
    if randomize:
        print("[INFO] Play uses TRAIN-like reference sampling (random motion phase on reset), same as teacher training.")
    player = DemoPlayer(args_cli=args_cli, randomize=randomize, debug_spawn=args_cli.debug_spawn)
    player.play(simulation_app=simulation_app)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
