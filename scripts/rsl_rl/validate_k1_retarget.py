# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Offline check: does a PKL clip still match AMASS SMPL keypoints under K1 FK?
# No Isaac Sim / no robot body change — only PyTorch + SMPL + Humanoid_Batch (same as grad_fit_k1).
#
# Typical run (PowerShell): cd to human2humanoid so data/smpl resolves, then:
#   python C:/path/to/HOVER/scripts/rsl_rl/validate_k1_retarget.py `
#     --amass_root D:/AMASS/AMASS_Complete `
#     --amass_npz D:/AMASS/.../clip.npz `
#     --pkl C:/path/to/HOVER/neural_wbc/data/data/motions/amass_all.pkl
# Uses --h2h_root (default: HOVER/third_party/human2humanoid) and chdirs there for SMPL assets.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

HOVER_ROOT = Path(__file__).resolve().parents[2]

# Same correspondences as third_party/human2humanoid/scripts/data_process/grad_fit_k1.py
SMPL_TO_K1_RETARGET = {
    "Trunk": "Pelvis",
    "Left_Shank": "L_Knee",
    "left_foot_link": "L_Ankle",
    "Right_Shank": "R_Knee",
    "right_foot_link": "R_Ankle",
    "Left_Arm_2": "L_Shoulder",
    "Left_Arm_3": "L_Elbow",
    "left_hand_link": "L_Hand",
    "Right_Arm_2": "R_Shoulder",
    "Right_Arm_3": "R_Elbow",
    "right_hand_link": "R_Hand",
}


def _load_amass_like_grad_fit(data_path: str):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
    if "mocap_framerate" not in entry_data:
        return None
    fps = int(entry_data["mocap_framerate"])
    poses = entry_data["poses"]
    trans = entry_data["trans"]
    betas = entry_data.get("betas", np.zeros(10))
    if poses.shape[1] < 72:
        poses = np.concatenate([poses, np.zeros((poses.shape[0], 72 - poses.shape[1]))], axis=1)
    pose_aa = poses[:, :72]
    return {
        "pose_aa": pose_aa,
        "trans": trans,
        "betas": betas[:10] if len(betas) >= 10 else np.pad(betas, (0, 10 - len(betas))),
        "fps": fps,
    }


def _clip_key(amass_root: str, npz_path: str) -> str:
    return os.path.relpath(npz_path, amass_root).replace("\\", "/").replace(".npz", "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SMPL vs K1 FK for one AMASS clip and matching PKL entry (no sim)."
    )
    parser.add_argument("--amass_root", type=str, required=True)
    parser.add_argument("--amass_npz", type=str, required=True, help="Full path to one AMASS .npz")
    parser.add_argument("--pkl", type=str, required=True)
    parser.add_argument(
        "--h2h_root",
        type=str,
        default=str(HOVER_ROOT / "third_party" / "human2humanoid"),
        help="human2humanoid repo root (for data/smpl, shape pkl, phc imports)",
    )
    parser.add_argument(
        "--shape_pkl",
        type=str,
        default="data/k1/shape_optimized_v1.pkl",
        help="Relative to h2h_root unless absolute",
    )
    parser.add_argument(
        "--mjcf",
        type=str,
        default=str(HOVER_ROOT / "neural_wbc" / "data" / "data" / "motion_lib" / "k1.xml"),
        help="K1 MJCF used by Humanoid_Batch (match grad_fit_k1 / training)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=512,
        help="Same cap as grad_fit_k1 (0 = no cap)",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Evaluate every Nth frame (faster smoke test)",
    )
    args = parser.parse_args()

    h2h = Path(args.h2h_root).resolve()
    if not h2h.is_dir():
        raise SystemExit(f"--h2h_root is not a directory: {h2h}")

    old_cwd = os.getcwd()
    sys.path.insert(0, str(h2h))
    os.chdir(h2h)
    try:
        from phc.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES
        from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch

        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        data_key = _clip_key(os.path.abspath(args.amass_root), os.path.abspath(args.amass_npz))
        motion = joblib.load(os.path.abspath(args.pkl))
        if data_key not in motion:
            raise SystemExit(
                f"PKL has no key {data_key!r}. "
                f"Check --amass_root matches retarget run (relative path must match)."
            )
        clip = motion[data_key]
        n_clip = int(np.asarray(clip["root_trans_offset"]).shape[0])

        raw = _load_amass_like_grad_fit(os.path.abspath(args.amass_npz))
        if raw is None:
            raise SystemExit("Invalid AMASS file (missing mocap_framerate?)")

        fps = raw["fps"]
        skip = max(1, int(fps // 30))
        pose_aa_np = raw["pose_aa"][::skip]
        trans_np = raw["trans"][::skip]
        if args.max_frames and args.max_frames > 0:
            pose_aa_np = pose_aa_np[: args.max_frames]
            trans_np = trans_np[: args.max_frames]
        n_amass = pose_aa_np.shape[0]

        if n_amass != n_clip:
            raise SystemExit(
                f"Frame count mismatch: AMASS after subsample/cap has {n_amass}, "
                f"PKL clip has {n_clip}. Use the same --max_frames / AMASS root as retarget."
            )

        shape_path = (
            Path(args.shape_pkl)
            if Path(args.shape_pkl).is_absolute()
            else h2h / args.shape_pkl
        )
        shape_new, _scale = joblib.load(shape_path)
        shape_new = shape_new.to(device)

        smpl = SMPL_Parser(model_path="data/smpl", gender="neutral").to(device)
        hb = Humanoid_Batch(
            mjcf_file=os.path.abspath(args.mjcf),
            extend_hand=False,
            extend_head=False,
            device=device,
        )

        k1_names = list(hb.model_names)
        k1_joint_pick_idx = [k1_names.index(n) for n in SMPL_TO_K1_RETARGET.keys()]
        smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(n) for n in SMPL_TO_K1_RETARGET.values()]

        idx = np.arange(0, n_clip, args.frame_stride, dtype=np.int64)

        pose_smpl = torch.tensor(pose_aa_np[idx], dtype=torch.float32, device=device)
        trans_smpl = torch.tensor(trans_np[idx], dtype=torch.float32, device=device)
        with torch.no_grad():
            _, joints_smpl = smpl.get_joints_verts(
                pose_smpl, shape_new.expand(pose_smpl.shape[0], -1), trans_smpl
            )

        pose_k1 = torch.tensor(np.asarray(clip["pose_aa"])[idx], dtype=torch.float32, device=device)
        trans_k1 = torch.tensor(np.asarray(clip["root_trans_offset"])[idx], dtype=torch.float32, device=device)
        with torch.no_grad():
            fk = hb.fk_batch(pose_k1.unsqueeze(0), trans_k1.unsqueeze(0), return_full=False)
            k1_world = fk["global_translation"][0]

        smpl_pick = joints_smpl[:, smpl_joint_pick_idx]
        k1_pick = k1_world[:, k1_joint_pick_idx]

        # Pelvis / trunk–centered: removes global translation + floor-lift ambiguity.
        smpl_pelvis = joints_smpl[:, SMPL_BONE_ORDER_NAMES.index("Pelvis")]
        k1_trunk = k1_world[:, k1_names.index("Trunk")]
        smpl_rel = smpl_pick - smpl_pelvis.unsqueeze(1)
        k1_rel = k1_pick - k1_trunk.unsqueeze(1)

        dist = (smpl_rel - k1_rel).norm(dim=-1)
        print(f"clip key: {data_key}")
        print(f"frames in clip: {n_clip}  (evaluated {len(idx)} with stride {args.frame_stride})")
        print(f"mean L2 keypoint error (pelvis-centered): {dist.mean().item():.4f} m")
        print(f"max  L2 keypoint error (pelvis-centered): {dist.max().item():.4f} m")
        per_kp = dist.mean(dim=0).cpu().numpy()
        print("per-keypoint mean (m):")
        for name, v in zip(SMPL_TO_K1_RETARGET.keys(), per_kp):
            print(f"  {name}: {v:.4f}")

        # World-space sanity (often dominated by root convention; large values are not always bugs).
        dist_w = (smpl_pick - k1_pick).norm(dim=-1)
        print(f"mean L2 keypoint error (world, uncorrected): {dist_w.mean().item():.4f} m")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
