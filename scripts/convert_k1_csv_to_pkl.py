"""Convert Booster K1 motion CSV files to the .pkl format expected by HOVER's MotionLibH1.

Usage:
    python scripts/convert_k1_csv_to_pkl.py \
        --csv_dir  third_party/booster_assets/motions/K1 \
        --output   neural_wbc/data/data/motions/k1_motions.pkl \
        --fps 30
"""

import argparse
import glob
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R

# K1 joint order in motion CSV (after the 7 base columns)
K1_JOINT_NAMES = [
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

# Rotation axis for each K1 joint (from URDF/MJCF)
K1_ROTATION_AXES = np.array([
    [0, 0, 1],  # AAHead_yaw
    [0, 1, 0],  # Head_pitch
    [0, 1, 0],  # ALeft_Shoulder_Pitch
    [1, 0, 0],  # Left_Shoulder_Roll
    [0, 1, 0],  # Left_Elbow_Pitch
    [0, 0, 1],  # Left_Elbow_Yaw
    [0, 1, 0],  # ARight_Shoulder_Pitch
    [1, 0, 0],  # Right_Shoulder_Roll
    [0, 1, 0],  # Right_Elbow_Pitch
    [0, 0, 1],  # Right_Elbow_Yaw
    [0, 1, 0],  # Left_Hip_Pitch
    [1, 0, 0],  # Left_Hip_Roll
    [0, 0, 1],  # Left_Hip_Yaw
    [0, 1, 0],  # Left_Knee_Pitch
    [0, 1, 0],  # Left_Ankle_Pitch
    [1, 0, 0],  # Left_Ankle_Roll
    [0, 1, 0],  # Right_Hip_Pitch
    [1, 0, 0],  # Right_Hip_Roll
    [0, 0, 1],  # Right_Hip_Yaw
    [0, 1, 0],  # Right_Knee_Pitch
    [0, 1, 0],  # Right_Ankle_Pitch
    [1, 0, 0],  # Right_Ankle_Roll
], dtype=np.float32)


def csv_to_motion_dict(csv_path: str, fps: int) -> dict:
    """Convert a single K1 CSV motion file to the dict format used by MotionLibH1.

    CSV columns: x, y, z, qx, qy, qz, qw, joint0, joint1, ..., joint21
    """
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    seq_len = data.shape[0]
    num_joints = 22

    # Base position
    root_trans = data[:, :3]  # [seq_len, 3]

    # Base orientation quaternion (CSV is xyzw, scipy expects xyzw)
    root_quat_xyzw = data[:, 3:7]
    root_rot = R.from_quat(root_quat_xyzw)
    root_aa = root_rot.as_rotvec().astype(np.float32)  # [seq_len, 3]

    # Joint angles -> axis-angle per joint
    joint_angles = data[:, 7:7 + num_joints]  # [seq_len, 22]
    # axis_angle = angle * axis  -> [seq_len, 22, 3]
    joint_aa = joint_angles[:, :, np.newaxis] * K1_ROTATION_AXES[np.newaxis, :, :]

    # Combine root + joints: [seq_len, 23, 3] (root is body 0)
    pose_aa = np.concatenate([root_aa[:, np.newaxis, :], joint_aa], axis=1)

    return {
        "root_trans_offset": root_trans,
        "pose_aa": pose_aa,
        "fps": fps,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert K1 CSV motions to HOVER pkl format")
    parser.add_argument("--csv_dir", type=str, required=True, help="Directory containing K1 CSV motion files")
    parser.add_argument("--csv_file", type=str, default=None, help="Single CSV file to convert (overrides --csv_dir)")
    parser.add_argument("--output", type=str, required=True, help="Output .pkl file path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second of the motion data")
    args = parser.parse_args()

    if args.csv_file:
        csv_files = [args.csv_file]
    else:
        csv_files = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))

    if not csv_files:
        print(f"No CSV files found in {args.csv_dir}")
        return

    all_motions = {}
    for csv_path in csv_files:
        key = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"Converting: {key} ({csv_path})")
        motion = csv_to_motion_dict(csv_path, args.fps)
        all_motions[key] = motion
        print(f"  -> {motion['pose_aa'].shape[0]} frames, {motion['pose_aa'].shape[1]} bodies")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(all_motions, f)

    print(f"\nSaved {len(all_motions)} motion(s) to {args.output}")


if __name__ == "__main__":
    main()
