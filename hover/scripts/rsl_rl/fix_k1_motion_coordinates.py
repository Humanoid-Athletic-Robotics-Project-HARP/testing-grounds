"""Fix coordinate system in K1 retargeted motion data.

The retargeting script (grad_fit_k1.py) applies smpl_root_align = sRot.from_quat([0.5,0.5,0.5,0.5]).inv()
to the SMPL root rotation. This puts root_trans_offset in SMPL frame (Y-up) and the root rotation in a
mixed frame where local Z ("up" on the K1 MJCF skeleton) maps to approximately world X.

Isaac Lab expects Z-up. Because positions and rotations ended up in different frames, we apply:

  Positions (root_trans_offset):
    SMPL Y-up -> MuJoCo Z-up via R_smpl2mujoco: v_fixed = v[..., [2, 0, 1]]
    (maps SMPL-X->MuJoCo-Y, SMPL-Y->MuJoCo-Z, SMPL-Z->MuJoCo-X)

  Rotations (pose_aa root):
    Ry(-90deg) left-multiplied onto the root rotation, which maps the robot's "up"
    direction from world-X to world-Z.

Joint DOF values (pose_aa[:, 1:] and dof) are LOCAL rotations and do not need conversion.
"""
import argparse
import os
import shutil

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

R_POS = np.array([2, 0, 1])  # permutation for SMPL Y-up -> MuJoCo Z-up positions
R_ROT = sRot.from_euler('y', -np.pi / 2)  # Ry(-90deg) for root rotation


def fix_clip(clip: dict) -> dict:
    pa = clip["pose_aa"]            # [N, num_bodies, 3]
    rt = clip["root_trans_offset"]  # [N, 3]

    root_rot_original = sRot.from_rotvec(pa[:, 0])
    root_rot_fixed = R_ROT * root_rot_original

    pa_fixed = pa.copy()
    pa_fixed[:, 0] = root_rot_fixed.as_rotvec().astype(np.float32)

    rt_fixed = rt[:, R_POS].copy().astype(np.float32)

    min_z = rt_fixed[:, 2].min()
    if min_z < 0.02:
        rt_fixed[:, 2] -= min_z - 0.02

    result = dict(clip)
    result["pose_aa"] = pa_fixed
    result["root_trans_offset"] = rt_fixed

    if "root_rot" in clip:
        rr_rot = sRot.from_quat(clip["root_rot"])  # scipy xyzw
        rr_fixed = (R_ROT * rr_rot).as_quat().astype(np.float32)
        result["root_rot"] = rr_fixed

    return result


def main():
    parser = argparse.ArgumentParser(description="Fix K1 motion coordinate system")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    if args.input is None:
        args.input = os.path.join(
            os.path.dirname(__file__), "..", "..", "neural_wbc", "data", "data", "motions", "amass_all.pkl"
        )
    args.input = os.path.abspath(args.input)

    if args.output is None:
        args.output = args.input

    print(f"Loading: {args.input}")
    data = joblib.load(args.input)
    keys = list(data.keys())
    print(f"  {len(keys)} clips")

    sample = data[keys[0]]
    root_aa = sample["pose_aa"][0, 0]
    mat = sRot.from_rotvec(root_aa).as_matrix()
    z_up = mat @ np.array([0, 0, 1])
    print(f"  Before fix - clip 0 root Z_up_in_world: {z_up}")
    if abs(z_up[2]) > 0.7:
        print("  Data appears to already be in Z-up convention. Aborting.")
        return

    data_fixed = {}
    for key in tqdm(keys, desc="Fixing coordinates"):
        data_fixed[key] = fix_clip(data[key])

    sample_f = data_fixed[keys[0]]
    root_aa_f = sample_f["pose_aa"][0, 0]
    mat_f = sRot.from_rotvec(root_aa_f).as_matrix()
    z_up_f = mat_f @ np.array([0, 0, 1])
    print(f"\n  After fix  - clip 0 root Z_up_in_world: {z_up_f}")
    print(f"  root_trans_offset frame 0: {sample_f['root_trans_offset'][0]}")

    up_count = 0
    for k in list(data_fixed.keys())[:100]:
        aa = data_fixed[k]["pose_aa"][0, 0]
        z = sRot.from_rotvec(aa).as_matrix() @ [0, 0, 1]
        if z[2] > 0.5:
            up_count += 1
    print(f"  Upright check (first 100 clips): {up_count}/100 have Z_up > 0.5")

    if not args.no_backup and args.output == args.input:
        bak = args.input + ".bak"
        if os.path.exists(bak):
            print(f"\n  Backup already exists: {bak}")
        else:
            print(f"\n  Creating backup: {bak}")
            shutil.copy2(args.input, bak)

    print(f"  Saving to: {args.output}")
    joblib.dump(data_fixed, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
