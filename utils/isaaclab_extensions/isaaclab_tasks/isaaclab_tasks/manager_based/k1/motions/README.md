# K1 Dance reference motions

**Goal:** Train the robot to **mimic the motion in the CSV while staying upright**. The CSV may come from animation (not physically validated), so the policy is rewarded for tracking the motion **only when the robot is sufficiently upright**; otherwise it should prioritize balance.

**Recommended workflow:**
1. **Train stand first** so the policy learns to balance:  
   `Isaac-K1-Stand-v0` (train to a good checkpoint).
2. **Train dance** with your CSV:  
   Set `K1_DANCE_CSV` to your file (or put `dance.csv` in this folder), then run `Isaac-K1-Dance-v0`. The dance task uses the same upright/height rewards as stand, plus a tracking reward that only counts when the base is upright (e.g. `base_up_proj >= 0.85`), so the robot learns to follow the motion when it can and stay balanced when the animation would make it fall.

**Run dance training (from repo root):**
```bash
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-K1-Dance-v0
```
On Windows: `isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-K1-Dance-v0`  
Optional: `--num_envs 16`, `--max_iterations 3000`, or `--resume --load_run <folder> --checkpoint model_123.pt` to resume from a checkpoint.

Use your own dance CSV by either:
- Putting it here as `dance.csv`, or
- Setting the **`K1_DANCE_CSV`** environment variable to the full path, e.g.  
  `set K1_DANCE_CSV=C:\Users\kylel\Humanoid\motions\custom\dance.csv`  
  then run the Isaac-K1-Dance-v0 task.

## Supported CSV formats

### 1) Humanoid-style (root + joint_0..joint_21, no time column)

- Columns: `root_x`, `root_y`, `root_z`, `root_qx`, `root_qy`, `root_qz`, `root_qw`,  
  `joint_0`, `joint_1`, ... `joint_21`.
- Joint values in **radians**. Time is inferred from row index: `t = frame_index / fps` (default 60 fps; set `motion_fps` in config if needed).

### 2) Time + named joints

- **First column**: time in seconds (`time` or `t`).
- **Remaining columns**: joint positions in radians, names matching K1 joint names below (or use `joint_column_map` in config).

### K1 joint order (for joint_0..joint_21)

Index 0–21 corresponds to:  
AAHead_yaw, ALeft_Shoulder_Pitch, ARight_Shoulder_Pitch, Left_Hip_Pitch, Right_Hip_Pitch, Head_pitch, Left_Shoulder_Roll, Right_Shoulder_Roll, Left_Hip_Roll, Right_Hip_Roll, Left_Elbow_Pitch, Right_Elbow_Pitch, Left_Hip_Yaw, Right_Hip_Yaw, Left_Elbow_Yaw, Right_Elbow_Yaw, Left_Knee_Pitch, Right_Knee_Pitch, Left_Ankle_Pitch, Right_Ankle_Pitch, Left_Ankle_Roll, Right_Ankle_Roll.

### Is there enough information to recreate the dance?

- **Yes, for the current setup:** The CSV provides **time** and **22 joint positions (rad)** per frame. The policy sees the **reference pose** at the current time and **phase** (0–1 over the loop), so it can track the motion while staying upright.
- **Dances with walking:** If your CSV includes **root_x, root_y, root_z, root_qx, root_qy, root_qz, root_qw** (humanoid-style), the loader enables **base position and heading tracking**. The policy then gets:
  - **ref_base_pos** (target position in world) and **ref_base_heading** (target yaw),
  - **base_xy** (current horizontal position),
  and a **base_position_tracking** reward (when upright) so it learns to walk to the reference positions and stay consistent with the motion.
- **For a richer dance:** Add more rows (keyframes) to the CSV so the motion is smoother and longer.
- **Optional later:** Reference joint velocities would help the policy anticipate motion; the loader currently supports only positions and interpolates between keyframes.
