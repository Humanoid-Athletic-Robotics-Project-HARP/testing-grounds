# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Load reference motion from CSV for K1 dance tracking.

Supported formats:
  - With time column: header has "time" (or "t") and joint columns.
  - Humanoid-style: root_x, root_y, root_z, root_qx, root_qy, root_qz, root_qw,
    joint_0, joint_1, ... joint_21 (no time column; time = frame_index / fps).
    When root_* columns are present, they are loaded so the policy can track base
    position and heading (e.g. for dances that involve walking to new positions).
    Note: MuJoCo exports free-joint quaternions as (qw, qx, qy, qz) in qpos.
    If those values were written under root_qx/root_qy/root_qz/root_qw labels,
    this loader auto-detects and remaps them to the expected semantic order.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .k1_stand_env_cfg import K1_JOINT_NAMES

DEFAULT_MOTION_FPS = 60.0

ROOT_POS_COLS = ("root_x", "root_y", "root_z")
ROOT_QUAT_COLS = ("root_qx", "root_qy", "root_qz", "root_qw")


def _quat_up_proj_z_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    """Compute world up-axis z projection for quaternion in (x, y, z, w)."""
    x = quat_xyzw[:, 0]
    y = quat_xyzw[:, 1]
    return 1.0 - 2.0 * (x * x + y * y)


class MotionLoaderCSV:
    """Load and sample K1 reference motion from a CSV file."""

    def __init__(
        self,
        csv_path: str,
        device: torch.device | str = "cpu",
        joint_column_map: dict[str, str] | None = None,
        fps: float | None = None,
    ):
        """Load motion from CSV.

        Args:
            csv_path: Path to the CSV file.
            device: Device for tensors.
            joint_column_map: Map from CSV column names to K1_JOINT_NAMES.
                If None, CSV columns must match K1_JOINT_NAMES exactly.
            fps: Frames per second of the motion. If None, inferred from time column delta.
        """
        assert os.path.isfile(csv_path), f"Motion CSV not found: {csv_path}"
        self.device = torch.device(device) if isinstance(device, str) else device
        self._joint_column_map = joint_column_map or {}

        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
        self.has_root = False
        if data.dtype.names is None:
            # No header: assume time in col 0, joints in K1_JOINT_NAMES order
            arr = np.genfromtxt(csv_path, delimiter=",", dtype=np.float64)
            times = arr[:, 0]
            self._joint_positions = np.asarray(arr[:, 1 : 1 + len(K1_JOINT_NAMES)], dtype=np.float32)
            if self._joint_positions.shape[1] != len(K1_JOINT_NAMES):
                raise ValueError(
                    f"CSV has {self._joint_positions.shape[1]} joint columns but K1 has {len(K1_JOINT_NAMES)}. "
                    "Use a header row with K1 joint names or provide joint_column_map."
                )
            n = len(times)
            self._root_positions = np.zeros((n, 3), dtype=np.float32)
            self._root_quats = np.zeros((n, 4), dtype=np.float32)
            self._root_quats[:, 3] = 1.0
            self._first_frame_root_pos = np.zeros((1, 3), dtype=np.float32)
            self._root_velocities = np.zeros((n, 3), dtype=np.float32)
            self._root_yaw_rates = np.zeros((n, 1), dtype=np.float32)
        else:
            names = list(data.dtype.names)
            # joint_0..joint_21 format (no time column); optionally load root_* for base tracking
            joint_cols = [f"joint_{i}" for i in range(len(K1_JOINT_NAMES))]
            if all(c in names for c in joint_cols):
                num_frames = len(data)
                self._joint_positions = np.zeros((num_frames, len(K1_JOINT_NAMES)), dtype=np.float32)
                for i in range(len(K1_JOINT_NAMES)):
                    self._joint_positions[:, i] = data[f"joint_{i}"]
                dt = 1.0 / (fps if fps is not None else DEFAULT_MOTION_FPS)
                times = np.arange(num_frames, dtype=np.float64) * dt
                # Load root position and quat when present (for dances with walking)
                self.has_root = all(c in names for c in ROOT_POS_COLS) and all(c in names for c in ROOT_QUAT_COLS)
                if self.has_root:
                    root_pos = np.stack([data[c] for c in ROOT_POS_COLS], axis=1).astype(np.float32)
                    self._first_frame_root_pos = root_pos[0:1].copy()
                    self._root_positions = root_pos - self._first_frame_root_pos
                    raw_root_quats = np.stack([data[c] for c in ROOT_QUAT_COLS], axis=1).astype(np.float32)
                    self._root_quats = self._resolve_root_quat_convention(raw_root_quats)
                    # Root velocity (world) from finite diff for walking rewards
                    self._root_velocities = np.zeros((num_frames, 3), dtype=np.float32)
                    if num_frames > 1:
                        self._root_velocities[:-1] = (self._root_positions[1:] - self._root_positions[:-1]) / dt
                        self._root_velocities[-1] = self._root_velocities[-2]
                    # Yaw rate from heading finite diff
                    q = self._root_quats
                    qw, qx, qy, qz = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
                    siny_cosp = 2.0 * (qw * qy - qz * qx)
                    cosy_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
                    yaw = np.arctan2(siny_cosp, cosy_cosp).astype(np.float32)
                    self._root_yaw_rates = np.zeros((num_frames, 1), dtype=np.float32)
                    if num_frames > 1:
                        dyaw = np.diff(yaw)
                        dyaw = np.clip(dyaw, -np.pi, np.pi)
                        self._root_yaw_rates[:-1, 0] = dyaw / dt
                        self._root_yaw_rates[-1, 0] = self._root_yaw_rates[-2, 0]
                else:
                    self._root_positions = np.zeros((num_frames, 3), dtype=np.float32)
                    self._root_quats = np.zeros((num_frames, 4), dtype=np.float32)
                    self._root_quats[:, 3] = 1.0
                    self._first_frame_root_pos = np.zeros((1, 3), dtype=np.float32)
                    self._root_velocities = np.zeros((num_frames, 3), dtype=np.float32)
                    self._root_yaw_rates = np.zeros((num_frames, 1), dtype=np.float32)
            else:
                time_col = None
                for c in ("time", "t", "Time", "T"):
                    if c in names:
                        time_col = c
                        break
                if time_col is None:
                    time_col = names[0]
                times = np.asarray(data[time_col], dtype=np.float64)
                n = len(times)
                self._joint_positions = np.zeros((n, len(K1_JOINT_NAMES)), dtype=np.float32)
                for i, k1_name in enumerate(K1_JOINT_NAMES):
                    csv_name = self._joint_column_map.get(k1_name, k1_name)
                    if csv_name in names:
                        self._joint_positions[:, i] = data[csv_name]
                    else:
                        raise KeyError(
                            f"CSV has no column for K1 joint '{k1_name}'. "
                            f"Available: {names}. Use joint_column_map or joint_0..joint_21."
                        )
                self._root_positions = np.zeros((n, 3), dtype=np.float32)
                self._root_quats = np.zeros((n, 4), dtype=np.float32)
                self._root_quats[:, 3] = 1.0
                self._first_frame_root_pos = np.zeros((1, 3), dtype=np.float32)
                self._root_velocities = np.zeros((n, 3), dtype=np.float32)
                self._root_yaw_rates = np.zeros((n, 1), dtype=np.float32)

        self._times = np.asarray(times, dtype=np.float64)
        self.duration = float(self._times[-1] - self._times[0]) if len(self._times) > 1 else 0.0
        self.num_frames = len(self._times)
        if fps is None and self.num_frames > 1:
            self.dt = float(self._times[1] - self._times[0])
            self.fps = 1.0 / self.dt
        else:
            self.fps = fps or DEFAULT_MOTION_FPS
            self.dt = 1.0 / self.fps

        self._pos_tensor = torch.tensor(self._joint_positions, dtype=torch.float32, device=self.device)
        # Joint velocities (rad/s) from finite difference for velocity tracking reward/obs
        self._joint_velocities = np.zeros_like(self._joint_positions, dtype=np.float32)
        if self.num_frames > 1:
            dt = 1.0 / self.fps
            self._joint_velocities[:-1] = (self._joint_positions[1:] - self._joint_positions[:-1]) / dt
            self._joint_velocities[-1] = self._joint_velocities[-2]
        # Sanity check: joint values must be in radians (K1 sim). If CSV is in degrees, error explodes and tracking ~0.
        max_abs = float(np.abs(self._joint_positions).max())
        if max_abs > 2 * np.pi:
            print(
                f"[WARNING] Motion CSV joint values have max |value| = {max_abs:.2f}. "
                "Expected radians (typical range ~0.5–2.0). If your CSV is in degrees, convert to radians "
                "or tracking reward will stay ~0."
            )
        else:
            print(f"Motion joint range (rad): min={self._joint_positions.min():.3f}, max={self._joint_positions.max():.3f}")
        if not hasattr(self, "_root_velocities"):
            self._root_velocities = np.zeros((self.num_frames, 3), dtype=np.float32)
            self._root_yaw_rates = np.zeros((self.num_frames, 1), dtype=np.float32)
        if getattr(self, "has_root", False):
            self._sanitize_root_quaternions()
            print(
                f"Motion loaded ({csv_path}): duration={self.duration:.2f}s, frames={self.num_frames}, "
                f"dt={self.dt:.4f}s, root tracking enabled (position + heading + velocity)"
            )
        else:
            print(
                f"Motion loaded ({csv_path}): duration={self.duration:.2f}s, frames={self.num_frames}, "
                f"dt={self.dt:.4f}s"
            )

    def _resolve_root_quat_convention(self, quats_from_csv: np.ndarray) -> np.ndarray:
        """Resolve root quaternion convention to semantic (qx, qy, qz, qw).

        Supported interpretations:
          A) Standard CSV semantics: [qx, qy, qz, qw]
          B) Common MuJoCo qpos dump mislabeled as root_q*: [qw, qx, qy, qz]
             (stored in CSV columns root_qx, root_qy, root_qz, root_qw respectively)
        """
        if quats_from_csv.shape[0] == 0:
            return quats_from_csv

        q_a = quats_from_csv.copy()
        # Remap B -> semantic XYZW
        # csv: [a,b,c,d] == [qw,qx,qy,qz]  => semantic [qx,qy,qz,qw] == [b,c,d,a]
        q_b = quats_from_csv[:, [1, 2, 3, 0]].copy()

        # Normalize both candidates before scoring.
        def _normalize(q: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            return q / norms

        q_a = _normalize(q_a)
        q_b = _normalize(q_b)

        up_a = _quat_up_proj_z_xyzw(q_a)
        up_b = _quat_up_proj_z_xyzw(q_b)
        score_a = float(np.mean(up_a))
        score_b = float(np.mean(up_b))

        # Choose the convention that yields more physically upright roots on average.
        # This avoids upside-down resets from mismatched CSV export conventions.
        if score_b > score_a + 0.2:
            print(
                "[INFO] Detected MuJoCo-style quaternion export under root_q* columns; "
                "remapping root quaternions from [qw,qx,qy,qz] to [qx,qy,qz,qw]."
            )
            return q_b.astype(np.float32)
        return q_a.astype(np.float32)

    def _sanitize_root_quaternions(self) -> None:
        """Normalize quaternions and enforce temporal sign continuity.

        CSV stores quaternions as (qx, qy, qz, qw). Since q and -q encode
        the same rotation, sign flips between neighboring frames can cause
        interpolation to traverse the long path and produce apparent flips.
        """
        if self._root_quats.shape[0] == 0:
            return
        q = self._root_quats.astype(np.float32, copy=True)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        bad = norms[:, 0] < 1e-8
        q[bad] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.maximum(norms, 1e-8)
        for i in range(1, q.shape[0]):
            if float(np.dot(q[i - 1], q[i])) < 0.0:
                q[i] *= -1.0
        self._root_quats = q

    def sample_at_times(self, times: torch.Tensor) -> torch.Tensor:
        """Sample joint positions at the given times (one per env). Loops time within [0, duration].

        Args:
            times: Shape (num_envs,) with time in seconds per env.

        Returns:
            Joint positions shape (num_envs, num_joints) on self.device.
        """
        t = times.cpu().numpy()
        t = np.clip(t, self._times[0], self._times[-1])
        if self.duration > 0:
            t = self._times[0] + (t - self._times[0]) % self.duration
        idx_0 = np.searchsorted(self._times, t, side="right") - 1
        idx_0 = np.clip(idx_0, 0, self.num_frames - 2)
        idx_1 = idx_0 + 1
        t0 = self._times[idx_0]
        t1 = self._times[idx_1]
        blend = np.clip((t - t0) / (t1 - t0 + 1e-8), 0.0, 1.0).astype(np.float32)
        pos0 = self._joint_positions[idx_0]
        pos1 = self._joint_positions[idx_1]
        # Interpolate angles on the shortest path to avoid wrap artifacts.
        dpos = np.arctan2(np.sin(pos1 - pos0), np.cos(pos1 - pos0)).astype(np.float32)
        out = pos0 + blend[:, None] * dpos
        return torch.tensor(out, dtype=torch.float32, device=self.device)

    def sample_joint_vel_at_times(self, times: torch.Tensor) -> torch.Tensor:
        """Sample joint velocities (rad/s) at the given times. Shape (num_envs, num_joints)."""
        return self._sample_at_times_impl(times, self._joint_velocities)

    def sample_at_frame_indices(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """Sample exact joint positions at integer frame indices."""
        idx = frame_indices.to(dtype=torch.long).clamp(0, self.num_frames - 1).cpu().numpy()
        return torch.tensor(self._joint_positions[idx], dtype=torch.float32, device=self.device)

    def sample_joint_vel_at_frame_indices(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """Sample exact joint velocities at integer frame indices."""
        idx = frame_indices.to(dtype=torch.long).clamp(0, self.num_frames - 1).cpu().numpy()
        return torch.tensor(self._joint_velocities[idx], dtype=torch.float32, device=self.device)

    def _sample_at_times_impl(self, times: torch.Tensor, values: np.ndarray) -> torch.Tensor:
        """Sample a (num_frames, dim) array at times; returns (num_envs, dim). Loops time."""
        t = times.cpu().numpy()
        t = np.clip(t, self._times[0], self._times[-1])
        if self.duration > 0:
            t = self._times[0] + (t - self._times[0]) % self.duration
        idx_0 = np.searchsorted(self._times, t, side="right") - 1
        idx_0 = np.clip(idx_0, 0, self.num_frames - 2)
        idx_1 = idx_0 + 1
        t0 = self._times[idx_0]
        t1 = self._times[idx_1]
        blend = np.clip((t - t0) / (t1 - t0 + 1e-8), 0.0, 1.0).astype(np.float32)
        v0 = values[idx_0]
        v1 = values[idx_1]
        out = (1 - blend)[:, None] * v0 + blend[:, None] * v1
        return torch.tensor(out, dtype=torch.float32, device=self.device)

    def sample_root_at_times(self, times: torch.Tensor) -> torch.Tensor:
        """Sample reference root position (displacement from motion start) at the given times.
        Returns shape (num_envs, 3): (x, y, z). When root data is not in CSV, returns zeros.
        """
        return self._sample_at_times_impl(times, self._root_positions)

    def sample_root_at_frame_indices(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """Sample exact root position (displacement from motion start) at integer frame indices."""
        idx = frame_indices.to(dtype=torch.long).clamp(0, self.num_frames - 1).cpu().numpy()
        return torch.tensor(self._root_positions[idx], dtype=torch.float32, device=self.device)

    def sample_root_quat_at_times(self, times: torch.Tensor) -> torch.Tensor:
        """Sample reference root quaternions (qx, qy, qz, qw) at times.

        Uses hemisphere-corrected normalized linear interpolation (nlerp) to
        avoid occasional orientation flips from quaternion sign ambiguity.
        """
        if not getattr(self, "has_root", False):
            out = torch.zeros(times.shape[0], 4, dtype=torch.float32, device=self.device)
            out[:, 3] = 1.0
            return out
        t = times.cpu().numpy()
        t = np.clip(t, self._times[0], self._times[-1])
        if self.duration > 0:
            t = self._times[0] + (t - self._times[0]) % self.duration
        idx_0 = np.searchsorted(self._times, t, side="right") - 1
        idx_0 = np.clip(idx_0, 0, self.num_frames - 2)
        idx_1 = idx_0 + 1
        t0 = self._times[idx_0]
        t1 = self._times[idx_1]
        blend = np.clip((t - t0) / (t1 - t0 + 1e-8), 0.0, 1.0).astype(np.float32)
        q0 = self._root_quats[idx_0].copy()
        q1 = self._root_quats[idx_1].copy()
        dots = np.sum(q0 * q1, axis=1, keepdims=True)
        q1 = np.where(dots < 0.0, -q1, q1)
        out = (1.0 - blend)[:, None] * q0 + blend[:, None] * q1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        bad = norms[:, 0] < 1e-8
        out = out / np.maximum(norms, 1e-8)
        out[bad] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return torch.tensor(out, dtype=torch.float32, device=self.device)

    def sample_root_quat_at_frame_indices(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """Sample exact root quaternion (qx, qy, qz, qw) at integer frame indices."""
        idx = frame_indices.to(dtype=torch.long).clamp(0, self.num_frames - 1).cpu().numpy()
        out = self._root_quats[idx].astype(np.float32, copy=True)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.maximum(norms, 1e-8)
        return torch.tensor(out, dtype=torch.float32, device=self.device)

    def sample_root_heading_at_times(self, times: torch.Tensor) -> torch.Tensor:
        """Sample reference yaw (heading) in radians at the given times.
        Returns shape (num_envs, 1). Derived from root quat (qx, qy, qz, qw). When no root quat, returns 0.
        """
        if not getattr(self, "has_root", False):
            return torch.zeros(times.shape[0], 1, dtype=torch.float32, device=self.device)
        q = self._root_quats
        qw, qx, qy, qz = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
        siny_cosp = 2.0 * (qw * qy - qz * qx)
        cosy_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        yaw = np.arctan2(siny_cosp, cosy_cosp).astype(np.float32)
        return self._sample_at_times_impl(times, yaw[:, np.newaxis])

    def sample_root_vel_at_times(self, times: torch.Tensor) -> torch.Tensor:
        """Sample reference root linear velocity (world frame) at the given times.
        Returns shape (num_envs, 3). Used for base_velocity_tracking reward so the robot walks.
        """
        return self._sample_at_times_impl(times, self._root_velocities)

    def sample_root_vel_at_frame_indices(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """Sample exact root linear velocity at integer frame indices."""
        idx = frame_indices.to(dtype=torch.long).clamp(0, self.num_frames - 1).cpu().numpy()
        return torch.tensor(self._root_velocities[idx], dtype=torch.float32, device=self.device)

    def sample_root_yaw_rate_at_times(self, times: torch.Tensor) -> torch.Tensor:
        """Sample reference yaw rate (rad/s) at the given times. Returns shape (num_envs, 1)."""
        return self._sample_at_times_impl(times, self._root_yaw_rates)

    def sample_root_yaw_rate_at_frame_indices(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """Sample exact root yaw rate (rad/s) at integer frame indices."""
        idx = frame_indices.to(dtype=torch.long).clamp(0, self.num_frames - 1).cpu().numpy()
        return torch.tensor(self._root_yaw_rates[idx], dtype=torch.float32, device=self.device)

    def sample_time_at_frame_indices(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """Sample exact timeline time (seconds) at integer frame indices."""
        idx = frame_indices.to(dtype=torch.long).clamp(0, self.num_frames - 1).cpu().numpy()
        return torch.tensor(self._times[idx], dtype=torch.float32, device=self.device)

    # -----------------------------------------------------------------
    # FK companion data (pre-computed body positions + COM)
    # -----------------------------------------------------------------
    def load_fk_data(self, npz_path: str) -> None:
        """Load pre-computed FK body positions and COM from companion .npz file.

        Expected keys:
          key_body_pos_rel: (num_frames, num_key_bodies, 3) relative to root
          com_pos_rel:      (num_frames, 3) relative to root
        """
        data = np.load(npz_path, allow_pickle=True)
        self._fk_key_body_pos_rel = data["key_body_pos_rel"].astype(np.float32)
        self._fk_com_pos_rel = data["com_pos_rel"].astype(np.float32)
        self.has_fk = True
        n_fk = self._fk_key_body_pos_rel.shape[0]
        if n_fk != self.num_frames:
            print(f"[WARNING] FK data has {n_fk} frames but motion has {self.num_frames}. Using min.")
        fk_names = data.get("key_body_names", [])
        print(f"FK data loaded ({npz_path}): {self._fk_key_body_pos_rel.shape[1]} key bodies, "
              f"{n_fk} frames, bodies={list(fk_names)}")

    def sample_fk_key_body_pos_rel_at_times(self, times: torch.Tensor) -> torch.Tensor:
        """Sample reference key body positions (relative to root) at times.
        Returns (num_envs, num_key_bodies, 3).
        """
        if not getattr(self, "has_fk", False):
            return torch.zeros(times.shape[0], 5, 3, dtype=torch.float32, device=self.device)
        n_bodies = self._fk_key_body_pos_rel.shape[1]
        flat = self._fk_key_body_pos_rel.reshape(self.num_frames, n_bodies * 3)
        sampled = self._sample_at_times_impl(times, flat)
        return sampled.reshape(times.shape[0], n_bodies, 3)

    def sample_fk_com_pos_rel_at_times(self, times: torch.Tensor) -> torch.Tensor:
        """Sample reference COM position (relative to root) at times. Returns (num_envs, 3)."""
        if not getattr(self, "has_fk", False):
            return torch.zeros(times.shape[0], 3, dtype=torch.float32, device=self.device)
        return self._sample_at_times_impl(times, self._fk_com_pos_rel)
