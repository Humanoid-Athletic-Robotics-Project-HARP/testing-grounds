# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""K1 Dance environment: DeepMimic-style motion imitation from CSV.

All reference buffers are in **asset joint order** for direct comparison with
robot.data.joint_pos / joint_vel. FK-based reference body positions and COM
are loaded from a pre-computed companion .npz file.

Uses Reference State Initialization (RSI) with full root + joint state.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv

from .k1_stand_env_cfg import K1_JOINT_NAMES, DEFAULT_Q
from .k1_dance_env_cfg import JOINT_POSE_WEIGHT_TENSOR, K1_KEY_BODY_NAMES, K1_FOOT_BODY_NAMES
from .motion_loader_csv import MotionLoaderCSV


def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angles), torch.cos(angles))


class K1DanceEnv(ManagerBasedRLEnv):
    """Manager-based RL env for DeepMimic-style dance motion imitation."""

    cfg: "K1DanceEnvCfg"  # noqa: F821

    def __init__(self, cfg: "K1DanceEnvCfg", render_mode: str | None = None, **kwargs):  # noqa: F821
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self._motion_loader = MotionLoaderCSV(
            csv_path=cfg.motion_file,
            device=self.device,
            joint_column_map=getattr(cfg, "joint_column_map", None),
            fps=getattr(cfg, "motion_fps", None),
        )
        self._motion_duration = self._motion_loader.duration
        self._has_root = getattr(self._motion_loader, "has_root", False)

        # Load FK companion data if it exists
        fk_path = getattr(cfg, "fk_data_file", None)
        if fk_path is None:
            csv_stem = os.path.splitext(os.path.basename(cfg.motion_file))[0]
            fk_path = os.path.join(os.path.dirname(cfg.motion_file), f"{csv_stem}_fk.npz")
        self._has_fk = False
        if os.path.isfile(fk_path):
            self._motion_loader.load_fk_data(fk_path)
            self._has_fk = True
        else:
            print(f"[WARNING] FK data not found at {fk_path}. "
                  f"End-effector and COM rewards will return 1.0 (no penalty). "
                  f"Run: python scripts/tools/precompute_dance_fk.py --csv {cfg.motion_file}")

        robot = self.scene["robot"]

        asset_joint_names = robot.joint_names
        n_asset_joints = len(asset_joint_names)

        # Build K1-to-asset joint ordering (with assertion, no idx=0 fallback)
        self._k1_to_asset_indices = []
        for name in asset_joint_names:
            if name not in K1_JOINT_NAMES:
                raise ValueError(f"Asset joint '{name}' not in K1_JOINT_NAMES: {K1_JOINT_NAMES}")
            self._k1_to_asset_indices.append(K1_JOINT_NAMES.index(name))
        self._k1_to_asset_indices = torch.tensor(self._k1_to_asset_indices, device=self.device, dtype=torch.long)

        # Per-joint pose weights reordered to asset order
        k1_weights = JOINT_POSE_WEIGHT_TENSOR.to(self.device)
        self._joint_pose_weights = k1_weights[self._k1_to_asset_indices]

        # Reference buffers -- ALL in asset joint order
        self._ref_joint_pos = torch.zeros(self.num_envs, n_asset_joints, dtype=torch.float32, device=self.device)
        self._ref_joint_vel = torch.zeros(self.num_envs, n_asset_joints, dtype=torch.float32, device=self.device)
        self._ref_joint_pos_future = torch.zeros(self.num_envs, 2, n_asset_joints, dtype=torch.float32, device=self.device)
        self._motion_phase_offset = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._motion_phase = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Root reference buffers
        self._ref_base_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._ref_base_vel = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._ref_root_quat = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self._ref_root_quat[:, 0] = 1.0
        self._ref_root_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

        # FK reference buffers (key body positions + COM, relative to root)
        n_key = len(K1_KEY_BODY_NAMES)
        self._ref_key_body_pos_rel = torch.zeros(self.num_envs, n_key, 3, dtype=torch.float32, device=self.device)
        self._ref_com_pos_rel = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

        first_frame = getattr(self._motion_loader, "_first_frame_root_pos", None)
        if first_frame is not None:
            self._spawn_base_pos = torch.tensor(first_frame, dtype=torch.float32, device=self.device).expand(self.num_envs, 3)
        else:
            self._spawn_base_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

        # Resolve key body and foot body indices, tracking FK file alignment
        self._key_body_ids = []
        self._key_fk_ids = []
        for i, body_name in enumerate(K1_KEY_BODY_NAMES):
            try:
                ids, _ = robot.find_bodies([body_name], preserve_order=True)
                if ids:
                    self._key_body_ids.append(ids[0])
                    self._key_fk_ids.append(i)
            except Exception:
                pass
        self._key_fk_ids = torch.tensor(self._key_fk_ids, device=self.device, dtype=torch.long)

        self._foot_body_ids = []
        for body_name in K1_FOOT_BODY_NAMES:
            try:
                ids, _ = robot.find_bodies([body_name], preserve_order=True)
                if ids:
                    self._foot_body_ids.append(ids[0])
            except Exception:
                pass

        # Body masses for COM computation at runtime
        # IsaacLab exposes parsed link masses as `default_mass` (num_instances, num_bodies).
        # NOTE: On some IsaacLab builds, default_mass is on CPU even when the env runs on CUDA.
        # Keep masses on the same device as body_com_pos_w to avoid device mismatch in reward_com.
        self._body_masses = robot.data.default_mass[0].to(dtype=torch.float32).clone().to(self.device)  # (num_bodies,)
        self._total_mass = self._body_masses.sum().clamp(min=1e-6)

        # Action tracking
        self._current_action = None
        self._prev_action = None

    def _k1_to_asset_order(self, k1_data: torch.Tensor) -> torch.Tensor:
        return k1_data[:, self._k1_to_asset_indices]

    def _get_looped_times(self, t: torch.Tensor) -> torch.Tensor:
        if self._motion_duration > 0:
            t_offset = t + self._motion_phase_offset * self._motion_duration
            return t_offset % self._motion_duration
        return t

    def _sample_ref_at_times(self, t_looped: torch.Tensor):
        pos_k1 = self._motion_loader.sample_at_times(t_looped)
        vel_k1 = self._motion_loader.sample_joint_vel_at_times(t_looped)
        return self._k1_to_asset_order(pos_k1), self._k1_to_asset_order(vel_k1)

    def _sample_ref_at_frame_indices(self, frame_indices: torch.Tensor):
        pos_k1 = self._motion_loader.sample_at_frame_indices(frame_indices)
        vel_k1 = self._motion_loader.sample_joint_vel_at_frame_indices(frame_indices)
        return self._k1_to_asset_order(pos_k1), self._k1_to_asset_order(vel_k1)

    def _sample_root_state_at_times(self, t_looped: torch.Tensor):
        pos = self._motion_loader.sample_root_at_times(t_looped)
        vel = self._motion_loader.sample_root_vel_at_times(t_looped)
        q_csv = self._motion_loader.sample_root_quat_at_times(t_looped)
        quat = torch.zeros(t_looped.shape[0], 4, dtype=torch.float32, device=self.device)
        quat[:, 0] = q_csv[:, 3]  # w
        quat[:, 1] = q_csv[:, 0]  # x
        quat[:, 2] = q_csv[:, 1]  # y
        quat[:, 3] = q_csv[:, 2]  # z
        quat = quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return pos, quat, vel

    def _sample_root_state_at_frame_indices(self, frame_indices: torch.Tensor):
        pos = self._motion_loader.sample_root_at_frame_indices(frame_indices)
        vel = self._motion_loader.sample_root_vel_at_frame_indices(frame_indices)
        q_csv = self._motion_loader.sample_root_quat_at_frame_indices(frame_indices)
        quat = torch.zeros(frame_indices.shape[0], 4, dtype=torch.float32, device=self.device)
        quat[:, 0] = q_csv[:, 3]  # w
        quat[:, 1] = q_csv[:, 0]  # x
        quat[:, 2] = q_csv[:, 1]  # y
        quat[:, 3] = q_csv[:, 2]  # z
        quat = quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return pos, quat, vel

    def _compute_sim_com_rel(self) -> torch.Tensor:
        """Compute simulated COM position relative to root. Returns (num_envs, 3)."""
        robot = self.scene["robot"]
        all_body_com_pos = robot.data.body_com_pos_w[:, :, :3]  # (N, num_bodies, 3)
        root_pos = robot.data.root_pos_w[:, :3].unsqueeze(1)
        masses = self._body_masses.unsqueeze(0).unsqueeze(2)  # (1, num_bodies, 1)

        com = (all_body_com_pos * masses).sum(dim=1) / self._total_mass  # (N, 3)
        return com - root_pos.squeeze(1)

    def _update_reference_motion(self) -> None:
        t = self.episode_length_buf.to(torch.float32) * self.step_dt
        t_looped = self._get_looped_times(t)

        self._ref_joint_pos, self._ref_joint_vel = self._sample_ref_at_times(t_looped)

        # Phase (0 to 1, looped)
        if self._motion_duration > 0:
            self._motion_phase = t_looped / self._motion_duration
        else:
            self._motion_phase.zero_()

        # Future reference frames
        dt_step = self.step_dt
        t_f1 = self._get_looped_times(t + dt_step)
        t_f2 = self._get_looped_times(t + 2 * dt_step)
        self._ref_joint_pos_future[:, 0], _ = self._sample_ref_at_times(t_f1)
        self._ref_joint_pos_future[:, 1], _ = self._sample_ref_at_times(t_f2)

        # Root state
        if self._has_root:
            self._ref_base_pos, self._ref_root_quat, self._ref_base_vel = \
                self._sample_root_state_at_times(t_looped)
            yaw_rate = self._motion_loader.sample_root_yaw_rate_at_times(t_looped)
            self._ref_root_ang_vel.zero_()
            self._ref_root_ang_vel[:, 2] = yaw_rate.squeeze(-1)

        # FK body positions + COM
        if self._has_fk:
            self._ref_key_body_pos_rel = self._motion_loader.sample_fk_key_body_pos_rel_at_times(t_looped)
            self._ref_com_pos_rel = self._motion_loader.sample_fk_com_pos_rel_at_times(t_looped)

    def _reset_idx(self, env_ids: Sequence[int]):
        """RSI: full root + joint state from reference."""
        super()._reset_idx(env_ids)
        if self._motion_duration <= 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device)
        if env_ids.dim() == 0:
            env_ids = env_ids.unsqueeze(0)
        n = env_ids.shape[0]

        # Discrete-frame RSI reset: initialize from exact CSV frames (no interpolation).
        frame_indices = torch.randint(
            low=0, high=self._motion_loader.num_frames, size=(n,), device=self.device, dtype=torch.long
        )
        frame_times = self._motion_loader.sample_time_at_frame_indices(frame_indices)
        if self._motion_duration > 0:
            self._motion_phase_offset[env_ids] = frame_times / self._motion_duration
        else:
            self._motion_phase_offset[env_ids].zero_()

        q_ref_asset, qd_ref_asset = self._sample_ref_at_frame_indices(frame_indices)
        robot = self.scene["robot"]
        robot.write_joint_state_to_sim(q_ref_asset, qd_ref_asset, env_ids=env_ids)

        if self._has_root:
            ref_pos_disp, ref_quat, ref_vel = self._sample_root_state_at_frame_indices(frame_indices)
            root_state = robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] = self.scene.env_origins[env_ids] + self._spawn_base_pos[env_ids] + ref_pos_disp
            root_state[:, 3:7] = ref_quat
            root_state[:, 7:10] = ref_vel
            yaw_rate = self._motion_loader.sample_root_yaw_rate_at_frame_indices(frame_indices)
            root_state[:, 12] = yaw_rate.squeeze(-1)
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
            robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)

    def reset(self, seed: int | None = None, options: dict | None = None):
        out = super().reset(seed=seed, options=options)
        self._update_reference_motion()
        return out

    def step(self, action: torch.Tensor):
        self._prev_action = self._current_action
        self._current_action = action.clone() if action is not None else None
        self._update_reference_motion()
        return super().step(action)
