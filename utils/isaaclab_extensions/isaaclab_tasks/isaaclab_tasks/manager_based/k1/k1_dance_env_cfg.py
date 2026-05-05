# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""K1 Dance environment config: MimicKit-style motion imitation.

Reward structure (MimicKit / DeepMimic 5-term + regularization):
  r_p:  weighted joint angle tracking            (w=0.50)
  r_v:  joint velocity tracking                  (w=0.10)
  r_rp: root position + rotation tracking        (w=0.15)
  r_rv: root linear + angular velocity tracking  (w=0.10)
  r_e:  end-effector position tracking            (w=0.10)
  r_c:  center-of-mass tracking                  (w=0.05)
  penalty: action smoothness                     (w=-0.01)

Actions: residual to reference (target = q_ref + scale * action).
Control rate: 30 Hz (decimation=4, dt=1/120).
Phase: explicit (sin, cos) observation.
Trained from scratch with Reference State Initialization (RSI).
"""

from __future__ import annotations

import os

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

import utils.isaaclab_extensions.isaaclab_tasks.isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

try:
    from isaaclab.actuators import ImplicitActuatorCfg
except Exception:
    from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg

from .k1_stand_env_cfg import (
    K1_CFG,
    K1_JOINT_NAMES,
    K1_LEFT_FOOT_BODY_NAME,
    K1_RIGHT_FOOT_BODY_NAME,
)
from .residual_position_action import ResidualJointPositionActionCfg

# ---- Body name constants ----
K1_KEY_BODY_NAMES = [
    "Head_2",
    "left_hand_link",
    "right_hand_link",
    K1_LEFT_FOOT_BODY_NAME,
    K1_RIGHT_FOOT_BODY_NAME,
]
K1_FOOT_BODY_NAMES = [K1_LEFT_FOOT_BODY_NAME, K1_RIGHT_FOOT_BODY_NAME]

# ---- Per-joint weights (K1_JOINT_NAMES order, reordered to asset order in env) ----
_JOINT_POSE_WEIGHTS = {
    "AAHead_yaw": 0.1, "Head_pitch": 0.1,
    "ALeft_Shoulder_Pitch": 0.3, "ARight_Shoulder_Pitch": 0.3,
    "Left_Shoulder_Roll": 0.3, "Right_Shoulder_Roll": 0.3,
    "Left_Elbow_Pitch": 0.2, "Right_Elbow_Pitch": 0.2,
    "Left_Elbow_Yaw": 0.2, "Right_Elbow_Yaw": 0.2,
    "Left_Hip_Pitch": 1.0, "Right_Hip_Pitch": 1.0,
    "Left_Hip_Roll": 1.0, "Right_Hip_Roll": 1.0,
    "Left_Hip_Yaw": 1.0, "Right_Hip_Yaw": 1.0,
    "Left_Knee_Pitch": 0.8, "Right_Knee_Pitch": 0.8,
    "Left_Ankle_Pitch": 0.5, "Right_Ankle_Pitch": 0.5,
    "Left_Ankle_Roll": 0.5, "Right_Ankle_Roll": 0.5,
}
JOINT_POSE_WEIGHT_TENSOR = torch.tensor(
    [_JOINT_POSE_WEIGHTS[name] for name in K1_JOINT_NAMES], dtype=torch.float32
)

# ---- Robot config ----
_LEG_REGEX = "Left_Hip_.*|Right_Hip_.*|Left_Knee_.*|Right_Knee_.*|Left_Ankle_.*|Right_Ankle_.*"
_NON_LEG_REGEX = (
    "AAHead_yaw|ALeft_Shoulder_Pitch|ARight_Shoulder_Pitch|Head_pitch|"
    "Left_Shoulder_Roll|Right_Shoulder_Roll|Left_Elbow_Pitch|Right_Elbow_Pitch|Left_Elbow_Yaw|Right_Elbow_Yaw"
)
K1_CFG_DANCE = K1_CFG.replace(
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={_LEG_REGEX: 200.0, _NON_LEG_REGEX: 80.0},
            damping={_LEG_REGEX: 10.0, _NON_LEG_REGEX: 4.0},
            effort_limit=200.0,
            velocity_limit=30.0,
        )
    }
)


@configclass
class K1DanceSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )
    robot = K1_CFG_DANCE.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# =============================================================================
# MimicKit-style reward functions (5 imitation terms + smoothness penalty)
# =============================================================================

def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def _reward_pose(env) -> torch.Tensor:
    """r_p: weighted joint angle tracking with angle wrapping.
    exp(-0.25 * weighted_sum_sq_err) -- scale matched to MimicKit.
    """
    if not hasattr(env, "_ref_joint_pos"):
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    n = env._ref_joint_pos.shape[1]
    w = env._joint_pose_weights
    diff = _wrap_to_pi(robot.data.joint_pos[:, :n] - env._ref_joint_pos)
    weighted_err = (diff ** 2 * w).sum(dim=1)
    return torch.exp(-0.25 * weighted_err)


def _reward_vel(env) -> torch.Tensor:
    """r_v: joint velocity tracking. exp(-0.01 * sum_sq_err) -- scale matched to MimicKit."""
    if not hasattr(env, "_ref_joint_vel"):
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    n = env._ref_joint_vel.shape[1]
    w = env._joint_pose_weights
    err = (w * (robot.data.joint_vel[:, :n] - env._ref_joint_vel) ** 2).sum(dim=1)
    return torch.exp(-0.01 * err)


def _reward_end_effector(env) -> torch.Tensor:
    """r_e: end-effector (key body) position tracking (relative to root).
    exp(-10 * sum ||p_sim - p_ref||^2) -- scale matched to MimicKit.
    Falls back to 1.0 if FK data not available.
    """
    if not getattr(env, "_has_fk", False):
        return torch.ones(env.num_envs, device=env.device)
    if not hasattr(env, "_key_body_ids") or len(env._key_body_ids) == 0:
        return torch.ones(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    root_pos = robot.data.root_pos_w[:, :3].unsqueeze(1)
    sim_key_rel = robot.data.body_pos_w[:, env._key_body_ids, :3] - root_pos
    ref_key_rel = env._ref_key_body_pos_rel[:, env._key_fk_ids, :]
    err = (sim_key_rel - ref_key_rel).square().sum(dim=(1, 2))
    return torch.exp(-10.0 * err)


def _reward_com(env) -> torch.Tensor:
    """r_c: center-of-mass position tracking (relative to root).
    exp(-10 * ||com_sim - com_ref||^2).
    Falls back to 1.0 if FK data not available.
    """
    if not getattr(env, "_has_fk", False):
        return torch.ones(env.num_envs, device=env.device)
    sim_com_rel = env._compute_sim_com_rel()
    ref_com_rel = env._ref_com_pos_rel
    err = (sim_com_rel - ref_com_rel).square().sum(dim=1)
    return torch.exp(-10.0 * err)


def _reward_root_pose(env) -> torch.Tensor:
    """r_rp: root position + rotation tracking (MimicKit style).
    exp(-5.0 * (pos_err + 0.1 * rot_err)).
    When CSV has no root data, returns 1.0 (no penalty).
    """
    if not getattr(env, "_has_root", False):
        return torch.ones(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    sim_root_pos = robot.data.root_pos_w[:, :3]
    ref_root_pos = env._ref_base_pos
    if hasattr(env, "scene") and hasattr(env.scene, "env_origins"):
        ref_root_pos = ref_root_pos + env.scene.env_origins[:, :3]
        if hasattr(env, "_spawn_base_pos"):
            ref_root_pos = ref_root_pos + env._spawn_base_pos[:, :3]
    pos_err = (sim_root_pos - ref_root_pos).square().sum(dim=1)

    sim_root_quat = robot.data.root_quat_w
    ref_root_quat = env._ref_root_quat
    dot = (sim_root_quat * ref_root_quat).sum(dim=1).abs().clamp(max=1.0)
    rot_err = 2.0 * torch.acos(dot)
    rot_err = rot_err ** 2

    return torch.exp(-5.0 * (pos_err + 0.1 * rot_err))


def _reward_root_vel(env) -> torch.Tensor:
    """r_rv: root linear + angular velocity tracking (MimicKit style).
    exp(-1.0 * (lin_vel_err + 0.1 * ang_vel_err)).
    When CSV has no root data, returns 1.0 (no penalty).
    """
    if not getattr(env, "_has_root", False):
        return torch.ones(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    sim_lin_vel = robot.data.root_lin_vel_w[:, :3]
    ref_lin_vel = env._ref_base_vel
    lin_vel_err = (sim_lin_vel - ref_lin_vel).square().sum(dim=1)

    sim_ang_vel = robot.data.root_ang_vel_w[:, :3]
    ref_ang_vel = getattr(env, "_ref_root_ang_vel", torch.zeros_like(sim_ang_vel))
    ang_vel_err = (sim_ang_vel - ref_ang_vel).square().sum(dim=1)

    return torch.exp(-1.0 * (lin_vel_err + 0.1 * ang_vel_err))


def _penalty_action_smoothness(env) -> torch.Tensor:
    """Penalize jittery actions: mean(||a_t - a_{t-1}||^2)."""
    if env._prev_action is None or env._current_action is None:
        return torch.zeros(env.num_envs, device=env.device)
    return ((env._current_action - env._prev_action) ** 2).mean(dim=1)


# =============================================================================
# Observations
# =============================================================================

def _base_pos_z(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


def _projected_gravity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def _ref_joint_pos_obs(env) -> torch.Tensor:
    if hasattr(env, "_ref_joint_pos"):
        return env._ref_joint_pos
    return torch.zeros(env.num_envs, 22, device=env.device, dtype=torch.float32)


def _ref_joint_vel_obs(env) -> torch.Tensor:
    if hasattr(env, "_ref_joint_vel"):
        return env._ref_joint_vel
    return torch.zeros(env.num_envs, 22, device=env.device, dtype=torch.float32)


def _ref_joint_pos_future_obs(env) -> torch.Tensor:
    if hasattr(env, "_ref_joint_pos_future"):
        return env._ref_joint_pos_future.reshape(env.num_envs, -1)
    return torch.zeros(env.num_envs, 44, device=env.device, dtype=torch.float32)


def _key_body_pos_rel_obs(env) -> torch.Tensor:
    if hasattr(env, "_key_body_ids") and len(env._key_body_ids) > 0:
        robot = env.scene["robot"]
        root_pos = robot.data.root_pos_w[:, :3].unsqueeze(1)
        key_pos = robot.data.body_pos_w[:, env._key_body_ids, :3]
        return (key_pos - root_pos).reshape(env.num_envs, -1)
    return torch.zeros(env.num_envs, 15, device=env.device, dtype=torch.float32)


def _phase_obs(env) -> torch.Tensor:
    """Explicit phase: (sin(2*pi*phi), cos(2*pi*phi))."""
    if hasattr(env, "_motion_phase"):
        ang = 2.0 * torch.pi * env._motion_phase
        return torch.stack([torch.sin(ang), torch.cos(ang)], dim=1)
    return torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_pos_z = ObsTerm(func=_base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=_projected_gravity)
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        actions = ObsTerm(func=mdp.last_action)
        ref_joint_pos = ObsTerm(func=_ref_joint_pos_obs)
        ref_joint_vel = ObsTerm(func=_ref_joint_vel_obs)
        ref_joint_pos_future = ObsTerm(func=_ref_joint_pos_future_obs)
        key_body_pos_rel = ObsTerm(func=_key_body_pos_rel_obs)
        phase = ObsTerm(func=_phase_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Actions: residual to reference
# =============================================================================
@configclass
class DanceActionsCfg:
    residual_joint_position = ResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        offset=0.0,
        clip={".*": (-1.0, 1.0)},
    )


# =============================================================================
# Termination
# =============================================================================
def _fall_termination(env, minimum_height: float, min_up_proj: float, min_elapsed_steps: int,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    after_grace = env.episode_length_buf >= min_elapsed_steps
    height_below = asset.data.root_pos_w[:, 2] < minimum_height
    up_proj = (-asset.data.projected_gravity_b[:, 2]).squeeze(-1)
    tipped = up_proj < min_up_proj
    return torch.logical_and(after_grace, torch.logical_or(height_below, tipped))


def _tracking_error_termination(env, max_joint_err: float, min_elapsed_steps: int,
                                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    if not hasattr(env, "_ref_joint_pos"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    asset = env.scene[asset_cfg.name]
    after_grace = env.episode_length_buf >= min_elapsed_steps
    n = env._ref_joint_pos.shape[1]
    diff = torch.atan2(
        torch.sin(asset.data.joint_pos[:, :n] - env._ref_joint_pos),
        torch.cos(asset.data.joint_pos[:, :n] - env._ref_joint_pos),
    )
    mean_abs_err = diff.abs().mean(dim=1)
    return torch.logical_and(after_grace, mean_abs_err > max_joint_err)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(
        func=_fall_termination,
        params={"minimum_height": 0.15, "min_up_proj": 0.2, "min_elapsed_steps": 10},
    )
    tracking_error = DoneTerm(
        func=_tracking_error_termination,
        params={"max_joint_err": 0.8, "min_elapsed_steps": 10},
    )


# =============================================================================
# Rewards -- MimicKit 5-term structure + smoothness regularization
# Weights: pose=0.50, vel=0.10, root_pose=0.15, root_vel=0.10, ee=0.10, com=0.05
# =============================================================================
@configclass
class RewardsCfg:
    reward_pose = RewTerm(func=_reward_pose, weight=0.50)
    reward_vel = RewTerm(func=_reward_vel, weight=0.10)
    reward_root_pose = RewTerm(func=_reward_root_pose, weight=0.15)
    reward_root_vel = RewTerm(func=_reward_root_vel, weight=0.10)
    reward_end_effector = RewTerm(func=_reward_end_effector, weight=0.10)
    reward_com = RewTerm(func=_reward_com, weight=0.05)
    penalty_smoothness = RewTerm(func=_penalty_action_smoothness, weight=-0.01)


# =============================================================================
# Events
# =============================================================================
@configclass
class EventCfg:
    # No event terms: RSI in _reset_idx handles all state initialization.
    pass


# =============================================================================
# Env config
# =============================================================================
K1_DANCE_MOTIONS_DIR = os.path.join(os.path.dirname(__file__), "motions")
_DEFAULT_DANCE_CSV = os.environ.get(
    "K1_DANCE_CSV",
    os.path.join(K1_DANCE_MOTIONS_DIR, "dance.csv"),
)


@configclass
class K1DanceEnvCfg(ManagerBasedRLEnvCfg):
    """MimicKit-style K1 Dance: 5-term imitation reward + smoothness, residual actions, 30Hz, RSI."""

    motion_file: str = _DEFAULT_DANCE_CSV
    joint_column_map: dict[str, str] | None = None
    motion_fps: float | None = None
    fk_data_file: str | None = None

    scene: K1DanceSceneCfg = K1DanceSceneCfg(num_envs=4096, env_spacing=3.0, clone_in_fabric=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: DanceActionsCfg = DanceActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4  # 30 Hz policy rate (dt=1/120 * 4 = 1/30)
        self.episode_length_s = 30.0

        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.min_velocity_iteration_count = 1

        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0


# =============================================================================
# RSL-RL PPO config
# =============================================================================
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class K1DanceRslRlPpoCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 50
    experiment_name = "k1_dance"
    run_name = "k1_dance"
    clip_actions = 1.0
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.1,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
