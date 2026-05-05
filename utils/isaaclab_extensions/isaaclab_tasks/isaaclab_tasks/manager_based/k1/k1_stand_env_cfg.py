# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import utils.isaaclab_extensions.isaaclab_tasks.isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

# -----------------------------
# Actuator cfg import (API path varies a bit by IsaacLab version)
# -----------------------------
try:
    from isaaclab.actuators import ImplicitActuatorCfg
except Exception:  # pragma: no cover
    # older layouts
    from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg


# -----------------------------------------------------------------------------
# Robot asset (K1)
# -----------------------------------------------------------------------------
# Use K1_USD_PATH env var if set, otherwise fall back to default location
_DEFAULT_K1_USD = r"C:\Users\kylel\Humanoid\sim\K1_22dof\K1_22dof.usd"
K1_USD_PATH = os.environ.get("K1_USD_PATH", _DEFAULT_K1_USD)

# Foot link body names (from K1 USD: left_foot_link, right_foot_link) for future contact/feet rewards
K1_LEFT_FOOT_BODY_NAME = "left_foot_link"
K1_RIGHT_FOOT_BODY_NAME = "right_foot_link"

K1_JOINT_NAMES = [
    "AAHead_yaw",
    "ALeft_Shoulder_Pitch",
    "ARight_Shoulder_Pitch",
    "Left_Hip_Pitch",
    "Right_Hip_Pitch",
    "Head_pitch",
    "Left_Shoulder_Roll",
    "Right_Shoulder_Roll",
    "Left_Hip_Roll",
    "Right_Hip_Roll",
    "Left_Elbow_Pitch",
    "Right_Elbow_Pitch",
    "Left_Hip_Yaw",
    "Right_Hip_Yaw",
    "Left_Elbow_Yaw",
    "Right_Elbow_Yaw",
    "Left_Knee_Pitch",
    "Right_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Right_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Ankle_Roll",
]

# Default joint positions for standing pose.
# URDF analysis: at all-zero angles the arms extend laterally (T-pose).
# Shoulder Roll (X-axis) controls arms up/down:
#   Left_Shoulder_Roll  ~ -1.4 rad (-80 deg) = arms down by sides
#   Right_Shoulder_Roll ~ +1.4 rad (+80 deg) = arms down (mirrored)
# Shoulder Pitch (Y-axis) controls forward/backward; 0 = neutral.
DEFAULT_Q = {
    "AAHead_yaw": 0.0,
    "ALeft_Shoulder_Pitch": 0.0,
    "ARight_Shoulder_Pitch": 0.0,
    "Left_Hip_Pitch": -0.25,
    "Right_Hip_Pitch": -0.25,
    "Head_pitch": 0.0,
    "Left_Shoulder_Roll": -1.4,
    "Right_Shoulder_Roll": 1.4,
    "Left_Hip_Roll": 0.0,
    "Right_Hip_Roll": 0.0,
    "Left_Elbow_Pitch": 0.0,
    "Right_Elbow_Pitch": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Left_Elbow_Yaw": 0.0,
    "Right_Elbow_Yaw": 0.0,
    "Left_Knee_Pitch": 0.5,
    "Right_Knee_Pitch": 0.5,
    "Left_Ankle_Pitch": -0.25,
    "Right_Ankle_Pitch": -0.25,
    "Left_Ankle_Roll": 0.0,
    "Right_Ankle_Roll": 0.0,
}

# IMPORTANT: IsaacLab requires actuators to be provided.
# This config gives a basic implicit actuator to every joint.
K1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=K1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            fix_root_link=False,

        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),  # match observed upright root ~0.60 m so it can settle without height penalty
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=DEFAULT_Q,
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Name can be anything; it's just a key in the dict.
        "all_joints": ImplicitActuatorCfg(
            # Field name differs across some IsaacLab revs; joint_names_expr is common.
            # If you get an error about an unexpected keyword, tell me the exact message
            # and I’ll adapt it to your installed version.
            joint_names_expr=[".*"],
            stiffness=80.0,   # stiffer to hold standing pose
            damping=4.0,      # more damping to reduce oscillation
            effort_limit=200.0,
            velocity_limit=30.0,
        )
    },
)


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------
@configclass
class K1StandSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    robot = K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# -----------------------------------------------------------------------------
# MDP
# -----------------------------------------------------------------------------
@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={".*": 60.0},
    )


def _base_pos_z(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Root height (z) so policy can learn to maintain standing height."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


def _upright_continuous(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Continuous reward for upright: base up-vector projected on world z (1 = vertical, 0 = horizontal)."""
    asset = env.scene[asset_cfg.name]
    return (-asset.data.projected_gravity_b[:, 2]).squeeze(-1)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_pos_z = ObsTerm(func=_base_pos_z)  # height so policy can maintain it
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (-0.05, 0.05)},
    )


def _fall_after_grace(
    env,
    minimum_height: float,
    min_up_proj: float,
    min_elapsed_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Single fall condition (after grace): height below threshold OR base tipped.
    Keeps termination stats clean: Episode_Termination/fall + time_out ≈ 1.
    - Height = root_pos_w[:, 2]. - up_proj = -projected_gravity_b[:, 2] (1 = vertical).
    """
    asset = env.scene[asset_cfg.name]
    after_grace = env.episode_length_buf >= min_elapsed_steps
    height_below = asset.data.root_pos_w[:, 2] < minimum_height
    up_proj = (-asset.data.projected_gravity_b[:, 2]).squeeze(-1)
    tipped = up_proj < min_up_proj
    return torch.logical_and(after_grace, torch.logical_or(height_below, tipped))


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    upright_continuous = RewTerm(func=_upright_continuous, weight=1.5)
    upright_bonus = RewTerm(func=mdp.upright_posture_bonus, weight=0.5, params={"threshold": 0.9})
    height = RewTerm(func=mdp.base_height_l2, weight=-1.5, params={"target_height": 0.61})
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.005)
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            "Left_Hip_Pitch", "Right_Hip_Pitch",
            "Left_Hip_Roll", "Right_Hip_Roll",
            "Left_Hip_Yaw", "Right_Hip_Yaw",
            "Left_Knee_Pitch", "Right_Knee_Pitch",
            "Left_Ankle_Pitch", "Right_Ankle_Pitch",
            "Left_Ankle_Roll", "Right_Ankle_Roll",
        ])},
    )
    joint_deviation_upper = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["AAHead_yaw", "Head_pitch",
                         "ALeft_Shoulder_Pitch", "ARight_Shoulder_Pitch",
                         "Left_Shoulder_Roll", "Right_Shoulder_Roll",
                         "Left_Elbow_Pitch", "Right_Elbow_Pitch",
                         "Left_Elbow_Yaw", "Right_Elbow_Yaw"],
        )},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Single "fall" = (height < 0.48 m OR up_proj < 0.6) after 240 steps. Stats partition: fall + time_out ≈ 1.
    fall = DoneTerm(
        func=_fall_after_grace,
        params={"minimum_height": 0.48, "min_up_proj": 0.6, "min_elapsed_steps": 240},
    )


# -----------------------------------------------------------------------------
# Env config
# -----------------------------------------------------------------------------
@configclass
class K1StandEnvCfg(ManagerBasedRLEnvCfg):
    # Use fewer envs by default to avoid GPU OOM / silent crashes; override with --num_envs
    scene: K1StandSceneCfg = K1StandSceneCfg(num_envs=8, env_spacing=3.0, clone_in_fabric=False)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0  # longer episodes so robot has time to settle and learn

        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.min_velocity_iteration_count = 1

        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0


# -----------------------------------------------------------------------------
# RSL-RL PPO cfg
# -----------------------------------------------------------------------------
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class K1StandRslRlPpoCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for K1 Stand task."""

    num_steps_per_env = 24
    max_iterations = 5000   # longer training for standing
    save_interval = 50   # save often so you can roll back to a good policy (e.g. ~258) if training collapses
    experiment_name = "k1_stand"
    run_name = "k1_stand"
    clip_actions = 1.0

    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # smaller initial exploration so robot doesn't flail as much
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
    )
    # Safer PPO: smaller LR + fewer epochs to avoid policy collapse after good iterations (e.g. 258–260).
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )