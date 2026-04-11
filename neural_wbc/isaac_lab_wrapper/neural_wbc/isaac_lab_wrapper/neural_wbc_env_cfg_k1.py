# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from neural_wbc.core.modes import NeuralWBCModes
from neural_wbc.data import get_data_path

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from .events import NeuralWBCPlayEventCfg, NeuralWBCTrainEventCfg
from .k1_cfg import K1_CFG
from .neural_wbc_env_cfg import NeuralWBCEnvCfg
from .rewards import NeuralWBCRewardCfg
from .terrain import HARD_ROUGH_TERRAINS_CFG, flat_terrain

@configclass
class NeuralWBCRewardCfgK1(NeuralWBCRewardCfg):
    # fmt: off
    # Order follows K1 joint_names:
    # AAHead_yaw, Head_pitch,
    # ALeft_Shoulder_Pitch, Left_Shoulder_Roll, Left_Elbow_Pitch, Left_Elbow_Yaw,
    # ARight_Shoulder_Pitch, Right_Shoulder_Roll, Right_Elbow_Pitch, Right_Elbow_Yaw,
    # Left_Hip_Pitch, Left_Hip_Roll, Left_Hip_Yaw, Left_Knee_Pitch, Left_Ankle_Pitch, Left_Ankle_Roll,
    # Right_Hip_Pitch, Right_Hip_Roll, Right_Hip_Yaw, Right_Knee_Pitch, Right_Ankle_Pitch, Right_Ankle_Roll
    torque_limits = [
        6.0, 6.0,
        14.0, 14.0, 14.0, 14.0,
        14.0, 14.0, 14.0, 14.0,
        30.0, 35.0, 20.0, 40.0, 20.0, 20.0,
        30.0, 35.0, 20.0, 40.0, 20.0, 20.0,
    ]
    joint_pos_limits = [
        (-1.0, 1.0), (-0.349, 0.855),
        (-3.316, 1.22), (-1.74, 1.57), (-2.27, 2.27), (-2.44, 0.0),
        (-3.316, 1.22), (-1.57, 1.74), (-2.27, 2.27), (0.0, 2.44),
        (-3.0, 2.21), (-0.4, 1.57), (-1.0, 1.0), (0.0, 2.23), (-0.87, 0.345), (-0.345, 0.345),
        (-3.0, 2.21), (-1.57, 0.4), (-1.0, 1.0), (0.0, 2.23), (-0.87, 0.345), (-0.345, 0.345),
    ]
    joint_vel_limits = [
        18.0, 18.0,
        18.0, 18.0, 18.0, 18.0,
        18.0, 18.0, 18.0, 18.0,
        7.1, 12.9, 18.1, 12.5, 18.1, 18.1,
        7.1, 12.9, 18.1, 12.5, 18.1, 18.1,
    ]
    # fmt: on


DISTILL_MASK_MODES_ALL_K1 = {
    "exbody": {
        "upper_body": [
            ".*Shoulder.*",
            ".*Elbow.*",
        ],
        "lower_body": ["root.*"],
    },
    "humanplus": {
        "upper_body": [
            ".*Shoulder.*",
            ".*Elbow.*",
        ],
        "lower_body": [
            ".*Hip.*",
            ".*Knee.*",
            ".*Ankle.*",
            "root.*",
        ],
    },
    "h2o": {
        "upper_body": [
            ".*Arm.*",
            ".*hand_link.*",
        ],
        "lower_body": [".*foot_link.*"],
    },
    "omnih2o": {
        "upper_body": [".*hand_link.*", "Head_2"],
    },
}


@configclass
class NeuralWBCEnvCfgK1(NeuralWBCEnvCfg):
    # General parameters:
    action_space = 22
    observation_space = 916
    state_space = 1001

    # Distillation parameters:
    single_history_dim = 72
    observation_history_length = 25

    # Mask setup for an OH2O specialist policy as default:
    distill_mask_sparsity_randomization_enabled = False
    distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL_K1["omnih2o"]}

    # Robot geometry / actuation parameters:
    actuators = {
        "head": IdealPDActuatorCfg(
            joint_names_expr=["AAHead_yaw", "Head_pitch"],
            effort_limit=6.0,
            velocity_limit=18.0,
            stiffness=0,
            damping=0,
        ),
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
                "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
                "Left_Knee_Pitch", "Right_Knee_Pitch",
            ],
            effort_limit={
                ".*_Hip_Pitch": 30.0,
                ".*_Hip_Roll": 35.0,
                ".*_Hip_Yaw": 20.0,
                ".*_Knee_Pitch": 40.0,
            },
            velocity_limit={
                ".*_Hip_Pitch": 7.1,
                ".*_Hip_Roll": 12.9,
                ".*_Hip_Yaw": 18.1,
                ".*_Knee_Pitch": 12.5,
            },
            stiffness=0,
            damping=0,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[
                "Left_Ankle_Pitch", "Left_Ankle_Roll",
                "Right_Ankle_Pitch", "Right_Ankle_Roll",
            ],
            effort_limit=20.0,
            velocity_limit=18.1,
            stiffness=0,
            damping=0,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll",
                "Left_Elbow_Pitch", "Left_Elbow_Yaw",
                "ARight_Shoulder_Pitch", "Right_Shoulder_Roll",
                "Right_Elbow_Pitch", "Right_Elbow_Yaw",
            ],
            effort_limit=14.0,
            velocity_limit=18.0,
            stiffness=0,
            damping=0,
        ),
    }

    robot: ArticulationCfg = K1_CFG.replace(prim_path="/World/envs/env_.*/Robot", actuators=actuators)

    # Body names in MJCF traversal order (matches skeleton file)
    body_names = [
        "Trunk",
        "Head_1",
        "Head_2",
        "Left_Arm_1",
        "Left_Arm_2",
        "Left_Arm_3",
        "left_hand_link",
        "Right_Arm_1",
        "Right_Arm_2",
        "Right_Arm_3",
        "right_hand_link",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Shank",
        "Left_Ankle_Cross",
        "left_foot_link",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Shank",
        "Right_Ankle_Cross",
        "right_foot_link",
    ]

    # Joint names in MJCF order
    joint_names = [
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

    # Lower body: hips + knees + ankles (indices 10-21)
    lower_body_joint_ids = list(range(10, 22))
    # Upper body: head + arms (indices 0-9)
    upper_body_joint_ids = list(range(0, 10))

    base_name = "Trunk"
    root_id = body_names.index(base_name)

    feet_name = ["left_foot_link", "right_foot_link"]
    feet_joint_ids = [15, 21]  # Left_Ankle_Roll, Right_Ankle_Roll

    # K1 has physical hand and head links -- no virtual extension needed
    extend_body_parent_names = []
    extend_body_names = []
    extend_body_pos = torch.zeros(0, 3)

    tracked_body_names = [
        "Trunk",
        "Head_1",
        "Head_2",
        "Left_Arm_1",
        "Left_Arm_2",
        "Left_Arm_3",
        "left_hand_link",
        "Right_Arm_1",
        "Right_Arm_2",
        "Right_Arm_3",
        "right_hand_link",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Shank",
        "Left_Ankle_Cross",
        "left_foot_link",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Shank",
        "Right_Ankle_Cross",
        "right_foot_link",
    ]

    # PD gains per joint
    stiffness = {
        "AAHead_yaw": 10.0,
        "Head_pitch": 10.0,
        "ALeft_Shoulder_Pitch": 30.0,
        "Left_Shoulder_Roll": 30.0,
        "Left_Elbow_Pitch": 30.0,
        "Left_Elbow_Yaw": 30.0,
        "ARight_Shoulder_Pitch": 30.0,
        "Right_Shoulder_Roll": 30.0,
        "Right_Elbow_Pitch": 30.0,
        "Right_Elbow_Yaw": 30.0,
        "Left_Hip_Pitch": 150.0,
        "Left_Hip_Roll": 150.0,
        "Left_Hip_Yaw": 100.0,
        "Left_Knee_Pitch": 200.0,
        "Left_Ankle_Pitch": 20.0,
        "Left_Ankle_Roll": 20.0,
        "Right_Hip_Pitch": 150.0,
        "Right_Hip_Roll": 150.0,
        "Right_Hip_Yaw": 100.0,
        "Right_Knee_Pitch": 200.0,
        "Right_Ankle_Pitch": 20.0,
        "Right_Ankle_Roll": 20.0,
    }

    damping = {
        "AAHead_yaw": 2.0,
        "Head_pitch": 2.0,
        "ALeft_Shoulder_Pitch": 5.0,
        "Left_Shoulder_Roll": 5.0,
        "Left_Elbow_Pitch": 5.0,
        "Left_Elbow_Yaw": 5.0,
        "ARight_Shoulder_Pitch": 5.0,
        "Right_Shoulder_Roll": 5.0,
        "Right_Elbow_Pitch": 5.0,
        "Right_Elbow_Yaw": 5.0,
        "Left_Hip_Pitch": 5.0,
        "Left_Hip_Roll": 5.0,
        "Left_Hip_Yaw": 5.0,
        "Left_Knee_Pitch": 5.0,
        "Left_Ankle_Pitch": 4.0,
        "Left_Ankle_Roll": 4.0,
        "Right_Hip_Pitch": 5.0,
        "Right_Hip_Roll": 5.0,
        "Right_Hip_Yaw": 5.0,
        "Right_Knee_Pitch": 5.0,
        "Right_Ankle_Pitch": 4.0,
        "Right_Ankle_Roll": 4.0,
    }

    mass_randomized_body_names = [
        "Trunk",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
    ]

    undesired_contact_body_names = [
        "Trunk",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Shank",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Shank",
    ]

    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    def __post_init__(self):
        super().__post_init__()

        self.rewards = NeuralWBCRewardCfgK1()
        self.reference_motion_manager.motion_path = get_data_path("motions/k1_fight_001.pkl")
        self.reference_motion_manager.skeleton_path = get_data_path("motion_lib/k1.xml")

        if self.terrain.terrain_generator == HARD_ROUGH_TERRAINS_CFG:
            self.events.update_curriculum.params["penalty_level_up_threshold"] = 125

        if self.mode == NeuralWBCModes.TRAIN:
            self.episode_length_s = 20.0
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "Trunk"
        elif self.mode == NeuralWBCModes.DISTILL:
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "Trunk"
            self.add_policy_obs_noise = False
            self.reset_mask = True
            num_regions = len(self.distill_mask_modes)
            if num_regions == 1:
                region_modes = list(self.distill_mask_modes.values())[0]
                if len(region_modes) == 1:
                    self.reset_mask = False
        elif self.mode == NeuralWBCModes.TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.add_policy_obs_noise = False
            self.resample_motions = False
            self.distill_mask_sparsity_randomization_enabled = False
            self.distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL_K1["omnih2o"]}
        elif self.mode == NeuralWBCModes.DISTILL_TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.distill_teleop_selected_keypoints_names = []
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.default_rfi_lim = 0.0
            self.add_policy_obs_noise = False
            self.resample_motions = False
            self.distill_mask_sparsity_randomization_enabled = False
            self.distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL_K1["omnih2o"]}
        else:
            raise ValueError(f"Unsupported mode {self.mode}")
