# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from dataclasses import dataclass

from inference_env.neural_wbc_env_cfg import NeuralWBCEnvCfg

from neural_wbc.core.mask import calculate_mask_length
from neural_wbc.data import get_data_path


@dataclass
class NeuralWBCEnvCfgK1(NeuralWBCEnvCfg):
    decimation = 4
    dt = 0.005
    max_episode_length_s = 3600
    action_scale = 0.25
    ctrl_delay_step_range = [2, 2]
    default_rfi_lim = 0
    robot = "mujoco_robot"

    # K1 has physical hand and head links -- no virtual extension needed
    extend_body_parent_names = []
    extend_body_names = []
    extend_body_pos = torch.zeros(0, 3)

    tracked_body_names = [
        "left_hand_link",
        "right_hand_link",
        "Head_2",
    ]

    # Distillation parameters:
    single_history_dim = 72  # 2*22 + 6 + 22
    observation_history_length = 25
    num_bodies = 23
    num_joints = 22
    mask_length = calculate_mask_length(
        num_bodies=num_bodies,
        num_joints=num_joints,
    )

    control_type = "Pos"
    robot_actuation_type = "Torque"

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

    effort_limit = {
        "AAHead_yaw": 6.0,
        "Head_pitch": 6.0,
        "ALeft_Shoulder_Pitch": 14.0,
        "Left_Shoulder_Roll": 14.0,
        "Left_Elbow_Pitch": 14.0,
        "Left_Elbow_Yaw": 14.0,
        "ARight_Shoulder_Pitch": 14.0,
        "Right_Shoulder_Roll": 14.0,
        "Right_Elbow_Pitch": 14.0,
        "Right_Elbow_Yaw": 14.0,
        "Left_Hip_Pitch": 30.0,
        "Left_Hip_Roll": 35.0,
        "Left_Hip_Yaw": 20.0,
        "Left_Knee_Pitch": 40.0,
        "Left_Ankle_Pitch": 20.0,
        "Left_Ankle_Roll": 20.0,
        "Right_Hip_Pitch": 30.0,
        "Right_Hip_Roll": 35.0,
        "Right_Hip_Yaw": 20.0,
        "Right_Knee_Pitch": 40.0,
        "Right_Ankle_Pitch": 20.0,
        "Right_Ankle_Roll": 20.0,
    }

    position_limit = {
        "AAHead_yaw": [-1.0, 1.0],
        "Head_pitch": [-0.349, 0.855],
        "ALeft_Shoulder_Pitch": [-3.316, 1.22],
        "Left_Shoulder_Roll": [-1.74, 1.57],
        "Left_Elbow_Pitch": [-2.27, 2.27],
        "Left_Elbow_Yaw": [-2.44, 0.0],
        "ARight_Shoulder_Pitch": [-3.316, 1.22],
        "Right_Shoulder_Roll": [-1.57, 1.74],
        "Right_Elbow_Pitch": [-2.27, 2.27],
        "Right_Elbow_Yaw": [0.0, 2.44],
        "Left_Hip_Pitch": [-3.0, 2.21],
        "Left_Hip_Roll": [-0.4, 1.57],
        "Left_Hip_Yaw": [-1.0, 1.0],
        "Left_Knee_Pitch": [0.0, 2.23],
        "Left_Ankle_Pitch": [-0.87, 0.345],
        "Left_Ankle_Roll": [-0.345, 0.345],
        "Right_Hip_Pitch": [-3.0, 2.21],
        "Right_Hip_Roll": [-1.57, 0.4],
        "Right_Hip_Yaw": [-1.0, 1.0],
        "Right_Knee_Pitch": [0.0, 2.23],
        "Right_Ankle_Pitch": [-0.87, 0.345],
        "Right_Ankle_Roll": [-0.345, 0.345],
    }

    robot_init_state = {
        "base_pos": [0.0, 0.0, 0.56],
        "base_quat": [1.0, 0.0, 0.0, 0.0],
        "joint_pos": {
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            "ALeft_Shoulder_Pitch": 0.0,
            "Left_Shoulder_Roll": 0.0,
            "Left_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": 0.0,
            "ARight_Shoulder_Pitch": 0.0,
            "Right_Shoulder_Roll": 0.0,
            "Right_Elbow_Pitch": 0.0,
            "Right_Elbow_Yaw": 0.0,
            "Left_Hip_Pitch": -0.28,
            "Left_Hip_Roll": 0.0,
            "Left_Hip_Yaw": 0.0,
            "Left_Knee_Pitch": 0.56,
            "Left_Ankle_Pitch": -0.28,
            "Left_Ankle_Roll": 0.0,
            "Right_Hip_Pitch": -0.28,
            "Right_Hip_Roll": 0.0,
            "Right_Hip_Yaw": 0.0,
            "Right_Knee_Pitch": 0.56,
            "Right_Ankle_Pitch": -0.28,
            "Right_Ankle_Roll": 0.0,
        },
        "joint_vel": {},
    }

    # Lower and upper body joint ids in the MJCF model.
    lower_body_joint_ids = list(range(10, 22))  # hips, knees, ankles
    upper_body_joint_ids = list(range(0, 10))   # head, shoulders, elbows

    def __post_init__(self):
        self.reference_motion_cfg.motion_path = get_data_path("motions/k1_fight_001.pkl")
        self.reference_motion_cfg.skeleton_path = get_data_path("motion_lib/k1.xml")
        self.reference_motion_cfg.fk_frame_rotation = [0.5, 0.5, 0.5, 0.5]
