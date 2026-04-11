# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Booster Robotics K1 humanoid robot."""

import os
import tempfile

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HOVER_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
_K1_URDF_PATH = os.path.join(_HOVER_ROOT, "third_party", "booster_assets", "robots", "K1", "K1_22dof.urdf")
_K1_USD_DIR = os.path.join(tempfile.gettempdir(), "IsaacLab", "k1_usd")

K1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_K1_URDF_PATH,
        usd_dir=_K1_USD_DIR,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        fix_base=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.56),
        joint_pos={
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
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=["AAHead_yaw", "Head_pitch"],
            effort_limit=6.0,
            velocity_limit=18.0,
            stiffness=10.0,
            damping=2.0,
        ),
        "legs": ImplicitActuatorCfg(
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
            stiffness={
                ".*_Hip_Pitch": 150.0,
                ".*_Hip_Roll": 150.0,
                ".*_Hip_Yaw": 100.0,
                ".*_Knee_Pitch": 200.0,
            },
            damping={
                ".*_Hip_Pitch": 5.0,
                ".*_Hip_Roll": 5.0,
                ".*_Hip_Yaw": 5.0,
                ".*_Knee_Pitch": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["Left_Ankle_Pitch", "Left_Ankle_Roll",
                              "Right_Ankle_Pitch", "Right_Ankle_Roll"],
            effort_limit=20.0,
            velocity_limit=18.1,
            stiffness=20.0,
            damping=4.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll",
                "Left_Elbow_Pitch", "Left_Elbow_Yaw",
                "ARight_Shoulder_Pitch", "Right_Shoulder_Roll",
                "Right_Elbow_Pitch", "Right_Elbow_Yaw",
            ],
            effort_limit=14.0,
            velocity_limit=18.0,
            stiffness=30.0,
            damping=5.0,
        ),
    },
)
"""Configuration for the Booster Robotics K1 Humanoid robot (22 DOF)."""
