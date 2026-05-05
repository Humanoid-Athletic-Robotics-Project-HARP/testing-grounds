# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Residual joint position action: q_cmd = q_ref(t) + scale * action.

Used for motion imitation so the policy outputs a small delta around the reference
and the reference carries the choreography. Requires the env to set _ref_joint_pos
each step (e.g. from a motion loader) in the same joint order as the articulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

from isaaclab.envs.mdp.actions import joint_actions
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ResidualJointPositionAction(joint_actions.JointAction):
    """Apply position targets as q_ref + scale * action. Env must provide _ref_joint_pos each step."""

    cfg: "ResidualJointPositionActionCfg"

    def __init__(self, cfg: "ResidualJointPositionActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._offset = 0.0

    def apply_actions(self) -> None:
        env = self._env
        asset: Articulation = self._asset
        if getattr(env, "_ref_joint_pos", None) is None:
            asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
            return
        # _ref_joint_pos is already in asset joint order (reordered by K1DanceEnv._k1_to_asset_order)
        target = env._ref_joint_pos + self.processed_actions
        asset.set_joint_position_target(target, joint_ids=self._joint_ids)


@configclass
class ResidualJointPositionActionCfg(JointActionCfg):
    """Config for residual position action: target = q_ref + scale * action."""

    class_type: type[ActionTerm] = ResidualJointPositionAction
