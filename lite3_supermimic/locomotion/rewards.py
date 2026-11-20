"""Identical differentiable locomotion rewards in simulation and imagination."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.locomotion.gait import GaitCfg, foot_schedule


@dataclass(kw_only=True)
class RewardWeights:
  track_velocity: float = 2.0
  track_yaw: float = 1.0
  upright: float = 0.5
  base_height: float = 0.5
  gait_height: float = 1.0
  vertical_velocity: float = -0.2
  tilt_rate: float = -0.03
  joint_deviation: float = -0.03
  stand_pose: float = -0.5
  action_rate: float = -0.05
  joint_limits: float = -2.0
  torque: float = -2e-5
  terminated: float = -200.0


def reward_terms(s, action, previous_action, command, consts, gait: GaitCfg):
  velocity = command[:, 3:6]
  _, _, stance, swing_height = foot_schedule(command, gait)
  height = st.feet_world_z(s) - st.FOOT_RADIUS
  # Stance feet stay on the floor, swing feet follow the gait's clearance arc.
  height_error = (height - swing_height).square()
  moving = (velocity[:, :2].norm(dim=-1) + velocity[:, 2].abs()) > gait.stand_speed
  q = s[:, st.Q] + consts.default_q
  limits = consts.soft_limits
  desired_q = consts.default_q + action * consts.action_scale
  torque = (consts.kp * (desired_q - q) - consts.kd * s[:, st.QD]).clamp(-consts.effort, consts.effort)
  return {
    "track_velocity": torch.exp(-(s[:, st.V_YAW][:, :2] - velocity[:, :2]).square().sum(-1) / 0.25),
    "track_yaw": torch.exp(-(s[:, st.W_B][:, 2] - velocity[:, 2]).square() / 0.25),
    "upright": torch.exp(-s[:, st.GRAV][:, :2].square().sum(-1) / 0.16),
    "base_height": torch.exp(-(s[:, 0] - 0.28).square() / 0.0025),
    "gait_height": (torch.exp(-height_error / 0.0009) * (0.5 + 0.5 * stance)).mean(-1),
    "vertical_velocity": s[:, 6].square(),
    "tilt_rate": s[:, st.W_B][:, :2].square().sum(-1),
    "joint_deviation": s[:, st.Q].square().sum(-1),
    "stand_pose": s[:, st.Q].square().sum(-1) * (~moving),
    "action_rate": (action - previous_action).square().sum(-1),
    "joint_limits": ((limits[..., 0] - q).clamp(min=0) + (q - limits[..., 1]).clamp(min=0)).sum(-1),
    "torque": torque.square().sum(-1),
  }


def weighted_reward(terms, weights: RewardWeights):
  return sum(getattr(weights, name) * value for name, value in terms.items())
