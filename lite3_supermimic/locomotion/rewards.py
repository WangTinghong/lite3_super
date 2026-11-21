"""Identical differentiable locomotion rewards in simulation and imagination."""
from __future__ import annotations
from dataclasses import dataclass, fields
import math
import torch
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.locomotion.gait import GaitCfg, foot_schedule
from lite3_supermimic.robot.constants import HIP_X_JOINT_INDICES


@dataclass(kw_only=True)
class RewardWeights:
  track_velocity: float = 2.0
  track_yaw: float = 1.0
  mid_straight_vx_error: float = 0.0
  mid_straight_vy_error: float = 0.0
  mid_straight_yaw_error: float = 0.0
  upright: float = 0.5
  base_height: float = 0.5
  gait_height: float = 1.0
  vertical_velocity: float = -0.2
  tilt_rate: float = -0.03
  joint_deviation: float = -0.03
  hipx_deviation: float = -1.0
  stand_pose: float = -0.5
  action_rate: float = -0.05
  joint_limits: float = -2.0
  torque: float = -2e-5
  terminated: float = -200.0

  def upper_reward_rate(self) -> float:
    bounded_terms = {"track_velocity", "track_yaw", "upright", "base_height", "gait_height", "terminated",
                     "mid_straight_vx_error", "mid_straight_vy_error", "mid_straight_yaw_error"}
    # These terms are in [0, 1]. A positive weight on any other term may
    # make the reward unbounded, so only cap when the bound is valid.
    positive_terms = [term for term in fields(self) if getattr(self, term.name) > 0]
    if any(term.name not in bounded_terms for term in positive_terms):
      return math.inf
    return sum(getattr(self, term.name) for term in positive_terms)


def reward_terms(s, action, previous_action, command, consts, gait: GaitCfg):
  target_velocity = command[:, :3]
  velocity = command[:, 3:6]
  planar_error = s[:, st.V_YAW][:, :2] - velocity[:, :2]
  yaw_error = s[:, st.W_B][:, 2] - velocity[:, 2]
  mid_speed_target = (target_velocity[:, 0] >= 1.10) & (target_velocity[:, 0] <= 1.30)
  straight_target = (target_velocity[:, 1:].abs() <= 1e-4).all(dim=-1)
  straight_command = (velocity[:, 1:].abs() <= 1e-4).all(dim=-1)
  trot = command[:, 10] == 1
  # Allow the initial acceleration, then ramp in the extra cost up to 1.10 m/s.
  speed_ramp = ((velocity[:, 0] - 0.90) / 0.20).clamp(0.0, 1.0)
  mid_straight = speed_ramp * (mid_speed_target & straight_target & straight_command & trot)
  _, _, stance, swing_height = foot_schedule(command, gait)
  height = st.feet_world_z(s) - st.FOOT_RADIUS
  # Stance feet stay on the floor, swing feet follow the gait's clearance arc.
  height_error = (height - swing_height).square()
  moving = (velocity[:, :2].norm(dim=-1) + velocity[:, 2].abs()) > gait.stand_speed
  joint_offset = s[:, st.Q]
  hipx_excess = (joint_offset[:, HIP_X_JOINT_INDICES].abs() - 0.1).clamp_min(0)
  # Leave room for balancing corrections, side steps, turns and bound.
  straight = torch.exp(-(velocity[:, 1] / 0.15).square() - (velocity[:, 2] / 0.35).square())
  hipx_weight = straight * (1.0 - command[:, 9])
  q = joint_offset + consts.default_q
  limits = consts.soft_limits
  desired_q = consts.default_q + action * consts.action_scale
  torque = (consts.kp * (desired_q - q) - consts.kd * s[:, st.QD]).clamp(-consts.effort, consts.effort)
  return {
    "track_velocity": torch.exp(-planar_error.square().sum(-1) / 0.25),
    "track_yaw": torch.exp(-yaw_error.square() / 0.25),
    "mid_straight_vx_error": -mid_straight * torch.expm1(-(planar_error[:, 0] / 0.15).square()),
    "mid_straight_vy_error": -mid_straight * torch.expm1(-(planar_error[:, 1] / 0.10).square()),
    "mid_straight_yaw_error": -mid_straight * torch.expm1(-(yaw_error / 0.20).square()),
    "upright": torch.exp(-s[:, st.GRAV][:, :2].square().sum(-1) / 0.16),
    "base_height": torch.exp(-(s[:, 0] - 0.28).square() / 0.0025),
    "gait_height": (torch.exp(-height_error / 0.0009) * (0.5 + 0.5 * stance)).mean(-1),
    "vertical_velocity": s[:, 6].square(),
    "tilt_rate": s[:, st.W_B][:, :2].square().sum(-1),
    "joint_deviation": joint_offset.square().sum(-1),
    "hipx_deviation": hipx_weight * hipx_excess.square().sum(-1),
    "stand_pose": joint_offset.square().sum(-1) * (~moving),
    "action_rate": (action - previous_action).square().sum(-1),
    "joint_limits": ((limits[..., 0] - q).clamp(min=0) + (q - limits[..., 1]).clamp(min=0)).sum(-1),
    "torque": torque.square().sum(-1),
  }


def weighted_reward(terms, weights: RewardWeights):
  return sum(getattr(weights, name) * value for name, value in terms.items())
