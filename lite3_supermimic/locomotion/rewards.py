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
  stair_front_shortfall: float = 0.0
  bound_pair_sync: float = 0.0
  bound_pair_timing: float = 0.0
  turn_contact_slip: float = 0.0
  turn_swing_clearance: float = 0.0
  turn_front_clearance: float = 0.0
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
                     "mid_straight_vx_error", "mid_straight_vy_error", "mid_straight_yaw_error",
                     "bound_pair_sync", "bound_pair_timing", "turn_contact_slip", "turn_swing_clearance",
                     "turn_front_clearance"}
    # These terms are in [0, 1]. A positive weight on any other term may
    # make the reward unbounded, so only cap when the bound is valid.
    positive_terms = [term for term in fields(self) if getattr(self, term.name) > 0]
    if any(term.name not in bounded_terms for term in positive_terms):
      return math.inf
    return sum(getattr(self, term.name) for term in positive_terms)


def bound_pair_errors(height, stance):
  # Use the same differentiable clearance proxy in simulation and imagination.
  support = torch.sigmoid((0.02 - height) / 0.008)
  fl, fr, hl, hr = support.unbind(-1)
  sync = 0.5 * ((fl - fr).square() + (hl - hr).square())
  # Match the front/rear contrast over the clock cycle, including flight.
  front_minus_hind = 0.5 * (fl + fr - hl - hr)
  expected = stance[:, :2].mean(-1) - stance[:, 2:].mean(-1)
  timing = (0.5 * (front_minus_hind - expected)).square()
  return sync, timing


def reward_terms(s, action, previous_action, command, consts, gait: GaitCfg, *,
                 foot_velocity_xy=None, front_clearance_enabled=False, foot_height=None, front_rise=None):
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
  height = st.feet_world_z(s) - st.FOOT_RADIUS if foot_height is None else foot_height
  bound_weight = (command[:, 10] == 2) * command[:, 9].square()
  pair_sync, pair_timing = bound_pair_errors(height, stance)
  turn = turn_slip = turn_clearance = front_clearance = torch.zeros_like(velocity[:, 2])
  if foot_velocity_xy is not None or front_clearance_enabled:
    turn = ((velocity[:, 2].abs() - 0.10) / 0.20).clamp(0., 1.)
    turn = turn * ((target_velocity[:, :2].abs().amax(dim=-1) <= 1e-4)
                   & (velocity[:, :2].norm(dim=-1) < 0.03))
  if foot_velocity_xy is not None:
    support = torch.sigmoid((0.012 - height) / 0.006)
    speed_error = -torch.expm1(-foot_velocity_xy.square().sum(-1) / 0.14**2)
    turn_slip = turn * (support * speed_error).mean(-1)
    clearance_shortfall = (swing_height.clamp(max=0.045) - height.clamp(min=0.)).clamp(0., 0.045)
    turn_clearance = turn * (clearance_shortfall / 0.045).square().mean(-1)
  if front_clearance_enabled:
    # In the model the shank capsule reaches 3 mm below the foot sphere.
    front_rise = height[:, :2] - 0.003
    front_target = swing_height[:, :2]
    front_shortfall = (front_target.clamp(max=0.035) - front_rise).clamp_min(0.)
    front_cost = -torch.expm1(-(front_shortfall / 0.035).square())
    front_clearance = turn * (front_cost * (front_target > 0.012)).mean(-1)
  stair_cost = torch.zeros_like(yaw_error)
  if front_rise is not None:
    # A swing near a riser should clear its tread by 2 cm at mid-swing.
    # Detach the terrain cue so reducing the deficit rewards lifting the foot.
    rise = front_rise.detach()
    arc = swing_height[:, :2] / max(gait.swing_height, 1e-6)
    deficit = ((rise + .02) * arc - height[:, :2]).clamp_min(0.)
    stair_cost = (arc * (rise / .02).clamp(max=1.) * deficit / .04).sum(-1)
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
    "stair_front_shortfall": stair_cost,
    "bound_pair_sync": bound_weight * pair_sync,
    "bound_pair_timing": bound_weight * pair_timing,
    "turn_contact_slip": turn_slip,
    "turn_swing_clearance": turn_clearance,
    "turn_front_clearance": front_clearance,
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
