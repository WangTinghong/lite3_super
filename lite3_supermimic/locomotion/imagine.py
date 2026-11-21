"""Phase-conditioned velocity learning through the existing world model.

The sampled desired velocity is held over the short imagined horizon; command
ramps, gait selection and the phase clock advance exactly as in simulation.
"""
import torch
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.supermimic.imagine import Imagination
from lite3_supermimic.locomotion import gait
from lite3_supermimic.locomotion.actions import action_bounds
from lite3_supermimic.locomotion.rewards import reward_terms, weighted_reward
from lite3_supermimic.locomotion.stairs import front_rise_ahead, local_foot_heights
from lite3_supermimic.target import terrain as tr

OBS_TERMS = st.OBS_TERMS[:-1] + (("velocity_gait", gait.OBS_DIM),)


def obs_frame(s, action, command, encoder_bias, gait_cfg, noise=False):
  frame = st.obs_frame(s, action, gait.observation(command, gait_cfg), encoder_bias, noise)
  frame["velocity_gait"] = frame.pop("target")
  return frame


class LocomotionImagination(Imagination):
  def __init__(self, cfg, consts, gait_cfg, history, device, hipx_target_limit=0.0, field=None, front_preview=False):
    self.cfg, self.c, self.gait_cfg = cfg, consts, gait_cfg
    self.h, self.device, self.terms = history, device, OBS_TERMS
    self.front_preview = front_preview
    if front_preview:
      self.terms += (("front_rise", 2),)
    self.hipx_target_limit = hipx_target_limit
    self.field = field
    self.max_return = float("inf")
    if cfg.gamma < 1:
      self.max_return = cfg.weights.upper_reward_rate() * cfg.dt / (1 - cfg.gamma)
    self.critic_dim = st.STATE_DIM + gait.STATE_DIM + 1 + st.CONTEXT_DIM
    self._step_fn = torch.compile(self._step, dynamic=False) if cfg.compile else self._step

  def _generator_state(self, seed):
    return seed["gen"]

  def _bootstrap_values(self, values):
    return values.clamp_max(self.max_return)

  def critic_input(self, s, pose, command, xi):
    # Sin/cos phase encoding removes the discontinuity at cycle wraparound.
    phase = command[:, 6:7] * (2 * torch.pi)
    encoded = torch.cat([command[:, :6], phase.sin(), phase.cos(), command[:, 7:]], -1)
    return torch.cat([s, encoded, xi], -1)

  def fall_prob(self, s):
    orientation = super().fall_prob(s)
    low = torch.sigmoid((0.12 - s[:, 0]) / 0.01)
    return 1 - (1 - orientation) * (1 - low)

  def _step(self, s, pose, a_prev, hist, command, xi, enc_bias, k, dt, actor, wm):
    crit_in = self.critic_input(s, pose, command, xi)
    requested_action = actor(hist)
    action = requested_action
    if self.hipx_target_limit:
      low, high = action_bounds(k["default_q"], k["action_scale"], k["soft_limits"], self.hipx_target_limit)
      action = requested_action.clamp(low, high)
    relief = s.new_zeros(s.shape[0], st.SCAN_POINTS)
    if self.field is not None:
      relief = tr.relief(pose[:, :2], pose[:, 2], self.field, grad=True)
    s_next, dpose = wm(s, action, a_prev, xi, relief)
    pose_next = st.apply_pose_delta(pose, dpose)
    constants = st.RobotConsts(k["default_q"], k["action_scale"], self.c.kp,
                               self.c.kd, self.c.effort, k["soft_limits"])
    foot_velocity_xy = None
    if self.cfg.weights.turn_contact_slip or self.cfg.weights.turn_swing_clearance:
      previous_feet = st.limb_points_world(s, pose)["feet"][..., :2]
      next_feet = st.limb_points_world(s_next, pose_next)["feet"][..., :2]
      foot_velocity_xy = (next_feet - previous_feet) / dt.reshape(-1, 1, 1).clamp_min(1e-6)
    rise = None
    if self.front_preview:
      front = st.limb_points_world(s_next, pose_next)["feet"][:, :2, :2]
      rise = front_rise_ahead(front, pose_next[:, 2], self.field)
    terms = reward_terms(s_next, action, a_prev, command, constants, self.gait_cfg,
                         foot_velocity_xy=foot_velocity_xy, front_rise=rise,
                         front_clearance_enabled=bool(self.cfg.weights.turn_front_clearance),
                         foot_height=local_foot_heights(s_next, pose_next, self.field)
                         if self.field is not None else None)
    death = self.fall_prob(s_next)
    # Clamping zeros the dynamics gradient outside the limits. This penalty
    # still pushes the actor's raw requests back into the executable range.
    excess = (requested_action - action).square().sum(-1)
    reward = (weighted_reward(terms, self.cfg.weights) + self.cfg.weights.terminated * death
              - 0.05 * excess) * self.cfg.dt
    command = gait.advance(command, self.cfg.dt, self.gait_cfg)
    frame = obs_frame(s_next, action, command, enc_bias, self.gait_cfg, noise=True)
    if self.front_preview:
      frame["front_rise"] = rise / .02
    hist = st.push_history(hist, frame, self.h, self.terms)
    return s_next, pose_next, action, hist, command, reward, death, crit_in
