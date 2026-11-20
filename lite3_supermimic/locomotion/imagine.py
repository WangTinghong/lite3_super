"""Phase-conditioned velocity learning through the existing world model.

The sampled desired velocity is held over the short imagined horizon; command
ramps, gait selection and the phase clock advance exactly as in simulation.
"""
import torch
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.supermimic.imagine import Imagination
from lite3_supermimic.locomotion import gait
from lite3_supermimic.locomotion.rewards import reward_terms, weighted_reward

OBS_TERMS = st.OBS_TERMS[:-1] + (("velocity_gait", gait.OBS_DIM),)


def obs_frame(s, action, command, encoder_bias, gait_cfg, noise=False):
  frame = st.obs_frame(s, action, gait.observation(command, gait_cfg), encoder_bias, noise)
  frame["velocity_gait"] = frame.pop("target")
  return frame


class LocomotionImagination(Imagination):
  def __init__(self, cfg, consts, gait_cfg, history, device):
    self.cfg, self.c, self.gait_cfg = cfg, consts, gait_cfg
    self.h, self.device, self.terms = history, device, OBS_TERMS
    self.critic_dim = st.STATE_DIM + gait.STATE_DIM + 1 + st.CONTEXT_DIM
    self._step_fn = torch.compile(self._step, dynamic=False) if cfg.compile else self._step

  def _generator_state(self, seed):
    return seed["gen"]

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
    action = actor(hist)
    relief = s.new_zeros(s.shape[0], st.SCAN_POINTS)
    s_next, dpose = wm(s, action, a_prev, xi, relief)
    pose_next = st.apply_pose_delta(pose, dpose)
    constants = st.RobotConsts(k["default_q"], k["action_scale"], self.c.kp,
                               self.c.kd, self.c.effort, k["soft_limits"])
    terms = reward_terms(s_next, action, a_prev, command, constants, self.gait_cfg)
    death = self.fall_prob(s_next)
    reward = (weighted_reward(terms, self.cfg.weights) + self.cfg.weights.terminated * death) * self.cfg.dt
    command = gait.advance(command, self.cfg.dt, self.gait_cfg)
    frame = obs_frame(s_next, action, command, enc_bias, self.gait_cfg, noise=True)
    hist = st.push_history(hist, frame, self.h, self.terms)
    return s_next, pose_next, action, hist, command, reward, death, crit_in
