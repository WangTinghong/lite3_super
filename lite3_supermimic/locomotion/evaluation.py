"""Shared simulator playback and metrics for Lite3 velocity policies."""
import math

import torch

from lite3_supermimic.locomotion.actions import action_bounds
from lite3_supermimic.locomotion.gait import GAITS, foot_schedule
from lite3_supermimic.robot.constants import HIP_X_JOINT_INDICES, LEGS
from lite3_supermimic.supermimic import state as st


class VelocityEvaluation:
  """Share command playback and metrics between the batch loop and mjlab viewer."""

  def __init__(self, env, *, vx=None, vy=0.0, yaw_rate=0.0, hipx_target_limit=0.0,
               vx_sequence=None, segment_seconds=5.0):
    self.unwrapped = env
    self.cfg, self.num_envs, self.device = env.cfg, env.num_envs, env.device
    self.term = env.command_manager.get_term("velocity")
    self.action_bounds = None
    if hipx_target_limit:
      consts = st.robot_consts(env)
      self.action_bounds = action_bounds(consts.default_q, consts.action_scale,
                                         consts.soft_limits, hipx_target_limit)
    self.vx_sequence = tuple(vx_sequence) if vx_sequence is not None else None
    if self.vx_sequence is not None:
      speeds = self.vx_sequence
    elif vx is not None:
      speeds = (vx,)
    else:
      speeds = (0.0, 0.2, 0.8, 1.8, 0.8, 0.0)
    self.command_cycle = torch.tensor(
      [(speed, vy, yaw_rate) for speed in speeds],
      device=self.device, dtype=self.term.state.dtype,
    )
    self.segment_steps = max(1, round(segment_seconds / env.step_dt))
    self.writer = None
    self.video_stride = max(1, round(1 / (25 * env.step_dt)))
    self.step_index = 0
    self.metrics = {name: self._empty_metrics() for name in ("stand", *GAITS)}
    self.segment_metrics = [self._empty_metrics() for _ in self.vx_sequence or ()]

  def _empty_metrics(self):
    return {"samples": 0, "velocity_squared_error": 0., "yaw_squared_error": 0.,
            "falls": 0, "foot_phase_agreement": 0., "vx_sum": 0., "command_vx_sum": 0.,
            "hipx_sum": self.term.state.new_zeros(len(LEGS)),
            "hipx_squared_sum": self.term.state.new_zeros(len(LEGS))}

  @staticmethod
  def _accumulate(values, valid, fell, state, command, agree, hipx):
    values["samples"] += int(valid.sum())
    values["falls"] += int(fell.sum())
    values["velocity_squared_error"] += float((state[valid, 4:6] - command[valid, 3:5]).square().sum())
    values["yaw_squared_error"] += float((state[valid, 9] - command[valid, 5]).square().sum())
    values["foot_phase_agreement"] += float(agree[valid].sum())
    values["vx_sum"] += float(state[valid, st.V_YAW.start].sum())
    values["command_vx_sum"] += float(command[valid, 3].sum())
    valid_hipx = hipx[valid]
    values["hipx_sum"] += valid_hipx.sum(dim=0)
    values["hipx_squared_sum"] += valid_hipx.square().sum(dim=0)

  def reset(self):
    self.obs, info = self.unwrapped.reset()
    return self.obs, info

  def get_observations(self):
    return self.obs

  def step(self, action):
    env = self.unwrapped
    segment = self.step_index // self.segment_steps
    if self.segment_metrics:
      segment = min(segment, len(self.segment_metrics) - 1)
    else:
      segment %= len(self.command_cycle)
    self.term.state[:, :3] = self.command_cycle[segment]
    command = self.term.state.clone()
    if self.action_bounds is not None:
      action = action.clamp(*self.action_bounds)
    result = env.step(action)
    self.obs, _, terminated, truncated, _ = result
    # Auto-reset states must be excluded from kinematic errors.
    keep = ~(terminated | truncated)
    s = st.robot_state(env)
    hipx = s[:, st.Q][:, HIP_X_JOINT_INDICES]
    gait_cfg = self.term.cfg.gait
    _, _, stance, _ = foot_schedule(command, gait_cfg)
    actual_contact = st.feet_world_z(s) - st.FOOT_RADIUS < 0.02
    agree = (actual_contact == (stance > 0.5)).float().mean(-1)
    moving = command[:, 3:5].norm(dim=-1) + command[:, 5].abs() > gait_cfg.stand_speed
    selected = torch.where(moving, command[:, 10].long() + 1, 0)
    for index, values in enumerate(self.metrics.values()):
      chosen = selected == index
      self._accumulate(values, chosen & keep, terminated & chosen, s, command, agree, hipx)
    if self.segment_metrics:
      self._accumulate(self.segment_metrics[segment], keep, terminated, s, command, agree, hipx)
    if self.writer and self.step_index % self.video_stride == 0:
      self.writer.add_image(env.render())
    self.step_index += 1
    return result

  def close(self):
    self.unwrapped.close()

  def _summary(self, values):
    n = values["samples"]
    result = {"falls": values["falls"], "valid_seconds": n * self.unwrapped.step_dt}
    if n:
      result.update({
        "velocity_rmse_m_s": math.sqrt(values["velocity_squared_error"] / n),
        "yaw_rmse_rad_s": math.sqrt(values["yaw_squared_error"] / n),
        "vx_mean_m_s": values["vx_sum"] / n,
        "command_vx_mean_m_s": values["command_vx_sum"] / n,
        "foot_phase_agreement": values["foot_phase_agreement"] / n,
        "hipx_mean_deg": dict(zip(LEGS, torch.rad2deg(values["hipx_sum"] / n).tolist())),
        "hipx_rms_deg": dict(zip(LEGS, torch.rad2deg((values["hipx_squared_sum"] / n).sqrt()).tolist())),
      })
    return result

  def summary(self):
    result = {name: self._summary(values) for name, values in self.metrics.items() if values["samples"]}
    if self.segment_metrics:
      segments = []
      for index, (speed, values) in enumerate(zip(self.vx_sequence, self.segment_metrics)):
        start = index * self.segment_steps
        if start >= self.step_index:
          break
        stop = min(start + self.segment_steps, self.step_index)
        if index == len(self.segment_metrics) - 1:
          stop = self.step_index
        dt = self.unwrapped.step_dt
        segments.append({"target_vx_m_s": speed, "start_s": start * dt, "end_s": stop * dt,
                         **self._summary(values)})
      result["segments"] = segments
    return result


def velocity_play_cfg(gait_cfg, num_envs, seed=None):
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Lite3-Velocity-Flat", play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  cmd_cfg = cfg.commands["velocity"]
  cmd_cfg.gait = gait_cfg
  cmd_cfg.resampling_time_range = (1e9, 1e9)
  cmd_cfg.standing_fraction = 1.0
  return cfg
