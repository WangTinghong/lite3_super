"""Resampled velocity commands and a continuous, speed-selected gait clock."""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from lite3_supermimic.locomotion import gait
from lite3_supermimic.locomotion.rewards import RewardWeights


class VelocityGaitCommand(CommandTerm):
  cfg: VelocityGaitCommandCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.state = gait.initial_state(self.num_envs, self.device)
    self.robot = env.scene["robot"]
    for name in ("velocity_error", "yaw_error", "bound_fraction"):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self):
    return gait.observation(self.state, self.cfg.gait)

  def reset(self, env_ids):
    self.state[env_ids] = gait.initial_state(len(env_ids), self.device)
    return super().reset(env_ids)

  def _resample_command(self, env_ids):
    n = len(env_ids)
    progress = min(1.0, self._env.common_step_counter / max(1, self.cfg.curriculum_steps))
    max_forward = self.cfg.initial_forward + progress * (self.cfg.max_forward - self.cfg.initial_forward)
    target = torch.empty(n, 3, device=self.device)
    target[:, 0].uniform_(-self.cfg.max_reverse * progress, max_forward)
    target[:, 1].uniform_(-self.cfg.max_lateral * progress, self.cfg.max_lateral * progress)
    target[:, 2].uniform_(-self.cfg.max_yaw * progress, self.cfg.max_yaw * progress)
    # Give high-speed straight running dedicated samples so bound is learnable.
    straight = target[:, 0] > self.cfg.gait.bound_exit
    target[straight, 1:] = 0
    standing = torch.rand(n, device=self.device) < self.cfg.standing_fraction
    target[standing] = 0
    self.state[env_ids, :3] = target

  def compute(self, dt, env_ids=None):
    if env_ids is None:
      self.state = gait.advance(self.state, dt, self.cfg.gait)
    super().compute(dt, env_ids)

  def _update_command(self, env_ids):
    pass

  def _update_metrics(self):
    from lite3_supermimic.target.interface import yaw_rotate_inverse
    d = self.robot.data
    velocity = yaw_rotate_inverse(d.root_link_lin_vel_w, d.heading_w)
    scale = self._env.max_episode_length
    self.metrics["velocity_error"] += (velocity[:, :2] - self.state[:, 3:5]).norm(dim=-1) / scale
    self.metrics["yaw_error"] += (d.root_link_ang_vel_b[:, 2] - self.state[:, 5]).abs() / scale
    self.metrics["bound_fraction"] += (self.state[:, 10] == 2).float() / scale



@dataclass(kw_only=True)
class VelocityGaitCommandCfg(CommandTermCfg):
  gait: gait.GaitCfg = field(default_factory=gait.GaitCfg)
  reward_weights: RewardWeights = field(default_factory=RewardWeights)
  curriculum_steps: int = 20_000
  initial_forward: float = 0.5
  max_forward: float = 2.0
  max_reverse: float = 0.3
  max_lateral: float = 0.3
  max_yaw: float = 0.8
  standing_fraction: float = 0.2

  def build(self, env):
    if not 0 <= self.standing_fraction <= 1:
      raise ValueError("standing_fraction must be in [0, 1]")
    if self.initial_forward <= 0 or self.max_forward < self.initial_forward:
      raise ValueError("max_forward must be >= initial_forward > 0")
    return VelocityGaitCommand(self, env)
