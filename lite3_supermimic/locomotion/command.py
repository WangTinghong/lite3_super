"""Resampled velocity commands and a continuous, speed-selected gait clock."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
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
    self.height_field = None
    if cfg.stair_level:
      from lite3_supermimic.locomotion.stairs import StairField
      self.height_field = StairField(cfg.stair_level)
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
    if self.cfg.stair_level:
      self.state[env_ids, :3] = 0.
      self.state[env_ids, 0] = torch.empty(n, device=self.device).uniform_(0.25, 0.38)
      return
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
    side_fraction = self.cfg.pure_side_fraction
    fast_fraction = self.cfg.fast_straight_fraction
    bound_fraction = self.cfg.bound_straight_fraction
    turn_fraction = self.cfg.pure_turn_fraction
    if side_fraction or fast_fraction or bound_fraction or turn_fraction:
      moving_sample = torch.rand(n, device=self.device)
      if side_fraction:
        side_step = ~standing & (moving_sample < side_fraction)
        direction = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        target[side_step] = 0.0
        target[side_step, 1] = 0.1 * direction[side_step]
      bands = (
        (fast_fraction, self.cfg.fast_straight_min, self.cfg.fast_straight_max),
        (bound_fraction, self.cfg.bound_straight_min, self.cfg.bound_straight_max),
      )
      lower = side_fraction
      for fraction, min_speed, max_speed in bands:
        upper = lower + fraction
        if fraction:
          straight_sample = ~standing & (moving_sample >= lower) & (moving_sample < upper)
          speed = torch.empty(n, device=self.device).uniform_(min_speed, max_speed)
          target[straight_sample, 0] = speed[straight_sample]
          target[straight_sample, 1:] = 0.0
        lower = upper
      if turn_fraction:
        turn = ~standing & (moving_sample >= lower) & (moving_sample < lower + turn_fraction)
        speed = torch.empty(n, device=self.device).uniform_(0.25, 0.45)
        direction = torch.where(torch.rand(n, device=self.device) < 0.5, -1., 1.)
        target[turn, :2] = 0.
        target[turn, 2] = (speed * direction)[turn]
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
  pure_side_fraction: float = 0.0
  pure_turn_fraction: float = 0.0
  fast_straight_fraction: float = 0.0
  fast_straight_min: float = 0.9
  fast_straight_max: float = 1.1
  bound_straight_fraction: float = 0.0
  bound_straight_min: float = 1.55
  bound_straight_max: float = 1.65
  stair_level: int = 0

  def build(self, env):
    if not 0 <= self.standing_fraction <= 1:
      raise ValueError("standing_fraction must be in [0, 1]")
    if not 0 <= self.pure_side_fraction <= 1:
      raise ValueError("pure_side_fraction must be in [0, 1]")
    if not 0 <= self.fast_straight_fraction <= 1 - self.pure_side_fraction:
      raise ValueError("fast_straight_fraction must be nonnegative and leave room for pure_side_fraction")
    if not (0 <= self.bound_straight_fraction
            <= 1 - (self.pure_side_fraction + self.fast_straight_fraction)):
      raise ValueError("bound_straight_fraction must be nonnegative and fit in the remaining moving commands")
    if not (0 <= self.pure_turn_fraction <= 1 - (
        self.pure_side_fraction + self.fast_straight_fraction + self.bound_straight_fraction)):
      raise ValueError("pure_turn_fraction must be nonnegative and fit in the remaining moving commands")
    if not 0 < self.fast_straight_min <= self.fast_straight_max < math.inf:
      raise ValueError("fast straight speeds must satisfy 0 < min <= max and be finite")
    if self.bound_straight_fraction and (
        self.gait.mode != "auto"
        or not self.gait.bound_enter < self.bound_straight_min <= self.bound_straight_max < math.inf
    ):
      raise ValueError("bound straight sampling requires auto gait and finite speeds above bound_enter")
    if self.initial_forward <= 0 or self.max_forward < self.initial_forward:
      raise ValueError("max_forward must be >= initial_forward > 0")
    return VelocityGaitCommand(self, env)
