"""Command dynamics shared by simulation and differentiable imagination.

Leg order is FR, FL, HR, HL (state sites), independent of action joint order.
Packed state: target velocity (3), ramped velocity (3), cycle phase (1),
walk/trot/bound mixture (3), selected gait (1). Velocity is in the yaw frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

STATE_DIM = 11
OBS_DIM = 12
GAITS = ("walk", "trot", "bound")


@dataclass(kw_only=True)
class GaitCfg:
  mode: str = "auto"
  walk_enter: float = 0.25
  walk_exit: float = 0.4
  bound_enter: float = 1.5
  bound_exit: float = 1.2
  bound_max_lateral: float = 0.25
  bound_max_yaw: float = 0.5
  transition_s: float = 0.5
  acceleration: float = 0.8
  yaw_acceleration: float = 1.5
  stand_speed: float = 0.05
  frequencies: tuple[float, float, float] = (1.5, 2.5, 3.0)
  duties: tuple[float, float, float] = (0.75, 0.55, 0.4)
  swing_height: float = 0.08

  def __post_init__(self):
    if self.mode not in ("auto", *GAITS):
      raise ValueError("gait mode must be auto, walk, trot or bound")
    if not 0 <= self.walk_enter < self.walk_exit < self.bound_exit < self.bound_enter:
      raise ValueError("gait thresholds must satisfy walk_enter < walk_exit < bound_exit < bound_enter")
    if min(self.transition_s, self.acceleration, self.yaw_acceleration, self.stand_speed,
           self.swing_height, *self.frequencies) <= 0:
      raise ValueError("gait times, accelerations, frequencies and heights must be positive")
    if len(self.frequencies) != 3 or len(self.duties) != 3 or not all(0 < d < 1 for d in self.duties):
      raise ValueError("each gait needs a frequency and duty factor in (0, 1)")


def initial_state(n: int, device: str) -> torch.Tensor:
  state = torch.zeros(n, STATE_DIM, device=device)
  state[:, 7] = 1.0
  return state


def advance(state: torch.Tensor, dt: float, cfg: GaitCfg) -> torch.Tensor:
  dt = torch.as_tensor(dt, device=state.device, dtype=state.dtype).expand(state.shape[0])
  target, velocity = state[:, :3], state[:, 3:6]
  limits = state.new_tensor([cfg.acceleration, cfg.acceleration, cfg.yaw_acceleration]) * dt[:, None]
  velocity = velocity + (target - velocity).clamp(-limits, limits)
  gait = state[:, 10].long()
  if cfg.mode == "auto":
    speed = velocity[:, :2].norm(dim=-1)
    walk = torch.where(gait == 0, speed < cfg.walk_exit, speed < cfg.walk_enter)
    forward = velocity[:, 0]
    bound = torch.where(gait == 2, forward > cfg.bound_exit, forward > cfg.bound_enter)
    bound &= (velocity[:, 1].abs() <= cfg.bound_max_lateral) & (velocity[:, 2].abs() <= cfg.bound_max_yaw)
    gait = torch.where(walk, 0, torch.where(bound, 2, 1))
  else:
    gait = torch.full_like(gait, GAITS.index(cfg.mode))
  desired = torch.nn.functional.one_hot(gait, 3).to(state.dtype)
  alpha = (1.0 - torch.exp(-dt / cfg.transition_s))[:, None]
  mixture = state[:, 7:10] + alpha * (desired - state[:, 7:10])
  frequency = (mixture * state.new_tensor(cfg.frequencies)).sum(-1)
  moving = (velocity[:, :2].norm(dim=-1) + velocity[:, 2].abs()) > cfg.stand_speed
  phase = (state[:, 6] + dt * frequency * moving).remainder(1.0)
  return torch.cat([target, velocity, phase[:, None], mixture, gait[:, None]], -1)


def foot_schedule(state: torch.Tensor, cfg: GaitCfg):
  # Walk lifts FR -> HL -> FL -> HR. Trot pairs FR/HL and FL/HR.
  # Bound pairs the two front feet and the two hind feet.
  offsets = state.new_tensor([[0.0, 0.5, 0.75, 0.25],
                              [0.0, 0.5, 0.5, 0.0],
                              [0.0, 0.0, 0.5, 0.5]])
  phase = (state[:, 6:7] + state[:, 7:10] @ offsets).remainder(1.0)
  duty = (state[:, 7:10] * state.new_tensor(cfg.duties)).sum(-1, keepdim=True)
  moving = (state[:, 3:5].norm(dim=-1) + state[:, 5].abs()) > cfg.stand_speed
  # Smooth stance probability avoids an unavailable gradient through contact.
  stance = torch.sigmoid((torch.cos(2 * math.pi * (phase - duty / 2))
                          - torch.cos(math.pi * duty)) / 0.1)
  swing_phase = ((phase - duty) / (1 - duty)).clamp(0, 1)
  height = cfg.swing_height * torch.sin(math.pi * swing_phase).square()
  stance = torch.where(moving[:, None], stance, torch.ones_like(stance))
  height = height * moving[:, None]
  return phase, duty, stance, height


def observation(state: torch.Tensor, cfg: GaitCfg) -> torch.Tensor:
  phase, duty, _, _ = foot_schedule(state, cfg)
  clock = torch.stack([torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], -1)
  return torch.cat([state[:, 3:6], clock.flatten(1), duty], -1)
