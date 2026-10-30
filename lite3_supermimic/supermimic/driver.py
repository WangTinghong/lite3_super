"""A compatible Lite3 actor can collect data for terrain world-model fitting.

Only Lite3 SuperMimic checkpoints with the same joint conventions are accepted.
"""
from __future__ import annotations

import math
import re

import torch


class ActorDriver:
  """One of our own checkpoints driving the env, e.g. a flat-ground chase-and-dodge
  policy on terrain, so the world model sees its jumps and ducks there. Its
  observation must be a prefix of the env's (true when the task only appends
  terms, like the terrain height scan)."""

  def __init__(self, path: str, env_obs_dim: int, device):
    from lite3_supermimic.supermimic.train import Actor

    ck = torch.load(path, map_location=device, weights_only=False)
    from lite3_supermimic.checkpoint import validate_checkpoint
    validate_checkpoint(ck)
    self.obs_dim = ck["obs_dim"]
    assert self.obs_dim <= env_obs_dim, (self.obs_dim, env_obs_dim)
    self.actor = Actor(self.obs_dim, tuple(ck["cfg"]["actor_hidden"]), 12).to(device)
    self.actor.load_state_dict(ck["actor"])
    self.actor.eval()

  @torch.no_grad()
  def act(self, obs: torch.Tensor) -> torch.Tensor:
    return self.actor(obs[:, : self.obs_dim])


class ChaseSteer:
  """Velocity commands for the driver: head for the chase ball (or, without one,
  a random waypoint in the arena) at a per-env random top speed, turning toward
  it like Isaac Lab's heading command."""

  def __init__(self, num_envs: int, device, vmax: tuple[float, float], radius: float):
    self.dev, self.vmax, self.radius = device, vmax, radius
    self.speed = torch.zeros(num_envs, device=device)
    self.goal = torch.zeros(num_envs, 2, device=device)
    self.timer = torch.zeros(num_envs, device=device)

  def resample(self, mask: torch.Tensor, center: torch.Tensor) -> None:
    k = int(mask.sum())
    if k == 0:
      return
    lo, hi = self.vmax
    self.speed[mask] = lo + (hi - lo) * torch.rand(k, device=self.dev)
    r = self.radius * torch.rand(k, device=self.dev).sqrt()
    th = 2 * math.pi * torch.rand(k, device=self.dev)
    self.goal[mask] = center[mask] + torch.stack([r * th.cos(), r * th.sin()], -1)
    self.timer[mask] = 2.0 + 3.0 * torch.rand(k, device=self.dev)

  def __call__(self, dt: float, pos: torch.Tensor, yaw: torch.Tensor, center: torch.Tensor,
               ball: torch.Tensor, has_ball: torch.Tensor) -> torch.Tensor:
    self.timer -= dt
    self.resample((self.timer <= 0) | ((self.goal - pos).norm(dim=-1) < 0.4), center)
    to = torch.where(has_ball.unsqueeze(-1), ball, self.goal) - pos
    err = torch.atan2(to[:, 1], to[:, 0]) - yaw
    err = torch.atan2(err.sin(), err.cos())
    v = torch.minimum(self.speed, 1.5 * to.norm(dim=-1))
    return torch.stack([v * err.cos().clamp(min=0), torch.zeros_like(v),
                        (1.5 * err).clamp(-1.5, 1.5)], -1)
