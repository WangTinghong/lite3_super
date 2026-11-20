"""Compare fine-tuned policies against their starting checkpoint in MuJoCo."""

from contextlib import contextmanager
from dataclasses import dataclass
import math
import random

import numpy as np
import torch

from lite3_supermimic.locomotion.evaluation import VelocityEvaluation, velocity_play_cfg


@contextmanager
def validation_seed(device, seed):
  """Make resets repeatable without changing training's random streams."""
  python_state, numpy_state = random.getstate(), np.random.get_state()
  cuda_devices = [device] if device.type == "cuda" else []
  with torch.random.fork_rng(devices=cuda_devices):
    try:
      random.seed(seed)
      np.random.seed(seed)
      torch.random.default_generator.manual_seed(seed)
      if cuda_devices:
        with torch.cuda.device(device):
          torch.cuda.manual_seed(seed)
      yield
    finally:
      random.setstate(python_state)
      np.random.set_state(numpy_state)


@dataclass(frozen=True)
class StraightMetrics:
  velocity_rmse: float
  yaw_rmse: float
  phase_agreement: float
  falls: int
  valid_steps: int

  def passes(self, reference: "StraightMetrics") -> bool:
    return (
      self.valid_steps > 0
      and math.isfinite(self.velocity_rmse)
      and math.isfinite(self.yaw_rmse)
      and math.isfinite(self.phase_agreement)
      and self.falls <= reference.falls
      and self.velocity_rmse <= reference.velocity_rmse * 1.15 + 0.01
      and self.yaw_rmse <= reference.yaw_rmse * 1.15 + 0.02
      and self.phase_agreement >= reference.phase_agreement - 0.05
    )

  def __str__(self):
    return (f"velocity_rmse={self.velocity_rmse:.3f} m/s | yaw_rmse={self.yaw_rmse:.3f} rad/s"
            f" | foot_phase={self.phase_agreement:.1%} | falls={self.falls}")


class StraightValidation:
  """Run a fixed 0.5 m/s command on repeatable initial conditions."""

  def __init__(self, gait_cfg, device, hipx_target_limit, *, num_envs=8, seconds=20.0, seed=42):
    from mjlab.envs import ManagerBasedRlEnv

    self.device = torch.device(device)
    self.seed = seed
    self.hipx_target_limit = hipx_target_limit
    with validation_seed(self.device, seed):
      self.env = ManagerBasedRlEnv(cfg=velocity_play_cfg(gait_cfg, num_envs), device=device)
    self.steps = max(1, round(seconds / self.env.step_dt))

  def close(self):
    self.env.close()

  @torch.no_grad()
  def evaluate(self, actor):
    with validation_seed(self.device, self.seed):
      evaluation = VelocityEvaluation(self.env, vx=0.5, hipx_target_limit=self.hipx_target_limit)
      evaluation.reset()
      for _ in range(self.steps):
        evaluation.step(actor(evaluation.get_observations()["actor"]))
    totals = evaluation.metrics.values()
    steps = sum(group["samples"] for group in totals)
    falls = sum(group["falls"] for group in totals)
    if steps == 0:
      return StraightMetrics(math.inf, math.inf, -math.inf, falls, steps)
    return StraightMetrics(
      math.sqrt(sum(group["velocity_squared_error"] for group in totals) / steps),
      math.sqrt(sum(group["yaw_squared_error"] for group in totals) / steps),
      sum(group["foot_phase_agreement"] for group in totals) / steps,
      falls, steps,
    )
