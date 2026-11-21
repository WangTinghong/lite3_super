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


@dataclass(frozen=True)
class MidStraightMetrics(StraightMetrics):
  vx_rmse: float
  vy_rmse: float
  vy_mean: float
  heading_abs_mean_deg: float

  def passes(self, reference: "MidStraightMetrics") -> bool:
    return (
      super().passes(reference)
      and self.falls == 0
      and all(math.isfinite(value) for value in (
        self.vx_rmse, self.vy_rmse, self.vy_mean, self.heading_abs_mean_deg,
      ))
      and self.vx_rmse <= reference.vx_rmse * 1.15 + 0.01
      and self.vy_rmse <= reference.vy_rmse + 0.008
      and abs(self.vy_mean) <= abs(reference.vy_mean) + 0.01
      and self.yaw_rmse <= reference.yaw_rmse + 0.01
      and self.heading_abs_mean_deg <= reference.heading_abs_mean_deg + 3.0
    )

  def __str__(self):
    return (f"{super().__str__()} | vx_rmse={self.vx_rmse:.3f} m/s"
            f" | vy_rmse={self.vy_rmse:.3f} m/s | vy_mean={self.vy_mean:+.3f} m/s"
            f" | heading_abs_mean={self.heading_abs_mean_deg:.1f} deg")


class StraightValidation:
  """Compare fixed straight commands on repeatable initial conditions."""

  def __init__(self, gait_cfg, device, hipx_target_limit, *, num_envs=8, seconds=20.0,
               seed=42, mid_straight=False, front_preview=False):
    from mjlab.envs import ManagerBasedRlEnv

    self.device = torch.device(device)
    self.seed = seed
    self.hipx_target_limit = hipx_target_limit
    self.checks = ((0.5, seed),)
    if mid_straight:
      self.checks += tuple((vx, mid_seed) for vx in (1.10, 1.30) for mid_seed in (42, 43))
    cfg = velocity_play_cfg(gait_cfg, num_envs, front_preview=front_preview)
    if mid_straight:
      cfg.episode_length_s = max(cfg.episode_length_s, 16.0)
    with validation_seed(self.device, seed):
      self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
    self.steps = max(1, round(seconds / self.env.step_dt))

  def close(self):
    self.env.close()

  @torch.no_grad()
  def evaluate(self, actor, vx=0.5, seed=None):
    mid_straight = vx != 0.5
    steps = round(15.0 / self.env.step_dt) if mid_straight else self.steps
    acceleration_steps = round(3.0 / self.env.step_dt) if mid_straight else None
    with validation_seed(self.device, self.seed if seed is None else seed):
      evaluation = VelocityEvaluation(
        self.env, vx=vx, hipx_target_limit=self.hipx_target_limit,
        acceleration_steps=acceleration_steps, total_steps=steps,
      )
      evaluation.reset()
      for _ in range(steps):
        evaluation.step(actor(evaluation.get_observations()["actor"]))
    totals = evaluation.metrics.values()
    falls = sum(group["falls"] for group in totals)
    if mid_straight:
      steady = evaluation.summary()["windows"]["steady_state"]
      heading = (steady["heading_abs_mean_deg"]
                 if steady["complete_heading_envs"] == self.env.num_envs else math.inf)
      return MidStraightMetrics(
        steady.get("velocity_rmse_m_s", math.inf),
        steady.get("yaw_rmse_rad_s", math.inf),
        steady.get("foot_phase_agreement", -math.inf),
        falls, evaluation.window_metrics["steady_state"]["samples"],
        steady.get("vx_rmse_m_s", math.inf),
        steady.get("vy_rmse_m_s", math.inf),
        steady.get("vy_mean_m_s", math.inf),
        heading,
      )
    samples = sum(group["samples"] for group in totals)
    if samples == 0:
      return StraightMetrics(math.inf, math.inf, -math.inf, falls, samples)
    return StraightMetrics(
      math.sqrt(sum(group["velocity_squared_error"] for group in totals) / samples),
      math.sqrt(sum(group["yaw_squared_error"] for group in totals) / samples),
      sum(group["foot_phase_agreement"] for group in totals) / samples,
      falls, samples,
    )
