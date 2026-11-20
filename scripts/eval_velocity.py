"""Evaluate speed tracking, stand/walk/trot/bound transitions and foot timing."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from contextlib import ExitStack
import json
import math
import os
import torch

from lite3_supermimic.checkpoint import validate_checkpoint
from lite3_supermimic.locomotion.gait import GaitCfg, GAITS, foot_schedule
from lite3_supermimic.supermimic import state as st


class VelocityEvaluation:
  """Share command playback and metrics between the batch loop and mjlab viewer."""

  def __init__(self, env, args):
    self.unwrapped = env
    self.cfg, self.num_envs, self.device = env.cfg, env.num_envs, env.device
    self.term = env.command_manager.get_term("velocity")
    self.command_cycle = torch.tensor(
      [(vx, args.vy, args.yaw_rate) for vx in (0.0, 0.2, 0.8, 1.8, 0.8, 0.0)],
      device=self.device, dtype=self.term.state.dtype,
    )
    if args.vx is not None:
      self.command_cycle[:, 0] = args.vx
    self.writer = None
    self.video_stride = max(1, round(1 / (25 * env.step_dt)))
    self.step_index = 0
    self.metrics = {
      name: {"samples": 0, "velocity_squared_error": 0., "yaw_squared_error": 0.,
             "falls": 0, "foot_phase_agreement": 0.} for name in ("stand", *GAITS)
    }

  def reset(self):
    self.obs, info = self.unwrapped.reset()
    return self.obs, info

  def get_observations(self):
    return self.obs

  def step(self, action):
    env = self.unwrapped
    segment = int(self.step_index * env.step_dt / 5) % len(self.command_cycle)
    self.term.state[:, :3] = self.command_cycle[segment]
    command = self.term.state.clone()
    result = env.step(action)
    self.obs, _, terminated, truncated, _ = result
    # Auto-reset states must be excluded from kinematic errors.
    keep = ~(terminated | truncated)
    s = st.robot_state(env)
    gait_cfg = self.term.cfg.gait
    _, _, stance, _ = foot_schedule(command, gait_cfg)
    actual_contact = st.feet_world_z(s) - st.FOOT_RADIUS < 0.02
    agree = (actual_contact == (stance > 0.5)).float().mean(-1)
    moving = command[:, 3:5].norm(dim=-1) + command[:, 5].abs() > gait_cfg.stand_speed
    selected = torch.where(moving, command[:, 10].long() + 1, 0)
    for index, values in enumerate(self.metrics.values()):
      chosen = selected == index
      valid = chosen & keep
      values["samples"] += int(valid.sum())
      values["falls"] += int((terminated & chosen).sum())
      values["velocity_squared_error"] += float((s[valid, 4:6] - command[valid, 3:5]).square().sum())
      values["yaw_squared_error"] += float((s[valid, 9] - command[valid, 5]).square().sum())
      values["foot_phase_agreement"] += float(agree[valid].sum())
    if self.writer and self.step_index % self.video_stride == 0:
      self.writer.add_image(env.render())
    self.step_index += 1
    return result

  def close(self):
    self.unwrapped.close()

  def summary(self):
    result = {}
    for name, values in self.metrics.items():
      n = values["samples"]
      if n:
        result[name] = {
          "velocity_rmse_m_s": math.sqrt(values["velocity_squared_error"] / n),
          "yaw_rmse_rad_s": math.sqrt(values["yaw_squared_error"] / n),
          "foot_phase_agreement": values["foot_phase_agreement"] / n,
          "falls": values["falls"], "valid_seconds": n * self.unwrapped.step_dt,
        }
    return result


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--num-envs", type=int, help="Defaults to 1 with --viewer, otherwise 64.")
  ap.add_argument("--seconds", type=float, default=60)
  ap.add_argument("--vx", type=float, default=None, help="Fixed command. Omit for an acceleration/deceleration cycle.")
  ap.add_argument("--vy", type=float, default=0)
  ap.add_argument("--yaw-rate", type=float, default=0)
  ap.add_argument("--gait", choices=("auto", *GAITS), default=None)
  ap.add_argument("--video", help="Optional MP4 of environment zero.")
  ap.add_argument("--viewer", action="store_true", help="Open a live MuJoCo window on this desktop.")
  args = ap.parse_args()
  if args.num_envs is None:
    args.num_envs = 1 if args.viewer else 64
  if args.num_envs <= 0 or args.seconds <= 0 or not all(math.isfinite(v) for v in (args.seconds, args.vy, args.yaw_rate)):
    ap.error("environment count and duration must be positive, commands must be finite")
  if args.vx is not None and not math.isfinite(args.vx):
    ap.error("--vx must be finite")
  if args.viewer and sys.platform.startswith("linux") and not (
      os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    ap.error("--viewer needs a graphical display. Run this command in a terminal inside the remote desktop.")
  if not torch.cuda.is_available():
    raise SystemExit("Velocity evaluation requires an NVIDIA CUDA GPU.")
  import lite3_supermimic.env
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from lite3_supermimic.supermimic.train import Actor
  ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  validate_checkpoint(ckpt)
  if ckpt.get("cfg", {}).get("task") != "Lite3-Velocity-Flat":
    raise ValueError("Expected a Lite3-Velocity-Flat checkpoint")
  cfg = load_env_cfg("Lite3-Velocity-Flat", play=True)
  cfg.scene.num_envs = args.num_envs
  cmd_cfg = cfg.commands["velocity"]
  cmd_cfg.gait = GaitCfg(**ckpt["gait_cfg"])
  if args.gait:
    cmd_cfg.gait.mode = args.gait
  cmd_cfg.resampling_time_range = (1e9, 1e9)
  cmd_cfg.standing_fraction = 1.0
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if args.video else None)
  with ExitStack() as stack:
    stack.callback(env.close)
    evaluation = VelocityEvaluation(env, args)
    obs, _ = evaluation.reset()
    actor = Actor(obs["actor"].shape[-1], tuple(ckpt["cfg"]["actor_hidden"]), 12).to(env.device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    steps = max(1, int(args.seconds / env.step_dt))
    if args.video:
      import mediapy
      Path(args.video).parent.mkdir(parents=True, exist_ok=True)
      frame = env.render()
      evaluation.writer = stack.enter_context(mediapy.VideoWriter(args.video, shape=frame.shape[:2], fps=25))
    if args.viewer:
      from mjlab.viewer import NativeMujocoViewer

      viewer = NativeMujocoViewer(evaluation, lambda obs: actor(obs["actor"]),
                                  enable_perturbations=False)
      viewer.run(num_steps=steps)
      if evaluation.writer and evaluation.step_index == 0:
        evaluation.writer.add_image(frame)
    else:
      with torch.no_grad():
        for _ in range(steps):
          evaluation.step(actor(evaluation.get_observations()["actor"]))
    print(json.dumps(evaluation.summary(), indent=2))


if __name__ == "__main__":
  main()
