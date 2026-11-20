"""Evaluate speed tracking, stand/walk/trot/bound transitions and foot timing."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import math
import torch

from lite3_supermimic.checkpoint import validate_checkpoint
from lite3_supermimic.locomotion.gait import GaitCfg, GAITS, foot_schedule
from lite3_supermimic.supermimic import state as st


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--seconds", type=float, default=60)
  ap.add_argument("--vx", type=float, default=None, help="Fixed command. Omit for an acceleration/deceleration cycle.")
  ap.add_argument("--vy", type=float, default=0)
  ap.add_argument("--yaw-rate", type=float, default=0)
  ap.add_argument("--gait", choices=("auto", *GAITS), default=None)
  ap.add_argument("--video", help="Optional MP4 of environment zero.")
  args = ap.parse_args()
  if args.num_envs <= 0 or args.seconds <= 0 or not all(math.isfinite(v) for v in (args.seconds, args.vy, args.yaw_rate)):
    ap.error("environment count and duration must be positive, commands must be finite")
  if args.vx is not None and not math.isfinite(args.vx):
    ap.error("--vx must be finite")
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
  writer = None
  try:
    obs, _ = env.reset()
    term = env.command_manager.get_term("velocity")
    actor = Actor(obs["actor"].shape[-1], tuple(ckpt["cfg"]["actor_hidden"]), 12).to(env.device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    # A 30 s cycle exercises stand, walk, trot, bound, deceleration and stand.
    speeds = (0.0, 0.2, 0.8, 1.8, 0.8, 0.0)
    metrics = {name: {"samples": 0, "velocity_squared_error": 0., "yaw_squared_error": 0.,
                      "falls": 0, "foot_phase_agreement": 0.} for name in ("stand", *GAITS)}
    steps = max(1, int(args.seconds / env.step_dt))
    if args.video:
      import mediapy
      Path(args.video).parent.mkdir(parents=True, exist_ok=True)
      frame = env.render()
      writer = mediapy.VideoWriter(args.video, shape=frame.shape[:2], fps=25)
      writer.__enter__()
    with torch.no_grad():
      for step in range(steps):
        vx = args.vx if args.vx is not None else speeds[int(step * env.step_dt / 5) % len(speeds)]
        term.state[:, :3] = term.state.new_tensor([vx, args.vy, args.yaw_rate])
        command = term.state.clone()
        obs, _, terminated, _, _ = env.step(actor(obs["actor"]))
        # Auto-reset states must be excluded from kinematic errors.
        keep = ~env.reset_buf
        s = st.robot_state(env)
        _, _, stance, _ = foot_schedule(command, cmd_cfg.gait)
        actual_contact = st.feet_world_z(s) - st.FOOT_RADIUS < 0.02
        agree = (actual_contact == (stance > 0.5)).float().mean(-1)
        moving = command[:, 3:5].norm(dim=-1) + command[:, 5].abs() > cmd_cfg.gait.stand_speed
        selected = torch.where(moving, command[:, 10].long() + 1, 0)
        for index, name in enumerate(metrics):
          chosen = selected == index
          valid = chosen & keep
          m = metrics[name]
          m["samples"] += int(valid.sum())
          m["falls"] += int((terminated & chosen).sum())
          m["velocity_squared_error"] += float((s[valid, 4:6] - command[valid, 3:5]).square().sum())
          m["yaw_squared_error"] += float((s[valid, 9] - command[valid, 5]).square().sum())
          m["foot_phase_agreement"] += float(agree[valid].sum())
        if writer and step % max(1, round(1 / (25 * env.step_dt))) == 0:
          writer.add_image(env.render())
    result = {}
    for name, values in metrics.items():
      n = values["samples"]
      if n:
        result[name] = {"velocity_rmse_m_s": math.sqrt(values["velocity_squared_error"] / n),
                        "yaw_rmse_rad_s": math.sqrt(values["yaw_squared_error"] / n),
                        "foot_phase_agreement": values["foot_phase_agreement"] / n,
                        "falls": values["falls"], "valid_seconds": n * env.step_dt}
    print(json.dumps(result, indent=2))
  finally:
    if writer:
      writer.__exit__(None, None, None)
    env.close()


if __name__ == "__main__":
  main()
