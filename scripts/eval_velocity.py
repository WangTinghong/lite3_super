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
from lite3_supermimic.locomotion.evaluation import VelocityEvaluation, velocity_play_cfg
from lite3_supermimic.locomotion.gait import GaitCfg, GAITS


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--num-envs", type=int, help="Defaults to 1 with --viewer, otherwise 64.")
  ap.add_argument("--seconds", type=float, help="Defaults to one custom sequence or 60 seconds.")
  ap.add_argument("--seed", type=int, help="Reuse the same seed to compare checkpoints.")
  forward = ap.add_mutually_exclusive_group()
  forward.add_argument("--vx", type=float, help="Fixed forward command in m/s.")
  forward.add_argument("--vx-sequence", type=float, nargs="+", help="Forward commands in m/s, in playback order.")
  ap.add_argument("--segment-seconds", type=float, default=5, help="Duration per speed command in a sequence.")
  ap.add_argument("--acceleration-seconds", type=float,
                  help="Split a fixed-command run into acceleration and steady-state windows.")
  ap.add_argument("--footfall-output", type=Path,
                  help="Save steady-state foot contact timing and pair metrics as JSON.")
  ap.add_argument("--step-height", type=float,
                  help="Place one test step of this height in meters directly ahead.")
  ap.add_argument("--vy", type=float, default=0)
  ap.add_argument("--yaw-rate", type=float, default=0)
  ap.add_argument("--hipx-target-limit", type=float, help="Override the checkpoint's action limits for comparison.")
  ap.add_argument("--gait", choices=("auto", *GAITS), default=None)
  ap.add_argument("--video", help="Optional MP4 of environment zero.")
  ap.add_argument("--viewer", action="store_true", help="Open a live MuJoCo window on this desktop.")
  ap.add_argument("--turn-front-assist", action="store_true",
                  help="Replay the tested front-leg lift offsets during a +0.35 rad/s in-place turn.")
  args = ap.parse_args()
  if args.num_envs is None:
    args.num_envs = 1 if args.viewer else 64
  durations = (args.segment_seconds,) if args.seconds is None else (args.seconds, args.segment_seconds)
  if args.num_envs <= 0 or not all(math.isfinite(d) and d > 0 for d in durations):
    ap.error("environment count and durations must be positive and finite")
  forward_speeds = args.vx_sequence or (() if args.vx is None else (args.vx,))
  if not all(math.isfinite(v) for v in (*forward_speeds, args.vy, args.yaw_rate)):
    ap.error("velocity commands must be finite")
  if args.acceleration_seconds is not None and (
      args.vx is None
      or not math.isfinite(args.acceleration_seconds)
      or not 0 < args.acceleration_seconds < (60 if args.seconds is None else args.seconds)):
    ap.error("--acceleration-seconds requires a fixed --vx command and a split within the run")
  if args.footfall_output is not None and args.acceleration_seconds is None:
    ap.error("--footfall-output requires --acceleration-seconds")
  if args.turn_front_assist and (
      args.vx != 0 or args.vy != 0 or args.yaw_rate != .35 or args.gait != "auto"
      or args.step_height is not None):
    ap.error("--turn-front-assist requires --gait auto --vx 0 --vy 0 --yaw-rate 0.35 on flat ground")
  if args.step_height is not None and (
      not math.isfinite(args.step_height) or args.step_height <= 0
      or args.vx is None or args.vx <= 0 or args.vy != 0 or args.yaw_rate != 0
      or args.footfall_output is not None):
    ap.error("--step-height requires a positive height and fixed straight command without --footfall-output")
  if args.hipx_target_limit is not None and (
      not math.isfinite(args.hipx_target_limit) or args.hipx_target_limit < 0):
    ap.error("--hipx-target-limit must be finite and nonnegative")
  if args.viewer and sys.platform.startswith("linux") and not (
      os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    ap.error("--viewer needs a graphical display. Run this command in a terminal inside the remote desktop.")
  if not torch.cuda.is_available():
    raise SystemExit("Velocity evaluation requires an NVIDIA CUDA GPU.")
  import lite3_supermimic.env
  from mjlab.envs import ManagerBasedRlEnv
  from lite3_supermimic.supermimic.train import Actor
  ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  validate_checkpoint(ckpt)
  if ckpt.get("cfg", {}).get("task") != "Lite3-Velocity-Flat":
    raise ValueError("Expected a Lite3-Velocity-Flat checkpoint")
  hipx_target_limit = ckpt["cfg"].get("hipx_target_limit", 0.)
  if args.hipx_target_limit is not None:
    hipx_target_limit = args.hipx_target_limit
  gait_cfg = GaitCfg(**ckpt["gait_cfg"])
  if args.gait:
    gait_cfg.mode = args.gait
  record_footfalls = args.footfall_output is not None
  cfg = velocity_play_cfg(gait_cfg, args.num_envs, args.seed, record_footfalls=record_footfalls,
                          step_height=args.step_height)
  step_dt = cfg.sim.mujoco.timestep * cfg.decimation
  if args.seconds is None and args.vx_sequence is not None:
    steps = len(args.vx_sequence) * max(1, round(args.segment_seconds / step_dt))
  else:
    seconds = 60 if args.seconds is None else args.seconds
    steps = max(1, int(seconds / step_dt))
  acceleration_steps = None
  if args.acceleration_seconds is not None:
    acceleration_steps = round(args.acceleration_seconds / step_dt)
    if not 0 < acceleration_steps < steps:
      ap.error("both evaluation windows must contain at least one simulation step")
  if args.vx_sequence is not None or acceleration_steps is not None:
    # A time-out reset would replace a deceleration test with another start from rest.
    cfg.episode_length_s = max(cfg.episode_length_s, steps * step_dt + 1.0)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if args.video else None)
  with ExitStack() as stack:
    stack.callback(env.close)
    evaluation = VelocityEvaluation(env, vx=args.vx, vy=args.vy, yaw_rate=args.yaw_rate,
                                    hipx_target_limit=hipx_target_limit,
                                    vx_sequence=args.vx_sequence, segment_seconds=args.segment_seconds,
                                    acceleration_steps=acceleration_steps, total_steps=steps,
                                    record_footfalls=record_footfalls, turn_front_assist=args.turn_front_assist)
    obs, _ = evaluation.reset()
    actor = Actor(obs["actor"].shape[-1], tuple(ckpt["cfg"]["actor_hidden"]), 12).to(env.device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
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
    if args.footfall_output is not None:
      args.footfall_output.parent.mkdir(parents=True, exist_ok=True)
      args.footfall_output.write_text(json.dumps(evaluation.footfall_summary(), indent=2) + "\n")
    print(json.dumps(evaluation.summary(), indent=2))


if __name__ == "__main__":
  main()
