"""Compare turn foot targets with MuJoCo and action-matched world-model predictions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lite3_supermimic.checkpoint import validate_checkpoint
from lite3_supermimic.locomotion.actions import action_bounds
from lite3_supermimic.locomotion.evaluation import VelocityEvaluation, velocity_play_cfg
from lite3_supermimic.locomotion.gait import GaitCfg, foot_schedule
from lite3_supermimic.locomotion.rewards import reward_terms
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.supermimic.train import Actor
from lite3_supermimic.supermimic.world_model import WorldModel


def summarize(samples):
  if not samples:
    return {}
  reference, actual, predicted = torch.cat(samples).unbind(dim=1)
  if not reference.numel():
    return {}
  result = {}
  for index, leg in enumerate(st.LEGS):
    target = reference[:, index]
    swing = target > 0.012
    error = predicted[swing, index] - actual[swing, index]
    result[leg] = {
      "swing_samples": int(swing.sum()),
      "target_over_12mm_fraction": float(swing.float().mean()),
      "target_mean_mm": float(1000 * target[swing].mean()) if error.numel() else None,
      "actual_mean_mm": float(1000 * actual[swing, index].mean()) if error.numel() else None,
      "predicted_mean_mm": float(1000 * predicted[swing, index].mean()) if error.numel() else None,
      "prediction_bias_mm": float(1000 * error.mean()) if error.numel() else None,
      "prediction_rmse_mm": float(1000 * error.square().mean().sqrt()) if error.numel() else None,
    }
  return result


@torch.no_grad()
def diagnose(env, actor, model, gait, yaw_rate, hipx_limit):
  dt = env.step_dt
  evaluation = VelocityEvaluation(env, vx=0., yaw_rate=yaw_rate)
  obs, _ = evaluation.reset()
  command = evaluation.term
  constants = st.robot_consts(env)
  limits = action_bounds(constants.default_q, constants.action_scale, constants.soft_limits,
                         hipx_limit) if hipx_limit else None
  previous_action = torch.zeros(env.num_envs, 12, device=env.device)
  context = st.ContextReader(env).read()
  relief = previous_action.new_zeros(env.num_envs, st.SCAN_POINTS)
  one_step, eight_steps, penalties = [], [], []
  for step in range(round(15 / dt)):
    state = st.robot_state(env)
    command.state[:, :3] = command.state.new_tensor([0., 0., yaw_rate])
    reference = command.state.clone()
    action = actor(obs["actor"])
    if limits is not None:
      action = action.clamp(*limits)
    predicted, _ = model(state, action, previous_action, context, relief)
    age = step % 8
    if age == 0:
      rollout = predicted
      rollout_valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
      rollout, _ = model(rollout, action, previous_action, context, relief)
    obs, _, terminated, truncated, _ = evaluation.step(action)
    valid = ~(terminated | truncated)
    rollout_valid &= valid
    if step >= round(3 / dt):
      actual = st.robot_state(env)
      desired = foot_schedule(reference, gait)[3]
      actual_height = st.feet_world_z(actual) - st.FOOT_RADIUS
      one_step.append(torch.stack((desired, actual_height, st.feet_world_z(predicted) - st.FOOT_RADIUS),
                                  dim=1)[valid])
      if age == 7:
        eight_steps.append(torch.stack((desired, actual_height, st.feet_world_z(rollout) - st.FOOT_RADIUS),
                                       dim=1)[rollout_valid])
      cost = reward_terms(actual, action, previous_action, reference, constants, gait,
                          front_clearance_enabled=True)["turn_front_clearance"]
      penalties.append(cost[valid])
    previous_action = torch.where(valid[:, None], action, 0.)
  cost = torch.cat(penalties)
  return {
    "yaw_rate_rad_s": yaw_rate,
    "steady_sample_count": cost.numel(),
    "front_clearance_penalty_nonzero_fraction": float((cost > 0).float().mean()) if cost.numel() else None,
    "front_clearance_penalty_unweighted_mean": float(cost.mean()) if cost.numel() else None,
    "one_step": summarize(one_step),
    "eight_step_action_replay": summarize(eight_steps),
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--yaw-rate", type=float, choices=(-0.35, 0.35), required=True)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if args.num_envs <= 0:
    parser.error("--num-envs must be positive")
  import lite3_supermimic.env
  from mjlab.envs import ManagerBasedRlEnv

  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  validate_checkpoint(checkpoint)
  cfg = checkpoint["cfg"]
  if cfg["task"] != "Lite3-Velocity-Flat":
    parser.error("expected a Lite3-Velocity-Flat checkpoint")
  gait = GaitCfg(**checkpoint["gait_cfg"])
  env_cfg = velocity_play_cfg(gait, args.num_envs, args.seed)
  env_cfg.episode_length_s = max(env_cfg.episode_length_s, 16.)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  try:
    obs, _ = env.reset()
    actor = Actor(obs["actor"].shape[-1], tuple(cfg["actor_hidden"]), 12).to(env.device)
    actor.load_state_dict(checkpoint["actor"])
    model = WorldModel(tuple(cfg["wm_hidden"])).to(env.device)
    model.load_state_dict(checkpoint["world_model"])
    result = diagnose(env, actor.eval(), model.eval(), gait, args.yaw_rate,
                      cfg.get("hipx_target_limit", 0.))
    weights = (checkpoint.get("command_cfg") or {}).get("reward_weights") or {}
    result["checkpoint_front_clearance_weight"] = weights.get("turn_front_clearance", 0.)
    print(json.dumps(result, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
