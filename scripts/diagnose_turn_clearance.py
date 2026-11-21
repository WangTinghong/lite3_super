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
from lite3_supermimic.locomotion.validation import validation_seed
from lite3_supermimic.robot.constants import JOINT_NAMES
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


@torch.no_grad()
def probe_actions(env, actor, model, gait, args, hipx_limit):
  leg = st.LEGS.index(args.probe_leg)
  joints = [JOINT_NAMES.index(f"{args.probe_leg}_{joint}_joint") for joint in ("HipY", "Knee")]
  limits = st.robot_consts(env)
  low, high = action_bounds(limits.default_q, limits.action_scale, limits.soft_limits, hipx_limit or float("inf"))
  context = st.ContextReader(env).read()
  relief = context.new_zeros(env.num_envs, st.SCAN_POINTS)
  command = env.command_manager.get_term("velocity")
  sensor = env.scene["footfall_ground_contact"]
  contact_columns = [sensor.primary_names.index(f"{args.probe_leg}_{part}_collision")
                     for part in ("FOOT", "SHANK")]
  target = context.new_tensor([0., 0., args.yaw_rate])
  dt = env.step_dt

  def step_sim(action):
    command.state[:, :3] = target
    return env.step(action)

  actions, states = [], []
  with validation_seed(torch.device(env.device), args.seed):
    obs, _ = env.reset()
    previous_height = 0.
    for step in range(round(5 / dt)):
      height = float(foot_schedule(command.state, gait)[3][0, leg])
      if step * dt >= 3. and previous_height <= .012 < height:
        break
      previous_height = height
      states.append(st.robot_state(env))
      action = actor(obs["actor"]).clamp(low, high)
      actions.append(action)
      obs, *_ = step_sim(action)
    else:
      raise RuntimeError("No front-leg swing onset found in the probe window")
    onset = len(actions)
    states.append(st.robot_state(env))
    for _ in range(args.probe_steps):
      action = actor(obs["actor"]).clamp(low, high)
      actions.append(action)
      obs, *_ = step_sim(action)

  def replay(start, delta):
    with validation_seed(torch.device(env.device), args.seed):
      env.reset()
      for action in actions[:start]:
        step_sim(action)
      predicted = st.robot_state(env)
      matched = (predicted - states[start]).abs().amax(dim=-1) < 1e-3
      previous = actions[start - 1]
      start_height = st.feet_world_z(predicted)[:, leg]
      actual_rises, predicted_rises, clean_air, forces = [], [], [], []
      for baseline in actions[start:start + args.probe_steps]:
        action = (baseline + delta).clamp(low, high)
        predicted, _ = model(predicted, action, previous, context, relief)
        _, _, terminated, truncated, _ = step_sim(action)
        actual = st.robot_state(env)
        matched &= ~(terminated | truncated) & (actual[:, st.GRAV.stop - 1] < -0.866)
        rise = st.feet_world_z(actual)[:, leg] - start_height
        actual_rises.append(rise)
        predicted_rises.append(st.feet_world_z(predicted)[:, leg] - start_height)
        contact = sensor.data
        clean_air.append((rise >= .010) & ~(contact.found[:, contact_columns] > 0).any(dim=-1))
        force_z = contact.force[:, contact_columns, 2]
        forces.append(force_z.abs().sum(dim=-1))
        previous = action
    air = torch.stack(clean_air)
    sustained = (air[:-2] & air[1:-1] & air[2:]).any(dim=0)
    return (torch.stack(actual_rises).amax(dim=0), torch.stack(predicted_rises).amax(dim=0),
            sustained, torch.stack(forces).mean(dim=0), matched)

  def mean_if_valid(values):
    return float(values.mean()) if values.numel() else None

  zero = context.new_zeros(12)
  results = []
  for lead in args.probe_lead_steps:
    start = onset - lead
    base_actual, base_predicted, base_air, base_force, base_valid = replay(start, zero)
    for scale in args.probe_scales:
      for hip, knee in ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        delta = zero.clone()
        delta[joints] = delta.new_tensor([hip * scale, knee * scale])
        actual, predicted, air, force, valid = replay(start, delta)
        valid &= base_valid
        results.append({
          "lead_steps": lead,
          "delta_normalized_hip_y_knee": [hip * scale, knee * scale],
          "valid_envs": int(valid.sum()),
          "baseline_sustained_lift_envs": int((base_air & valid).sum()),
          "sustained_lift_envs": int((air & valid).sum()),
          "mujoco_peak_rise_mm": mean_if_valid(1000 * actual[valid]),
          "mujoco_peak_rise_gain_mm": mean_if_valid(1000 * (actual - base_actual)[valid]),
          "model_peak_rise_gain_mm": mean_if_valid(1000 * (predicted - base_predicted)[valid]),
          "mean_vertical_support_n": mean_if_valid(force[valid]),
          "baseline_vertical_support_n": mean_if_valid(base_force[valid]),
        })
  results.sort(key=lambda row: (row["sustained_lift_envs"], row["valid_envs"],
                               row["mujoco_peak_rise_mm"] or 0.), reverse=True)
  return {"leg": args.probe_leg, "yaw_rate_rad_s": args.yaw_rate, "seed": args.seed,
          "swing_onset_s": onset * dt, "horizon_s": args.probe_steps * dt,
          "sample_dt_s": dt, "perturbations": results}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--yaw-rate", type=float, choices=(-0.35, 0.35), required=True)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--probe-leg", choices=("FL", "FR"),
                      help="Replay matched actions with small HipY and knee target perturbations.")
  parser.add_argument("--probe-scales", type=float, nargs="+", default=[.15])
  parser.add_argument("--probe-lead-steps", type=int, nargs="+", default=[0])
  parser.add_argument("--probe-steps", type=int, default=8)
  args = parser.parse_args()
  if (not all(0 < scale <= 1 for scale in args.probe_scales)
      or not all(0 <= lead <= 12 for lead in args.probe_lead_steps)
      or not 3 <= args.probe_steps <= 50):
    parser.error("probe scales must be in (0, 1], lead steps in [0, 12], and horizon in [3, 50]")
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
  env_cfg = velocity_play_cfg(gait, args.num_envs, args.seed, record_footfalls=bool(args.probe_leg),
                              front_preview=cfg.get("stair_front_clearance", False),
                              edge_preview=cfg.get("stair_edge_clearance", False))
  if args.probe_leg:
    sensor = next(s for s in env_cfg.scene.sensors if s.name == "footfall_ground_contact")
    sensor.fields = ("found", "force")
    sensor.reduce = "netforce"
  env_cfg.episode_length_s = max(env_cfg.episode_length_s, 16.)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  try:
    obs, _ = env.reset()
    actor = Actor(obs["actor"].shape[-1], tuple(cfg["actor_hidden"]), 12).to(env.device)
    actor.load_state_dict(checkpoint["actor"])
    model = WorldModel(tuple(cfg["wm_hidden"])).to(env.device)
    model.load_state_dict(checkpoint["world_model"])
    actor.eval()
    model.eval()
    hipx_limit = cfg.get("hipx_target_limit", 0.)
    if args.probe_leg:
      result = probe_actions(env, actor, model, gait, args, hipx_limit)
    else:
      result = diagnose(env, actor, model, gait, args.yaw_rate, hipx_limit)
    weights = (checkpoint.get("command_cfg") or {}).get("reward_weights") or {}
    result["checkpoint_front_clearance_weight"] = weights.get("turn_front_clearance", 0.)
    print(json.dumps(result, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
