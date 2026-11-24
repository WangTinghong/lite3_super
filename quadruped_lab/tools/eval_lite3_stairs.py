# Copyright (c) 2026-2027 zh
"""同一接触几何下比较 Lite3 正面上楼梯与平地直行。"""

import argparse
from functools import partial
import json
import math
from pathlib import Path
from statistics import median
import sys
from unittest.mock import patch

# 命令从 quadruped_lab 执行，也支持工具放在远程工作目录外。
sys.path.insert(0, str(Path.cwd()))

import torch
from mjlab.sensor import ContactMatch, ContactSensorCfg

import src.assets.robots.lite3.lite3 as robot
from src.assets.robots.lite3.constants import FOOT_COLLISION_RADIUS_M
from src.tasks.locomotion.go2_supermimic.config import env_cfgs
from src.tasks.locomotion.go2_supermimic.mdp.reference_state import POSE
from src.tasks.locomotion.go2_supermimic.mdp.terrains import TerrainGeometry
from src.tasks.locomotion.go2_supermimic.rl.playback import make_playback
from src.tasks.locomotion.go2_supermimic.rl.transition_adapter import Go2TransitionAdapter


STAIRS = "pyramid_stairs_inv_3"
FRONT_GEOMS = ("FL_FOOT_collision", "FR_FOOT_collision", "FL_SHANK_collision", "FR_SHANK_collision")
SLOTS = 4
_make_env_cfg = env_cfgs.make_go2_supermimic_env_cfg


def stair_contacts(cfg, **kwargs):
    """额外的接触法向传感器仅供评估。"""
    env_cfg = _make_env_cfg(cfg, **kwargs)
    env_cfg.scene.sensors += (ContactSensorCfg(
        name="front_riser_contacts",
        primary=ContactMatch(mode="geom", pattern=FRONT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "normal", "pos"), reduce="maxforce", num_slots=SLOTS,
    ),)
    return env_cfg


def reset_on(env, family, x, level=0):
    adapter = env.adapter
    column = adapter.map.family_names.index(family)

    def fixed_origins(ids):
        adapter.terrain_curriculum.family_ids[ids] = column
        adapter.terrain_curriculum.levels[ids] = level
        Go2TransitionAdapter._set_origins(adapter, ids)

    adapter._set_origins = fixed_origins
    pose = adapter.raw_env.event_manager.get_term_cfg("reset_root").params["pose_range"]
    pose.update(x=(x, x), y=(0., 0.), yaw=(0., 0.))
    adapter.set_play_commands((0.30, 0., 0.))
    torch.manual_seed(adapter.cfg.env.seed)
    return env.reset()


def heading_change(initial_yaw, yaw, first, last, dt):
    """展开航向角，回报窗口起点、净变化和平均变化速度。"""
    if last <= first:
        return None
    jump = torch.diff(torch.cat((initial_yaw.reshape(1), yaw)))
    unwrapped = jump.sin().atan2(jump.cos()).cumsum(0)
    change = torch.rad2deg(unwrapped[last] - unwrapped[first]).item()
    return {
        "window_s": (last - first) * dt,
        "entry_yaw_deg": torch.rad2deg(unwrapped[first]).item(),
        "delta_yaw_deg": change,
        "yaw_rate_deg_s": change / ((last - first) * dt),
    }


def first_riser_geometry(adapter, origin, radius):
    """从已编译碰撞地形查询正前方第一个立面，沿中心线采样间距约 2 mm。"""
    geometry = TerrainGeometry.from_model(adapter.raw_env.sim.mj_model, terrain_map=adapter.map)
    xs = torch.linspace(0., radius - 0.10, math.ceil((radius - 0.10) / .002) + 1)
    points = origin.cpu() + torch.stack((xs, torch.zeros_like(xs)), dim=-1)
    heights = geometry.height(points)
    rises = (heights.diff() > .02).nonzero().flatten()
    if not len(rises):
        raise ValueError("No ascending riser found on the stair centerline")
    first = int(rises[0])
    edge = (xs[first:first + 2].mean()).item()
    return edge, heights[first + 1].item()


def first_riser_metrics(trace, root_x, dt):
    """统计首阶事件；时刻为相对回合开始的控制周期采样值。"""
    by_foot = {}
    near_steps, crossed_steps = [], []

    def first(mask):
        indices = mask.nonzero().flatten()
        return int(indices[0]) if len(indices) else None

    def time(index):
        return (index + 1) * dt if index is not None else None

    for leg, values in zip(("FL", "FR"), trace.transpose(0, 1), strict=True):
        x, clearance, support, force = values.T
        support = support.bool()
        near = first(x + FOOT_COLLISION_RADIUS_M >= -.10)
        edge = first(x + FOOT_COLLISION_RADIUS_M >= 0.)
        crossed = first(x - FOOT_COLLISION_RADIUS_M >= 0.)
        hit = first(force > 0.)
        lifts = (support[:-1] & ~support[1:]).nonzero().flatten() + 1

        def observed_liftoff(event):
            if event is None:
                return None
            candidates = lifts[lifts <= event]
            if not len(candidates):
                return None
            lift = int(candidates[-1])
            return lift if not support[lift:event].any() else None

        encountered = min(i for i in (edge, hit) if i is not None) if edge is not None or hit is not None else None
        first_lift = observed_liftoff(encountered)
        hit_lift = observed_liftoff(hit)
        crossing_lift = observed_liftoff(crossed)
        first_swing_clear = (first_lift == crossing_lift) if first_lift is not None and crossing_lift is not None else None

        by_foot[leg] = {
            "edge_reached_s": time(edge),
            "foot_fully_across_s": time(crossed),
            "first_riser_hit_s": time(hit),
            "first_riser_peak_force_n": force.max().item(),
            "edge_clearance_mm": (1000 * clearance[edge].item()) if edge is not None else None,
            "liftoff_to_first_riser_hit_s": (hit - hit_lift) * dt if hit_lift is not None else None,
            "liftoff_to_foot_fully_across_s": (crossed - crossing_lift) * dt if crossing_lift is not None else None,
            "cleared_in_first_swing": first_swing_clear,
            "crossed_without_riser_hit": crossed is not None and (hit is None or hit > crossed),
        }
        if near is not None:
            near_steps.append(near)
        if crossed is not None:
            crossed_steps.append(crossed)

    low_speed_time = None
    if near_steps:
        start = min(near_steps)
        end = max(crossed_steps) if len(crossed_steps) == 2 else len(root_x) - 1
        progress = root_x[start:end + 1].diff() / dt
        low_speed_time = int((progress < .08).sum()) * dt
    return {"front_feet": by_foot, "near_edge_forward_below_0p08_s": low_speed_time}


@torch.inference_mode()
def rollout(env, policy, stairs, stair_level=0, start_x=0.65):
    adapter = env.adapter
    dt = adapter.step_dt
    obs = reset_on(env, STAIRS if stairs else "flat", start_x if stairs else 0.,
                   stair_level if stairs else 0)
    initial_yaw = adapter.task.yaw(adapter.get_state()).cpu()
    origin = adapter.raw_env.scene.env_origins[:, :2].clone()
    radius = adapter.map.size[0] * 0.5
    top = adapter.task.surface_height(origin + origin.new_tensor((radius - 0.10, 0.)))
    if stairs:
        edge_x, tread_z = first_riser_geometry(adapter, origin[0], radius)
    foot_ids = adapter.task.foot_ids
    sensor = adapter.raw_env.scene["front_riser_contacts"]
    front_indices = [sensor.primary_names.index(name) for name in FRONT_GEOMS]
    alive = torch.ones(adapter.num_envs, dtype=torch.bool, device=adapter.device)
    rows = []
    timing_rows = []
    for _ in range(round((15. if stairs else 7.) / dt)):
        obs, _, terminated, truncated, info = env.step(policy(obs))
        alive &= ~(terminated | truncated)
        state = info["final_state"]
        yaw = adapter.task.yaw(state)
        relative_x = state.physical[:, POSE.start] - origin[:, 0]
        xy = adapter.robot.data.site_pos_w[:, foot_ids, :2]
        height = adapter.task.surface_height(xy)
        inside = ((xy - origin[:, None]).abs() < radius - 0.05).all(dim=(1, 2))
        on_top = alive & inside & (height >= top[:, None] - 0.005).all(dim=1)

        contacts = sensor.data
        force = contacts.force[:, :, 0].abs()
        facing = contacts.normal[:, :, 0].abs() >= 0.50
        steep = contacts.normal[:, :, 2].abs() <= 0.70
        riser = torch.where((contacts.found > 0) & facing & steep & (force >= 1.), force, 0.)
        per_geom = riser.reshape(adapter.num_envs, len(sensor.primary_names), SLOTS).amax(-1)
        rows.append(torch.cat((alive[:, None], yaw[:, None], relative_x[:, None], on_top[:, None],
                               per_geom[:, front_indices]), dim=1))
        if stairs:
            foot_positions = adapter.robot.data.site_pos_w[:, foot_ids[:2]]
            edge_force = torch.where((contacts.pos[:, :, 0] - origin[:, :1] - edge_x).abs() <= .03, riser, 0.)
            support = (contacts.found > 0) & ~steep & (force >= 1.)
            slot_shape = (adapter.num_envs, len(sensor.primary_names), SLOTS)
            indices = front_indices[:2]
            timing_rows.append(torch.stack((
                foot_positions[..., 0] - origin[:, :1] - edge_x,
                foot_positions[..., 2] - FOOT_COLLISION_RADIUS_M - tread_z,
                support.reshape(slot_shape).any(-1)[:, indices],
                edge_force.reshape(slot_shape).amax(-1)[:, indices],
            ), dim=-1))
    timing = torch.stack(timing_rows).cpu() if timing_rows else None
    result = summarize(torch.stack(rows).cpu(), initial_yaw, dt, stairs, timing)
    if stairs:
        result["first_riser_local_x_m"] = edge_x
        result["sample_period_s"] = dt
    return result


def summarize(rows, initial_yaw, dt, stairs, timing=None):
    by_env = []
    for index, trace in enumerate(rows.transpose(0, 1)):
        valid = trace[trace[:, 0] > 0]
        result = {"env": index, "ended_early": len(valid) < len(trace)}
        by_env.append(result)
        if not len(valid):
            continue
        result["max_local_x_m"] = valid[:, 2].max().item()
        if stairs:
            top = valid[:, 3] > 0
            changes = torch.diff(torch.cat((top.new_zeros(1), top, top.new_zeros(1))).int())
            runs = list(zip((changes > 0).nonzero().flatten().tolist(),
                            (changes < 0).nonzero().flatten().tolist(), strict=True))
            stable = next(((first, end) for first, end in runs if (end - first) * dt >= 0.30), None)
            result["reached_top"] = stable is not None
            approach = valid[:stable[0]] if stable is not None else valid
            if timing is not None and len(approach):
                result["first_riser"] = first_riser_metrics(timing[:len(approach), index], approach[:, 2], dt)
            result["riser_contact"] = {}
            for name, force in zip(FRONT_GEOMS, approach[:, 4:].T, strict=True):
                hit = force > 0
                previous = torch.cat((hit.new_zeros(1), hit[:-1]))[:len(hit)]
                result["riser_contact"][name] = {
                    "events": int((hit & ~previous).sum()),
                    "peak_normal_force_n": force.max().item() if len(force) else 0.,
                }
            if stable is not None:
                result["top_entry_s"] = (stable[0] + 1) * dt
                result["heading"] = heading_change(initial_yaw[index], valid[:, 1], stable[0], stable[1] - 1, dt)
        elif len(valid) >= round(7. / dt):
            result["heading"] = heading_change(initial_yaw[index], valid[:, 1], round(2. / dt) - 1, len(valid) - 1, dt)

    heading = [row["heading"] for row in by_env if row.get("heading")]
    result = {"attempts": len(by_env), "ended_early": sum(row["ended_early"] for row in by_env),
              "heading_windows": len(heading), "by_env": by_env}
    if stairs:
        result["reached_top"] = sum(row.get("reached_top", False) for row in by_env)
        top_times = [row["top_entry_s"] for row in by_env if "top_entry_s" in row]
        result["median_top_entry_s"] = median(top_times) if top_times else None
        near_edge = [row["first_riser"] for row in by_env if "first_riser" in row]
        if near_edge:
            slow_times = [r["near_edge_forward_below_0p08_s"] for r in near_edge
                          if r["near_edge_forward_below_0p08_s"] is not None]
            result["median_near_edge_forward_below_0p08_s"] = median(slow_times) if slow_times else None
            first_feet = {}
            for leg in ("FL", "FR"):
                feet = [r["front_feet"][leg] for r in near_edge]
                clearances = [r["edge_clearance_mm"] for r in feet if r["edge_clearance_mm"] is not None]
                first_feet[leg] = {
                    "median_edge_clearance_mm": median(clearances) if clearances else None,
                    "riser_hits": sum(r["first_riser_hit_s"] is not None for r in feet),
                    "identified_first_swings": sum(r["cleared_in_first_swing"] is not None for r in feet),
                    "cleared_in_first_swing": sum(r["cleared_in_first_swing"] is True for r in feet),
                    "crossed_without_riser_hit": sum(r["crossed_without_riser_hit"] for r in feet),
                }
            result["first_riser_front_feet"] = first_feet
        result["riser_contact_events"] = {
            name: sum(row.get("riser_contact", {}).get(name, {}).get("events", 0) for row in by_env)
            for name in FRONT_GEOMS
        }
    if heading:
        result["median_signed_yaw_rate_deg_s"] = median(row["yaw_rate_deg_s"] for row in heading)
        result["median_abs_yaw_rate_deg_s"] = median(abs(row["yaw_rate_deg_s"]) for row in heading)
    return result


def evaluate(checkpoint, device, num_envs, stair_level=0, start_x=0.65, stairs_only=False):
    with patch.object(env_cfgs, "make_go2_supermimic_env_cfg", stair_contacts):
        env, policy = make_playback("Lite3-SuperMimic-Rough", checkpoint_file=checkpoint,
                                    device=device, num_envs=num_envs)
    try:
        if not 0 <= stair_level < env.adapter.map.rows:
            raise ValueError(f"Stair level must be between 0 and {env.adapter.map.rows - 1}")
        result = {"checkpoint": str(checkpoint), "stair_level": stair_level, "stair_start_x_m": start_x,
                  "stairs": rollout(env, policy, True, stair_level, start_x)}
        if not stairs_only:
            result["flat"] = rollout(env, policy, False)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("candidates", type=Path, nargs="*")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--stair-level", type=int, default=0)
    parser.add_argument("--start-x", type=float, default=0.65, help="Starting X offset on the stair tile (m).")
    parser.add_argument("--stairs-only", action="store_true", help="Skip the flat-ground control run.")
    args = parser.parse_args()
    if args.num_envs < 1 or args.stair_level < 0 or not math.isfinite(args.start_x):
        parser.error("Expected a positive environment count, nonnegative stair level and finite start offset")
    with patch.object(robot, "get_spec", partial(robot.get_spec, shank_radius_m=0.021)):
        for checkpoint in (args.baseline, *args.candidates):
            report = evaluate(checkpoint, args.device, args.num_envs, args.stair_level, args.start_x, args.stairs_only)
            print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
