# Copyright (c) 2026-2027 zh
"""
内容：
    在固定的平地直行回放中比较实际步态，记录足部和小腿接触。

用法：
    python tools/compare_lite3_trot.py CHECKPOINT [CHECKPOINT ...] --events trot_events.json
"""

import argparse
import json
import math
import sys
from itertools import pairwise
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.tasks.locomotion.go2_supermimic.mdp.gait_rewards import DIAGONAL_PAIRS
from src.tasks.locomotion.go2_supermimic.mdp.reference_state import CONTACT_FOUND, FEET, HIPS
from src.tasks.locomotion.go2_supermimic.mdp.state_spec import FOOT_NAMES
from src.tasks.locomotion.go2_supermimic.rl.playback import make_playback
from src.tasks.locomotion.go2_supermimic.rl.transition_adapter import Go2TransitionAdapter

WARMUP_S = 2.0
PLAYBACK_S = 9.0
VX_MPS = 0.60
MIN_SWING_S = 0.06
PAIR_WINDOW_S = 0.12


def swing_intervals(contact):
    """取窗口内离地、落地均可观察到的完整摆动。"""
    changes = (contact[1:] != contact[:-1]).nonzero().flatten() + 1
    return [(lift, land) for lift, land in pairwise(changes.tolist()) if not contact[lift]]


def pair_swings(first, second, window):
    """按离地顺序在指定窗口内一对一匹配，保留两侧未匹配摆动。"""
    matched = []
    i = j = 0
    while i < len(first) and j < len(second):
        delta = first[i][0] - second[j][0]
        if delta < -window:
            i += 1
        elif delta > window:
            j += 1
        else:
            matched.append((first[i], second[j]))
            i += 1
            j += 1
    return matched


def median_or_none(values):
    return median(values) if values else None


@torch.inference_mode()
def compare(checkpoint, device, num_envs, vx_mps=VX_MPS):
    env, policy = make_playback("Lite3-SuperMimic-Rough", checkpoint_file=checkpoint,
                                device=device, num_envs=num_envs)
    try:
        adapter = env.adapter
        flat = adapter.map.family_names.index("flat")

        def fixed_origins(ids):
            adapter.terrain_curriculum.family_ids[ids] = flat
            adapter.terrain_curriculum.levels[ids] = 0
            Go2TransitionAdapter._set_origins(adapter, ids)

        adapter._set_origins = fixed_origins
        pose = adapter.raw_env.event_manager.get_term_cfg("reset_root").params["pose_range"]
        pose.update(x=(0., 0.), y=(0., 0.), yaw=(0., 0.))
        sensor = adapter.raw_env.scene["leg_ground_contact"]
        shank_columns = [sensor.primary_names.index(f"{leg}_SHANK_collision") for leg in FOOT_NAMES]
        adapter.set_play_commands((vx_mps, 0., 0.))
        obs = env.reset()
        dt = adapter.step_dt
        valid = torch.ones(num_envs, dtype=torch.bool, device=adapter.device)
        falls = torch.zeros_like(valid)
        samples = []
        for step in range(round(PLAYBACK_S / dt)):
            obs, _, terminated, truncated, info = env.step(policy(obs))
            falls |= valid & terminated
            valid &= ~(terminated | truncated)
            if step < round(WARMUP_S / dt):
                continue
            state = info["final_state"]
            physical = state.physical
            vx = adapter.task.tracking_velocity(state)[:, :1]
            foot = physical[:, CONTACT_FOUND]
            shank = sensor.data.found[:, shank_columns] > 0
            clearance = adapter.task.feet_height(physical) - .025
            reach = (physical[:, FEET] - physical[:, HIPS]).reshape(num_envs, 4, 3)[..., 0]
            x = adapter.robot.data.site_pos_w[:, adapter.task.foot_ids, 0]
            samples.append(torch.cat((valid[:, None], vx, foot, shank, clearance, reach, x), dim=1))
        rows = torch.stack(samples).cpu()
        count_falls = int(falls.sum())
    finally:
        env.close()

    report, events = summarize(rows, dt, vx_mps)
    report["checkpoint"] = events["checkpoint"] = str(checkpoint)
    report["envs_with_fall"] = count_falls
    return report, events


def summarize(rows, dt, vx_mps=VX_MPS):
    legs = {leg: {"foot_swings": 0, "short_foot_swings": 0, "swings_with_shank_contact": 0,
                  "sustained_swings": 0, "clearance_mm": [], "reach_mm": [], "stride_mm": [], "cycle_s": []}
            for leg in FOOT_NAMES}
    pairs = {"-".join(pair): {"matched": 0, "first_unmatched": 0, "second_unmatched": 0,
                             "mismatch_frames": 0, "lift_ms": [], "land_ms": []}
             for pair in DIAGONAL_PAIRS}
    event_rows = []
    valid_count = 0
    for env_index, track in enumerate(rows.transpose(0, 1)):
        track = track[track[:, 0].bool()]
        valid_count += len(track)
        foot, shank, clearance, reach, x = track[:, 2:].split(4, dim=1)
        foot, shank = foot > .5, shank > .5
        contact = foot | shank
        trace = {}
        sustained_by_leg = {}
        for index, leg in enumerate(FOOT_NAMES):
            stats = legs[leg]
            foot_swings = swing_intervals(foot[:, index])
            swings = swing_intervals(contact[:, index])
            sustained = [(lift, land) for lift, land in swings if (land - lift) * dt + 1e-9 >= MIN_SWING_S]
            sustained_by_leg[leg] = sustained
            stats["foot_swings"] += len(foot_swings)
            stats["sustained_swings"] += len(sustained)
            stats["short_foot_swings"] += sum((land - lift) * dt + 1e-9 < MIN_SWING_S for lift, land in foot_swings)
            stats["swings_with_shank_contact"] += sum(bool(shank[lift:land, index].any()) for lift, land in foot_swings)
            trace[leg] = {
                "foot_swings_s": [[lift * dt, land * dt] for lift, land in foot_swings],
                "foot_and_shank_clear_swings_s": [[lift * dt, land * dt] for lift, land in swings],
            }
            for lift, land in sustained:
                stats["clearance_mm"].append(float(clearance[lift:land, index].max()) * 1000)
                stats["reach_mm"].append(float(reach[land, index]) * 1000)
            complete = set(sustained)
            # 丢弃短摆动后的相邻跨步，避免把两个步周期合并。
            for previous, current in pairwise(swings):
                if previous in complete and current in complete:
                    first_land, second_land = previous[1], current[1]
                    stats["stride_mm"].append(float(x[second_land, index] - x[first_land, index]) * 1000)
                    stats["cycle_s"].append((second_land - first_land) * dt)
        for first_leg, second_leg in DIAGONAL_PAIRS:
            stats = pairs[f"{first_leg}-{second_leg}"]
            first, second = sustained_by_leg[first_leg], sustained_by_leg[second_leg]
            matches = pair_swings(first, second, round(PAIR_WINDOW_S / dt))
            stats["matched"] += len(matches)
            stats["first_unmatched"] += len(first) - len(matches)
            stats["second_unmatched"] += len(second) - len(matches)
            mismatch = contact[:, FOOT_NAMES.index(first_leg)] != contact[:, FOOT_NAMES.index(second_leg)]
            stats["mismatch_frames"] += int(mismatch.sum())
            for (lift_a, land_a), (lift_b, land_b) in matches:
                stats["lift_ms"].append(abs(lift_a - lift_b) * dt * 1000)
                stats["land_ms"].append(abs(land_a - land_b) * dt * 1000)
        event_rows.append({"env": env_index, "valid_frames": len(track), "feet": trace})

    speeds = rows[..., 1][rows[..., 0].bool()]
    report = {
        "target_vx_mps": vx_mps,
        "actual_vx_mps": float(speeds.mean()) if speeds.numel() else None,
        "vx_rmse_mps": float((speeds - vx_mps).square().mean().sqrt()) if speeds.numel() else None,
        "valid_frames": valid_count,
        "feet": {leg: {key: median_or_none(value) if isinstance(value, list) else value
                       for key, value in stats.items()} for leg, stats in legs.items()},
        "diagonal_pairs": {
            pair: {"matched_swings": stats["matched"], "first_leg_unmatched": stats["first_unmatched"],
                   "second_leg_unmatched": stats["second_unmatched"],
                   "contact_mismatch_fraction": stats["mismatch_frames"] / valid_count if valid_count else None,
                   "median_liftoff_gap_ms": median_or_none(stats["lift_ms"]),
                   "median_touchdown_gap_ms": median_or_none(stats["land_ms"])}
            for pair, stats in pairs.items()
        },
    }
    return report, {"warmup_s": WARMUP_S, "sample_period_s": dt, "environments": event_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--vx", type=float, default=VX_MPS, help="Forward command in m/s.")
    parser.add_argument("--events", type=Path, help="Path for individual liftoff and touchdown timestamps.")
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if not math.isfinite(args.vx) or args.vx <= 0:
        parser.error("--vx must be finite and positive")
    all_events = []
    for checkpoint in args.checkpoints:
        report, events = compare(checkpoint, args.device, args.num_envs, args.vx)
        print(json.dumps(report, indent=2), flush=True)
        all_events.append(events)
    if args.events is not None:
        args.events.write_text(json.dumps(all_events, indent=2) + "\n")
