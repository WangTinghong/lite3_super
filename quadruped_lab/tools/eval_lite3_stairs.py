# Copyright (c) 2026-2027 zh
"""同一接触几何下比较 Lite3 正面上楼梯与平地直行。"""

import argparse
from contextlib import nullcontext
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
from src.tasks.locomotion.go2_supermimic.mdp.reference_rotations import quat_to_mat
from src.tasks.locomotion.go2_supermimic.mdp.reference_state import POSE
from src.tasks.locomotion.go2_supermimic.mdp.terrains import TerrainGeometry
from src.tasks.locomotion.go2_supermimic.rl.playback import make_playback
from src.tasks.locomotion.go2_supermimic.rl.transition_adapter import Go2TransitionAdapter


STAIRS = "pyramid_stairs_inv_3"
STEERING_MODES = ("off", "heading", "centerline")
FRONT_GEOMS = ("FL_FOOT_collision", "FR_FOOT_collision", "FL_SHANK_collision", "FR_SHANK_collision")
TORSO_COLLISION = "TORSO_collision"
TORSO_VISUAL = "TORSO_visual"
SLOTS = 4
TOP_HOLD_S = 0.30
_make_env_cfg = env_cfgs.make_go2_supermimic_env_cfg


def stair_contacts(cfg, **kwargs):
    """额外的接触法向传感器仅供评估。"""
    env_cfg = _make_env_cfg(cfg, **kwargs)
    env_cfg.scene.sensors += (ContactSensorCfg(
        name="front_riser_contacts",
        primary=ContactMatch(mode="geom", pattern=FRONT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "normal", "pos"), reduce="maxforce", num_slots=SLOTS,
    ), ContactSensorCfg(
        name="stair_torso_contacts",
        primary=ContactMatch(mode="geom", pattern=TORSO_COLLISION, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "pos", "normal"), reduce="maxforce", num_slots=SLOTS,
    ),)
    return env_cfg


def stair_commands(state, *, center_y, vx, steering):
    """沿楼梯世界 +X 轴保持航向，按需限制相对踏面中心的横向漂移。"""
    pose = state.physical[:, POSE]
    yaw = pose[:, 2]
    heading_error = yaw.sin().atan2(yaw.cos())
    yaw_rate = (-1.5 * heading_error).clamp(-.30, .30)
    lateral = torch.zeros_like(yaw_rate)
    if steering == "centerline":
        offset = pose[:, 1] - center_y
        # 将轴向横移修正投影到机身侧向，留出 2 cm 的落足波动范围。
        correction = -.65 * offset.sign() * (offset.abs() - .02).clamp_min(0.)
        lateral = (correction * yaw.cos()).clamp(-.08, .08)
    return torch.stack((torch.full_like(yaw_rate, vx), lateral, yaw_rate), dim=-1)


def reset_on(env, family, x, level=0, *, vx=0.30, steering="off"):
    adapter = env.adapter
    column = adapter.map.family_names.index(family)

    def fixed_origins(ids):
        adapter.terrain_curriculum.family_ids[ids] = column
        adapter.terrain_curriculum.levels[ids] = level
        Go2TransitionAdapter._set_origins(adapter, ids)

    adapter._set_origins = fixed_origins
    pose = adapter.raw_env.event_manager.get_term_cfg("reset_root").params["pose_range"]
    pose.update(x=(x, x), y=(0., 0.), yaw=(0., 0.))
    env.set_command_provider(None)
    adapter.set_play_commands((vx, 0., 0.))
    torch.manual_seed(adapter.cfg.env.seed)
    obs = env.reset()
    if steering != "off":
        center_y = adapter.raw_env.scene.env_origins[:, 1].clone()
        env.set_command_provider(partial(stair_commands, center_y=center_y, vx=vx, steering=steering))
        obs = env.get_observations()
    return obs


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
    near_steps, crossed_steps, encounter_steps = [], [], []

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
        if encountered is not None:
            encounter_steps.append(encountered)
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
    approach_speed = None
    if encounter_steps:
        end = min(encounter_steps) - 1  # 避开开始接触或首次到达立面的采样帧。
        start = max(0, end - round(.20 / dt))
        if end > start:
            approach_speed = (root_x[end] - root_x[start]).item() / ((end - start) * dt)
    return {"front_feet": by_foot, "near_edge_forward_below_0p08_s": low_speed_time,
            "approach_forward_speed_mps": approach_speed}


def riser_forces(contacts):
    """识别法线近水平且朝向楼梯进场轴的接触力。"""
    force = contacts.force[:, :, 0].abs()
    facing = contacts.normal[:, :, 0].abs() >= .50
    steep = contacts.normal[:, :, 2].abs() <= .70
    return torch.where((contacts.found > 0) & facing & steep & (force >= 1.), force, 0.)


def first_riser_risk(trace, dt):
    """以物理子步累积首阶双前腿接触和碰撞后的低头速度。"""
    front, rear = trace[:, :2], trace[:, 2]
    hits = front > 0
    both = hits.all(dim=-1)
    changes = torch.diff(torch.cat((both.new_zeros(1), both, both.new_zeros(1))).int())
    starts = (changes > 0).nonzero().flatten()
    ends = (changes < 0).nonzero().flatten()
    first = hits.any(dim=-1).nonzero().flatten()
    peak_pitch, peak_pitch_rate = None, None
    if first.numel():
        start = int(first[0])
        window = trace[start:start + max(1, round(.20 / dt)), 3:5]
        peak_pitch, peak_pitch_rate = window.amax(dim=0).clamp_min(0.).tolist()
    return {
        "dual_front_contact_s": int(both.sum()) * dt,
        "longest_dual_front_contact_s": int((ends - starts).amax()) * dt if starts.numel() else 0.,
        "peak_first_riser_front_contact_n": front.amax().item() if front.numel() else 0.,
        "peak_nose_down_pitch_after_first_hit_deg": peak_pitch,
        "peak_nose_down_rate_after_first_hit_deg_s": peak_pitch_rate,
        "peak_rear_forward_force_during_dual_contact_n": rear[both].amax().item() if starts.numel() else None,
    }


def front_shell_vertices(model, reference):
    """取机身碰撞盒前端之外的视觉网格点，保留网格局部坐标。"""
    visual = model.geom(f"robot/{TORSO_VISUAL}")
    box = model.geom(f"robot/{TORSO_COLLISION}")
    mesh = int(visual.dataid[0])
    start = model.mesh_vertadr[mesh]
    vertices = reference.new_tensor(model.mesh_vert[start:start + model.mesh_vertnum[mesh]].copy())
    body_rotation = quat_to_mat(reference.new_tensor(visual.quat.copy()))
    body_x = vertices @ body_rotation[0] + float(visual.pos[0])
    return visual.id, vertices[body_x >= float(box.pos[0] + box.size[0])]


def torso_ascent_safety(trace, dt):
    """分开报告求解器接触与视觉前壳在首级踏面上的几何净空。"""
    torso_hit, first_torso_hit, torso_force, shell_clearance, approach, contact_x, second_riser_hit = trace[:, 5:].unbind(-1)
    hit = torso_hit.bool()
    observed = torch.isfinite(shell_clearance)
    clearance = shell_clearance[observed].amin().item() if observed.any() else None
    hit_steps = hit.nonzero().flatten()
    pitch = trace[:, 3].clamp_min(0.)
    near_first = approach.bool()
    return {
        "torso_contact_s": int(hit.sum()) * dt,
        "first_torso_contact_time_s": (int(hit_steps[0]) + 1) * dt if hit_steps.numel() else None,
        "first_torso_contact_local_x_m": contact_x[hit_steps[0]].item() if hit_steps.numel() else None,
        "first_riser_torso_contact_s": int(first_torso_hit.sum()) * dt,
        "second_riser_torso_contact_s": int(second_riser_hit.sum()) * dt,
        "peak_torso_contact_force_n": torso_force.amax().item(),
        "first_riser_min_front_shell_clearance_mm": clearance * 1000 if clearance is not None else None,
        "first_riser_front_shell_overlap": (clearance < 0.) if clearance is not None else None,
        "peak_nose_down_pitch_deg": pitch.amax().item(),
        "peak_nose_down_pitch_near_first_riser_deg": pitch[near_first].amax().item() if near_first.any() else None,
    }


@torch.no_grad()
def rollout(env, policy, stairs, stair_level=0, start_x=0.65, *, vx=0.30, steering="off"):
    adapter = env.adapter
    dt = adapter.step_dt
    obs = reset_on(env, STAIRS if stairs else "flat", start_x if stairs else 0.,
                   stair_level if stairs else 0, vx=vx, steering=steering if stairs else "off")
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
    lateral_rows = []
    timing_rows = []
    risk_rows = []
    slot_shape = (adapter.num_envs, len(sensor.primary_names), SLOTS)
    capture = nullcontext()
    if stairs:
        world_edge = origin[:, :1] + edge_x
        step_width = adapter.map.generator.sub_terrains[STAIRS].step_width
        tread_end = world_edge + step_width
        rear_sensor = adapter.raw_env.scene["feet_ground_contact"]
        rear_indices = [rear_sensor.primary_names.index(f"{leg}_FOOT_collision") for leg in ("HL", "HR")]
        torso_sensor = adapter.raw_env.scene["stair_torso_contacts"]
        sim = adapter.raw_env.sim
        shell_geom, shell_vertices = front_shell_vertices(sim.mj_model, origin)
        manager = adapter.raw_env.metrics_manager
        compute_substep = manager.compute_substep

        def record_substep():
            compute_substep()
            contacts = sensor.data
            at_edge = (contacts.pos[..., 0] - world_edge).abs() <= .03
            front = torch.where(at_edge, riser_forces(contacts), 0.).reshape(slot_shape).amax(-1)
            front = front[:, front_indices].reshape(-1, 2, 2).amax(dim=1)
            rear = rear_sensor.data.force[:, rear_indices, 0].clamp_min(0.).sum(-1)
            data = adapter.robot.data
            gravity, omega = data.projected_gravity_b, data.root_link_ang_vel_b
            cosine = gravity[:, 1:].norm(dim=-1)
            pitch = gravity[:, 0].atan2(cosine)
            pitch_rate = (omega[:, 2] * gravity[:, 1] - omega[:, 1] * gravity[:, 2]) / cosine.clamp_min(1e-6)
            angles = torch.rad2deg(torch.stack((pitch, pitch_rate), dim=-1))
            torso = torso_sensor.data
            torso_hit = torso.found > 0
            torso_x = torso.pos[..., 0]
            first_hit = torso_hit & (torso_x >= world_edge - .03) & (torso_x <= tread_end)
            second_hit = ((torso_x - tread_end).abs() <= .03) & (riser_forces(torso) > 0.)
            torso_force, torso_slot = torch.where(torso_hit, torso.force[..., 0].abs(), -1.).max(-1)
            torso_contact_x = torso_x.gather(-1, torso_slot[:, None]).squeeze(-1) - origin[:, 0]
            rotation = sim.data.geom_xmat[:, shell_geom]
            shell = shell_vertices @ rotation.transpose(-1, -2) + sim.data.geom_xpos[:, shell_geom, None]
            shell_x, shell_y, shell_z = shell.unbind(-1)
            # 只看首级踏面内，避开不同踏面和立面边缘约2毫米的定位误差。
            over_first = ((shell_x >= world_edge + .002) & (shell_x <= tread_end - .002)
                          & ((shell_y - origin[:, 1:2]).abs() <= edge_x - .002))
            clearance = torch.where(over_first, shell_z - tread_z, torch.inf).amin(dim=-1)
            approach = ((shell_x.amax(dim=-1) >= world_edge[:, 0] - .25)
                        & (data.root_link_pos_w[:, 0] <= world_edge[:, 0]))
            body = torch.stack((torso_hit.any(-1), first_hit.any(-1), torso_force.clamp_min(0.),
                                clearance, approach, torso_contact_x, second_hit.any(-1)), dim=-1)
            risk_rows.append(torch.cat((front, rear[:, None], angles, body), dim=-1))

        capture = patch.object(manager, "compute_substep", record_substep)
    with capture:
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
            riser = riser_forces(contacts)
            per_geom = riser.reshape(slot_shape).amax(-1)
            rows.append(torch.cat((alive[:, None], yaw[:, None], relative_x[:, None], on_top[:, None],
                                   per_geom[:, front_indices]), dim=1))
            if stairs:
                lateral_rows.append(state.physical[:, POSE.start + 1] - origin[:, 1])
                foot_positions = adapter.robot.data.site_pos_w[:, foot_ids[:2]]
                edge_force = torch.where((contacts.pos[..., 0] - world_edge).abs() <= .03, riser, 0.)
                support = ((contacts.found > 0) & (contacts.normal[..., 2].abs() > .70)
                           & (contacts.force[..., 0].abs() >= 1.))
                indices = front_indices[:2]
                timing_rows.append(torch.stack((
                    foot_positions[..., 0] - world_edge,
                    foot_positions[..., 2] - FOOT_COLLISION_RADIUS_M - tread_z,
                    support.reshape(slot_shape).any(-1)[:, indices],
                    edge_force.reshape(slot_shape).amax(-1)[:, indices],
                ), dim=-1))
    timing = torch.stack(timing_rows).cpu() if timing_rows else None
    rows = torch.stack(rows).cpu()
    alignment_data = (torch.stack(lateral_rows).cpu(), (edge_x, edge_x + step_width), start_x) if stairs else None
    result = summarize(rows, initial_yaw, dt, stairs, timing, alignment_data=alignment_data)
    if stairs:
        result["first_riser_local_x_m"] = edge_x
        result["sample_period_s"] = dt
        sub_dt = adapter.raw_env.physics_dt
        substeps = round(dt / sub_dt)
        if len(risk_rows) != len(rows) * substeps:
            raise RuntimeError("The first-riser sensor did not record every physics substep")
        samples = torch.stack(risk_rows).cpu()
        risks = []
        bodies = []
        for entry in result["by_env"]:
            index = entry["env"]
            valid = int(rows[:, index, 0].sum())
            end_step = round(entry["top_entry_s"] / dt) if "top_entry_s" in entry else valid + entry["ended_early"]
            risk = first_riser_risk(samples[:end_step * substeps, index], sub_dt)
            entry["first_riser_risk"] = risk
            risks.append(risk)
            if entry.get("reached_top", False):
                end_step = min(valid, end_step - 1 + math.ceil(TOP_HOLD_S / dt))
            body = torso_ascent_safety(samples[:end_step * substeps, index], sub_dt)
            entry["torso_ascent_safety"] = body
            bodies.append(body)
        overlap = [r["dual_front_contact_s"] for r in risks]
        peaks = [r["peak_nose_down_rate_after_first_hit_deg_s"] for r in risks
                 if r["peak_nose_down_rate_after_first_hit_deg_s"] is not None]
        result["first_riser_risk"] = {
            "sample_period_s": sub_dt,
            "envs_with_dual_front_contact": sum(t > 0. for t in overlap),
            "total_dual_front_contact_s": sum(overlap),
            "max_contiguous_dual_front_contact_s": max(r["longest_dual_front_contact_s"] for r in risks),
            "max_first_riser_front_contact_n": max(r["peak_first_riser_front_contact_n"] for r in risks),
            "median_peak_nose_down_rate_after_first_hit_deg_s": median(peaks) if peaks else None,
        }
        shell_clearances = [b["first_riser_min_front_shell_clearance_mm"] for b in bodies
                            if b["first_riser_min_front_shell_clearance_mm"] is not None]
        approach_pitch = [b["peak_nose_down_pitch_near_first_riser_deg"] for b in bodies
                          if b["peak_nose_down_pitch_near_first_riser_deg"] is not None]
        result["torso_ascent_safety"] = {
            "envs_with_torso_contact": sum(b["torso_contact_s"] > 0. for b in bodies),
            "envs_with_first_riser_torso_contact": sum(b["first_riser_torso_contact_s"] > 0. for b in bodies),
            "envs_with_second_riser_torso_contact": sum(b["second_riser_torso_contact_s"] > 0. for b in bodies),
            "envs_with_first_riser_shell_measurement": sum(b["first_riser_front_shell_overlap"] is not None for b in bodies),
            "envs_with_first_riser_shell_overlap": sum(b["first_riser_front_shell_overlap"] is True for b in bodies),
            "min_first_riser_front_shell_clearance_mm": min(shell_clearances) if shell_clearances else None,
            "max_nose_down_pitch_deg": max(b["peak_nose_down_pitch_deg"] for b in bodies),
            "max_nose_down_pitch_near_first_riser_deg": max(approach_pitch) if approach_pitch else None,
            "reached_top_without_torso_contact": sum(
                e.get("reached_top", False) and b["torso_contact_s"] == 0.
                for e, b in zip(result["by_env"], bodies, strict=True)),
            "reached_top_with_clear_nose_and_no_torso_contact": sum(
                e.get("reached_top", False) and b["torso_contact_s"] == 0. and b["first_riser_front_shell_overlap"] is False
                for e, b in zip(result["by_env"], bodies, strict=True)),
        }
    return result


def alignment_window(lateral, heading):
    """保留有符号末值和峰值，避免左右跑偏相互抵消。"""
    if not lateral.numel():
        return None
    return {
        "end_lateral_m": lateral[-1].item(),
        "peak_abs_lateral_m": lateral.abs().max().item(),
        "end_heading_deg": heading[-1].item(),
        "peak_abs_heading_deg": heading.abs().max().item(),
    }


def ascent_alignment(trace, lateral, edges, start_x, dt):
    yaw = trace[:, 1]
    heading = torch.rad2deg(yaw.sin().atan2(yaw.cos()))
    x = trace[:, 2]
    # 按机身首次穿过两道立面的时刻划分，避免倒退后重复计入前一段。
    boundaries = [0]
    for edge in edges:
        crossings = (x[boundaries[-1]:] >= edge).nonzero().flatten()
        boundaries.append(boundaries[-1] + int(crossings[0]) if crossings.numel() else len(x))
    boundaries.append(len(x))
    phases = {}
    for name, first, end in zip(("approach", "first_to_second_riser", "later_ascent"),
                                boundaries[:-1], boundaries[1:], strict=True):
        phases[name] = alignment_window(lateral[first:end], heading[first:end])
    return {
        **alignment_window(lateral, heading),
        "mean_forward_mps": (x[-1].item() - start_x) / (len(x) * dt),
        "phases": phases,
    }


def summarize(rows, initial_yaw, dt, stairs, timing=None, *, alignment_data=None):
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
            stable = next(((first, end) for first, end in runs if (end - first) * dt >= TOP_HOLD_S), None)
            result["reached_top"] = stable is not None
            approach = valid[:stable[0]] if stable is not None else valid
            if alignment_data is not None:
                lateral, edges, start_x = alignment_data
                count = stable[0] + 1 if stable is not None else len(valid)
                result["alignment"] = ascent_alignment(valid[:count], lateral[:count, index], edges, start_x, dt)
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
        alignment = [row["alignment"] for row in by_env if "alignment" in row]
        if alignment:
            result["ascent_alignment"] = {
                "measured_envs": len(alignment),
                "median_abs_end_lateral_m": median(abs(r["end_lateral_m"]) for r in alignment),
                "median_peak_abs_lateral_m": median(r["peak_abs_lateral_m"] for r in alignment),
                "max_peak_abs_lateral_m": max(r["peak_abs_lateral_m"] for r in alignment),
                "median_peak_abs_heading_deg": median(r["peak_abs_heading_deg"] for r in alignment),
                "median_mean_forward_mps": median(r["mean_forward_mps"] for r in alignment),
            }
        top_times = [row["top_entry_s"] for row in by_env if "top_entry_s" in row]
        result["median_top_entry_s"] = median(top_times) if top_times else None
        near_edge = [row["first_riser"] for row in by_env if "first_riser" in row]
        if near_edge:
            approach_speeds = [r["approach_forward_speed_mps"] for r in near_edge
                               if r["approach_forward_speed_mps"] is not None]
            result["approach_speed_samples"] = len(approach_speeds)
            result["median_approach_forward_speed_mps"] = median(approach_speeds) if approach_speeds else None
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


def evaluate(checkpoint, device, num_envs, stair_level=0, start_x=0.65, stairs_only=False, *, vx=0.30, steering="off"):
    with patch.object(env_cfgs, "make_go2_supermimic_env_cfg", stair_contacts):
        env, policy = make_playback("Lite3-SuperMimic-Rough", checkpoint_file=checkpoint,
                                    device=device, num_envs=num_envs)
    try:
        if not 0 <= stair_level < env.adapter.map.rows:
            raise ValueError(f"Stair level must be between 0 and {env.adapter.map.rows - 1}")
        result = {"checkpoint": str(checkpoint), "stair_level": stair_level, "stair_start_x_m": start_x,
                  "command_vx_mps": vx, "steering": steering,
                  "stairs": rollout(env, policy, True, stair_level, start_x, vx=vx, steering=steering)}
        if not stairs_only:
            result["flat"] = rollout(env, policy, False, vx=vx)
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
    parser.add_argument("--vx", type=float, default=0.30, help="Forward command in m/s.")
    parser.add_argument("--steering", choices=STEERING_MODES, default="off")
    parser.add_argument("--stairs-only", action="store_true", help="Skip the flat-ground control run.")
    args = parser.parse_args()
    if args.num_envs < 1 or args.stair_level < 0 or not math.isfinite(args.start_x) or not math.isfinite(args.vx) or args.vx <= 0:
        parser.error("Expected a positive environment count, nonnegative stair level, finite start offset and positive forward speed")
    with patch.object(robot, "get_spec", partial(robot.get_spec, shank_radius_m=0.021)):
        for checkpoint in (args.baseline, *args.candidates):
            report = evaluate(checkpoint, args.device, args.num_envs, args.stair_level, args.start_x, args.stairs_only,
                              vx=args.vx, steering=args.steering)
            print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
