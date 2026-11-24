# Copyright (c) 2026-2027 zh
"""内容：实现指定HIM接触奖励及其已授权的想象软接触近似。"""

import math

import torch

from .state_spec import AIR_TIME, CONTACT, CONTACT_TIME, ELAPSED_TIME, FOOT_NAMES
from .reference_state import FEET, HIPS, POSE


DIAGONAL_PAIRS = (("FL", "HR"), ("FR", "HL"))
_DIAGONAL_FIRST = tuple(FOOT_NAMES.index(first) for first, _ in DIAGONAL_PAIRS)
_DIAGONAL_SECOND = tuple(FOOT_NAMES.index(second) for _, second in DIAGONAL_PAIRS)


def gait_phase(state, dt, period):
    """与来源整数episode_length_buf的取余顺序一致，消除浮点累时边界漂移。"""
    steps = torch.round(state.memory[:, ELAPSED_TIME] / dt)
    return torch.remainder(steps, period / dt) / (period / dt)


def gait_reward_terms(task, state, next_state):
    """二值接触严格等于来源；概率接触按逐脚期望匹配及连续计时近似。"""
    cfg = task.cfg.reward.gait
    phase = gait_phase(next_state, task.dt, cfg.period)
    offsets = task.gait_offsets
    leg_phase = (phase[:, None] + offsets) % 1.0
    desired = (leg_phase < cfg.stance_fraction).to(phase.dtype)
    contact = next_state.memory[:, CONTACT]
    active = (state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs() > cfg.command_threshold).to(phase.dtype)
    matches = desired * contact + (1 - desired) * (1 - contact)
    stance_duration = cfg.period * cfg.stance_fraction
    swing_duration = cfg.period * (1 - cfg.stance_fraction)
    # 想象记忆保存概率加权时长，先恢复条件时长；二值真实状态原样保留。
    contact_duration = next_state.memory[:, CONTACT_TIME] / contact.clamp_min(1e-6)
    air_duration = next_state.memory[:, AIR_TIME] / (1 - contact).clamp_min(1e-6)
    stance_score = (1 - (contact_duration / stance_duration - 1).abs()).clamp_min(0)
    swing_score = (1 - (air_duration / swing_duration - 1).abs()).clamp_min(0)
    durations = desired * contact * stance_score + (1 - desired) * (1 - contact) * swing_score
    terms = {"foot_gait": matches.mean(-1) * active, "feet_air_time": durations.mean(-1) * active}
    scales = task.cfg.reward.scales
    if (scales.swing_clearance_shortfall or scales.front_swing_clearance or scales.front_reach_shortfall
            or scales.diagonal_contact_mismatch):
        vx, vy, yaw = state.command.unbind(-1)
        straight_forward = ((vx - 0.2) / 0.2).clamp(0., 1.) * (vy.abs() <= 0.1) * (yaw.abs() <= 0.25)
    if scales.swing_clearance_shortfall or scales.front_reach_shortfall:
        swing = ((leg_phase - cfg.stance_fraction) / (1 - cfg.stance_fraction)).clamp(0., 1.)
    if scales.diagonal_contact_mismatch or scales.front_swing_clearance:
        # 局部起伏增大时，逐步放宽平地步态约束。
        xy = next_state.physical[:, POSE][:, :2].detach()
        probes = xy.new_tensor(((0., 0.), (.4, 0.), (-.4, 0.), (0., .4), (0., -.4)))
        heights = task.surface_height(xy[:, None] + probes)
        relief = heights.amax(-1) - heights.amin(-1)
        flat = (1. - relief / 0.03).clamp(0., 1.)
    if scales.diagonal_contact_mismatch:
        mismatch = (contact[:, _DIAGONAL_FIRST] - contact[:, _DIAGONAL_SECOND]).square().mean(-1)
        terms["diagonal_contact_mismatch"] = straight_forward * flat * mismatch
    if scales.swing_clearance_shortfall or scales.front_swing_clearance:
        # 足端中心与半径 25 mm 的小腿碰撞端点重合。
        clearance = task.feet_height(next_state.physical) - 0.025
    if scales.swing_clearance_shortfall:
        arc = torch.sin(math.pi * swing).square()
        target = cfg.swing_clearance_target
        shortfall = ((target * arc - clearance) / target).clamp(0., 2.)
        terms["swing_clearance_shortfall"] = straight_forward * (arc * shortfall).mean(-1)
    if scales.front_swing_clearance:
        # 支撑时足端相对髋向后移动，向前摆时才奖励净空。
        before = state.physical
        after = next_state.physical
        forward = ((after[:, FEET] - after[:, HIPS]) - (before[:, FEET] - before[:, HIPS]))
        forward_speed = forward.reshape(-1, 4, 3)[:, :2, 0] / task.dt
        swing_weight = (forward_speed / 0.6).clamp(0., 1.)
        height_score = (clearance[:, :2] / cfg.swing_clearance_target).clamp(0., 1.)
        terms["front_swing_clearance"] = straight_forward * flat * (height_score * swing_weight).mean(-1)
    if scales.front_reach_shortfall:
        physical = next_state.physical
        reach = (physical[:, FEET] - physical[:, HIPS]).reshape(-1, 4, 3)[:, :2, 0]
        late = ((swing[:, :2] - 0.5) * 2).clamp(0., 1.)
        late_weight = torch.sin(math.pi * late).square()
        # 按一半支撑期的前进距离确定前伸目标，并限制高速目标。
        target = (0.5 * vx * stance_duration).clamp(0., 0.13)
        shortfall = ((target[:, None] - reach) / 0.10).clamp(0., 2.)
        terms["front_reach_shortfall"] = straight_forward * (late_weight * shortfall).mean(-1)
    return terms


def posture_reward(task, state, next_state):
    """严格采用go2_ppo的逐关节标准差与速度分段。"""
    cfg = task.cfg.reward.posture
    magnitude = state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs()
    standing = magnitude < cfg.walking_threshold
    walking = (magnitude >= cfg.walking_threshold) & (magnitude < cfg.running_threshold)
    std = (task.posture_std[0] * standing[:, None] + task.posture_std[1] * walking[:, None]
           + task.posture_std[2] * (magnitude >= cfg.running_threshold)[:, None])
    return torch.exp(-(next_state.physical[:, 10:22].square() / std.square()).mean(-1))
