# Copyright (c) 2026-2027 zh
"""内容：实现指定HIM接触奖励及其已授权的想象软接触近似。"""

import math

import torch

from .state_spec import AIR_TIME, CONTACT, CONTACT_TIME, ELAPSED_TIME


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
    if task.cfg.reward.scales.swing_clearance_shortfall:
        # 足端中心与半径 25 mm 的小腿碰撞端点重合。
        clearance = task.feet_height(next_state.physical) - 0.025
        swing = ((leg_phase - cfg.stance_fraction) / (1 - cfg.stance_fraction)).clamp(0., 1.)
        arc = torch.sin(math.pi * swing).square()
        shortfall = ((0.035 * arc - clearance) / 0.035).clamp(0., 2.)
        vx, vy, yaw = state.command.unbind(-1)
        straight_forward = ((vx - 0.2) / 0.2).clamp(0., 1.) * (vy.abs() <= 0.1) * (yaw.abs() <= 0.25)
        terms["swing_clearance_shortfall"] = straight_forward * (arc * shortfall).mean(-1)
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
