# Copyright (c) 2026-2027 zh
"""内容：实现HIM速度采样与heading反馈，真实和想象使用同一命令状态。"""

import torch
from mjlab.utils.lab_api.math import wrap_to_pi

from .state_spec import (
    COMMAND_MODE, COMMAND_OVERRIDE, COMMAND_TIMER, FIXED_STRAIGHT_MODE,
    HEADING_TARGET, HIGH_STAIR_HEADING_MODE, STANDING,
)


def sample_commands(cfg, context, yaw):
    """采样速度、指令模式和窗口时长。"""
    bounds = context[:, 19:25].reshape(-1, 3, 2)
    random = torch.rand_like(bounds[..., 0])
    command = bounds[..., 0] + random * (bounds[..., 1] - bounds[..., 0])
    heading = cfg.heading_range[0] + torch.rand_like(yaw) * (cfg.heading_range[1] - cfg.heading_range[0])
    standing = torch.rand_like(yaw) <= cfg.rel_standing_envs
    mode_sample = torch.rand_like(yaw)
    forward = mode_sample <= cfg.rel_forward_envs
    mode = forward.to(yaw.dtype)
    command = torch.stack((torch.where(forward, command[:, 0].abs().clamp_min(0.3), command[:, 0]),
                           torch.where(forward, torch.zeros_like(yaw), command[:, 1]), command[:, 2]), -1)
    if cfg.rel_fixed_straight_envs:
        # 复用同一个随机样本，固定直行只占行走窗口。
        straight = ~standing & (mode_sample >= 1.0 - cfg.rel_fixed_straight_envs)
        mode = torch.where(straight, FIXED_STRAIGHT_MODE, mode)
        fixed = command.new_tensor((cfg.fixed_straight_speed, 0., 0.))
        command = torch.where(straight[:, None], fixed, command)
        heading = torch.where(straight, yaw, heading)
    lower, upper = cfg.resampling_time
    timer = lower + (upper - lower) * torch.rand_like(yaw)
    return command, timer, heading, standing, mode


def heading_command(cfg, command, yaw, heading, standing, mode):
    """常规行走使用航向反馈，固定直行保留零偏航指令。"""
    error = wrap_to_pi(heading - yaw)
    angular = (cfg.heading_control_stiffness * error).clamp(*cfg.initial_ranges[2])
    angular = torch.where(mode == FIXED_STRAIGHT_MODE, 0., angular)
    if cfg.high_stair_heading_feedback:
        stair_yaw = (1.5 * error).clamp(-.30, .30)
        angular = torch.where(mode == HIGH_STAIR_HEADING_MODE, stair_yaw, angular)
    result = torch.cat((command[:, :2], angular[:, None]), -1)
    return torch.where(standing[:, None].bool(), torch.zeros_like(result), result)


def advance_commands(cfg, state, physical, dt):
    """在终点yaw刷新下一决策命令；固定覆盖命令不受heading重写。"""
    memory = state.memory
    sampled, timer, heading, standing, mode = sample_commands(cfg, state.context, physical[:, 72])
    expired = memory[:, COMMAND_TIMER] - dt <= 0.0
    command = torch.where(expired[:, None], sampled, state.command)
    timer = torch.where(expired, timer, memory[:, COMMAND_TIMER] - dt)
    heading = torch.where(expired, heading, memory[:, HEADING_TARGET])
    standing = torch.where(expired, standing, memory[:, STANDING].bool())
    mode = torch.where(expired, mode, memory[:, COMMAND_MODE])
    command = heading_command(cfg, command, physical[:, 72], heading, standing, mode)
    command = torch.where(memory[:, COMMAND_OVERRIDE, None].bool(), state.command, command)
    return command, timer, heading, standing, mode
