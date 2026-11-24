# Copyright (c) 2026-2027 zh
"""按平地与上升台阶几何计算训练奖励。"""

import torch

from src.assets.robots.lite3.constants import FOOT_COLLISION_RADIUS_M

from .reference_state import FEET, POSE, yaw_rotate_inverse
from .state_spec import CONTACT, CONTACT_TIME


def flat_ground_weight(task, physical):
    """沿用局部五点探测，地面起伏达到 3 cm 时关闭平地评分。"""
    xy = physical[:, POSE][:, :2].detach()
    probes = xy.new_tensor(((0., 0.), (.4, 0.), (-.4, 0.), (0., .4), (0., -.4)))
    heights = task.surface_height(xy[:, None] + probes)
    return (1. - (heights.amax(-1) - heights.amin(-1)) / 0.03).clamp(0., 1.)


def front_riser_clearance_shortfall(task, state, next_state):
    """用前足的上方踏面估计净空，真实与想象共用。"""
    with torch.no_grad():
        yaw = task.yaw(state)
        front = state.physical[:, FEET].reshape(-1, 4, 3)[:, :2]
        xy = state.physical[:, POSE][:, None, :2] + yaw_rotate_inverse(front, -yaw[:, None])[..., :2]
        direction = torch.stack((yaw.cos(), yaw.sin()), dim=-1)
        offsets = xy.new_tensor((0., .04, .08, .12))
        points = xy[:, :, None] + direction[:, None, None] * offsets[None, None, :, None]
        heights = task.surface_height(points)
        # 陡升阈值抑制平地和通常的坡面；最远探针留出提前抬腿的空间。
        rise = (heights[..., 1:] - heights[..., :-1]).amax(-1)
        riser = ((rise - 0.02) / 0.02).clamp(0., 1.)
        upper_tread = heights[..., 1:].amax(-1)
        # 用本步起点的接触判断摆动，保留第一次触碰立面当步的评分。
        contact = state.memory[:, CONTACT].clamp(0., 1.)
        front_contact = contact[:, :2]
        swing = 1. - front_contact
        prelift = task.cfg.reward.front_riser_prelift_s
        if prelift:
            # 想象计时按接触概率加权，先恢复条件支撑时长。
            stance_time = state.memory[:, CONTACT_TIME][:, :2] / front_contact.clamp_min(1e-6)
            gait = task.cfg.reward.gait
            stance_target = gait.period * gait.stance_fraction
            ready = (1. - (stance_target - stance_time) / prelift).clamp(0., 1.)
            other_contacts = contact.sum(-1, keepdim=True) - front_contact
            support = (other_contacts / 2.).clamp(0., 1.)
            # 支撑末段渐入，足端仍承重时至多使用一半摆动权重。
            swing = swing + 0.5 * front_contact * ready * support

    sole_radius = max(FOOT_COLLISION_RADIUS_M, task.cfg.experiment.lite3_shank_collision_radius_m)
    front_z = next_state.physical[:, FEET].reshape(-1, 4, 3)[:, :2, 2]
    foot_bottom = task.world_position(next_state)[:, 2:3] + front_z - sole_radius
    shortfall = (upper_tread + task.cfg.reward.front_riser_clearance_margin - foot_bottom).clamp_min(0.)
    return (riser * swing * shortfall / (shortfall + 0.04)).mean(-1)


def terrain_reward_terms(task, state, next_state, velocity):
    cfg = task.cfg.reward
    scales = cfg.scales
    terms = {}
    if scales.flat_forward_velocity_error or scales.front_riser_clearance_shortfall:
        vx, vy, yaw = state.command.unbind(-1)
        speed = ((vx - 0.20) / 0.10).clamp(0., 1.) * ((0.85 - vx) / 0.15).clamp(0., 1.)
        straight = speed * (vy.abs() <= 0.10) * (yaw.abs() <= 0.25)
    if scales.flat_forward_velocity_error:
        error = (velocity[:, 0] - vx) / cfg.flat_forward_velocity_error_scale
        terms["flat_forward_velocity_error"] = (
            straight * flat_ground_weight(task, next_state.physical) * -torch.expm1(-error.square()))
    if scales.front_riser_clearance_shortfall:
        terms["front_riser_clearance_shortfall"] = straight * front_riser_clearance_shortfall(task, state, next_state)
    return terms
