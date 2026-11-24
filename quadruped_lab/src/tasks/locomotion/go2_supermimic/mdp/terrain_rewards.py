# Copyright (c) 2026-2027 zh
"""按平地与上升台阶几何计算训练奖励。"""

import math
import torch

from src.assets.robots.lite3.constants import FOOT_COLLISION_RADIUS_M

from .reference_state import FEET, FOOT_CONTACT_FORCE, POSE, normalized_gravity, yaw_rotate_inverse
from .state_spec import CONTACT, CONTACT_TIME, CONTEXT_ORIGIN


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


def first_riser_swing_shortfall(task, state, next_state):
    """高台阶起步层的前摆净空，按预测足端与阶边的距离渐入。"""
    radius = FOOT_COLLISION_RADIUS_M
    with torch.no_grad():
        yaw = task.yaw(state)
        direction = torch.stack((yaw.cos(), yaw.sin()), dim=-1)
        front = state.physical[:, FEET].reshape(-1, 4, 3)[:, :2]
        xy = state.physical[:, POSE][:, None, :2] + yaw_rotate_inverse(front, -yaw[:, None])[..., :2]
        # 从足跟一侧开始确认起步层，向前探测足尖即将到达的首个高立面。
        offsets = xy.new_tensor((-.05, -.025, 0., .025, .05, .075, .10, .125, .15, .175, .20, .225))
        points = xy[:, :, None] + direction[:, None, None] * (offsets + radius)[None, None, :, None]
        heights = task.surface_height(points)
        rise = heights - state.context[:, CONTEXT_ORIGIN][:, 2, None, None]
        high = rise >= .12
        first = high.int().argmax(-1, keepdim=True)
        edge = (offsets[first.squeeze(-1)] - .0125).clamp_min(0.)
        tread = heights.gather(-1, (first + 1).clamp_max(offsets.numel() - 1)).squeeze(-1)
        # 起步层已经跨过后关闭；摆动起点保留首次撞立面当步的反馈。
        active = high.any(-1) & (rise[..., 0].abs() < .04)
        swing = 1. - state.memory[:, CONTACT][:, :2].clamp(0., 1.)
        vx, vy, wz = state.command.unbind(-1)
        forward = ((vx - .20) / .10).clamp(0., 1.) * (vy.abs() <= vx) * (wz.abs() <= .50)

    next_yaw = task.yaw(next_state)
    next_front = next_state.physical[:, FEET].reshape(-1, 4, 3)[:, :2]
    next_world = task.world_position(next_state)[:, None] + yaw_rotate_inverse(next_front, -next_yaw[:, None])
    advance = ((next_world[..., :2] - xy) * direction[:, None]).sum(-1)
    proximity = (1. - (edge - advance) / .18).clamp(0., 1.).square()
    shortfall = (tread + .008 - (next_world[..., 2] - radius)).clamp_min(0.)
    return forward * (active * swing * proximity * shortfall / (shortfall + .04)).mean(-1)


def first_riser_body_shortfall(task, state, next_state):
    """首个高立面附近限制低头和前壳净空不足，供真实与想象共用。"""
    with torch.no_grad():
        xy = state.physical[:, POSE][:, :2]
        yaw = task.yaw(state)
        direction = torch.stack((yaw.cos(), yaw.sin()), dim=-1)
        offsets = xy.new_tensor((0., .24, .28, .32, .36, .40, .44, .48, .52, .56))
        heights = task.surface_height(xy[:, None] + direction[:, None] * offsets[None, :, None])
        rise = heights - state.context[:, CONTEXT_ORIGIN][:, 2:3]
        high = rise >= .12
        first = high.int().argmax(-1, keepdim=True)
        tread = heights.gather(-1, (first + 1).clamp_max(offsets.numel() - 1)).squeeze(-1)
        proximity = (1. - (offsets[first[:, 0]] - .28) / .28).clamp(0., 1.)
        vx, vy, wz = state.command.unbind(-1)
        forward = ((vx - .20) / .10).clamp(0., 1.) * (vy.abs() <= vx) * (wz.abs() <= .50)
        weight = forward * high.any(-1) * (rise[:, 0].abs() < .04) * proximity

    gx, gy, gz = normalized_gravity(next_state.physical).unbind(-1)
    cosine = torch.linalg.vector_norm(torch.stack((gy, gz), dim=-1), dim=-1)
    pitch = torch.atan2(gx, cosine.clamp_min(1e-6))  # 正值表示低头。
    pitch_excess = ((pitch - math.radians(8.)) / math.radians(8.)).clamp_min(0.).square()
    # 用本模型实测外壳前端和碰撞盒下前沿分别作保守净空估计。
    shell = -.279 * gx + .029 * gz - .08 * gy.abs()
    box = -.22 * gx + .07 * gz - .08 * gy.abs()
    low_front_z = task.world_position(next_state)[:, 2] + torch.minimum(shell, box)
    height_excess = ((tread + .02 - low_front_z) / .06).clamp_min(0.).square()
    return weight * (.6 * pitch_excess / (1. + pitch_excess) + .4 * height_excess / (1. + height_excess))


def _raised_treads(task, xy, direction):
    """沿多个前缘采样，以局部高度差识别每个高立面和上方踏面。"""
    offsets = xy.new_tensor((-.10, -.075, -.05, -.025, 0., .025, .05, .075,
                             .10, .125, .15, .175, .20, .225, .25))
    heights = task.surface_height(xy[:, :, None] + direction[:, None, None] * offsets[None, None, :, None])
    # 比较相距7.5厘米的点，覆盖高度栅格在台阶边缘的插值带。
    rise = heights[..., 3:] - heights[..., :-3]
    gate = ((rise - .105) / .035).clamp(0., 1.)
    upper = torch.cat((heights[..., 4:], heights[..., -1:]), dim=-1)
    return upper, gate, offsets[3:] - .025


def _edge_proximity(distance, advance, reach):
    """在前缘接近阶边时渐入，完全跨过后退出。"""
    approaching = (1. - (distance - advance) / reach).clamp(0., 1.).square()
    # 按步起点决定退出，避免通过本步向前穿进立面来减轻惩罚。
    leaving = ((distance + .09) / .055).clamp(0., 1.)
    return approaching * leaving


def _forward_weight(command):
    vx, vy, wz = command.unbind(-1)
    return ((vx - .20) / .10).clamp(0., 1.) * (vy.abs() <= vx) * (wz.abs() <= .50)


def next_tread_front_foot_shortfall(task, state, next_state):
    """把净空反馈延伸到阶边受阻，普通承重前足仍可保持支撑。"""
    radius = FOOT_COLLISION_RADIUS_M
    with torch.no_grad():
        yaw = task.yaw(state)
        direction = torch.stack((yaw.cos(), yaw.sin()), dim=-1)
        front = state.physical[:, FEET].reshape(-1, 4, 3)[:, :2]
        xy = state.physical[:, POSE][:, None, :2] + yaw_rotate_inverse(front, -yaw[:, None])[..., :2]
        tread, riser, edge = _raised_treads(task, xy + radius * direction[:, None], direction)
        contact = state.memory[:, CONTACT][:, :2].clamp(0., 1.)
        force = state.physical[:, FOOT_CONTACT_FORCE].reshape(-1, 4, 3)[:, :2]
        along = (force[..., :2] * direction[:, None]).sum(-1).abs()
        blocked = ((along - .5 * force[..., 2].abs()) / 12.).clamp(0., 1.)
        at_edge = ((.055 - edge.abs()) / .04).clamp(0., 1.)
        eligible = (1. - contact)[..., None] + (contact * blocked)[..., None] * at_edge
        forward = _forward_weight(state.command)

    next_yaw = task.yaw(next_state)
    next_front = next_state.physical[:, FEET].reshape(-1, 4, 3)[:, :2]
    points = task.world_position(next_state)[:, None] + yaw_rotate_inverse(next_front, -next_yaw[:, None])
    advance = ((points[..., :2] - xy) * direction[:, None]).sum(-1)
    proximity = _edge_proximity(edge, advance[..., None], .18)
    deficit = (tread + .008 - (points[..., 2, None] - radius)).clamp_min(0.)
    per_edge = riser * eligible * proximity * deficit / (deficit + .04)
    return forward * per_edge.amax(-1).mean(-1)


def next_tread_body_shortfall(task, state, next_state):
    """在机身前沿分别检查外壳和碰撞盒，连续跟随楼梯踏面。"""
    xy = state.physical[:, POSE][:, :2]
    # 外壳前端和碰撞盒下前沿采用与诊断模型一致的保守纵向探针。
    lengths = xy.new_tensor((.279, .22))
    with torch.no_grad():
        yaw = task.yaw(state)
        direction = torch.stack((yaw.cos(), yaw.sin()), dim=-1)
        anchors = xy[:, None] + direction[:, None] * lengths[None, :, None]
        tread, riser, edge = _raised_treads(task, anchors, direction)
        forward = _forward_weight(state.command)

    gx, gy, gz = normalized_gravity(next_state.physical).unbind(-1)
    root = task.world_position(next_state)
    advance = ((root[:, :2] - xy) * direction).sum(-1)
    proximity = _edge_proximity(edge, advance[:, None, None], .22)
    depths = xy.new_tensor((.029, .07))
    bottom = root[:, 2:3] - gx[:, None] * lengths + gz[:, None] * depths - .08 * gy.abs()[:, None]
    deficit = ((tread + .02 - bottom[..., None]) / .06).clamp_min(0.).square()
    per_edge = riser * proximity * deficit / (1. + deficit)
    return forward * per_edge.amax(dim=(-2, -1))


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
    if scales.first_riser_swing_shortfall:
        terms["first_riser_swing_shortfall"] = first_riser_swing_shortfall(task, state, next_state)
    if scales.first_riser_body_shortfall:
        terms["first_riser_body_shortfall"] = first_riser_body_shortfall(task, state, next_state)
    if scales.next_tread_front_foot_shortfall:
        terms["next_tread_front_foot_shortfall"] = next_tread_front_foot_shortfall(task, state, next_state)
    if scales.next_tread_body_shortfall:
        terms["next_tread_body_shortfall"] = next_tread_body_shortfall(task, state, next_state)
    return terms
