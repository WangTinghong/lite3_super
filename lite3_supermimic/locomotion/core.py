"""Shared tensor definitions for actual and imagined locomotion (no mjlab dependency)."""
from __future__ import annotations

import torch

from lite3_supermimic.supermimic import state as st
from lite3_supermimic.target import terrain as tr
from .config import Stage

# Same dimensions in all four stages, including a height scan on the plane.
TERMS = st.OBS_TERMS[:-1] + (("velocity_command", 3), ("height_scan", tr.SCAN_POINTS))


def sample_commands(n: int, vmax: torch.Tensor, stage: Stage) -> torch.Tensor:
    r = torch.rand(n, 3, device=vmax.device)
    cmd = torch.stack([
        stage.backward + r[:, 0] * (vmax - stage.backward),
        (2 * r[:, 1] - 1) * stage.lateral,
        (2 * r[:, 2] - 1) * stage.yaw,
    ], -1)
    # Independent components include pure turns, straight runs and standing.
    cmd[:, :2] *= (torch.rand(n, 1, device=vmax.device) > 0.1)
    cmd[:, 1:] *= (torch.rand(n, 1, device=vmax.device) > 0.3)
    cmd = cmd * (torch.rand(n, 1, device=vmax.device) >= stage.stand_fraction)
    if stage.stair_focus_prob:
        # Use the same sampler in the real environment and imagined rollouts.
        limit = vmax.clamp(max=1.0)
        forward = 0.25 + torch.rand(n, device=vmax.device) * (limit - 0.25).clamp(min=0)
        straight = torch.stack((forward, torch.zeros_like(forward), torch.zeros_like(forward)), -1)
        cmd = torch.where((torch.rand(n, device=vmax.device) < stage.stair_focus_prob)[:, None],
                          straight, cmd)
    return cmd


def advance_commands(packed: torch.Tensor, dt: float, stage: Stage) -> torch.Tensor:
    """Packed command = (vx, vy, yaw rate, seconds left, forward speed ceiling)."""
    left = packed[:, 3] - dt
    replace = left <= 0
    cmd = sample_commands(len(packed), packed[:, 4], stage)
    duration = torch.empty_like(left).uniform_(*stage.interval)
    return torch.cat([torch.where(replace[:, None], cmd, packed[:, :3]),
                      torch.where(replace, duration, left)[:, None], packed[:, 4:5]], -1)


def observation(s, action, command, bias, scan, noise=False):
    frame = st.obs_frame(s, action, command, bias, noise=noise, height_scan=scan)
    frame["velocity_command"] = frame.pop("target")
    return {name: frame[name] for name, _ in TERMS}


def feet_world(s, pose, field):
    feet = st.limb_points_world(s, pose)["feet"]
    # s.Z is height above the ground UNDER THE BASE, not a world z coordinate.
    lift = tr.ground(pose[:, :2], field)
    return feet + torch.stack([0 * lift, 0 * lift, lift], -1)[:, None]


def foot_clearances(s, pose, field):
    feet = feet_world(s, pose, field)
    return feet[..., 2] - tr.ground(feet[..., :2], field) - st.FOOT_RADIUS


def stair_clearance_terms(s, pose, command, field):
    """Geometry-based, differentiable proxies shared by simulation and imagination.

    A foot approaching a riser must clear the landing surface. Samples along
    each lower leg discourage using the shank as a contact surface. The local
    height gradient penalizes a diagonal approach to an edge.
    """
    zero = s[:, 0] * 0
    if field is None:
        return zero, zero, zero
    points = st.limb_points_world(s, pose)
    base_ground = tr.ground(pose[:, :2], field)
    lift = torch.stack((zero, zero, base_ground), -1)[:, None]
    feet, knees = points["feet"] + lift, points["knees"] + lift
    forward = torch.stack((pose[:, 2].cos(), pose[:, 2].sin()), -1)
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), -1)
    foot_xy = feet[..., :2]
    here = tr.ground(foot_xy, field)
    ahead = torch.maximum(tr.ground(foot_xy + 0.12 * forward[:, None], field),
                          tr.ground(foot_xy + 0.25 * forward[:, None], field))
    rise = (ahead - here).clamp(min=0, max=0.25)
    foot_clear = feet[..., 2] - here - st.FOOT_RADIUS
    # Positive only near a rising edge. A landing foot already touching its
    # current surface is not asked to lift until it starts swinging.
    swing = 0.2 + 0.8 * torch.sigmoid((foot_clear - 0.015) / 0.01)
    foot_penalty = ((0.04 + rise - foot_clear).clamp(min=0).square()
                    * swing * (rise / 0.04).clamp(0, 1)).sum(-1)

    # The locomotion-only collision capsule stops 3 cm above the foot centre.
    # Sample its centreline before that endpoint; terrain penetration predicts
    # likely riser contact without using simulator-only contact flags.
    shank_penalty = zero
    for t in (0.45, 0.70, 0.85):
        point = knees + t * (feet - knees)
        deficit = (tr.ground(point[..., :2], field) + 0.025 - point[..., 2]).clamp(min=0)
        shank_penalty = shank_penalty + deficit.square().sum(-1)

    base = pose[:, :2]
    grad_f = tr.ground(base + 0.30 * forward, field) - tr.ground(base - 0.30 * forward, field)
    grad_l = tr.ground(base + 0.30 * lateral, field) - tr.ground(base - 0.30 * lateral, field)
    edge = ((grad_f.abs() + grad_l.abs() - 0.04) / 0.08).clamp(0, 1)
    alignment = edge * grad_l.square() / (grad_f.square() + grad_l.square() + 0.0025)
    moving_forward = ((command[:, 0] - 0.15) / 0.15).clamp(0, 1)
    return (foot_penalty * moving_forward,
            shank_penalty * moving_forward,
            alignment * moving_forward)


def stair_hipx_target_penalty(s, pose, a, command, c, field):
    """Discourage front-leg outward targets only while approaching a rising edge.

    The target receives a direct actor gradient; a soft hinge still permits
    lateral balance corrections and ordinary turning away from the riser.
    """
    if field is None:
        return s[:, 0] * 0
    feet = feet_world(s, pose, field)[:, :2]  # FR, FL
    forward = torch.stack((pose[:, 2].cos(), pose[:, 2].sin()), -1)
    xy = feet[..., :2]
    here = tr.ground(xy, field)
    ahead = torch.maximum(tr.ground(xy + 0.12 * forward[:, None], field),
                          tr.ground(xy + 0.25 * forward[:, None], field))
    edge = ((ahead - here - 0.01) / 0.04).clamp(0, 1)[:, [1, 0]]  # FL, FR
    target = c.default_q[:, 0:6:3] + a[:, 0:6:3] * c.action_scale[:, 0:6:3]
    outward = target * target.new_tensor([-1, 1])
    straight = ((command[:, 0] - 0.25) / 0.25).clamp(0, 1)
    straight = straight * (1 - command[:, 1].abs() / 0.15).clamp(0, 1)
    straight = straight * (1 - command[:, 2].abs() / 0.3).clamp(0, 1)
    return (((outward - 0.35).clamp(min=0).square() * edge).sum(-1) * straight)


def stair_foothold_reward(s0, s1, p0, p1, command, field, dt):
    """Reward a front foot settling onto a higher tread while the base is below it.

    All quantities come from the same state and height map in sim and WM.
    This complements clearance shaping, which only encourages lifting the foot.
    """
    if field is None:
        return s1[:, 0] * 0
    feet0 = feet_world(s0, p0, field)[:, :2]
    feet1 = feet_world(s1, p1, field)[:, :2]  # FR, FL
    tread = tr.ground(feet1[..., :2], field)
    base_ground = tr.ground(p1[:, None, :2], field)
    above_base = ((tread - base_ground - 0.02) / 0.05).clamp(0, 1)
    clearance = feet1[..., 2] - tread - st.FOOT_RADIUS
    settled = torch.exp(-((clearance - 0.01) / 0.05).square())
    foot_speed2 = ((feet1[..., :2] - feet0[..., :2]) / dt).square().sum(-1)
    stable = torch.exp(-foot_speed2 / 0.5**2)
    forward = ((command[:, 0] - 0.2) / 0.2).clamp(0, 1)
    straight = (1 - command[:, 1].abs() / 0.15).clamp(0, 1)
    straight = straight * (1 - command[:, 2].abs() / 0.3).clamp(0, 1)
    return (above_base * settled * stable).sum(-1) * 0.5 * forward * straight


def reward_terms(s0, s1, p0, p1, a, prev, command, c, dt, field=None,
                 stair_refine=False, stair_hipx_refine=False, stair_foothold_refine=False):
    """Identical state-based shaping on sim/WM; contact is a smooth height proxy.

    Contact-force measurements are evaluation metrics, not hidden reward inputs
    unavailable to the world model. Termination is handled separately.
    """
    q = s1[:, st.Q] + c.default_q
    tau = (c.kp * (a * c.action_scale + c.default_q - q) - c.kd * s1[:, st.QD])
    tau = tau.clamp(-c.effort, c.effort)
    clear = foot_clearances(s1, p1, field)
    vf = (feet_world(s1, p1, field) - feet_world(s0, p0, field)) / dt
    speed2 = vf[..., :2].square().sum(-1)
    contact = torch.sigmoid((0.015 - clear) / 0.01)
    # A soft contact proxy keeps the same gait objective differentiable in the
    # world-model rollout. Apply only to near-straight, moderate forward motion;
    # standing, turning and terrain adaptation should remain unconstrained.
    gait_gate = ((command[:, 0] - 0.3) / 0.2).clamp(0, 1)
    gait_gate = gait_gate * (1 - command[:, 1].abs() / 0.3).clamp(0, 1)
    gait_gate = gait_gate * (1 - command[:, 2].abs() / 0.5).clamp(0, 1)
    diagonal_mismatch = ((contact[:, 0] - contact[:, 3]).abs()
                         + (contact[:, 1] - contact[:, 2]).abs()) * 0.5
    flight = (1 - contact).prod(-1)
    # Joint order is FL, FR, HL, HR; contact order is FR, FL, HR, HL.
    # Every HipX axis points toward -x, so negative angles on the left and
    # positive angles on the right move the leg outward.
    hipx_outward = q[:, 0::3] * q.new_tensor([-1, 1, -1, 1])
    target_q = c.default_q + a * c.action_scale
    target_hipx_outward = target_q[:, 0::3] * q.new_tensor([-1, 1, -1, 1])
    swing = 1 - contact[:, [1, 0, 3, 2]]
    hipx_swing_outward = (hipx_outward - 0.25).clamp(min=0).square() * swing
    hipx_target_outward = (target_hipx_outward - 0.25).clamp(min=0).square() * swing
    relief = tr.relief(p1[:, :2], p1[:, 2], field)
    rough = (relief.amax(-1) - relief.amin(-1)).clamp(0, 0.12)
    desired_clearance = 0.04 + rough[:, None] * 0.5
    stair_foot, shank, alignment = (stair_clearance_terms(s1, p1, command, field)
                                    if stair_refine else (s1[:, 0] * 0,) * 3)
    stair_hipx = (stair_hipx_target_penalty(s1, p1, a, command, c, field)
                   if stair_hipx_refine else s1[:, 0] * 0)
    stair_foothold = (stair_foothold_reward(s0, s1, p0, p1, command, field, dt)
                      if stair_foothold_refine else s1[:, 0] * 0)
    return {
        "track_linear": torch.exp(-(s1[:, 4:6] - command[:, :2]).square().sum(-1) / 0.25),
        "track_yaw": torch.exp(-(s1[:, 9] - command[:, 2]).square() / 0.25),
        "upright": torch.exp(-s1[:, 1:3].square().sum(-1) / 0.4**2),
        "base_height": (s1[:, 0] - 0.28).square(),
        "roll_pitch_rate": s1[:, 7:9].square().sum(-1),
        "vertical_velocity": s1[:, 6].square(),
        "diagonal_contact": gait_gate * diagonal_mismatch,
        "all_airborne": gait_gate * flight,
        "hipx_swing_outward": gait_gate * hipx_swing_outward.sum(-1),
        "hipx_target_outward": gait_gate * hipx_target_outward.sum(-1),
        "action_rate": (a - prev).square().sum(-1),
        "joint_limits": ((c.soft_limits[..., 0] - q).clamp(min=0)
                         + (q - c.soft_limits[..., 1]).clamp(min=0)).sum(-1),
        "joint_torques": tau.square().sum(-1),
        "joint_acceleration": ((s1[:, st.QD] - s0[:, st.QD]) / dt).square().sum(-1),
        "foot_slip": (contact * speed2).sum(-1),
        "foot_clearance": ((desired_clearance - clear).clamp(min=0).square()
                           * speed2 * (1 - contact)).sum(-1),
        "stair_foot_clearance": stair_foot,
        "shank_clearance": shank,
        "stair_alignment": alignment,
        "stair_hipx_target_outward": stair_hipx,
        "stair_foothold": stair_foothold,
        "stand_still": (command.abs().sum(-1) < 0.05) * s1[:, st.Q].square().sum(-1),
    }


def validate_checkpoint(ckpt):
    from lite3_supermimic.checkpoint import validate_checkpoint as validate_robot
    from .config import HISTORY, SCHEMA
    validate_robot(ckpt)
    if ckpt.get("task_family") != SCHEMA:
        raise ValueError("This entry point requires a locomotion checkpoint; chase/PPO checkpoints are incompatible.")
    terms = tuple(tuple(t) for t in ckpt.get("obs_terms", ()))
    if terms != TERMS or ckpt.get("history") != HISTORY:
        raise ValueError("Locomotion observation schema/history mismatch")
