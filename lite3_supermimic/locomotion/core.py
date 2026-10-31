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
    return cmd * (torch.rand(n, 1, device=vmax.device) >= stage.stand_fraction)


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


def reward_terms(s0, s1, p0, p1, a, prev, command, c, dt, field=None):
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
    relief = tr.relief(p1[:, :2], p1[:, 2], field)
    rough = (relief.amax(-1) - relief.amin(-1)).clamp(0, 0.12)
    desired_clearance = 0.04 + rough[:, None] * 0.5
    return {
        "track_linear": torch.exp(-(s1[:, 4:6] - command[:, :2]).square().sum(-1) / 0.25),
        "track_yaw": torch.exp(-(s1[:, 9] - command[:, 2]).square() / 0.25),
        "upright": torch.exp(-s1[:, 1:3].square().sum(-1) / 0.4**2),
        "base_height": (s1[:, 0] - 0.28).square(),
        "roll_pitch_rate": s1[:, 7:9].square().sum(-1),
        "vertical_velocity": s1[:, 6].square(),
        "action_rate": (a - prev).square().sum(-1),
        "joint_limits": ((c.soft_limits[..., 0] - q).clamp(min=0)
                         + (q - c.soft_limits[..., 1]).clamp(min=0)).sum(-1),
        "joint_torques": tau.square().sum(-1),
        "joint_acceleration": ((s1[:, st.QD] - s0[:, st.QD]) / dt).square().sum(-1),
        "foot_slip": (contact * speed2).sum(-1),
        "foot_clearance": ((desired_clearance - clear).clamp(min=0).square()
                           * speed2 * (1 - contact)).sum(-1),
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
