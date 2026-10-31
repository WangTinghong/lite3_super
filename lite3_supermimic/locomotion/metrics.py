"""Physical-rollout metrics; contact data comes from the simulator, not the WM."""
from __future__ import annotations

import torch
from .core import feet_world, foot_clearances


def read_contacts(sensor):
    """Match named feet: mjlab's sensor order may follow XML (FL, FR, HL, HR)."""
    from lite3_supermimic.robot.constants import FOOT_GEOM_NAMES
    names = [name.split("/")[-1] for name in sensor.primary_names]
    order = [names.index(name) for name in FOOT_GEOM_NAMES]
    n = sensor.data.found.shape[0]
    found = sensor.data.found.reshape(n, len(names), -1).any(-1)[:, order]
    forces = sensor.data.force.reshape(n, len(names), -1, 3).sum(2)[:, order]
    return found, forces


class Metrics:
    def __init__(self, n, device):
        self.n, self.samples, self.elapsed = n, 0, 0.0
        self.falls, self.completed, self.survived = 0, 0, 0
        self.sums = {k: 0.0 for k in ("linear_sq", "yaw_sq", "speed", "action_rate", "slip_sq", "slip_count")}
        self.contact_time = torch.zeros(4, device=device)
        self.touchdowns = torch.zeros(4, device=device)
        self.stride_sum = torch.zeros(4, device=device)
        self.stride_count = torch.zeros(4, device=device)
        self.last_touch = torch.zeros(n, 4, 3, device=device)
        self.has_touch = torch.zeros(n, 4, dtype=torch.bool, device=device)
        self.prev_contact = torch.zeros_like(self.has_touch)
        self.prev_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.clear_sum = torch.zeros(4, device=device)
        self.swing_count = torch.zeros(4, device=device)
        self.max_force = 0.0

    def update(self, s0, p0, s1, p1, cmd, a, prev, contact, force, terminated, truncated, dt, field=None):
        self.samples += self.n
        self.elapsed += dt
        self.falls += int(terminated.sum())
        done = terminated | truncated
        self.completed += int(done.sum())
        self.survived += int((truncated & ~terminated).sum())
        self.sums["linear_sq"] += float((s1[:, 4:6] - cmd[:, :2]).square().sum())
        self.sums["yaw_sq"] += float((s1[:, 9] - cmd[:, 2]).square().sum())
        self.sums["speed"] += float(s1[:, 4:6].norm(dim=-1).sum())
        self.sums["action_rate"] += float((a - prev).square().sum())
        feet = feet_world(s1, p1, field)
        vf = (feet - feet_world(s0, p0, field)) / dt
        stable_contact = contact & self.prev_contact & self.prev_valid[:, None]
        self.sums["slip_sq"] += float((vf[..., :2].square().sum(-1) * stable_contact).sum())
        self.sums["slip_count"] += int(stable_contact.sum())
        landed = contact & ~self.prev_contact & self.prev_valid[:, None]
        measured = landed & self.has_touch
        self.stride_sum += ((feet - self.last_touch)[..., :2].norm(dim=-1) * measured).sum(0)
        self.stride_count += measured.sum(0)
        self.touchdowns += landed.sum(0)
        self.last_touch = torch.where(landed[..., None], feet, self.last_touch)
        self.has_touch |= landed
        self.contact_time += contact.sum(0) * dt
        clear = foot_clearances(s1, p1, field).clamp(min=0)
        self.clear_sum += (clear * ~contact).sum(0)
        self.swing_count += (~contact).sum(0)
        self.max_force = max(self.max_force, float(force.norm(dim=-1).max()))
        self.prev_contact = contact.clone()
        self.prev_valid[:] = True
        self.prev_valid[done] = False
        self.prev_contact[done] = False
        self.has_touch[done] = False

    def result(self):
        total = max(self.samples, 1)
        return {
            "falls": self.falls,
            "falls_per_robot_minute": self.falls / max(self.n * self.elapsed / 60, 1e-9),
            "completed_episodes": self.completed,
            "nonfall_completion_fraction": self.survived / self.completed if self.completed else None,
            "linear_rmse_mps": (self.sums["linear_sq"] / total)**0.5,
            "yaw_rmse_radps": (self.sums["yaw_sq"] / total)**0.5,
            "mean_speed_mps": self.sums["speed"] / total,
            "action_change_l2": self.sums["action_rate"] / total,
            "stance_slip_rms_mps": (self.sums["slip_sq"] / max(self.sums["slip_count"], 1))**0.5,
            "max_sampled_foot_force_N": self.max_force,
            "leg_order": ["FR", "FL", "HR", "HL"],
            "duty_factor": (self.contact_time / max(self.n * self.elapsed, 1e-9)).tolist(),
            "touchdowns_per_second": (self.touchdowns / max(self.n * self.elapsed, 1e-9)).tolist(),
            "mean_stride_m": (self.stride_sum / self.stride_count.clamp(min=1)).tolist(),
            "measured_strides": self.stride_count.tolist(),
            "mean_swing_clearance_m": (self.clear_sum / self.swing_count.clamp(min=1)).tolist(),
            "seconds_per_env": self.elapsed,
        }
