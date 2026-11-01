"""Physical-rollout metrics; contact data comes from the simulator, not the WM."""
from __future__ import annotations

import torch
from .core import feet_world, foot_clearances
from lite3_supermimic.target import terrain as tr


def _read_contacts(sensor, geom_names):
    """Reorder mjlab's XML sensor order into the explicit FR, FL, HR, HL order."""
    names = [name.split("/")[-1] for name in sensor.primary_names]
    order = [names.index(name) for name in geom_names]
    n = sensor.data.found.shape[0]
    found = sensor.data.found.reshape(n, len(names), -1).any(-1)[:, order]
    forces = sensor.data.force.reshape(n, len(names), -1, 3).sum(2)[:, order]
    return found, forces


def read_contacts(sensor):
    from lite3_supermimic.robot.constants import FOOT_GEOM_NAMES
    return _read_contacts(sensor, FOOT_GEOM_NAMES)


def read_shank_contacts(sensor):
    from lite3_supermimic.robot.constants import SHANK_GEOM_NAMES
    return _read_contacts(sensor, SHANK_GEOM_NAMES)


class Metrics:
    def __init__(self, n, device):
        self.n, self.samples, self.elapsed = n, 0, 0.0
        self.falls, self.completed, self.survived = 0, 0, 0
        self.sums = {k: 0.0 for k in ("linear_sq", "yaw_sq", "speed", "forward_vx", "lateral_vy",
                                     "action_rate", "slip_sq", "slip_count")}
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
        self.shank_contact_time = torch.zeros(4, device=device)
        self.shank_without_foot_time = torch.zeros(4, device=device)
        self.shank_contact_events = torch.zeros(4, device=device)
        self.prev_shank_contact = torch.zeros(n, 4, dtype=torch.bool, device=device)
        self.max_shank_force = 0.0
        self.start_pose = torch.zeros(n, 3, device=device)
        self.start_ground = torch.zeros(n, device=device)
        self.max_ground_gain = torch.zeros(n, device=device)
        self.max_forward_progress = torch.zeros(n, device=device)
        self.episode_ground_gain_sum = 0.0
        self.episode_forward_progress_sum = 0.0
        self.hipx_pos_sum = torch.zeros(4, device=device)
        self.hipx_pos_near_limit = torch.zeros(4, device=device)
        self.hipx_target_over_limit = torch.zeros(4, device=device)
        self.hipx_samples = 0
        self.forward_command_samples = 0
        self.stall_samples = 0

    def update_joint(self, joint_pos, joint_target):
        """Joint order FL, FR, HL, HR; report FR, FL, HR, HL."""
        signs = joint_pos.new_tensor([-1, 1, -1, 1])
        outward = (joint_pos[:, 0::3] * signs)[:, [1, 0, 3, 2]]
        target = (joint_target[:, 0::3] * signs)[:, [1, 0, 3, 2]]
        self.hipx_pos_sum += outward.sum(0)
        self.hipx_pos_near_limit += (outward > 0.50).sum(0)
        self.hipx_target_over_limit += (target > 0.523).sum(0)
        self.hipx_samples += len(joint_pos)

    def update(self, s0, p0, s1, p1, cmd, a, prev, contact, force, terminated, truncated, dt,
               field=None, shank_contact=None, shank_force=None):
        new_episode = ~self.prev_valid
        self.start_pose[new_episode] = p0[new_episode]
        ground0 = tr.ground(p0[:, :2], field)
        self.start_ground[new_episode] = ground0[new_episode]
        self.max_ground_gain[new_episode] = 0
        self.max_forward_progress[new_episode] = 0
        gain = (tr.ground(p1[:, :2], field) - self.start_ground).clamp(min=0)
        self.max_ground_gain = torch.maximum(self.max_ground_gain, gain)
        dx = p1[:, :2] - self.start_pose[:, :2]
        heading = self.start_pose[:, 2]
        forward = dx[:, 0] * heading.cos() + dx[:, 1] * heading.sin()
        self.max_forward_progress = torch.maximum(self.max_forward_progress, forward)
        done = terminated | truncated
        self.episode_ground_gain_sum += float(self.max_ground_gain[done].sum())
        self.episode_forward_progress_sum += float(self.max_forward_progress[done].sum())
        forward_command = cmd[:, 0] >= 0.25
        self.forward_command_samples += int(forward_command.sum())
        self.stall_samples += int((forward_command & (s1[:, 4] < 0.1)).sum())
        self.samples += self.n
        self.elapsed += dt
        self.falls += int(terminated.sum())
        self.completed += int(done.sum())
        self.survived += int((truncated & ~terminated).sum())
        self.sums["linear_sq"] += float((s1[:, 4:6] - cmd[:, :2]).square().sum())
        self.sums["yaw_sq"] += float((s1[:, 9] - cmd[:, 2]).square().sum())
        self.sums["speed"] += float(s1[:, 4:6].norm(dim=-1).sum())
        self.sums["forward_vx"] += float(s1[:, 4].sum())
        self.sums["lateral_vy"] += float(s1[:, 5].sum())
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
        if shank_contact is not None:
            self.shank_contact_time += shank_contact.sum(0) * dt
            self.shank_without_foot_time += (shank_contact & ~contact).sum(0) * dt
            self.shank_contact_events += (shank_contact & ~self.prev_shank_contact
                                          & self.prev_valid[:, None]).sum(0)
            self.prev_shank_contact = shank_contact.clone()
            if shank_force is not None:
                self.max_shank_force = max(self.max_shank_force,
                                           float(shank_force.norm(dim=-1).max()))
        self.prev_contact = contact.clone()
        self.prev_valid[:] = True
        self.prev_valid[done] = False
        self.prev_contact[done] = False
        self.has_touch[done] = False
        self.prev_shank_contact[done] = False

    def result(self):
        total = max(self.samples, 1)
        return {
            "falls": self.falls,
            "falls_per_robot_minute": self.falls / max(self.n * self.elapsed / 60, 1e-9),
            "completed_episodes": self.completed,
            "nonfall_completion_fraction": self.survived / self.completed if self.completed else None,
            "mean_episode_max_ground_gain_m": self.episode_ground_gain_sum / self.completed if self.completed else None,
            "mean_episode_max_forward_progress_m": self.episode_forward_progress_sum / self.completed if self.completed else None,
            "forward_command_stall_fraction": self.stall_samples / self.forward_command_samples if self.forward_command_samples else None,
            "hipx_outward_mean_rad": (self.hipx_pos_sum / max(self.hipx_samples, 1)).tolist(),
            "hipx_pos_over_0p5_fraction": (self.hipx_pos_near_limit / max(self.hipx_samples, 1)).tolist(),
            "hipx_target_over_limit_fraction": (self.hipx_target_over_limit / max(self.hipx_samples, 1)).tolist(),
            "linear_rmse_mps": (self.sums["linear_sq"] / total)**0.5,
            "yaw_rmse_radps": (self.sums["yaw_sq"] / total)**0.5,
            "mean_speed_mps": self.sums["speed"] / total,
            "mean_body_forward_vx_mps": self.sums["forward_vx"] / total,
            "mean_body_lateral_vy_mps": self.sums["lateral_vy"] / total,
            "action_change_l2": self.sums["action_rate"] / total,
            "stance_slip_rms_mps": (self.sums["slip_sq"] / max(self.sums["slip_count"], 1))**0.5,
            "max_sampled_foot_force_N": self.max_force,
            "shank_contact_fraction": (self.shank_contact_time / max(self.n * self.elapsed, 1e-9)).tolist(),
            "shank_without_foot_fraction": (self.shank_without_foot_time / max(self.n * self.elapsed, 1e-9)).tolist(),
            "shank_contacts_per_second": (self.shank_contact_events / max(self.n * self.elapsed, 1e-9)).tolist(),
            "max_sampled_shank_force_N": self.max_shank_force,
            "leg_order": ["FR", "FL", "HR", "HL"],
            "duty_factor": (self.contact_time / max(self.n * self.elapsed, 1e-9)).tolist(),
            "touchdowns_per_second": (self.touchdowns / max(self.n * self.elapsed, 1e-9)).tolist(),
            "mean_stride_m": (self.stride_sum / self.stride_count.clamp(min=1)).tolist(),
            "measured_strides": self.stride_count.tolist(),
            "mean_swing_clearance_m": (self.clear_sum / self.swing_count.clamp(min=1)).tolist(),
            "seconds_per_env": self.elapsed,
        }
