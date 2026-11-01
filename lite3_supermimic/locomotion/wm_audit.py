"""Read-only world-model prediction errors during real locomotion rollouts."""
from __future__ import annotations

import torch

from lite3_supermimic.supermimic import state as st
from lite3_supermimic.target import terrain as tr
from .core import feet_world, foot_clearances


class WorldModelAudit:
    def __init__(self, model, num_envs: int, device, horizon: int = 5):
        self.model, self.horizon = model, horizon
        self.age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.roll_s = self.roll_pose = None
        self.start_near = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.stats: dict[str, dict[str, float]] = {}

    @staticmethod
    def near_riser(s, pose, field):
        if field is None:
            return torch.zeros(len(s), dtype=torch.bool, device=s.device)
        feet = feet_world(s, pose, field)[:, :2, :2]
        heading = torch.stack((pose[:, 2].cos(), pose[:, 2].sin()), -1)
        here = tr.ground(feet, field)
        ahead = torch.maximum(tr.ground(feet + 0.12 * heading[:, None], field),
                              tr.ground(feet + 0.25 * heading[:, None], field))
        return ((ahead - here).amax(-1) > 0.04)

    @staticmethod
    def errors(pred_s, pred_pose, real_s, real_pose, field):
        pred_foot_z = pred_s[:, st.FEET].reshape(-1, 4, 3)[:, :2, 2]
        real_foot_z = real_s[:, st.FEET].reshape(-1, 4, 3)[:, :2, 2]
        pred_knee_z = pred_s[:, st.KNEES].reshape(-1, 4, 3)[:, :2, 2]
        real_knee_z = real_s[:, st.KNEES].reshape(-1, 4, 3)[:, :2, 2]
        pred_foot_xy = feet_world(pred_s, pred_pose, field)[:, :2, :2]
        real_foot_xy = feet_world(real_s, real_pose, field)[:, :2, :2]
        pred_clear = foot_clearances(pred_s, pred_pose, field)[:, :2]
        real_clear = foot_clearances(real_s, real_pose, field)[:, :2]
        return {
            "base_vx_abs_mps": (pred_s[:, 4] - real_s[:, 4]).abs(),
            "yaw_rate_abs_radps": (pred_s[:, 9] - real_s[:, 9]).abs(),
            "base_xy_abs_m": (pred_pose[:, :2] - real_pose[:, :2]).norm(dim=-1),
            "front_foot_xy_abs_m": (pred_foot_xy - real_foot_xy).norm(dim=-1).mean(-1),
            "front_foot_z_abs_m": (pred_foot_z - real_foot_z).abs().mean(-1),
            "front_knee_z_abs_m": (pred_knee_z - real_knee_z).abs().mean(-1),
            "front_clearance_abs_m": (pred_clear - real_clear).abs().mean(-1),
        }

    def _add(self, name, mask, errors):
        count = int(mask.sum())
        if count == 0:
            return
        sums = self.stats.setdefault(name, {"samples": 0})
        sums["samples"] += count
        for key, values in errors.items():
            sums[key] = sums.get(key, 0.0) + float(values[mask].sum())

    @torch.no_grad()
    def update(self, s0, pose0, s1, pose1, action, prev_action, context, field,
               done, foot_contact, shank_contact):
        valid = ~done
        near = self.near_riser(s0, pose0, field)
        relief = tr.relief(pose0[:, :2], pose0[:, 2], field)
        predicted_s, delta = self.model(s0, action, prev_action, context, relief)
        predicted_pose = st.apply_pose_delta(pose0, delta)
        errors = self.errors(predicted_s, predicted_pose, s1, pose1, field)
        self._add("one_step_all", valid, errors)
        self._add("one_step_near_riser", valid & near, errors)
        self._add("one_step_elsewhere", valid & ~near, errors)
        shank_only = (shank_contact[:, :2] & ~foot_contact[:, :2]).any(-1)
        self._add("one_step_front_shank_only", valid & shank_only, errors)

        restart = self.age == 0
        if self.roll_s is None:
            self.roll_s, self.roll_pose = s0, pose0
        else:
            self.roll_s = torch.where(restart[:, None], s0, self.roll_s)
            self.roll_pose = torch.where(restart[:, None], pose0, self.roll_pose)
        self.start_near = torch.where(restart, near, self.start_near)
        roll_relief = tr.relief(self.roll_pose[:, :2], self.roll_pose[:, 2], field)
        self.roll_s, roll_delta = self.model(
            self.roll_s, action, prev_action, context, roll_relief)
        self.roll_pose = st.apply_pose_delta(self.roll_pose, roll_delta)
        next_age = self.age + 1
        complete = valid & (next_age == self.horizon)
        if bool(complete.any()):
            long_errors = self.errors(self.roll_s, self.roll_pose, s1, pose1, field)
            self._add("five_step_all", complete, long_errors)
            self._add("five_step_near_riser", complete & self.start_near, long_errors)
            self._add("five_step_elsewhere", complete & ~self.start_near, long_errors)
        self.age = torch.where(done | complete, 0, next_age)

    def result(self):
        return {group: {"samples": values["samples"], **{
            key: value / values["samples"] for key, value in values.items() if key != "samples"
        }} for group, values in self.stats.items()}
