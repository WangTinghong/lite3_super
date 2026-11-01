"""Velocity-command BPTT rollouts using the same observation and reward functions as sim."""
from __future__ import annotations

import torch

from lite3_supermimic.supermimic import state as st
from lite3_supermimic.supermimic.imagine import Imagination as BaseImagination
from lite3_supermimic.target import terrain as tr
from .amp import flatness_gate
from .core import TERMS, advance_commands, observation, reward_terms


class Imagination(BaseImagination):
    def __init__(self, cfg, consts, stage, history, device, field=None, amp=None):
        self.cfg, self.c, self.stage = cfg, consts, stage
        self.h, self.device, self.field = history, device, field
        self.amp, self.amp_scale, self.last_amp_reward = amp, 0.0, 0.0
        self.stair_refine = cfg.weights.stair_foot_clearance != 0.0
        self.stair_hipx_refine = cfg.weights.stair_hipx_target_outward != 0.0
        self.stair_foothold_refine = cfg.weights.stair_foothold != 0.0
        # Include command timer and speed ceiling: both affect future commands.
        self.critic_dim = st.STATE_DIM + 5 + st.CONTEXT_DIM + tr.SCAN_POINTS
        self._step_fn = torch.compile(self._step, dynamic=False) if cfg.compile else self._step

    def fall_prob(self, s):
        tilted = super().fall_prob(s)
        low = torch.sigmoid((0.12 - s[:, 0]) / 0.01)
        return 1 - (1 - tilted) * (1 - low)

    def scan(self, s, pose, grad=False):
        return s[:, st.Z] + tr.relief(pose[:, :2], pose[:, 2], self.field, grad=grad)

    def critic_input(self, s, pose, commands, xi):
        return torch.cat([s, commands, xi, self.scan(s, pose)], -1)

    def _step(self, s, pose, prev, hist, commands, xi, bias, c, actor, wm):
        cfg = self.cfg
        x = self.critic_input(s, pose, commands, xi)
        a = actor(hist)
        relief = tr.relief(pose[:, :2], pose[:, 2], self.field, grad=cfg.sensor_grad)
        next_s, dp = wm(s, a, prev, xi, relief)
        next_pose = st.apply_pose_delta(pose, dp)
        terms = reward_terms(s, next_s, pose, next_pose, a, prev, commands[:, :3],
                             c, cfg.dt, self.field, stair_refine=self.stair_refine,
                             stair_hipx_refine=self.stair_hipx_refine,
                             stair_foothold_refine=self.stair_foothold_refine)
        d = self.fall_prob(next_s)
        r = (sum(getattr(cfg.weights, name) * v for name, v in terms.items())
             + cfg.weights.terminated * d) * cfg.dt
        # Like mjlab: reward uses the old command; resampling precedes the next observation.
        commands = advance_commands(commands, cfg.dt, self.stage)
        frame = observation(next_s, a, commands[:, :3], bias,
                            self.scan(next_s, next_pose, cfg.sensor_grad), noise=True)
        hist = st.push_history(hist, frame, self.h, TERMS)
        return next_s, next_pose, a, hist, commands, r, d, x

    def rollout(self, wm, actor, critic, seed, hint_weight=0.0):
        del hint_weight  # Only used by chase/dodge; inherited optimiser interface.
        cfg, ids = self.cfg, seed["env"]
        c = st.RobotConsts(default_q=self.c.default_q[ids], action_scale=self.c.action_scale[ids],
                           soft_limits=self.c.soft_limits[ids], kp=self.c.kp,
                           kd=self.c.kd, effort=self.c.effort)
        s, pose, prev, hist = seed["s"], seed["pose"], seed["a_prev"], seed["obs"]
        commands, xi, bias = seed["gen"], seed["xi"], seed["enc_bias"]
        survival = torch.ones(len(s), device=s.device)
        ret = torch.zeros_like(survival)
        rewards, survivals, critic_inputs = [], [], []
        amp_rewards = []
        discount = 1.0
        for _ in range(cfg.horizon):
            survivals.append(survival)
            previous_s, command = s, commands[:, :3]
            s, pose, prev, hist, commands, r, d, x = self._step_fn(
                s, pose, prev, hist, commands, xi, bias, c, actor, wm)
            if self.amp is not None and self.amp_scale > 0:
                gate = ((command[:, 0] - 0.3) / 0.2).clamp(0, 1)
                gate = gate * (1 - command[:, 1].abs() / 0.15).clamp(0, 1)
                gate = gate * (1 - command[:, 2].abs() / 0.3).clamp(0, 1)
                gate = gate * flatness_gate(x[:, -tr.SCAN_POINTS:]).detach()
                style = self.amp.reward(previous_s, s, c.default_q) * gate
                r = r + self.amp_scale * style * cfg.dt
                amp_rewards.append(style.detach())
            rewards.append(r)
            critic_inputs.append(x)
            ret = ret + discount * survival * r
            survival = survival * (1 - d)
            discount *= cfg.gamma
        self.last_amp_reward = (float(torch.stack(amp_rewards).mean()) if amp_rewards else 0.0)
        last = self.critic_input(s, pose, commands, xi)
        ret = ret + discount * survival * critic(last).squeeze(-1)
        return ret, {"rewards": torch.stack(rewards, 1),
                     "surv": torch.stack(survivals + [survival], 1),
                     "critic_in": torch.stack(critic_inputs + [last], 1)}
