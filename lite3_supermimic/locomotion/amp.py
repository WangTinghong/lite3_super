"""Optional adversarial gait prior from Lite3 forward-motion recordings.

The discriminator sees joint positions and velocities at two consecutive
control steps. Task rewards still determine speed, balance and terrain safety.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from lite3_supermimic.supermimic import state as st


MOTION_RATES = {"lite3_直行慢速.json": (1.0,),
                "lite3_直行快速.json": (1.0, 1.15, 1.25)}
FEATURE_DIM = 24


def flatness_gate(relief: torch.Tensor) -> torch.Tensor:
    """Use flat demonstrations only where the local scan is nearly level."""
    spread = relief.amax(-1) - relief.amin(-1)
    return ((0.04 - spread) / 0.03).clamp(0, 1)


def _interpolate(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    lo = indices.floor().long()
    alpha = (indices - lo).unsqueeze(-1)
    return values[lo] * (1 - alpha) + values[lo + 1] * alpha


def load_forward_motions(directory: str | Path, dt: float, device: str) -> torch.Tensor:
    """Return expert transition pairs [N, 2, 24] in Lite3 joint order.

    Recorded joint axes are opposite to the Lite3 URDF/MJCF. Fast clips are
    also played up to 1.25x to cover roughly 1.0 m/s without a 1.0 m/s clip.
    """
    pairs = []
    for name, rates in MOTION_RATES.items():
        path = Path(directory).expanduser() / name
        with path.open(encoding="utf-8") as stream:
            motion = json.load(stream)
        frames = torch.as_tensor(motion["Frames"], dtype=torch.float32)
        if frames.ndim != 2 or frames.shape[1] != 61 or len(frames) < 3:
            raise ValueError(f"{path}: expected at least three 61-field frames")
        if not bool(torch.isfinite(frames).all()):
            raise ValueError(f"{path}: non-finite motion frame")
        frame_dt = float(motion["FrameDuration"])
        if abs(frame_dt - dt) > 1e-6:
            raise ValueError(f"{path}: frame duration {frame_dt} differs from control dt {dt}")
        q, qd = -frames[:, 7:19], -frames[:, 37:49]
        for rate in rates:
            t = torch.arange(0, len(frames) - 1 - rate, rate)
            if len(t) == 0:
                raise ValueError(f"{path}: too short for {rate}x playback")
            start = torch.cat((_interpolate(q, t), rate * _interpolate(qd, t)), -1)
            end = torch.cat((_interpolate(q, t + rate), rate * _interpolate(qd, t + rate)), -1)
            pairs.append(torch.stack((start, end), dim=1))
    return torch.cat(pairs).to(device)


def amp_features(s: torch.Tensor, default_q: torch.Tensor) -> torch.Tensor:
    """Same joint coordinates as the validated expert clips: FL/FR/HL/HR."""
    return torch.cat((s[:, st.Q] + default_q, s[:, st.QD]), -1)


class AMPStyle(nn.Module):
    def __init__(self, motion_dir: str, dt: float, device: str, lr: float = 1e-4):
        super().__init__()
        self.expert = load_forward_motions(motion_dir, dt, device)
        flat = self.expert.reshape(-1, FEATURE_DIM)
        self.register_buffer("mean", flat.mean(0))
        self.register_buffer("scale", flat.std(0, unbiased=False).clamp(min=0.1))
        self.discriminator = nn.Sequential(
            nn.Linear(2 * FEATURE_DIM, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1),
        ).to(device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=1e-4)

    def _features(self, pair: torch.Tensor) -> torch.Tensor:
        return ((pair - self.mean) / self.scale).clamp(-10, 10).flatten(1)

    def score(self, pair: torch.Tensor) -> torch.Tensor:
        return self.discriminator(self._features(pair)).squeeze(-1)

    def reward(self, s0: torch.Tensor, s1: torch.Tensor, default_q: torch.Tensor) -> torch.Tensor:
        pair = torch.stack((amp_features(s0, default_q), amp_features(s1, default_q)), 1)
        return torch.sigmoid(self.score(pair))

    def fit(self, replay, default_q: torch.Tensor, batch: int, updates: int) -> dict[str, float]:
        """Train on real simulator transitions only; skip resets and non-forward commands."""
        if updates == 0:
            return {}
        for p in self.discriminator.parameters():
            p.requires_grad_(True)
        losses, policy_scores, expert_scores = [], [], []
        for _ in range(updates):
            env, idx, valid = replay.sample_windows(4 * batch, 1)
            command = replay.gen[env, idx[:, 0], :3]
            keep = ((valid[:, 0] > 0.5) & (command[:, 0] >= 0.4)
                    & (command[:, 1].abs() <= 0.15) & (command[:, 2].abs() <= 0.3)
                    & (flatness_gate(replay.relief[env, idx[:, 0]].float()) > 0.5))
            selected = keep.nonzero().flatten()
            if len(selected) == 0:
                continue
            selected = selected[torch.randint(len(selected), (batch,), device=selected.device)]
            ids, slots = env[selected], idx[selected]
            q0 = amp_features(replay.s[ids, slots[:, 0]], default_q[ids])
            q1 = amp_features(replay.s[ids, slots[:, 1]], default_q[ids])
            policy = torch.stack((q0, q1), 1).detach()
            expert = self.expert[torch.randint(len(self.expert), (batch,), device=policy.device)]
            pred_p, pred_e = self.score(policy), self.score(expert)
            adversarial = 0.5 * ((pred_p + 1).square().mean() + (pred_e - 1).square().mean())
            expert_input = self._features(expert).detach().requires_grad_(True)
            gp_pred = self.discriminator(expert_input)
            gradient = torch.autograd.grad(gp_pred.sum(), expert_input, create_graph=True)[0]
            loss = adversarial + 10.0 * gradient.square().sum(-1).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(adversarial.detach()))
            policy_scores.append(float(pred_p.detach().mean()))
            expert_scores.append(float(pred_e.detach().mean()))
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        if not losses:
            return {"amp_batches": 0.0}
        return {"amp_d_loss": sum(losses) / len(losses),
                "amp_d_policy": sum(policy_scores) / len(policy_scores),
                "amp_d_expert": sum(expert_scores) / len(expert_scores),
                "amp_batches": float(len(losses))}
