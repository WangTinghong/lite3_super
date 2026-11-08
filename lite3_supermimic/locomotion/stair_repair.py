"""Constrained stair-action refinement from a known working policy.

Only the FR HipY and HR HipX output rows may change. The FR target comes from
the measured beneficial stair probe; the HR term is a soft limit, not a cap.
"""
from __future__ import annotations

import torch
from torch import nn

FR_HIPY = 4
HR_HIPX = 9


def restrict_actor_outputs(actor: nn.Module) -> None:
    """Freeze the feature extractor and mask every output row except 4 and 9.

    The caller must also stop updating the actor's observation normalizer.
    This is installed after loading a checkpoint and after clearing old Adam
    moments for a new pilot, otherwise zero gradients could still move rows.
    """
    for param in actor.parameters():
        param.requires_grad_(False)
    final = actor.net[-1]
    if not isinstance(final, nn.Linear) or final.out_features != 12:
        raise ValueError("Expected a 12-action final linear layer")
    final.weight.requires_grad_(True)
    final.bias.requires_grad_(True)
    row_mask = torch.zeros_like(final.bias)
    row_mask[[FR_HIPY, HR_HIPX]] = 1
    final.weight.register_hook(lambda grad: grad * row_mask[:, None])
    final.bias.register_hook(lambda grad: grad * row_mask)


@torch.no_grad()
def assert_restricted_actor(actor: nn.Module, source: nn.Module) -> None:
    """Refuse to save a repair pilot if any of the other ten actions can drift."""
    active = {FR_HIPY, HR_HIPX}
    unchanged = [i for i in range(12) if i not in active]
    actor_state, source_state = actor.state_dict(), source.state_dict()
    if actor_state.keys() != source_state.keys():
        raise RuntimeError("Source actor parameter names changed")
    for (name, value), (source_name, expected) in zip(
            actor_state.items(), source_state.items()):
        if name != source_name:
            raise RuntimeError("Source actor parameter names changed")
        if name.endswith(".weight") and value.shape[:1] == (12,) and name.startswith("net."):
            value, expected = value[unchanged], expected[unchanged]
        elif name.endswith(".bias") and value.shape == (12,) and name.startswith("net."):
            value, expected = value[unchanged], expected[unchanged]
        if not torch.equal(value, expected):
            raise RuntimeError(f"Stair repair changed frozen actor tensor: {name}")


def repair_penalty(actions: torch.Tensor, reference: torch.Tensor,
                   state: torch.Tensor, pose: torch.Tensor, command: torch.Tensor,
                   field, action_scale: torch.Tensor, default_q: torch.Tensor,
                   fr_delta: float = -0.1, fr_weight: float = 30.0,
                   hr_weight: float = 2.0, hr_limit: float = 0.5):
    """Real-state actor loss; labels never come from an imagined contact head."""
    from .core import foot_clearances
    from .stair_probe import front_leg_probe
    with torch.no_grad():
        contact = foot_clearances(state, pose, field) <= 0.015
        teacher, gate = front_leg_probe(reference, state, pose, command, field,
                                        contact, action_scale, hipy_delta=fr_delta)
    fr_loss = (actions[:, FR_HIPY] - teacher[:, FR_HIPY]).square().mean()
    scale = torch.broadcast_to(action_scale, actions.shape)
    default = torch.broadcast_to(default_q, actions.shape)
    hr_target = default[:, HR_HIPX] + actions[:, HR_HIPX] * scale[:, HR_HIPX]
    forward = ((command[:, 0] > 0.1) & (command[:, 1].abs() < 0.05)
               & (command[:, 2].abs() < 0.05))
    hr_swing = forward & ~contact[:, 2]
    excess = (hr_target - hr_limit).clamp(min=0)
    hr_loss = (excess.square() * hr_swing).mean()
    stats = {"repair_fr_teacher_mse": float(fr_loss.detach()),
             "repair_fr_gate_fraction": float((gate > 0).float().mean()),
             "repair_hr_excess_mse": float(hr_loss.detach()),
             "repair_hr_swing_fraction": float(hr_swing.float().mean())}
    return fr_weight * fr_loss + hr_weight * hr_loss, stats
