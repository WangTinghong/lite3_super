"""Small evaluation-only joint-target interventions; not a learned controller."""
import torch
from lite3_supermimic.target import terrain as tr
from .core import feet_world, foot_clearances


def front_leg_probe(action, state, pose, command, field, foot_contact,
                    action_scale, hipy_delta=0.0, knee_delta=0.0):
    gate = torch.zeros(len(action), device=action.device)
    if field is None or (hipy_delta == 0 and knee_delta == 0):
        return action, gate
    foot = feet_world(state, pose, field)[:, 0, :2]  # FR in state/contact order
    heading = torch.stack((pose[:, 2].cos(), pose[:, 2].sin()), -1)
    rise = torch.maximum(tr.ground(foot + .12 * heading, field),
                         tr.ground(foot + .25 * heading, field)) - tr.ground(foot, field)
    gate = ((rise - .04) / .04).clamp(0, 1)
    gate *= ((foot_clearances(state, pose, field)[:, 0] - .005) / .03).clamp(0, 1)
    gate *= ~foot_contact[:, 0]
    gate *= ((command[:, 0] > .1) & (command[:, 1].abs() < .05)
             & (command[:, 2].abs() < .05))
    adjusted = action.clone()
    scale = torch.broadcast_to(action_scale, action.shape)
    # Native action order: FL, FR, HL, HR; each HipX, HipY, Knee.
    adjusted[:, 4] += gate * hipy_delta / scale[:, 4]
    adjusted[:, 5] += gate * knee_delta / scale[:, 5]
    return adjusted, gate
