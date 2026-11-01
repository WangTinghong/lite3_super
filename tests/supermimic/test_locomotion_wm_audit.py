"""World-model audit measures prediction error without changing the policy."""

import pytest
import torch

from lite3_supermimic.locomotion.wm_audit import WorldModelAudit
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.target.terrain import HeightField


class FrontFootBiasModel:
    def __call__(self, state, action, previous, context, relief):
        next_state = state.clone()
        next_state[:, st.FEET].reshape(-1, 4, 3)[:, :2, 2] += 0.01
        return next_state, torch.zeros(len(state), 3)


def test_audit_reports_near_riser_and_open_loop_foot_error():
    x = torch.linspace(-1, 1, 101)
    field = HeightField((x >= 0.2).float().mul(0.18).expand(101, 101).clone(),
                        (-1., -1.), 0.02)
    state = torch.zeros(2, st.STATE_DIM)
    state[:, st.Z] = 0.28
    state[:, st.FEET] = torch.tensor([0.2, -0.1, -0.258, 0.2, 0.1, -0.258,
                                     -0.2, -0.1, -0.258, -0.2, 0.1, -0.258])
    pose = torch.tensor([[-0.15, 0., 0.], [-0.8, 0., 0.]])
    zeros = torch.zeros(2, 12)
    contacts = torch.zeros(2, 4, dtype=torch.bool)
    shanks = contacts.clone()
    audit = WorldModelAudit(FrontFootBiasModel(), 2, "cpu")
    assert audit.near_riser(state, pose, field).tolist() == [True, False]

    for step in range(5):
        done = torch.tensor([False, step == 2])
        shanks[0, 0] = step == 1
        audit.update(state, pose, state, pose, zeros, zeros, torch.zeros(2, 4),
                     field, done, contacts, shanks)
    result = audit.result()
    assert result["one_step_near_riser"]["samples"] == 5
    assert result["one_step_elsewhere"]["samples"] == 4
    assert result["one_step_front_shank_only"]["samples"] == 1
    assert result["five_step_near_riser"]["samples"] == 1
    assert "five_step_elsewhere" not in result
    assert result["one_step_near_riser"]["front_foot_z_abs_m"] == pytest.approx(0.01, abs=1e-6)
    assert result["five_step_near_riser"]["front_foot_z_abs_m"] == pytest.approx(0.05, abs=1e-6)
