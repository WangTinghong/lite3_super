"""A stair pilot must leave ten of twelve actions exactly unchanged."""
import torch
from torch import nn
import pytest

from lite3_supermimic.locomotion.stair_repair import (
    assert_restricted_actor, repair_penalty, restrict_actor_outputs,
)
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.target.terrain import HeightField


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 8), nn.ELU(), nn.Linear(8, 12))

    def forward(self, x):
        return self.net(x)


def test_only_selected_output_rows_move_after_adam_updates():
    torch.manual_seed(3)
    actor = TinyActor()
    obs = torch.randn(32, 5)
    before = actor(obs).detach().clone()
    initial = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    restrict_actor_outputs(actor)
    for _ in range(10):
        loss = -actor(obs).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    after = actor(obs).detach()
    unchanged = [i for i in range(12) if i not in (4, 9)]
    torch.testing.assert_close(after[:, unchanged], before[:, unchanged], rtol=0, atol=0)
    assert (after[:, [4, 9]] - before[:, [4, 9]]).abs().max() > 0.01
    torch.testing.assert_close(actor.net[-1].weight[unchanged],
                               initial["net.2.weight"][unchanged], rtol=0, atol=0)
    torch.testing.assert_close(actor.net[-1].bias[unchanged],
                               initial["net.2.bias"][unchanged], rtol=0, atol=0)


def test_checkpoint_guard_rejects_a_frozen_row_or_normalizer_change():
    actor, source = TinyActor(), TinyActor()
    source.load_state_dict(actor.state_dict())
    with torch.no_grad():
        actor.net[-1].bias[4] += 0.1
        actor.net[-1].bias[9] -= 0.1
    assert_restricted_actor(actor, source)
    with torch.no_grad():
        actor.net[-1].bias[0] += 0.01
    try:
        assert_restricted_actor(actor, source)
    except RuntimeError as exc:
        assert "frozen actor" in str(exc)
    else:
        raise AssertionError("Guard accepted an altered frozen action")


def test_real_state_repair_loss_moves_fr_hipy_and_hr_extreme_inward():
    x = torch.linspace(-1, 1, 101)
    field = HeightField((x >= 0.2).float().mul(0.18).expand(101, 101).clone(),
                        (-1., -1.), 0.02)
    state = torch.zeros(1, st.STATE_DIM)
    state[:, st.Z] = 0.28
    state[:, st.FEET] = torch.tensor([0.2, -0.1, -0.2, 0.2, 0.1, -0.2,
                                     -0.2, -0.1, -0.2, -0.2, 0.1, -0.2])
    pose = torch.tensor([[-0.15, 0.0, 0.0]])
    command = torch.tensor([[1.0, 0.0, 0.0]])
    reference = torch.zeros(1, 12)
    actions = reference.clone()
    actions[:, 9] = 2.4
    actions.requires_grad_()
    scale = torch.full((1, 12), 0.25)
    loss, stats = repair_penalty(actions, reference, state, pose, command, field,
                                 scale, torch.zeros_like(scale))
    assert stats["repair_fr_gate_fraction"] == 1.0
    assert stats["repair_hr_excess_mse"] > 0
    loss.backward()
    assert actions.grad[0, 4] > 0  # Gradient descent moves FR HipY negative.
    assert actions.grad[0, 9] > 0  # Gradient descent reduces excessive HR HipX.
    assert torch.count_nonzero(actions.grad[:, [0, 1, 2, 3, 5, 6, 7, 8, 10, 11]]) == 0


def test_repair_requires_source_and_policy_anchor():
    from lite3_supermimic.locomotion.train import TrainCfg
    with pytest.raises(ValueError, match="stair-repair"):
        TrainCfg(stage="terrain", harder_stairs=True, stair_repair=True,
                 policy_anchor_weight=10.0)
    with pytest.raises(ValueError, match="stair-repair"):
        TrainCfg(stage="terrain", harder_stairs=True, stair_repair=True,
                 resume="source.pt")
    cfg = TrainCfg(stage="terrain", harder_stairs=True, stair_repair=True,
                   resume="source.pt", policy_anchor_weight=10.0)
    assert cfg.stair_repair
