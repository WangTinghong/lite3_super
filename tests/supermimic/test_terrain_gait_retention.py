"""AMP gait retention uses level real transitions without constraining obstacles."""
from types import SimpleNamespace

import pytest
import torch

from lite3_supermimic.locomotion.amp import MOTION_RATES, flatness_gate
from lite3_supermimic.locomotion.train import TrainCfg, Trainer
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.supermimic.buffer import ReplayBuffer


def test_flatness_gate_and_terrain_configuration(tmp_path):
    relief = torch.zeros(2, 77)
    relief[1, 0] = 0.1
    assert torch.equal(flatness_gate(relief), torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError, match="gait-anchor-weight"):
        TrainCfg(stage="terrain", gait_anchor_weight=0.2)
    for name in MOTION_RATES:
        (tmp_path / name).write_text("{}")
    cfg = TrainCfg(stage="terrain", amp_motion_dir=str(tmp_path), resume="source.pt",
                   gait_anchor_weight=0.2)
    assert cfg.stage == "terrain" and cfg.gait_anchor_weight == 0.2


def test_anchor_step_rehearses_only_level_forward_states():
    torch.manual_seed(0)
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = SimpleNamespace(imag_batch=16, gait_anchor_weight=1.0,
                                  grad_clip=1.0, grad_clip_elem=1.0)
    trainer.buf = ReplayBuffer(4, 12, 4, "cpu", gen_dim=5)
    for _ in range(8):
        trainer.buf.add(torch.zeros(4, st.STATE_DIM), torch.zeros(4, 3),
                        torch.zeros(4, 12), torch.zeros(4, 12), torch.ones(4, 4),
                        torch.tensor([[0.8, 0., 0., 2., 1.]]).expand(4, 5),
                        torch.zeros(4, dtype=torch.bool), torch.zeros(4, 77))
    trainer.actor = torch.nn.Linear(4, 12, bias=False)
    trainer.anchor_actor = torch.nn.Linear(4, 12, bias=False)
    trainer.anchor_actor.requires_grad_(False)
    trainer.opt_actor = torch.optim.SGD(trainer.actor.parameters(), lr=0.01)
    trainer.it, trainer.actor_frozen_until = 20, 10
    before = (trainer.actor(torch.ones(1, 4)) - trainer.anchor_actor(torch.ones(1, 4))).square().mean()
    stats = trainer.retain_gait()
    after = (trainer.actor(torch.ones(1, 4)) - trainer.anchor_actor(torch.ones(1, 4))).square().mean()
    assert stats["anchor_batch"] > 0
    assert after < before
    trainer.buf.relief[..., 0] = 0.1
    assert trainer.retain_gait() == {"anchor_batch": 0.0}
