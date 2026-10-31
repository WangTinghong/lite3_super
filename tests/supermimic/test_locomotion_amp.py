"""Check the AMP data contract and real-transition discriminator update."""
import json

import pytest
import torch

from lite3_supermimic.locomotion.amp import AMPStyle, load_forward_motions
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.supermimic.buffer import ReplayBuffer


def motions(tmp_path):
    for name in ("lite3_直行慢速.json", "lite3_直行快速.json"):
        frames = []
        for i in range(12):
            frame = [0.0] * 61
            frame[7:19] = [0.1 * i, 1.0, -1.8] * 4
            frame[37:49] = [5.0, 0.0, 0.0] * 4
            frames.append(frame)
        (tmp_path / name).write_text(json.dumps({"FrameDuration": 0.02, "Frames": frames}))
    return tmp_path


def test_loader_converts_joint_axes_and_time_warps_fast_demo(tmp_path):
    directory = motions(tmp_path)
    pairs = load_forward_motions(directory, 0.02, "cpu")
    assert pairs.shape[1:] == (2, 24)
    assert pairs[0, 0, 1] == -1.0
    assert pairs[0, 0, 2] == 1.8
    assert pairs[0, 0, 12] == -5.0
    # 1.25x playback interpolates positions and scales joint velocities.
    assert torch.any(torch.isclose(pairs[:, 0, 12], torch.tensor(-6.25)))
    with pytest.raises(ValueError, match="frame duration"):
        load_forward_motions(directory, 0.04, "cpu")


def test_discriminator_uses_valid_forward_real_transitions_and_actor_gradient(tmp_path):
    amp = AMPStyle(str(motions(tmp_path)), 0.02, "cpu")
    replay = ReplayBuffer(8, 16, 4, "cpu", gen_dim=5)
    for i in range(8):
        s = torch.zeros(8, st.STATE_DIM)
        s[:, st.Q] = 0.01 * i
        s[:, st.QD] = 0.02 * i
        replay.add(s, torch.zeros(8, 3), torch.zeros(8, 12), torch.zeros(8, 12),
                   torch.zeros(8, 4), torch.tensor([[1., 0., 0., 2., 1.]]).expand(8, 5),
                   torch.zeros(8, dtype=torch.bool), torch.zeros(8, 77))
    default_q = torch.tensor([[0., -1., 1.8] * 4]).expand(8, 12)
    before = amp.discriminator[0].weight.detach().clone()
    stats = amp.fit(replay, default_q, batch=16, updates=1)
    assert stats["amp_batches"] == 1
    assert not torch.equal(before, amp.discriminator[0].weight)
    assert not any(p.requires_grad for p in amp.discriminator.parameters())
    s0 = replay.s[:, 0].clone().requires_grad_()
    s1 = replay.s[:, 1].clone().requires_grad_()
    amp.reward(s0, s1, default_q).mean().backward()
    assert s1.grad[:, st.Q].abs().sum() > 0
