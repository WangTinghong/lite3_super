import torch
from lite3_supermimic.locomotion.stair_probe import front_leg_probe
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.target.terrain import HeightField


def test_probe_changes_only_airborne_front_right_near_uphill_edge():
    x = torch.linspace(-1, 1, 101)
    field = HeightField((x >= .2).float().mul(.18).expand(101, 101).clone(), (-1., -1.), .02)
    s = torch.zeros(4, st.STATE_DIM)
    s[:, st.Z] = .28
    s[:, st.FEET] = torch.tensor([.2, -.1, -.2, .2, .1, -.2,
                                 -.2, -.1, -.2, -.2, .1, -.2])
    p = torch.zeros(4, 3)
    p[:, 0] = -.15
    p[2, 0] = -.8
    cmd = torch.zeros(4, 3)
    cmd[:, 0] = .5
    cmd[3, 0] = -.5
    contact = torch.zeros(4, 4, dtype=torch.bool)
    contact[1, 0] = True
    a = torch.zeros(4, 12)
    scale = torch.full((12,), .25)
    corrected, gate = front_leg_probe(a, s, p, cmd, field, contact, scale, -.1, -.05)
    expected = a.clone()
    expected[0, 4], expected[0, 5] = -.4, -.2
    torch.testing.assert_close(corrected, expected)
    torch.testing.assert_close(gate, torch.tensor([1., 0., 0., 0.]))
    assert torch.count_nonzero(a) == 0
    unchanged, disabled = front_leg_probe(a, s, p, cmd, field, contact, scale)
    assert unchanged is a and not disabled.any()
    unchanged, disabled = front_leg_probe(a, s, p, cmd, None, contact, scale, -.1)
    assert unchanged is a and not disabled.any()
