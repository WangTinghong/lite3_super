"""Robot-specific migration checks, including real MJCF compilation."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from lite3_supermimic.checkpoint import validate_checkpoint
from lite3_supermimic.robot.constants import DEFAULT_JOINT_POS, JOINT_NAMES, FOOT_GEOM_NAMES
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.supermimic.world_model import WorldModel
from lite3_supermimic.target.terrain import SCAN_POINTS

ROOT = Path(__file__).resolve().parents[2]
XML = ROOT / 'lite3_supermimic/robot/lite3.xml'


def test_native_joint_axes_and_limits_match_urdf():
    source = ET.parse(ROOT / 'resources/robots/lite3/urdf/Lite3.urdf')
    target = ET.parse(XML)
    names = tuple(j.attrib['name'] for j in target.findall('.//body/joint'))
    assert names == JOINT_NAMES
    for name in names:
        urdf = source.find(f".//joint[@name='{name}']")
        mjcf = target.find(f".//joint[@name='{name}']")
        assert mjcf.attrib['axis'] == urdf.find('axis').attrib['xyz']
        limit = urdf.find('limit')
        assert np.allclose(np.fromstring(mjcf.attrib['range'], sep=' '),
                           [float(limit.attrib['lower']), float(limit.attrib['upper'])])


def test_mjcf_compiles_with_native_feet_and_sensors():
    mujoco = pytest.importorskip('mujoco')
    model = mujoco.MjModel.from_xml_path(str(XML))
    assert model.nq == 19 and model.nv == 18
    assert model.nu == 0  # mjlab creates the actuators, no duplicate legacy motors
    assert [model.joint(i).name for i in range(1, model.njnt)] == list(JOINT_NAMES)
    for name in FOOT_GEOM_NAMES:
        assert model.geom(name).id >= 0
    for name in (*st.HIP_BODIES, *st.KNEE_BODIES):
        assert model.body(name).id >= 0
    for name in st.FOOT_SITES:
        assert model.site(name).id >= 0
    assert model.sensor('imu_ang_vel').dim == 3
    data = mujoco.MjData(model)
    data.qpos[2] = 0.32
    data.qpos[3] = 1
    data.qpos[7:] = [next(v for k, v in DEFAULT_JOINT_POS.items() if re.fullmatch(k, n))
                       for n in JOINT_NAMES]
    mujoco.mj_forward(model, data)
    feet = np.array([data.site_xpos[model.site(n).id] for n in st.FOOT_SITES])
    assert np.all(feet[:, 2] > 0.022)
    assert np.all(feet[:2, 0] > feet[2:, 0])


def test_checkpoint_rejects_go2_and_reordered_lite3():
    with pytest.raises(ValueError, match='Go2/PPO'):
        validate_checkpoint({'actor': {}})
    with pytest.raises(ValueError, match='joint order'):
        validate_checkpoint({'robot': 'lite3', 'joint_names': JOINT_NAMES[::-1]})
    validate_checkpoint({'robot': 'lite3', 'joint_names': JOINT_NAMES})


def test_world_model_is_differentiable_through_multiple_steps():
    torch.manual_seed(9)
    wm = WorldModel((32, 32))
    for param in wm.parameters():
        param.requires_grad_(False)
    s = torch.randn(4, st.STATE_DIM)
    a = torch.randn(4, 12, requires_grad=True)
    prev = torch.zeros_like(a)
    xi = torch.zeros(4, st.CONTEXT_DIM)
    relief = torch.zeros(4, SCAN_POINTS)
    for _ in range(3):
        s, delta = wm(s, a, prev, xi, relief)
        prev = a
    (s.square().mean() + delta.square().mean()).backward()
    assert torch.isfinite(a.grad).all()
    assert a.grad.abs().sum() > 0
    assert all(p.grad is None for p in wm.parameters())


def test_mjlab_entity_builds_actuators_and_stands_under_pd():
    mujoco = pytest.importorskip('mujoco')
    pytest.importorskip('mjlab')
    from mjlab.entity import Entity
    from lite3_supermimic.robot.lite3 import get_lite3_robot_cfg
    from lite3_supermimic.robot.constants import STIFFNESS, DAMPING, EFFORT_LIMIT
    entity = Entity(get_lite3_robot_cfg())
    spec = entity.spec
    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[5, 5, 0.1], contype=1, conaffinity=1)
    model = spec.compile()
    assert model.nu == 12
    for name in FOOT_GEOM_NAMES:
        assert model.geom(name).condim.item() == 6
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    target = data.qpos[7:].copy()
    for _ in range(1000):
        data.ctrl[:] = np.clip(STIFFNESS * (target - data.qpos[7:]) - DAMPING * data.qvel[6:],
                               -EFFORT_LIMIT, EFFORT_LIMIT)
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert 0.15 < data.qpos[2] < 0.4
    assert abs(data.qpos[3]) > 0.9


def test_actor_imagination_update_has_finite_gradients():
    from lite3_supermimic.supermimic.train import Actor, Critic
    from lite3_supermimic.supermimic.imagine import ImagineCfg, Imagination
    from lite3_supermimic.supermimic.buffer import pack_generator
    from lite3_supermimic.target.generator import BallTarget, BallTargetCfg
    torch.manual_seed(12)
    n, h = 4, 5
    cfg = BallTargetCfg()
    gen = BallTarget(cfg, n, 'cpu')
    gen.spawn(torch.arange(n), torch.zeros(n, 2), torch.zeros(n), torch.zeros(n, 2),
              torch.ones(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool))
    consts = st.RobotConsts(
        default_q=torch.zeros(n, 12), action_scale=torch.full((n, 12), 0.25),
        kp=torch.full((12,), 20.), kd=torch.full((12,), 0.7),
        effort=torch.full((12,), 30.), soft_limits=torch.tensor([-2., 2.]).expand(n, 12, 2),
    )
    imag = Imagination(ImagineCfg(horizon=3, compile=False), consts, cfg, h, 'cpu')
    wm = WorldModel((32, 32))
    actor = Actor(350, (32, 32), 12)
    critic = Critic(imag.critic_dim, (32, 32))
    for net in (wm, critic):
        for param in net.parameters():
            param.requires_grad_(False)
    # A small residual model avoids an arbitrary untrained rollout exploding.
    for param in wm.net.parameters():
        param.data.mul_(0.01)
    s = torch.zeros(n, st.STATE_DIM)
    s[:, st.Z] = 0.28
    s[:, 3] = -1
    seed = dict(s=s, pose=torch.zeros(n, 3), a_prev=torch.zeros(n, 12),
                obs=torch.zeros(n, 350), xi=torch.zeros(n, 4), enc_bias=torch.zeros(n, 12),
                env=torch.arange(n), gen=pack_generator(gen))
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in actor.parameters()]
    objective, data = imag.rollout(wm, actor, critic, seed)
    assert torch.isfinite(objective).all()
    (-objective.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
    opt.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, actor.parameters()))
    assert torch.isfinite(imag.gae_targets(critic, data)).all()
    assert all(p.grad is None for p in wm.parameters())


def test_invalid_buffer_config_is_rejected_before_allocating():
    from lite3_supermimic.supermimic.train import TrainCfg
    with pytest.raises(ValueError, match='buffer_len'):
        TrainCfg(buffer_len=8, wm_window=8)


def test_registered_tasks_and_native_actions():
    pytest.importorskip('mjlab')
    import lite3_supermimic.env
    from mjlab.tasks.registry import load_env_cfg
    for task, dim in [('Lite3-Chase-Flat', 350), ('Lite3-Chase-Obstacles', 430),
                      ('Lite3-Chase-Terrain', 735)]:
        cfg = load_env_cfg(task)
        assert cfg.decimation * cfg.sim.mujoco.timestep == 0.02
        assert cfg.actions['joint_pos'].scale == {'.*_joint': 0.25}
        assert cfg.observations['actor'].history_length * sum(d for _, d in st.obs_terms(cfg)) == dim
        assert cfg.scene.entities['robot'].init_state.joint_pos == DEFAULT_JOINT_POS
        assert 'push_robot' not in cfg.events
        assert cfg.rewards['terminated'].weight < 0


def test_flat_actor_expands_to_lidar_without_changing_output():
    from lite3_supermimic.supermimic.train import Actor, expand_actor_new_terms
    torch.manual_seed(7)
    old = Actor(350, (32,), 12)
    new = Actor(430, (32,), 12)
    terms = st.OBS_TERMS + (('lidar', 16),)
    expand_actor_new_terms(old.state_dict(), new, st.OBS_TERMS, terms, 5)
    x = torch.randn(4, 350)
    torch.testing.assert_close(new(torch.cat([x, torch.randn(4, 80)], dim=-1)), old(x))
