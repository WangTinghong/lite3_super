"""Locomotion semantics plus regression protection for the existing Chase tasks."""
from types import SimpleNamespace

import pytest
import torch

from lite3_supermimic.locomotion.config import (
    HISTORY, SCHEMA, STAGES, TASKS, WEIGHTS, gait_refine_weights,
    hipx_refine_weights, hipx_target_refine_weights, stair_refine_weights,
)
from lite3_supermimic.locomotion.core import (
    TERMS, advance_commands, feet_world, foot_clearances, observation, reward_terms,
    stair_clearance_terms, stair_hipx_target_penalty, stair_foothold_reward,
    sample_commands, validate_checkpoint,
)
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.robot.constants import JOINT_NAMES
from lite3_supermimic.target.terrain import HeightField


def inputs(n=4):
    s = torch.zeros(n, st.STATE_DIM)
    s[:, 0], s[:, 3] = 0.28, -1
    s[:, st.FEET] = torch.tensor([0.2, -0.1, -0.258, 0.2, 0.1, -0.258,
                                 -0.2, -0.1, -0.258, -0.2, 0.1, -0.258])
    pose = torch.zeros(n, 3)
    a = torch.zeros(n, 12)
    c = st.RobotConsts(default_q=a.clone(), action_scale=torch.full_like(a, 0.25),
                       kp=torch.full((12,), 20.), kd=torch.full((12,), .7),
                       effort=torch.full((12,), 30.), soft_limits=torch.tensor([-2., 2.]).expand(n, 12, 2))
    return s, pose, a, c


def test_commands_resample_on_expiry_and_hold_otherwise():
    torch.manual_seed(5)
    stage = STAGES["fast"]
    packed = torch.tensor([[0.7, 0.1, 0.2, 0.01, 2.0], [0.7, 0.1, 0.2, 2., 2.]])
    nxt = advance_commands(packed, .02, stage)
    torch.testing.assert_close(nxt[1, :3], packed[1, :3])
    assert nxt[1, 3] == pytest.approx(1.98)
    assert stage.interval[0] <= nxt[0, 3] <= stage.interval[1]
    assert not torch.equal(nxt[0, :3], packed[0, :3])
    commands = sample_commands(10_000, torch.full((10_000,), 2.), stage)
    assert commands[:, 0].min() >= stage.backward and commands[:, 0].max() <= 2
    assert commands[:, 1].abs().max() <= stage.lateral
    assert commands[:, 2].abs().max() <= stage.yaw
    assert 0.1 < float((commands == 0).all(-1).float().mean()) < 0.3


def test_stair_focus_commands_are_forward_and_match_imagined_resampling():
    from dataclasses import replace
    stage = replace(STAGES["terrain"], stair_focus_prob=1.0)
    vmax = torch.full((256,), 1.5)
    sampled = sample_commands(256, vmax, stage)
    assert torch.all((sampled[:, 0] >= 0.25) & (sampled[:, 0] <= 1.0))
    assert torch.count_nonzero(sampled[:, 1:]) == 0
    packed = torch.cat((torch.zeros(256, 3), torch.zeros(256, 1), vmax[:, None]), -1)
    next_command = advance_commands(packed, .02, stage)
    assert torch.count_nonzero(next_command[:, 1:3]) == 0


def test_stair_refinement_detects_riser_and_diagonal_approach():
    x = torch.linspace(-1, 1, 101)
    field = HeightField((x >= 0.2).float().mul(0.18).expand(101, 101).clone(),
                        (-1., -1.), .02)
    s, p, _, _ = inputs(1)
    p[:, 0] = -0.15  # Front feet are just before the 18 cm riser.
    cmd = torch.tensor([[0.5, 0., 0.]])
    low_foot, _, _ = stair_clearance_terms(s, p, cmd, field)
    lifted = s.clone()
    lifted[:, st.FEET].reshape(1, 4, 3)[:, :2, 2] += 0.20
    high_foot, _, _ = stair_clearance_terms(lifted, p, cmd, field)
    assert low_foot.item() > high_foot.item() >= 0
    p[:, 0] = 0
    shank_low = s.clone()
    shank_low[:, st.KNEES].reshape(1, 4, 3)[0, 0] = torch.tensor([0.35, -0.1, -0.1])
    shank_low[:, st.FEET].reshape(1, 4, 3)[0, 0, 0] = 0.05
    shank_high = shank_low.clone()
    shank_high[:, st.FEET].reshape(1, 4, 3)[0, 0, 2] += 0.20
    _, low_collision, _ = stair_clearance_terms(shank_low, p, cmd, field)
    _, high_collision, _ = stair_clearance_terms(shank_high, p, cmd, field)
    assert low_collision.item() > high_collision.item() >= 0
    p[:, 0] = 0.05
    _, _, aligned = stair_clearance_terms(s, p, cmd, field)
    p[:, 2] = torch.pi / 4
    _, _, diagonal = stair_clearance_terms(s, p, cmd, field)
    assert diagonal.item() > aligned.item()
    assert stair_refine_weights().stair_foot_clearance < 0
    assert WEIGHTS.stair_foot_clearance == 0


def test_stair_refinement_requires_harder_terrain_and_resume():
    from lite3_supermimic.locomotion.train import TrainCfg
    with pytest.raises(ValueError, match="stair-refine"):
        TrainCfg(stage="terrain", harder_stairs=True, stair_refine=True)
    with pytest.raises(ValueError, match="stair-refine"):
        TrainCfg(stage="terrain", stair_refine=True, resume="model.pt")
    assert TrainCfg(stage="terrain", harder_stairs=True, stair_refine=True,
                    resume="model.pt").stair_refine
    with pytest.raises(ValueError, match="stair-hipx-weight"):
        TrainCfg(stage="terrain", harder_stairs=True, resume="model.pt",
                 stair_hipx_weight=2.0)
    with pytest.raises(ValueError, match="stair-foothold-weight"):
        TrainCfg(stage="terrain", harder_stairs=True, resume="model.pt",
                 stair_foothold_weight=0.5)


def test_stair_hipx_target_penalty_is_edge_gated_and_differentiable():
    x = torch.linspace(-1, 1, 101)
    field = HeightField((x >= 0.2).float().mul(0.18).expand(101, 101).clone(),
                        (-1., -1.), .02)
    s, p, a, c = inputs(1)
    p[:, 0] = -0.15
    a[:, 0], a[:, 3] = -2.4, 2.4  # FL and FR outward targets are 0.6 rad.
    cmd = torch.tensor([[0.5, 0., 0.]])
    action = a.clone().requires_grad_()
    penalty = stair_hipx_target_penalty(s, p, action, cmd, c, field)
    assert penalty.item() > 0
    penalty.sum().backward()
    assert action.grad[0, 0] < 0 and action.grad[0, 3] > 0
    assert stair_hipx_target_penalty(s, p, a, cmd, c, None).item() == 0
    assert stair_hipx_target_penalty(s, p, a, cmd * 0, c, field).item() == 0
    turn = torch.tensor([[0.5, 0., 0.5]])
    assert stair_hipx_target_penalty(s, p, a, turn, c, field).item() == 0
    weights = stair_refine_weights()
    assert weights.stair_hipx_target_outward == 0


def test_stair_foothold_rewards_front_foot_settling_on_upper_tread():
    x = torch.linspace(-1, 1, 101)
    field = HeightField((x >= 0.2).float().mul(0.18).expand(101, 101).clone(),
                        (-1., -1.), .02)
    s, p, _, _ = inputs(1)
    feet = s[:, st.FEET].reshape(1, 4, 3)
    feet[:, :2, 0], feet[:, :2, 2] = 0.35, -0.05
    command = torch.tensor([[0.5, 0., 0.]])
    assert stair_foothold_reward(s, s, p, p, command, field, .02).item() == 0
    moved = s.clone()
    moved[:, st.FEET].reshape(1, 4, 3)[:, :2, 0] -= 0.01
    next_pose = p.clone()
    next_pose[:, 0] += 0.01  # Base advances at 0.5 m/s; feet stay on the tread.
    landed = stair_foothold_reward(s, moved, p, next_pose, command, field, .02)
    hovering = s.clone()
    hovering[:, st.FEET].reshape(1, 4, 3)[:, :2, 0] -= 0.01
    hovering[:, st.FEET].reshape(1, 4, 3)[:, :2, 2] += 0.15
    high = stair_foothold_reward(s, hovering, p, next_pose, command, field, .02)
    assert landed.item() > high.item() >= 0
    assert stair_foothold_reward(s, s, p, p, command, None, .02).item() == 0
    assert stair_foothold_reward(s, moved, p, next_pose, command * 0, field, .02).item() == 0
    assert stair_refine_weights().stair_foothold == 0


def test_eval_metrics_report_progress_and_hipx_target_limit():
    from lite3_supermimic.locomotion.metrics import Metrics
    s, p, a, _ = inputs(1)
    end = p.clone()
    end[:, 0] = 1.0
    joint = torch.zeros(1, 12)
    joint[0, 3] = 0.51  # FR near the 0.523 rad limit.
    target = joint.clone()
    target[0, 3] = 0.60
    metrics = Metrics(1, "cpu")
    metrics.update_joint(joint, target)
    metrics.update(s, p, s, end, torch.tensor([[0.5, 0., 0.]]), a, a,
                   torch.zeros(1, 4, dtype=torch.bool), torch.zeros(1, 4, 3),
                   torch.tensor([False]), torch.tensor([True]), .02)
    result = metrics.result()
    assert result["mean_episode_max_forward_progress_m"] == pytest.approx(1.0)
    assert result["hipx_outward_mean_rad"][0] == pytest.approx(0.51)
    assert result["hipx_pos_over_0p5_fraction"][0] == 1.0
    assert result["hipx_target_over_limit_fraction"][0] == 1.0


def test_eval_counts_shank_before_foot_touchdown():
    from lite3_supermimic.locomotion.metrics import Metrics
    s, p, a, _ = inputs(1)
    metrics = Metrics(1, "cpu")
    command = torch.zeros(1, 3)
    force = torch.zeros(1, 4, 3)
    done = torch.tensor([False])
    first = torch.tensor([[False, False, False, False]])
    shank = torch.tensor([[True, False, False, False]])
    metrics.update(s, p, s, p, command, a, a, first, force, done, done, .02,
                   shank_contact=shank, shank_force=force)
    landed = torch.tensor([[True, True, False, False]])
    metrics.update(s, p, s, p, command, a, a, landed, force, done, done, .02,
                   shank_contact=first, shank_force=force)
    fractions = metrics.result()["touchdown_without_prior_shank_only_fraction"]
    assert fractions[:2] == [0.0, 1.0]


def test_contact_clearance_uses_each_foot_local_ground():
    s, p, _, _ = inputs(1)
    # z = x + 1, base ground is 1; front/rear feet sit over different ground.
    x = torch.linspace(-1, 1, 21)
    field = HeightField((x + 1).expand(21, 21).clone(), (-1., -1.), .1)
    feet = feet_world(s, p, field)
    torch.testing.assert_close(feet[..., 2], torch.full((1, 4), 1.022))
    torch.testing.assert_close(foot_clearances(s, p, field), torch.tensor([[-.2, -.2, .2, .2]]), atol=1e-6, rtol=0)
    # Translating the entire ground vertically must not change local rewards.
    shifted = HeightField(field.h + 4, field.origin, field.res)
    torch.testing.assert_close(foot_clearances(s, p, field), foot_clearances(s, p, shifted), atol=1e-6, rtol=0)


def test_tracking_reward_prefers_command_matching_and_standing():
    s, p, a, c = inputs()
    cmd = torch.tensor([[1., 0., .5]]).expand(4, 3)
    matched = s.clone()
    matched[:, 4], matched[:, 9] = 1., .5
    r0 = reward_terms(s, s, p, p, a, a, cmd, c, .02)
    r1 = reward_terms(s, matched, p, p, a, a, cmd, c, .02)
    assert torch.all(r1["track_linear"] > r0["track_linear"])
    assert torch.all(r1["track_yaw"] > r0["track_yaw"])
    assert set(r1) == set(vars(WEIGHTS)) - {"terminated"}
    assert all(torch.isfinite(v).all() for v in r1.values())
    assert all(torch.count_nonzero(r1[name]) == 0 for name in
               ("stair_foot_clearance", "shank_clearance", "stair_alignment"))


def test_optional_gait_reward_distinguishes_diagonal_stance_from_hopping():
    s, p, a, c = inputs(4)
    feet = s[:, st.FEET].reshape(4, 4, 3)
    feet[0, [1, 2], 2] += 0.10  # FR+HL: a diagonal stance.
    feet[1, 3, 2] += 0.10  # FR+FL+HR: FR lands before HL.
    feet[2, :, 2] += 0.10  # All four airborne.
    feet[3, :, 2] += 0.10  # Airborne while turning: gait terms are gated off.
    command = torch.tensor([[1., 0., 0.], [1., 0., 0.],
                            [1., 0., 0.], [1., 0., 0.6]])
    r = reward_terms(s, s, p, p, a, a, command, c, .02)
    assert r["diagonal_contact"][0] < r["diagonal_contact"][1]
    assert r["all_airborne"][0] < r["all_airborne"][2]
    assert r["diagonal_contact"][3] == 0
    assert r["all_airborne"][3] == 0
    assert WEIGHTS.diagonal_contact == WEIGHTS.all_airborne == 0
    assert gait_refine_weights().diagonal_contact < 0
    assert gait_refine_weights().all_airborne < 0


def test_hipx_refinement_penalizes_outward_swing_only_for_straight_motion():
    s, p, a, c = inputs(4)
    # Joint order: FL, FR, HL, HR. Negative FL and positive HR swing outward.
    s[[0, 2, 3], st.Q.start] = -0.4
    s[[0, 2, 3], st.Q.start + 9] = 0.4
    s[1, st.Q.start], s[1, st.Q.start + 9] = 0.4, -0.4
    feet = s[:, st.FEET].reshape(4, 4, 3)
    feet[:3, [1, 2], 2] += 0.10  # FL and HR in swing; row 3 stays in stance.
    cmd = torch.tensor([[1., 0., 0.], [1., 0., 0.],
                        [1., 0., 0.6], [1., 0., 0.]])
    state = s.clone().requires_grad_()
    penalty = reward_terms(s, state, p, p, a, a, cmd, c, .02)["hipx_swing_outward"]
    assert penalty[0] > penalty[3] > penalty[1] == penalty[2] == 0
    penalty[0].backward()
    assert state.grad[0, st.Q.start] < 0
    assert state.grad[0, st.Q.start + 9] > 0
    assert WEIGHTS.hipx_swing_outward == gait_refine_weights().hipx_swing_outward == 0
    assert hipx_refine_weights().hipx_swing_outward < 0


def test_hipx_target_refinement_has_direct_action_gradient():
    s, p, a, c = inputs(3)
    feet = s[:, st.FEET].reshape(3, 4, 3)
    feet[:2, [1, 2], 2] += 0.10  # FL and HR swing; row 2 stands.
    a[0, 0], a[0, 9] = -1.6, 1.6  # Target angles -0.4 and +0.4 rad.
    a[1, 0], a[1, 9] = 1.6, -1.6  # Same magnitude, inward.
    a[2, 0], a[2, 9] = -1.6, 1.6
    action = a.clone().requires_grad_()
    cmd = torch.tensor([[1., 0., 0.]]).expand(3, 3)
    penalty = reward_terms(s, s, p, p, action, a, cmd, c, .02)["hipx_target_outward"]
    assert penalty[0] > penalty[2] > penalty[1] == 0
    penalty[0].backward()
    assert action.grad[0, 0] < 0
    assert action.grad[0, 9] > 0
    assert WEIGHTS.hipx_target_outward == hipx_refine_weights().hipx_target_outward == 0
    assert hipx_target_refine_weights().hipx_target_outward < 0


def test_observation_history_has_no_target_slot():
    s, p, a, _ = inputs()
    scan = torch.zeros(4, 77)
    cmd = torch.randn(4, 3)
    frame = observation(s, a, cmd, a, scan)
    assert tuple(frame) == tuple(k for k, _ in TERMS)
    assert "target" not in frame
    assert HISTORY * sum(d for _, d in TERMS) == 610
    hist = torch.zeros(4, 610)
    for _ in range(HISTORY):
        hist = st.push_history(hist, frame, HISTORY, TERMS)
    start = 42 * HISTORY
    torch.testing.assert_close(hist[:, start:start + 15], cmd.repeat(1, HISTORY))


def test_task_family_rejects_chase_even_if_dimensions_match():
    ckpt = dict(robot="lite3", joint_names=JOINT_NAMES, obs_terms=TERMS, history=HISTORY)
    with pytest.raises(ValueError, match="chase/PPO"):
        validate_checkpoint(ckpt)
    validate_checkpoint(dict(ckpt, task_family=SCHEMA))
    with pytest.raises(ValueError, match="schema/history"):
        validate_checkpoint(dict(ckpt, task_family=SCHEMA, history=1))


def test_imagined_policy_update_and_stage_weight_transfer():
    from lite3_supermimic.locomotion.imagine import Imagination
    from lite3_supermimic.supermimic.imagine import ImagineCfg
    from lite3_supermimic.supermimic.train import Actor, Critic
    from lite3_supermimic.supermimic.world_model import WorldModel
    s, p, a, c = inputs()
    actor = Actor(610, (32, 32), 12)
    wm = WorldModel((32, 32))
    for param in wm.net.parameters():
        param.data.mul_(.01)
    wm.requires_grad_(False)
    for stage in STAGES.values():
        imag = Imagination(ImagineCfg(horizon=3, weights=WEIGHTS), c, stage, HISTORY, "cpu")
        critic = Critic(imag.critic_dim, (32, 32)).requires_grad_(False)
        seed = dict(s=s, pose=p, a_prev=a, obs=torch.zeros(4, 610), env=torch.arange(4),
                    gen=torch.tensor([[1., 0., 0., 2., 2.]]).expand(4, 5),
                    xi=torch.zeros(4, 4), enc_bias=a)
        actor.zero_grad(set_to_none=True)
        ret, data = imag.rollout(wm, actor, critic, seed)
        (-ret.mean()).backward()
        grads = [param.grad for param in actor.parameters() if param.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
        assert torch.isfinite(imag.gae_targets(critic, data)).all()
        clone = Actor(610, (32, 32), 12)
        clone.load_state_dict(actor.state_dict())
        torch.testing.assert_close(actor(seed["obs"]), clone(seed["obs"]))


def test_all_task_configs_preserve_chase_and_share_locomotion_schema():
    pytest.importorskip("mjlab")
    from lite3_supermimic.locomotion.env import make_env_cfg
    from mjlab.tasks.registry import load_env_cfg
    for stage in STAGES:
        cfg = make_env_cfg(stage)
        assert list(cfg.commands) == ["velocity"]
        assert tuple(cfg.observations["actor"].terms) == tuple(k for k, _ in TERMS)
        assert cfg.observations["actor"].history_length == HISTORY
        assert cfg.actions["joint_pos"].scale == {".*_joint": .25}
        assert cfg.decimation * cfg.sim.mujoco.timestep == .02
        assert "chase_tag" not in cfg.rewards
        assert load_env_cfg(TASKS[stage]).commands["velocity"].stage == stage
    for task, dim in (("Lite3-Chase-Flat", 350), ("Lite3-Chase-Obstacles", 430), ("Lite3-Chase-Terrain", 735)):
        cfg = load_env_cfg(task)
        assert list(cfg.commands) == ["target"]
        assert "chase_tag" in cfg.rewards and "hit" in cfg.rewards
        assert sum(d for _, d in st.obs_terms(cfg)) * 5 == dim


def test_fixed_terrain_level_survives_reset_and_curriculum_uses_performance():
    pytest.importorskip("mjlab")
    from lite3_supermimic.locomotion.env import terrain_curriculum
    terrain = SimpleNamespace(terrain_levels=torch.tensor([1, 1]), terrain_types=torch.tensor([0, 0]),
                               terrain_origins=torch.randn(5, 1, 3), env_origins=torch.zeros(2, 3))
    term = SimpleNamespace(cfg=SimpleNamespace(fixed_level=None), count=torch.tensor([100., 100.]),
                           error_sum=torch.tensor([10., 100.]), distance=torch.tensor([4., 4.]))
    env = SimpleNamespace(scene=SimpleNamespace(terrain=terrain),
                           command_manager=SimpleNamespace(get_term=lambda _: term),
                           termination_manager=SimpleNamespace(terminated=torch.tensor([False, True])))
    ids = torch.arange(2)
    terrain_curriculum(env, ids)
    assert terrain.terrain_levels.tolist() == [2, 0]
    term.cfg.fixed_level = 4
    for _ in range(3):
        terrain_curriculum(env, ids)
        assert terrain.terrain_levels.tolist() == [4, 4]
        torch.testing.assert_close(terrain.env_origins, terrain.terrain_origins[4, 0].expand(2, 3))


def test_gait_metrics_exclude_reset_teleports_and_use_real_elapsed_time():
    from lite3_supermimic.locomotion.metrics import Metrics
    s, p, a, _ = inputs(2)
    metric = Metrics(2, "cpu")
    contact = torch.ones(2, 4, dtype=torch.bool)
    no = torch.zeros(2, dtype=torch.bool)
    metric.update(s, p, s, p, torch.zeros(2, 3), a, a, contact, torch.zeros(2, 4, 3), no, no, .02)
    assert metric.touchdowns.sum() == 0
    metric.update(s, p, s, p, torch.zeros(2, 3), a, a, contact, torch.zeros(2, 4, 3), ~no, no, .02)
    result = metric.result()
    assert result["falls"] == 2 and result["completed_episodes"] == 2
    assert result["falls_per_robot_minute"] == pytest.approx(1500.)
    assert result["stance_slip_rms_mps"] == 0
    assert result["nonfall_completion_fraction"] == 0
    assert metric.has_touch.sum() == 0


def test_contact_metrics_reorder_xml_feet_to_state_order():
    from lite3_supermimic.locomotion.metrics import read_contacts
    sensor = SimpleNamespace(
        primary_names=[f"robot/{leg}_FOOT_collision" for leg in ("FL", "FR", "HL", "HR")],
        data=SimpleNamespace(found=torch.tensor([[1, 0, 1, 0]]),
                             force=torch.arange(12).float().reshape(1, 4, 3)))
    contact, force = read_contacts(sensor)
    assert contact.tolist() == [[False, True, False, True]]
    torch.testing.assert_close(force, sensor.data.force[:, [1, 0, 3, 2]])


def test_shank_contacts_are_measured_separately_from_foot_contacts():
    from lite3_supermimic.locomotion.metrics import Metrics, read_shank_contacts
    sensor = SimpleNamespace(
        primary_names=[f"robot/{leg}_SHANK_collision" for leg in ("FL", "FR", "HL", "HR")],
        data=SimpleNamespace(found=torch.tensor([[0, 1, 0, 0]]),
                             force=torch.tensor([[[0., 0., 0.], [3., 4., 0.],
                                                  [0., 0., 0.], [0., 0., 0.]]])))
    shank_contact, shank_force = read_shank_contacts(sensor)
    assert shank_contact.tolist() == [[True, False, False, False]]
    s, p, a, _ = inputs(1)
    metric = Metrics(1, "cpu")
    foot_contact = torch.zeros(1, 4, dtype=torch.bool)
    foot_force = torch.zeros(1, 4, 3)
    no = torch.zeros(1, dtype=torch.bool)
    for _ in range(2):
        metric.update(s, p, s, p, torch.zeros(1, 3), a, a, foot_contact, foot_force,
                      no, no, .02, shank_contact=shank_contact, shank_force=shank_force)
    result = metric.result()
    assert result["shank_contact_fraction"] == pytest.approx([1., 0., 0., 0.])
    assert result["shank_without_foot_fraction"] == pytest.approx([1., 0., 0., 0.])
    assert result["shank_contacts_per_second"] == pytest.approx([0., 0., 0., 0.])
    assert result["max_sampled_shank_force_N"] == pytest.approx(5.)
    assert result["max_sampled_foot_force_N"] == 0


def test_locomotion_collision_exposes_foot_without_modifying_shared_xml():
    mujoco = pytest.importorskip("mujoco")
    from lite3_supermimic.locomotion.env import make_env_cfg
    from lite3_supermimic.robot.lite3 import get_spec
    original = get_spec()
    specific = make_env_cfg("flat").scene.entities["robot"].spec_fn()
    sensors = {sensor.name for sensor in make_env_cfg("terrain").scene.sensors}
    assert "feet_ground_contact" in sensors
    assert "shanks_ground_contact" in sensors
    for leg in ("FR", "FL", "HR", "HL"):
        old = original.geom(f"{leg}_SHANK_collision")
        new = specific.geom(f"{leg}_SHANK_collision")
        foot = specific.geom(f"{leg}_FOOT_collision")
        assert old.fromto[5] == pytest.approx(-.21012)
        assert new.fromto[5] == pytest.approx(-.18012)
        assert foot.size[0] == pytest.approx(.022)
    specific.compile()
