"""Regressions for command timing, leg pairing and imagined locomotion learning."""
from types import SimpleNamespace
import pytest
import torch
from lite3_supermimic.locomotion.gait import GaitCfg, initial_state, advance, foot_schedule
from lite3_supermimic.locomotion.imagine import LocomotionImagination, OBS_TERMS
from lite3_supermimic.locomotion.rewards import RewardWeights, reward_terms, weighted_reward
from lite3_supermimic.supermimic import state as st


def consts(n):
  return st.RobotConsts(torch.zeros(n, 12), torch.full((n, 12), 0.25),
    torch.full((12,), 20.), torch.full((12,), .7), torch.full((12,), 30.),
    torch.tensor([-2., 2.]).expand(n, 12, 2))


def test_speed_ramp_gait_hysteresis_and_deceleration():
  cfg = GaitCfg()
  s = initial_state(1, 'cpu')
  s[:, 0] = 2
  step = advance(s, .02, cfg)
  assert step[0, 3] == pytest.approx(.016)
  for _ in range(300):
    s = advance(s, .02, cfg)
  assert s[0, 10] == 2 and s[0, 9] > .99
  s[:, :3] = torch.tensor([1.3, 0., 0.])
  for _ in range(100):
    s = advance(s, .02, cfg)
  assert s[0, 10] == 2
  s[:, :3] = 0
  for _ in range(300):
    s = advance(s, .02, cfg)
  assert s[0, 10] == 0
  assert torch.equal(s[:, 3:6], torch.zeros(1, 3))
  _, _, stance, height = foot_schedule(s, cfg)
  torch.testing.assert_close(stance, torch.ones_like(stance))
  torch.testing.assert_close(height, torch.zeros_like(height))


@pytest.mark.parametrize('mode,pairs', [('trot', ((0, 3), (1, 2))), ('bound', ((0, 1), (2, 3)))])
def test_leg_pairs_and_phase_continuity(mode, pairs):
  cfg = GaitCfg(mode=mode)
  s = initial_state(1, 'cpu')
  s[:, 0] = .8
  for _ in range(300):
    s = advance(s, .02, cfg)
  phase, _, _, _ = foot_schedule(s, cfg)
  for a, b in pairs:
    assert abs(float(phase[0, a] - phase[0, b])) < 1e-4
  old = s[:, 6].clone()
  s[:, 0] = 1.8
  nxt = advance(s, .02, cfg)
  delta = (nxt[:, 6] - old).remainder(1)
  assert 0 < delta < .1


def test_per_environment_zero_dt_does_not_advance_reset_gait():
  s = initial_state(2, 'cpu')
  s[:, 0] = 2
  nxt = advance(s, torch.tensor([0., .02]), GaitCfg())
  torch.testing.assert_close(nxt[0], s[0])
  assert nxt[1, 3] > 0


def test_bound_is_limited_to_forward_running():
  cfg = GaitCfg()
  for velocity in ((-2., 0., 0.), (2., .4, 0.), (2., 0., 1.)):
    s = initial_state(1, 'cpu')
    s[:, :3] = torch.tensor(velocity)
    for _ in range(300):
      s = advance(s, .02, cfg)
    assert s[0, 10] != 2


def test_velocity_task_registration_and_observation_layout():
  import lite3_supermimic.env
  from mjlab.tasks.registry import load_env_cfg
  cfg = load_env_cfg('Lite3-Velocity-Flat')
  assert set(cfg.commands) == {'velocity'}
  assert st.obs_terms(cfg) == OBS_TERMS
  assert sum(dim for _, dim in OBS_TERMS) * 5 == 270
  assert 'target' not in cfg.observations['actor'].terms
  assert cfg.scene.terrain.terrain_generator is None


def test_env_and_imagined_rewards_share_command_timing():
  from lite3_supermimic.locomotion.env_cfg import locomotion_reward
  from lite3_supermimic.supermimic.imagine import ImagineCfg
  cfg = GaitCfg()
  c = consts(2)
  state = torch.zeros(2, st.STATE_DIM)
  state[:, 0], state[:, 3] = .28, -1
  state[:, st.V_YAW.start] = 1.15
  command = initial_state(2, 'cpu')
  command[:, 0], command[:, 3], command[:, 10] = 1.30, 1.20, 1
  action = torch.full((2, 12), .1)
  previous = torch.zeros_like(action)
  weights = RewardWeights(mid_straight_vx_error=-.40)
  expected = weighted_reward(reward_terms(state, action, previous, command, c, cfg), weights)
  # Environment wrapper must use the same pre-advance command and action history.
  term = SimpleNamespace(state=command, robot_consts=c, cfg=SimpleNamespace(gait=cfg))
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: term),
    action_manager=SimpleNamespace(action=action, prev_action=previous),
    cfg=SimpleNamespace(commands={'velocity': SimpleNamespace(reward_weights=weights)}))
  from unittest.mock import patch
  with patch.object(st, 'robot_state', return_value=state):
    torch.testing.assert_close(locomotion_reward(env), expected)
  imag = LocomotionImagination(ImagineCfg(dt=.02, weights=weights), c, cfg, 5, 'cpu')
  class Model:
    def __call__(self, *args):
      return state, torch.zeros(2, 3)
  result = imag._step(state, torch.zeros(2, 3), previous, torch.zeros(2, 270), command,
    torch.zeros(2, 4), torch.zeros(2, 12),
    {'default_q': c.default_q, 'action_scale': c.action_scale, 'soft_limits': c.soft_limits},
    torch.full((2,), .02), lambda _: action, Model())
  death = imag.fall_prob(state)
  torch.testing.assert_close(result[5], (expected + weights.terminated * death) * .02)
  torch.testing.assert_close(result[4], advance(command, .02, cfg))


def test_mid_speed_reward_is_axis_specific_and_applies_only_to_straight_trot():
  n = 10
  command = initial_state(n, 'cpu')
  command[:, 0], command[:, 3], command[:, 10] = 1.30, 1.20, 1
  command[4, 0:4:3] = .5                   # Low target and effective command.
  command[5, 1] = .08                      # Lateral target just arrived.
  command[6, 2] = -.2                      # Turning target just arrived.
  command[7, 4] = .08                      # Previous lateral command still settling.
  command[8, 10] = 2                       # Bound held during deceleration.
  command[9, 0] = 1.6                      # High-speed target during acceleration.
  state = torch.zeros(n, st.STATE_DIM)
  state[:, st.V_YAW.start] = command[:, 3]
  state[1, st.V_YAW.start] -= .06
  state[2, st.V_YAW.start + 1] = .03
  state[3, st.W_B.stop - 1] = .08
  state[4:, st.V_YAW.start + 1] = .03
  state.requires_grad_(True)
  action = torch.zeros(n, 12)
  terms = reward_terms(state, action, action, command, consts(n), GaitCfg())
  axes = torch.stack([terms[f'mid_straight_{axis}_error'] for axis in ('vx', 'vy', 'yaw')], -1)
  assert torch.equal(axes[1:4] > 0, torch.eye(3, dtype=torch.bool))
  assert torch.count_nonzero(axes[0]) == 0
  assert torch.count_nonzero(axes[4:]) == 0
  axes.sum().backward()
  assert state.grad[1, st.V_YAW.start] < 0
  assert state.grad[2, st.V_YAW.start + 1] > 0
  assert state.grad[3, st.W_B.stop - 1] > 0


def test_hipx_reward_penalizes_both_sides_and_relaxes_during_maneuvers():
  from lite3_supermimic.robot.constants import HIP_X_JOINT_INDICES

  n = 5
  positions = torch.zeros(n, st.STATE_DIM)
  positions[:, st.Q.start + 1] = .8  # HipY flexion is outside this penalty.
  hipx = torch.tensor([-.3, .3, -.3, .3])
  positions[:, st.Q.start + torch.tensor(HIP_X_JOINT_INDICES)] = hipx
  positions[0, st.Q] *= .25
  positions.requires_grad_(True)
  command = initial_state(n, 'cpu')
  command[:, 3] = .5
  command[2, 4] = .3
  command[3, 5] = .7
  command[4, 7:10] = torch.tensor([0., 0., 1.])
  zero_action = torch.zeros(n, 12)
  penalty = reward_terms(positions, zero_action, zero_action, command, consts(n), GaitCfg())['hipx_deviation']
  torch.testing.assert_close(penalty[0], torch.tensor(0.))
  torch.testing.assert_close(penalty[1], torch.tensor(.16))
  assert (penalty[2:4] < .02 * penalty[1]).all()
  torch.testing.assert_close(penalty[4], torch.tensor(0.))
  penalty.sum().backward()
  grad = positions.grad[1, st.Q]
  assert (grad[list(HIP_X_JOINT_INDICES)] * hipx > 0).all()
  assert grad[1] == 0


@pytest.mark.parametrize('anchor_weight', [0.0, 2.0])
def test_resuming_legacy_rewards_refits_critic_on_continued_timeline(tmp_path, anchor_weight):
  from dataclasses import asdict
  from lite3_supermimic.robot.constants import JOINT_NAMES
  from lite3_supermimic.supermimic.train import Actor, Critic, Trainer, TrainCfg
  from lite3_supermimic.supermimic.world_model import WorldModel

  path = tmp_path / 'model_7000.pt'
  trainer = Trainer.__new__(Trainer)
  trainer.cfg = TrainCfg(continue_run=True, resume=str(path), actor_anchor_weight=anchor_weight)
  trainer.actor_anchor = None
  trainer.dev, trainer.locomotion = 'cpu', True
  trainer.policy_start, trainer.actor_frozen_until = 300, 0
  trainer.obs = torch.zeros(1, 270)
  gait, weights = GaitCfg(), RewardWeights()
  trainer.term = SimpleNamespace(cfg=SimpleNamespace(gait=gait, reward_weights=weights))
  trainer.env = SimpleNamespace(common_step_counter=0)
  trainer.wm = WorldModel((16, 16))
  trainer.actor = Actor(270, (16, 16), 12)
  trainer.critic = Critic(86, (16, 16))
  initial_critic = {key: value.clone() for key, value in trainer.critic.state_dict().items()}
  old_weights = asdict(weights)
  old_weights.pop('hipx_deviation')
  old_critic = {key: torch.zeros_like(value) for key, value in initial_critic.items()}
  checkpoint = dict(robot='lite3', joint_names=JOINT_NAMES, iteration=7000,
                  cfg={'task': trainer.cfg.task}, gait_cfg=asdict(gait),
                  command_cfg={'reward_weights': old_weights}, critic=old_critic,
                  actor=trainer.actor.state_dict(), world_model=trainer.wm.state_dict())
  torch.save(checkpoint, path)
  trainer.load(str(path))
  assert trainer.env.common_step_counter == trainer.start_it == 7000
  assert trainer.actor_frozen_until == 7500
  for key, value in trainer.critic.state_dict().items():
    torch.testing.assert_close(value, initial_critic[key])
  if anchor_weight:
    frozen_anchor = {key: value.clone() for key, value in trainer.actor_anchor.state_dict().items()}
    assert all(not parameter.requires_grad for parameter in trainer.actor_anchor.parameters())
    joint_bias = next(reversed(trainer.actor.net.state_dict()))
    checkpoint['actor_anchor'] = frozen_anchor
    with torch.no_grad():
      trainer.actor.net.state_dict()[joint_bias].add_(.5)
    checkpoint['actor'] = trainer.actor.state_dict()
    torch.save(checkpoint, path)
    trainer.load(str(path))
    output_difference = trainer.actor(trainer.obs) - trainer.actor_anchor(trainer.obs)
    torch.testing.assert_close(output_difference, torch.full_like(output_difference, .5))
  else:
    assert trainer.actor_anchor is None


def test_imagined_joint_targets_obey_action_bounds_and_can_leave_saturation():
  from lite3_supermimic.locomotion.actions import action_bounds
  from lite3_supermimic.robot.constants import HIP_X_JOINT_INDICES
  from lite3_supermimic.supermimic.imagine import ImagineCfg

  c = consts(1)
  low, high = action_bounds(c.default_q, c.action_scale, c.soft_limits, .25)
  torch.testing.assert_close(low[:, HIP_X_JOINT_INDICES], torch.full((1, 4), -1.))
  torch.testing.assert_close(high[:, HIP_X_JOINT_INDICES], torch.ones(1, 4))
  assert high[0, 1] == 8.  # HipY's bounds are determined by its soft joint limits.
  imag = LocomotionImagination(ImagineCfg(weights=RewardWeights()), c, GaitCfg(), 5, 'cpu', .25)
  state = torch.zeros(1, st.STATE_DIM)
  state[:, 0], state[:, 3] = .28, -1
  request = torch.full((1, 12), 20., requires_grad=True)

  def model(state, action, previous, context, relief):
    torch.testing.assert_close(action, high)
    return state, torch.zeros(1, 3)

  result = imag._step(state, torch.zeros(1, 3), torch.zeros(1, 12), torch.zeros(1, 270),
    initial_state(1, 'cpu'), torch.zeros(1, 4), torch.zeros(1, 12),
    {'default_q': c.default_q, 'action_scale': c.action_scale, 'soft_limits': c.soft_limits},
    torch.full((1,), .02), lambda _: request, model)
  torch.testing.assert_close(result[2], high)
  result[5].sum().backward()
  assert (request.grad < 0).all()


@pytest.mark.parametrize('bootstrap_weight', [0., 1.])
def test_imagined_critic_bootstrap_stays_within_velocity_reward_bound(bootstrap_weight):
  from lite3_supermimic.supermimic.imagine import ImagineCfg

  cfg = ImagineCfg(horizon=3, actor_bootstrap_weight=bootstrap_weight, weights=RewardWeights())
  imag = LocomotionImagination(cfg, consts(1), GaitCfg(), 5, 'cpu')
  s = torch.zeros(1, st.STATE_DIM)
  s[:, 0], s[:, 3] = .28, -1
  seed = dict(s=s, pose=torch.zeros(1, 3), a_prev=torch.zeros(1, 12), obs=torch.zeros(1, 270),
              xi=torch.zeros(1, 4), enc_bias=torch.zeros(1, 12), env=torch.arange(1), gen=initial_state(1, 'cpu'))
  def critic(x):
    return x[..., :1] * 0 + 1000.

  objective, data = imag.rollout(
    lambda state, *args: (state, torch.zeros(1, 3)), lambda obs: torch.zeros(1, 12), critic, seed,
  )
  expected = bootstrap_weight * 10 * cfg.gamma ** cfg.horizon * data['surv'][:, -1]
  torch.testing.assert_close(data['terminal_value'], torch.tensor([1000.]))
  torch.testing.assert_close(data['bootstrap_return'], expected)
  torch.testing.assert_close(objective, data['step_return'] + expected)
  assert (imag.gae_targets(critic, data) <= 10.00001).all()


def test_locomotion_imagination_updates_actor_with_finite_gradients():
  from lite3_supermimic.supermimic.train import Actor, Critic
  from lite3_supermimic.supermimic.imagine import ImagineCfg
  from lite3_supermimic.supermimic.world_model import WorldModel
  torch.manual_seed(12)
  n = 4
  cfg = GaitCfg()
  imag = LocomotionImagination(ImagineCfg(horizon=4, weights=RewardWeights()), consts(n), cfg, 5, 'cpu')
  wm, actor = WorldModel((32, 32)), Actor(270, (32, 32), 12)
  critic = Critic(imag.critic_dim, (32, 32))
  for param in wm.parameters():
    param.requires_grad_(False)
  for param in wm.net.parameters():
    param.data.mul_(.01)
  s = torch.zeros(n, st.STATE_DIM)
  s[:, 0], s[:, 3] = .28, -1
  command = initial_state(n, 'cpu')
  command[:, 0] = 1.8
  seed = dict(s=s, pose=torch.zeros(n, 3), a_prev=torch.zeros(n, 12),
    obs=torch.zeros(n, 270), xi=torch.zeros(n, 4), enc_bias=torch.zeros(n, 12),
    env=torch.arange(n), gen=command)
  before = [p.detach().clone() for p in actor.parameters()]
  objective, data = imag.rollout(wm, actor, critic, seed)
  (-objective.mean()).backward()
  assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
  torch.optim.Adam(actor.parameters(), lr=1e-3).step()
  assert any(not torch.equal(a, b) for a, b in zip(before, actor.parameters()))
  assert all(p.grad is None for p in wm.parameters())
  assert torch.isfinite(imag.gae_targets(critic, data)).all()


@pytest.mark.parametrize('validate', [False, True])
def test_training_loop_saves_only_validated_candidates_when_enabled(tmp_path, validate, capsys):
  from collections import defaultdict
  from unittest.mock import Mock
  from lite3_supermimic.locomotion.validation import StraightMetrics
  from lite3_supermimic.supermimic.train import Trainer, TrainCfg
  trainer = Trainer.__new__(Trainer)
  trainer.cfg = TrainCfg(iterations=2, warmup_iterations=0, log_interval=1, save_interval=1)
  trainer.locomotion = True
  trainer.buf = SimpleNamespace(size=20)
  trainer.term = SimpleNamespace(state=initial_state(1, 'cpu'))
  trainer.archive = SimpleNamespace(size=0)
  trainer.writer = SimpleNamespace(add_scalar=lambda *args: None)
  trainer.start_it = trainer.env_steps = trainer.nonfinite_steps = trainer.actor_frozen_until = 0
  trainer.policy_start = trainer.ramp_start = trainer.collapse_strikes = 0
  trainer.lr_scale = 1.
  trainer.ep_stats = defaultdict(list)
  trainer.ep_stats['Metrics/velocity/velocity_error'].append(.1)
  trainer.last_ckpt = None
  trainer.opt_wm = trainer.opt_actor = trainer.opt_critic = SimpleNamespace(param_groups=[{}])
  saved = []
  trainer.collect = lambda: None
  trainer.fit_world_model = lambda: .01
  trainer.improve_policy = lambda: {'imagined_return': 1.}
  def save(it):
    saved.append(it)
    return tmp_path / f'model_{it}.pt'
  trainer.save = save
  if validate:
    trainer.actor = object()
    reference = StraightMetrics(.048, .053, .85, 0, 1000)
    accepted = StraightMetrics(.05, .06, .84, 0, 1000)
    regressed = StraightMetrics(.22, .505, .71, 3, 1000)
    validator = Mock()
    validator.evaluate.side_effect = [reference, accepted, regressed]
    trainer._run(validator)
    assert saved == [1]
    assert 'FAIL' in capsys.readouterr().out
  else:
    trainer.run()
    assert saved == [1, 2]


@pytest.mark.parametrize('kwargs', [{'gait': 'unknown'}, {'max_forward': -1},
                                    {'bound_enter': 1.0}, {'velocity_curriculum_iterations': 0},
                                    {'fast_straight_min': 1.3, 'fast_straight_max': 1.1},
                                    {'fast_straight_max': float('inf')}])
def test_invalid_velocity_training_settings_fail_early(kwargs):
  from lite3_supermimic.supermimic.train import TrainCfg
  with pytest.raises(ValueError):
    TrainCfg(**kwargs)


def test_command_manager_reset_and_per_environment_timing():
  from lite3_supermimic.locomotion.command import VelocityGaitCommandCfg
  data = SimpleNamespace(root_link_lin_vel_w=torch.zeros(2, 3), heading_w=torch.zeros(2),
                         root_link_ang_vel_b=torch.zeros(2, 3))
  env = SimpleNamespace(num_envs=2, device='cpu', scene={'robot': SimpleNamespace(data=data)},
                        step_dt=.02, common_step_counter=0, max_episode_length=1000)
  command = VelocityGaitCommandCfg(resampling_time_range=(4., 8.), standing_fraction=0).build(env)
  command.reset(torch.arange(2))
  command.state[:, :3] = torch.tensor([2., 0., 0.])
  previous = command.state.clone()
  command.compute(torch.tensor([0., .02]))
  torch.testing.assert_close(command.state[0], previous[0])
  assert command.state[1, 3] > 0
  other = command.state[1].clone()
  command.reset(torch.tensor([0]))
  command.compute(0., env_ids=torch.tensor([0]))
  torch.testing.assert_close(command.state[1], other)
  assert command.command.shape == (2, 12)


def test_pure_side_sampling_retains_standing_and_covers_both_directions():
  from lite3_supermimic.locomotion.command import VelocityGaitCommandCfg

  env = SimpleNamespace(num_envs=256, device='cpu', scene={'robot': None},
                        common_step_counter=7500)
  cfg = VelocityGaitCommandCfg(resampling_time_range=(4., 8.), pure_side_fraction=1.0)
  term = cfg.build(env)
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(42)
    term.reset(torch.arange(env.num_envs))
  targets = term.state[:, :3]
  torch.testing.assert_close(targets[:, [0, 2]], torch.zeros(env.num_envs, 2))
  torch.testing.assert_close(targets[:, 1].unique(), torch.tensor([-.1, 0., .1]))
  torch.testing.assert_close(term.state[:, 3:6], torch.zeros(env.num_envs, 3))


@pytest.mark.parametrize('band', [{}, {'fast_straight_min': 1.1, 'fast_straight_max': 1.3}])
def test_fast_straight_sampling_preserves_the_side_and_standing_allocations(band):
  from lite3_supermimic.locomotion.command import VelocityGaitCommandCfg

  env = SimpleNamespace(num_envs=4096, device='cpu', scene={'robot': None},
                        common_step_counter=7600)
  cfg = VelocityGaitCommandCfg(resampling_time_range=(4., 8.), max_forward=.8,
                               pure_side_fraction=.35, fast_straight_fraction=.25, **band)
  command = cfg.build(env)
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(42)
    command.reset(torch.arange(env.num_envs))
  vx, vy, yaw = command.state[:, :3].unbind(-1)
  fast = vx >= cfg.fast_straight_min
  side = (vx == 0) & (vy.abs() == .1) & (yaw == 0)
  stand = (vx == 0) & (vy == 0) & (yaw == 0)
  assert ((vx[fast] <= cfg.fast_straight_max) & (vy[fast] == 0) & (yaw[fast] == 0)).all()
  assert vx[~fast].max() < .62
  assert fast.float().mean().item() == pytest.approx(.8 * .25, abs=.025)
  assert side.float().mean().item() == pytest.approx(.8 * .35, abs=.025)
  assert stand.float().mean().item() == pytest.approx(.2, abs=.025)


def test_velocity_evaluation_separates_the_initial_and_final_stop(monkeypatch):
  from unittest.mock import Mock
  from lite3_supermimic.locomotion.evaluation import VelocityEvaluation

  term = SimpleNamespace(state=initial_state(1, 'cpu'), cfg=SimpleNamespace(gait=GaitCfg()))
  env = SimpleNamespace(cfg=None, num_envs=1, device='cpu', step_dt=.02,
                        command_manager=SimpleNamespace(get_term=lambda name: term), step=Mock())
  samples = []

  def step(action):
    samples.append(term.state[0, 0].item())
    terminated = torch.tensor([len(samples) == 4])
    return {}, None, terminated, torch.tensor([False]), {}

  env.step.side_effect = step
  monkeypatch.setattr(st, 'robot_state', lambda env: torch.zeros(1, st.STATE_DIM))
  evaluation = VelocityEvaluation(env, vx_sequence=[0., .5, 0.], segment_seconds=.04)
  for _ in range(7):
    evaluation.step(torch.zeros(1, 12))
  assert samples == [0., 0., .5, .5, 0., 0., 0.]
  summary = evaluation.summary()
  segments = summary['segments']
  assert [segment['target_vx_m_s'] for segment in segments] == [0., .5, 0.]
  assert [segment['falls'] for segment in segments] == [0, 1, 0]
  assert segments[1]['valid_seconds'] == pytest.approx(.02)
  assert segments[2]['start_s'] == pytest.approx(.08)
  assert segments[2]['end_s'] == pytest.approx(.14)
  assert summary['stand']['falls'] == 1


def test_side_step_metrics_preserve_direction_and_split_tracking_error(monkeypatch):
  from lite3_supermimic.locomotion.evaluation import VelocityEvaluation

  term = SimpleNamespace(state=initial_state(1, 'cpu'), cfg=SimpleNamespace(gait=GaitCfg()))
  term.state[0, 4] = -.1
  done = torch.tensor([False])
  env = SimpleNamespace(cfg=None, num_envs=1, device='cpu', step_dt=.02,
                        command_manager=SimpleNamespace(get_term=lambda name: term),
                        step=lambda action: ({}, None, done, done, {}))
  state = torch.zeros(1, st.STATE_DIM)
  state[0, st.V_YAW] = torch.tensor([.03, -.08, 0.])
  monkeypatch.setattr(st, 'robot_state', lambda env: state)
  evaluation = VelocityEvaluation(env, vx=0., vy=-.1)
  evaluation.step(torch.zeros(1, 12))
  walk = evaluation.summary()['walk']
  assert walk['vx_mean_m_s'] == pytest.approx(.03)
  assert walk['vy_mean_m_s'] == pytest.approx(-.08)
  assert walk['command_vy_mean_m_s'] == pytest.approx(-.1)
  assert walk['vx_rmse_m_s'] == pytest.approx(.03)
  assert walk['vy_rmse_m_s'] == pytest.approx(.02)
  assert walk['velocity_rmse_m_s'] ** 2 == pytest.approx(.0013)
