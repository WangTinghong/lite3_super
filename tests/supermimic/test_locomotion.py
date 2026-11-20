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
  command = initial_state(2, 'cpu')
  command[:, 0] = 2
  action = torch.full((2, 12), .1)
  previous = torch.zeros_like(action)
  weights = RewardWeights()
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


def test_training_loop_logs_and_saves_without_chase_generator(tmp_path):
  from collections import defaultdict
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
  trainer.run()
  assert saved == [1, 2]


@pytest.mark.parametrize('kwargs', [{'gait': 'unknown'}, {'max_forward': -1},
                                    {'bound_enter': 1.0}, {'velocity_curriculum_iterations': 0}])
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
