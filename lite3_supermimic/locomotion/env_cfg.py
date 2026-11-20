"""Lite3 flat-ground velocity tracking and gait transitions."""
import math
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from lite3_supermimic.env.env_cfg import _configure_lite3, ACTOR_HISTORY
from lite3_supermimic.locomotion.command import VelocityGaitCommandCfg
from lite3_supermimic.locomotion.rewards import reward_terms, weighted_reward
from lite3_supermimic.supermimic import state as st


def locomotion_reward(env):
  term = env.command_manager.get_term("velocity")
  # Immutable action mapping is cached once, outside the simulation step loop.
  if not hasattr(term, "robot_consts"):
    term.robot_consts = st.robot_consts(env)
  return weighted_reward(reward_terms(
    st.robot_state(env), env.action_manager.action, env.action_manager.prev_action,
    term.state, term.robot_consts, term.cfg.gait,
  ), env.cfg.commands["velocity"].reward_weights)


def low_body(env):
  return env.scene["robot"].data.root_link_pos_w[:, 2] < 0.12


def flat_velocity_env_cfg(play=False):
  cfg = make_velocity_env_cfg()
  _configure_lite3(cfg)
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.commands = {"velocity": VelocityGaitCommandCfg(resampling_time_range=(4.0, 8.0), debug_vis=False)}
  cfg.observations = {"actor": ObservationGroupCfg(
    terms={
      "base_ang_vel": ObservationTermCfg(func=vel_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"}, noise=Unoise(n_min=-0.2, n_max=0.2)),
      "projected_gravity": ObservationTermCfg(func=vel_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05)),
      "joint_pos": ObservationTermCfg(func=vel_mdp.joint_pos_rel, params={"biased": True},
        noise=Unoise(n_min=-0.01, n_max=0.01)),
      "joint_vel": ObservationTermCfg(func=vel_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)),
      "actions": ObservationTermCfg(func=vel_mdp.last_action),
      "velocity_gait": ObservationTermCfg(func=vel_mdp.generated_commands, params={"command_name": "velocity"}),
    }, concatenate_terms=True, enable_corruption=not play, history_length=ACTOR_HISTORY,
  )}
  cfg.rewards = {
    "locomotion": RewardTermCfg(func=locomotion_reward, weight=1.0),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated,
      weight=cfg.commands["velocity"].reward_weights.terminated),
  }
  cfg.terminations["fell_over"] = TerminationTermCfg(func=vel_mdp.bad_orientation,
    params={"limit_angle": math.radians(60.0)})
  cfg.terminations["low_body"] = TerminationTermCfg(func=low_body)
  cfg.curriculum = {}
  return cfg
