"""Opt-in mjlab registration. Original Chase configs/registration remain untouched."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
import torch
from mjlab.managers import CurriculumTermCfg, TerminationTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg

from lite3_supermimic.env.env_cfg import lite3_chase_env_cfg
from lite3_supermimic.robot.constants import SHANK_GEOM_NAMES
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.target import terrain as tr
from .config import HISTORY, STAGES, TASKS, WEIGHTS
from .core import reward_terms, sample_commands


@dataclass(kw_only=True)
class VelocityCommandCfg(CommandTermCfg):
    stage: str = "flat"
    curriculum_steps: int = 15_000
    fixed_command: tuple[float, float, float] | None = None
    fixed_level: int | None = None
    field_res: float = 0.02
    stair_refine: bool = False
    stair_hipx_refine: bool = False
    stair_focus_prob: float = 0.0
    min_terrain_level: int = 0
    cache_dir: str = "logs/locomotion_heightfields"

    def build(self, env):
        return VelocityCommand(self, env)


class VelocityCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.stage = replace(STAGES[cfg.stage], stair_focus_prob=cfg.stair_focus_prob)
        self.robot = env.scene["robot"]
        self.height_field = None
        if self.stage.terrain:
            exclude = {int(g) for g in self.robot.indexing.geom_ids}
            self.height_field = tr.env_height_field(env, exclude, 6.5, cfg.field_res, cfg.cache_dir)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.vmax = torch.full((self.num_envs,), self.stage.initial_forward, device=self.device)
        self.prev_s = torch.zeros(self.num_envs, st.STATE_DIM, device=self.device)
        self.prev_pose = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_action = torch.zeros(self.num_envs, 12, device=self.device)
        self.count = torch.zeros(self.num_envs, device=self.device)
        self.error_sum = torch.zeros_like(self.count)
        self.distance = torch.zeros_like(self.count)
        self._reward_step = -1
        self._reward_cache = {}
        self.consts = None  # Action manager is constructed after command manager.

    @property
    def command(self):
        return self._command

    def packed(self):
        return torch.cat([self.command, self.time_left[:, None], self.vmax[:, None]], -1)

    def scan(self):
        s = st.robot_state(self._env, self.height_field)
        pose = st.planar_pose(self._env)
        return s[:, st.Z] + tr.relief(pose[:, :2], pose[:, 2], self.height_field)

    def _resample_command(self, env_ids):
        progress = min(1.0, self._env.common_step_counter / max(1, self.cfg.curriculum_steps))
        top = self.stage.initial_forward + progress * (self.stage.forward - self.stage.initial_forward)
        limit = torch.full((len(env_ids),), top, device=self.device)
        if self.stage.terrain:
            level = self._env.scene.terrain.terrain_levels[env_ids]
            limit *= (1 - self.stage.terrain_speed_reduction * level).clamp(min=0.35)
        self.vmax[env_ids] = limit
        if self.cfg.fixed_command is None:
            self._command[env_ids] = sample_commands(len(env_ids), limit, self.stage)
        else:
            self._command[env_ids] = torch.tensor(self.cfg.fixed_command, device=self.device)

    def _update_metrics(self):
        pass  # Accumulate at reward computation, including the actual terminal step.

    def _update_command(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        self.prev_s[ids] = st.robot_state(self._env, self.height_field)[ids]
        self.prev_pose[ids] = st.planar_pose(self._env)[ids]
        self.prev_action[ids] = self._env.action_manager.action[ids]

    def reset(self, env_ids):
        extras = super().reset(env_ids)
        valid = self.count[env_ids] > 0
        if valid.any():
            ids = env_ids[valid]
            extras["linear_error_mps"] = (self.error_sum[ids] / self.count[ids]).mean().item()
            extras["distance_m"] = self.distance[ids].mean().item()
        self.count[env_ids] = 0
        self.error_sum[env_ids] = 0
        self.distance[env_ids] = 0
        return extras

    def rewards(self):
        env = self._env
        if self._reward_step != env.common_step_counter:
            if self.consts is None:
                self.consts = st.robot_consts(env)
            s, pose = st.robot_state(env, self.height_field), st.planar_pose(env)
            self._reward_cache = reward_terms(
                self.prev_s, s, self.prev_pose, pose, env.action_manager.action,
                self.prev_action, self.command, self.consts, env.step_dt, self.height_field,
                stair_refine=self.cfg.stair_refine,
                stair_hipx_refine=self.cfg.stair_hipx_refine)
            self.count += 1
            self.error_sum += (s[:, 4:6] - self.command[:, :2]).norm(dim=-1)
            self.distance += (pose[:, :2] - self.prev_pose[:, :2]).norm(dim=-1)
            self._reward_step = env.common_step_counter
        return self._reward_cache

    def _debug_vis_impl(self, visualizer):
        data = self.robot.data
        for i in visualizer.get_env_indices(self.num_envs):
            start = data.root_link_pos_w[i].cpu().numpy() + np.array([0, 0, 0.15])
            yaw = float(data.heading_w[i])
            vx, vy = self.command[i, :2].cpu().tolist()
            delta = np.array([math.cos(yaw) * vx - math.sin(yaw) * vy,
                              math.sin(yaw) * vx + math.cos(yaw) * vy, 0])
            visualizer.add_arrow(start, start + delta * 0.4, (0.1, 0.8, 0.2, 1), width=0.02)
            v = data.root_link_lin_vel_w[i].cpu().numpy().copy()
            v[2] = 0
            visualizer.add_arrow(start, start + v * 0.4, (1, 0.85, 0.1, 1), width=0.02)


def scan(env):
    return env.command_manager.get_term("velocity").scan()


def reward(env, name):
    return env.command_manager.get_term("velocity").rewards()[name]


def fell_over(env):
    term = env.command_manager.get_term("velocity")
    s = st.robot_state(env, term.height_field)
    return (-s[:, 3] < math.cos(math.radians(70))) | (s[:, 0] < 0.12)


def tile_boundary(env):
    xy = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return xy.abs().amax(-1) > 5.0


def terrain_curriculum(env, env_ids):
    """Promote on successful travel/tracking; demote on falls; fixed eval overrides both."""
    terrain = env.scene.terrain
    term = env.command_manager.get_term("velocity")
    levels = terrain.terrain_levels[env_ids].clone()
    if term.cfg.fixed_level is not None:
        levels.fill_(term.cfg.fixed_level)
    else:
        seen = term.count[env_ids] > 20
        error = term.error_sum[env_ids] / term.count[env_ids].clamp(min=1)
        failed = env.termination_manager.terminated[env_ids]
        good = seen & ~failed & (error < 0.35) & (term.distance[env_ids] > 3.0)
        levels += good.long() - (seen & failed).long()
        levels.clamp_(getattr(term.cfg, "min_terrain_level", 0),
                      terrain.terrain_origins.shape[0] - 1)
    terrain.terrain_levels[env_ids] = levels
    terrain.env_origins[env_ids] = terrain.terrain_origins[levels, terrain.terrain_types[env_ids]]
    return {"mean_level": terrain.terrain_levels.float().mean()}


def extended_stairs(cfg):
    """Extend the existing stair curriculum without changing the original rows.

    Rows 0..4 remain 2..10 cm per step; rows 5..8 add 12..18 cm.
    Starting at rows 0..4 lets a transferred policy reach the new rows by
    successful travel, rather than spawning on the hardest stairs immediately.
    """
    terrain = cfg.scene.terrain
    gen = terrain.terrain_generator
    gen.num_rows = 9
    for name in ("stairs", "stairs_inv"):
        gen.sub_terrains[name].step_height_range = (0.02, 0.18)
    terrain.max_init_terrain_level = 4
    # Height-field ray casting is a CPU operation over the entire 9 x 7 tile
    # layout. The default 2 cm grid would need tens of millions of rays here.
    # A 10 cm grid still samples each 40 cm stair tread four times.
    cfg.commands["velocity"].field_res = 0.10
    return cfg


def make_env_cfg(stage="flat", play=False, harder_stairs=False):
    spec = STAGES[stage]
    # Borrow only robot/physics/sensor setup and proprioceptive observation config.
    cfg = lite3_chase_env_cfg(play=False, terrain=spec.terrain)
    # In the original simplified collision model the shank capsule protrudes
    # farther than the foot sphere. Keep the chassis/actuators unchanged, but
    # expose each foot in this task so its contact sensor measures stance.
    robot_cfg = cfg.scene.entities["robot"]
    original_spec = robot_cfg.spec_fn

    def locomotion_spec():
        model_spec = original_spec()
        for leg in ("FR", "FL", "HR", "HL"):
            shank = model_spec.geom(f"{leg}_SHANK_collision")
            shank.fromto[5] += 0.03
        return model_spec

    robot_cfg.spec_fn = locomotion_spec
    # A foot-only sensor misses stair climbing by pushing the riser with the
    # shank. Keep this diagnostic separate from policy observations and the
    # differentiable world-model reward.
    cfg.scene.sensors += (ContactSensorCfg(
        name="shanks_ground_contact",
        primary=ContactMatch(mode="geom", pattern=SHANK_GEOM_NAMES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1,
    ),)
    cfg.commands = {"velocity": VelocityCommandCfg(
        stage=stage, resampling_time_range=spec.interval, debug_vis=True)}
    actor = cfg.observations["actor"]
    actor.terms.pop("target")
    actor.terms.pop("height_scan", None)
    actor.terms["velocity_command"] = ObservationTermCfg(
        func=mdp.generated_commands, params={"command_name": "velocity"})
    actor.terms["height_scan"] = ObservationTermCfg(
        func=scan, noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02))
    actor.history_length = HISTORY
    actor.enable_corruption = not play
    cfg.rewards = {name: RewardTermCfg(func=reward, weight=weight, params={"name": name})
                   for name, weight in vars(WEIGHTS).items() if name != "terminated"}
    cfg.rewards["terminated"] = RewardTermCfg(func=mdp.is_terminated, weight=WEIGHTS.terminated)
    cfg.terminations["fell_over"] = TerminationTermCfg(func=fell_over)
    cfg.metrics = {}
    cfg.events.pop("quiet_solver_warnings", None)  # API differs between Warp versions.
    cfg.curriculum = {}
    cfg.episode_length_s = 20.0
    if stage == "adaptive":
        # Disturbed initial states are observed by the WM; do not inject hidden
        # impulses into transitions of a deterministic dynamics model.
        cfg.events["reset_base"].params["velocity_range"] = {
            "x": (-0.4, 0.4), "y": (-0.3, 0.3), "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3)}
    if spec.terrain:
        cfg.scene.terrain.max_init_terrain_level = 0
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(func=terrain_curriculum)
        cfg.terminations["tile_boundary"] = TerminationTermCfg(func=tile_boundary, time_out=True)
        if harder_stairs:
            extended_stairs(cfg)
    return cfg


def register_tasks():
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.rl import RslRlOnPolicyRunnerCfg
    for stage, name in TASKS.items():
        register_mjlab_task(task_id=name, env_cfg=make_env_cfg(stage),
                           play_env_cfg=make_env_cfg(stage, True), rl_cfg=RslRlOnPolicyRunnerCfg())


register_tasks()
