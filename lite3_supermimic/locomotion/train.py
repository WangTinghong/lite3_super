"""Independent locomotion entry point, reusing SuperMimic replay/WM/BPTT optimisers."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from lite3_supermimic.supermimic import state as st
from lite3_supermimic.supermimic.buffer import ReplayBuffer, WindowArchive
from lite3_supermimic.supermimic.imagine import ImagineCfg
from lite3_supermimic.supermimic.train import Actor, Critic, TrainCfg as BaseCfg, Trainer as BaseTrainer, clip_grads, lr_at
from lite3_supermimic.supermimic.world_model import WorldModel
from lite3_supermimic.target import terrain as tr
from .config import (
    HISTORY, SCHEMA, STAGES, TASKS, WEIGHTS, gait_refine_weights,
    hipx_refine_weights, hipx_target_refine_weights,
    stair_refine_weights,
)
from .core import TERMS, foot_clearances, validate_checkpoint
from .amp import AMPStyle, MOTION_RATES, flatness_gate
from .imagine import Imagination


@dataclass(kw_only=True)
class TrainCfg(BaseCfg):
    device: str = "cuda:0"
    stage: str = "flat"
    gait_refine: bool = False
    hipx_refine: bool = False
    hipx_target_refine: bool = False
    amp_motion_dir: str = ""
    amp_weight: float = 0.5
    amp_start_iteration: int = 500
    amp_ramp_iterations: int = 1000
    amp_batch: int = 512
    amp_updates: int = 2
    amp_lr: float = 1e-4
    gait_anchor_weight: float = 0.0
    harder_stairs: bool = False
    stair_refine: bool = False
    task: str = TASKS["flat"]
    speed_curriculum_iterations: int = 15_000
    warmup_iterations: int = 1000
    wm_updates: int = 4
    actor_lr_scale: float = 0.3
    dodge_shaping: float = 0.0
    run_name: str = ""
    log_root: str = "logs/lite3_locomotion"

    def __post_init__(self):
        super().__post_init__()
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {tuple(STAGES)}")
        if self.gait_refine and self.stage != "flat":
            raise ValueError("--gait-refine currently supports only --stage flat")
        if self.hipx_refine and not self.gait_refine:
            raise ValueError("--hipx-refine requires --gait-refine")
        if self.hipx_target_refine and not (self.gait_refine and self.hipx_refine):
            raise ValueError("--hipx-target-refine requires --gait-refine --hipx-refine")
        if self.amp_motion_dir:
            if self.stage not in ("flat", "terrain") or (self.stage == "flat" and not self.gait_refine):
                raise ValueError("--amp-motion-dir requires flat + gait-refine or terrain")
            missing = [name for name in MOTION_RATES
                       if not (Path(self.amp_motion_dir).expanduser() / name).is_file()]
            if missing:
                raise ValueError(f"AMP motion files missing from {self.amp_motion_dir}: {missing}")
        if self.amp_weight < 0 or self.amp_start_iteration < 0 or self.amp_ramp_iterations <= 0:
            raise ValueError("AMP weight/start/ramp must be nonnegative/nonnegative/positive")
        if self.amp_batch <= 0 or self.amp_updates < 0 or self.amp_lr <= 0:
            raise ValueError("AMP batch/updates/lr must be positive/nonnegative/positive")
        if self.gait_anchor_weight < 0 or (self.gait_anchor_weight > 0 and
                                           (self.stage != "terrain" or not self.resume)):
            raise ValueError("--gait-anchor-weight requires a terrain run resumed from a checkpoint")
        if self.harder_stairs and self.stage != "terrain":
            raise ValueError("--harder-stairs requires --stage terrain")
        if self.stair_refine and (not self.harder_stairs or not self.resume or self.gait_refine):
            raise ValueError("--stair-refine requires --stage terrain --harder-stairs and --resume")
        if self.speed_curriculum_iterations <= 0:
            raise ValueError("speed_curriculum_iterations must be positive")
        if self.wm_prioritized_frac + self.wm_archive_frac > 1:
            raise ValueError("world-model sampling fractions must sum to <= 1")
        if self.driver_policy or self.expert_policy:
            raise ValueError("Locomotion uses --resume from a locomotion checkpoint, not chase experts")
        self.task = TASKS[self.stage]
        self.run_name = self.run_name or f"lite3_locomotion_{self.stage}"


class Trainer(BaseTrainer):
    def __init__(self, cfg):
        from . import env as registration  # noqa: F401
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Locomotion training requires an NVIDIA CUDA GPU")
        # Validate before constructing expensive terrain or allocating GPU buffers.
        if cfg.resume:
            checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=False)
            validate_checkpoint(checkpoint)
            changed_stage = checkpoint["cfg"]["stage"] != cfg.stage
            if bool(checkpoint["cfg"].get("gait_refine", False)) != cfg.gait_refine:
                cfg.reset_critic = True  # The reward changed; the old value target is stale.
            if bool(checkpoint["cfg"].get("hipx_refine", False)) != cfg.hipx_refine:
                cfg.reset_critic = True
            if bool(checkpoint["cfg"].get("hipx_target_refine", False)) != cfg.hipx_target_refine:
                cfg.reset_critic = True
            if bool(checkpoint["cfg"].get("amp_motion_dir", "")) != bool(cfg.amp_motion_dir):
                cfg.reset_critic = True
            if bool(checkpoint["cfg"].get("harder_stairs", False)) != cfg.harder_stairs:
                cfg.reset_critic = True
                if cfg.continue_run:
                    raise ValueError("Changing stair geometry requires a new run; omit --continue-run")
            if bool(checkpoint["cfg"].get("stair_refine", False)) != cfg.stair_refine:
                cfg.reset_critic = True
            if cfg.continue_run and changed_stage:
                raise ValueError("--continue-run requires the same stage; omit it for stage transfer")
            if changed_stage:
                cfg.reset_critic = True
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        env_cfg = load_env_cfg(cfg.task)
        if cfg.harder_stairs:
            from .env import extended_stairs
            extended_stairs(env_cfg)
        self.weights = (stair_refine_weights() if cfg.stair_refine else
                        hipx_target_refine_weights() if cfg.hipx_target_refine else
                        hipx_refine_weights() if cfg.hipx_refine else
                        gait_refine_weights() if cfg.gait_refine else WEIGHTS)
        if cfg.gait_refine or cfg.stair_refine:
            for name, value in vars(self.weights).items():
                if name != "terminated":
                    env_cfg.rewards[name].weight = value
        env_cfg.scene.num_envs, env_cfg.seed = cfg.num_envs, cfg.seed
        cmd_cfg = env_cfg.commands["velocity"]
        cmd_cfg.curriculum_steps = cfg.speed_curriculum_iterations * cfg.collect_steps
        if cfg.stair_refine:
            env_cfg.scene.terrain.max_init_terrain_level = 8
            cmd_cfg.stair_refine = True
            cmd_cfg.stair_focus_prob = 0.65
            cmd_cfg.min_terrain_level = 4
        self.env_cfg = env_cfg
        self.env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
        if cfg.stair_refine:
            terrain = self.env.scene.terrain
            terrain.terrain_levels.clamp_(min=cmd_cfg.min_terrain_level)
            terrain.env_origins[:] = terrain.terrain_origins[
                terrain.terrain_levels, terrain.terrain_types]
        self.dev = self.env.device
        self.term = self.env.command_manager.get_term("velocity")
        self.field = self.term.height_field
        self.history, self.terms = HISTORY, TERMS
        from lite3_supermimic.robot.constants import JOINT_NAMES
        if tuple(self.env.scene["robot"].joint_names) != JOINT_NAMES:
            raise ValueError("Unexpected Lite3 joint order")
        obs, _ = self.env.reset()
        self.obs = obs["actor"]
        obs_dim = HISTORY * sum(d for _, d in TERMS)
        if self.obs.shape[-1] != obs_dim:
            raise ValueError("Locomotion observation dimension mismatch")
        self.consts = st.robot_consts(self.env)
        self.amp = (AMPStyle(cfg.amp_motion_dir, self.env.step_dt, self.dev, cfg.amp_lr)
                    if cfg.amp_motion_dir else None)
        self.ctx = st.ContextReader(self.env)
        # Startup COM randomisation has already run. Its reference must be the
        # original MuJoCo model, not each world's already-randomised COM.
        base_id = int(self.ctx.base_body)
        self.ctx.base_ipos0 = torch.as_tensor(self.env.sim.mj_model.body_ipos[base_id].copy(),
                                            dtype=torch.float32, device=self.dev).expand(cfg.num_envs, 3)
        self.xi = self.ctx.read()
        self.enc_bias = self.env.scene["robot"].data.encoder_bias.clone()
        self.wm = WorldModel(cfg.wm_hidden).to(self.dev)
        self.actor = Actor(obs_dim, cfg.actor_hidden, 12).to(self.dev)
        self.anchor_actor = (Actor(obs_dim, cfg.actor_hidden, 12).to(self.dev)
                             if cfg.gait_anchor_weight > 0 else None)
        self.imag = Imagination(ImagineCfg(horizon=cfg.horizon, dt=self.env.step_dt,
                                compile=cfg.compile, sensor_grad=cfg.sensor_grad, weights=self.weights),
                                self.consts, self.term.stage, HISTORY, self.dev, self.field,
                                amp=self.amp)
        self.critic = Critic(self.imag.critic_dim, cfg.critic_hidden).to(self.dev)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.buf = ReplayBuffer(cfg.num_envs, cfg.buffer_len, obs_dim, self.dev, gen_dim=5)
        self.archive = WindowArchive(cfg.archive_size, cfg.wm_window, st.CONTEXT_DIM, self.dev)
        self.explore = torch.arange(cfg.num_envs, device=self.dev) < int(cfg.explore_frac * cfg.num_envs)
        self.ou = torch.zeros(cfg.num_envs, 12, device=self.dev)
        self.a_prev = torch.zeros_like(self.ou)
        self.ep_stats = defaultdict(list)
        self.env_steps = 0
        self.actor_frozen_until = 0
        self.policy_start = cfg.warmup_iterations
        self.lr_scale, self.ramp_start = 1.0, 0
        self.nonfinite_steps, self.start_it = 0, 0
        self.last_ckpt = None
        self.window_falls, self.window_steps = 0, 0
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        self.log_dir = Path(cfg.log_root) / f"{stamp}_{cfg.run_name}"
        self.log_dir.mkdir(parents=True, exist_ok=False)
        self.writer = SummaryWriter(str(self.log_dir))
        if cfg.resume:
            self.load(cfg.resume)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.dev, weights_only=False)
        validate_checkpoint(ckpt)
        super().load(path)
        if self.anchor_actor is not None:
            self.anchor_actor.load_state_dict(ckpt.get("gait_anchor_actor", ckpt["actor"]))
            self.anchor_actor.eval()
            self.anchor_actor.requires_grad_(False)
        if self.amp is not None and "amp_style" in ckpt:
            self.amp.load_state_dict(ckpt["amp_style"])
            if "amp_optimizer" in ckpt:
                self.amp.optimizer.load_state_dict(ckpt["amp_optimizer"])
        if self.cfg.continue_run:
            self.env.common_step_counter = ckpt.get("command_steps", self.start_it * self.cfg.collect_steps)
            self.env_steps = ckpt.get("env_steps", 0)
            # Replay is not checkpointed: refill and rewarm on restart.
            self.policy_start = self.start_it + self.cfg.warmup_iterations
            self.actor_frozen_until = max(self.actor_frozen_until, self.policy_start)
            levels = ckpt.get("terrain_levels")
            if self.field is not None and levels is not None:
                terrain = self.env.scene.terrain
                if len(levels) == self.cfg.num_envs:
                    terrain.terrain_levels[:] = levels.to(self.dev)
                    terrain.env_origins[:] = terrain.terrain_origins[
                        terrain.terrain_levels, terrain.terrain_types]
            # New command/observation must reflect the restored schedule and origins.
            self.obs = self.env.reset()[0]["actor"]

    @torch.no_grad()
    def collect(self):
        env, cfg = self.env, self.cfg
        for _ in range(cfg.collect_steps):
            s, pose = st.robot_state(env, self.field), st.planar_pose(env)
            packed = self.term.packed().clone()
            relief = self._relief(pose)
            a = self.actor(self.obs)
            dt = env.step_dt
            self.ou = (self.ou * (1 - cfg.explore_theta * dt)
                       + cfg.explore_noise * (2 * cfg.explore_theta * dt)**0.5 * torch.randn_like(a))
            a = a + torch.where(self.explore[:, None], self.ou, cfg.action_noise * torch.randn_like(a))
            obs, _, terminated, truncated, extras = env.step(a)
            done = terminated | truncated
            self.buf.add(s, pose, a, self.a_prev, self.obs, packed, done, relief)
            keep = ~done
            if keep.any():
                s1 = st.robot_state(env, self.field)
                dp = st.pose_delta(pose, st.planar_pose(env))
                self.wm.update_stats(s[keep], a[keep], self.a_prev[keep], self.xi[keep],
                                     relief[keep], s1[keep], dp[keep])
                self.ep_stats["locomotion/linear_error_mps"].append(
                    float((s1[keep, 4:6] - packed[keep, :2]).norm(dim=-1).mean()))
                self.ep_stats["locomotion/yaw_error_radps"].append(
                    float((s1[keep, 9] - packed[keep, 2]).abs().mean()))
                self.ep_stats["locomotion/speed_mps"].append(float(s1[keep, 4:6].norm(dim=-1).mean()))
            self.actor.norm.update(self.obs)
            self.window_falls += int(terminated.sum())
            self.window_steps += env.num_envs
            self.ou[done] = 0
            self.a_prev = torch.where(done[:, None], torch.zeros_like(a), a)
            self.obs = obs["actor"]
            self.env_steps += env.num_envs
            for key, val in extras.get("log", {}).items():
                self.ep_stats[key].append(float(val))

    def _relief(self, pose):
        return tr.relief(pose[:, :2], pose[:, 2], self.field)

    def _window_contacts(self, w):
        s, p = w["s"], w["pose"]
        return (foot_clearances(s.reshape(-1, st.STATE_DIM), p.reshape(-1, 3), self.field)
                .reshape(*s.shape[:2], 4) < 0.015)

    def _rarity(self, w):
        contact = self._window_contacts(w).float()
        return ((contact[:, 1:] - contact[:, :-1]).abs().sum((1, 2))
                + (contact.sum(-1) == 0).float().sum(-1)
                + 4 * (w["valid"][:, -1] < 0.5).float())

    def _archive_worthy(self, w):
        flight = (~self._window_contacts(w).any(-1)).sum(-1) >= 2
        return flight | (w["s"][:, :, 6].abs().amax(-1) > 0.6) | (w["valid"][:, -1] < 0.5)

    def save(self, it):
        path = self.log_dir / f"model_{it}.pt"
        terrain = self.env.scene.terrain
        data = dict(supermimic=True, robot="lite3", task_family=SCHEMA,
                    joint_names=list(self.env.scene["robot"].joint_names), iteration=it,
                    cfg=asdict(self.cfg), obs_terms=TERMS, history=HISTORY, obs_dim=self.obs.shape[-1],
                    actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    world_model=self.wm.state_dict(), opt_wm=self.opt_wm.state_dict(),
                    opt_actor=self.opt_actor.state_dict(), opt_critic=self.opt_critic.state_dict(),
                    env_steps=self.env_steps, command_steps=self.env.common_step_counter,
                    terrain_levels=terrain.terrain_levels if self.field is not None else None)
        if self.amp is not None:
            data["amp_style"] = self.amp.state_dict()
            data["amp_optimizer"] = self.amp.optimizer.state_dict()
        if self.anchor_actor is not None:
            data["gait_anchor_actor"] = self.anchor_actor.state_dict()
        temporary = path.with_suffix(".tmp")
        torch.save(data, temporary)
        temporary.replace(path)
        return path

    def retain_gait(self) -> dict[str, float]:
        """Rehearse the flat AMP actor on real, level, forward terrain states."""
        cfg = self.cfg
        env, slot = self.buf.sample_states(cfg.imag_batch)
        command = self.buf.gen[env, slot, :3]
        relief = self.buf.relief[env, slot].float()
        keep = ((command[:, 0] >= 0.4) & (command[:, 1].abs() <= 0.15)
                & (command[:, 2].abs() <= 0.3) & (flatness_gate(relief) > 0.5))
        if not bool(keep.any()):
            return {"anchor_batch": 0.0}
        obs = self.buf.obs[env[keep], slot[keep]].float()
        with torch.no_grad():
            target = self.anchor_actor(obs)
        loss = (self.actor(obs) - target).square().mean()
        if self.it > self.actor_frozen_until:
            self.opt_actor.zero_grad(set_to_none=True)
            (cfg.gait_anchor_weight * loss).backward()
            norm = clip_grads(self.actor.parameters(), cfg.grad_clip, cfg.grad_clip_elem)
            if bool(torch.isfinite(norm)):
                self.opt_actor.step()
        return {"anchor_mse": float(loss.detach()), "anchor_batch": float(len(obs))}

    def run(self):
        cfg, started = self.cfg, time.time()
        try:
            while self.buf.size <= cfg.wm_window + 1:
                self.collect()
            for it in range(self.start_it + 1, cfg.iterations + 1):
                self.it = it
                start = max(self.actor_frozen_until, self.policy_start)
                ramp = min(1.0, max(0, it - start) / max(1, cfg.actor_ramp_iterations))
                for opt, scale in ((self.opt_wm, 1), (self.opt_actor, ramp * cfg.actor_lr_scale),
                                   (self.opt_critic, 1)):
                    for group in opt.param_groups:
                        group["lr"] = lr_at(cfg, it) * scale
                self.collect()
                stats = {"wm_loss": self.fit_world_model()}
                if self.amp is not None:
                    stats.update(self.amp.fit(self.buf, self.consts.default_q,
                                              cfg.amp_batch, cfg.amp_updates))
                    progress = max(0.0, (it - cfg.amp_start_iteration) / cfg.amp_ramp_iterations)
                    self.imag.amp_scale = cfg.amp_weight * min(1.0, progress)
                    stats["amp_scale"] = self.imag.amp_scale
                if it > self.policy_start:
                    stats.update(self.improve_policy())
                    if self.anchor_actor is not None:
                        stats.update(self.retain_gait())
                    if self.amp is not None:
                        stats["amp_imagined_reward"] = self.imag.last_amp_reward
                if it % cfg.log_interval == 0:
                    mins = (time.time() - started) / 60
                    fall_rate = self.window_falls / max(1, self.window_steps) / self.env.step_dt * 60
                    line = [f"it {it}/{cfg.iterations}", f"{mins:.1f} min", f"falls/min {fall_rate:.3g}"]
                    for key, val in stats.items():
                        self.writer.add_scalar(f"train/{key}", val, it)
                        line.append(f"{key} {val:.4g}")
                    for key, vals in self.ep_stats.items():
                        val = sum(vals) / len(vals)
                        self.writer.add_scalar(key, val, it)
                        if key.startswith("locomotion/"):
                            line.append(f"{key.split('/')[-1]} {val:.3g}")
                    self.writer.add_scalar("locomotion/falls_per_minute", fall_rate, it)
                    self.writer.add_scalar("train/nonfinite_skipped", self.nonfinite_steps, it)
                    self.writer.add_scalar("time/minutes", mins, it)
                    self.writer.add_scalar("steps/environment", self.env_steps, it)
                    self.ep_stats.clear()
                    self.window_falls, self.window_steps = 0, 0
                    print(" | ".join(line), flush=True)
                if it % cfg.save_interval == 0 or it == cfg.iterations:
                    self.last_ckpt = self.save(it)
                    print(f"saved {self.last_ckpt}", flush=True)
        finally:
            self.writer.close()
            self.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = asdict(TrainCfg())
    hidden = {"task", "terrain_curriculum_iterations", "target_curriculum_iterations",
              "collapse_falls", "expert_policy", "expert_frac"}
    for name, value in defaults.items():
        if name in hidden or name.startswith(("driver_", "dodge_")):
            continue
        kwargs = {"default": value}
        if isinstance(value, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        elif isinstance(value, tuple):
            kwargs.update(type=int, nargs="+")
        else:
            kwargs["type"] = type(value)
        if name == "stage":
            kwargs["choices"] = tuple(STAGES)
        parser.add_argument("--" + name.replace("_", "-"), **kwargs)
    opts = vars(parser.parse_args())
    cfg = TrainCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in opts.items()})
    Trainer(cfg).run()
