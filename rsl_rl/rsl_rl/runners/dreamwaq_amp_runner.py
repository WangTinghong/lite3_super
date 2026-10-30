import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms import PPO_DreamWaQAMP
from rsl_rl.datasets import Lite3AMPLoader
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCriticDreamWaQ
from rsl_rl.runners.dreamwaq_runner import DreamWaQRunner


class DreamWaQAMPRunner(DreamWaQRunner):
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        expected_dof_names = [
            f"{leg}_{joint}_joint"
            for leg in ("FL", "FR", "HL", "HR")
            for joint in ("HipX", "HipY", "Knee")
        ]
        if list(self.env.dof_names) != expected_dof_names:
            raise ValueError(
                "Lite3 AMP requires DOF order FL/FR/HL/HR with HipX/HipY/Knee; "
                f"got {list(self.env.dof_names)}"
            )

        num_critic_obs = self.env.num_privileged_obs
        history_dim = self.env.cfg.env.num_history_obs
        actor_critic = ActorCriticDreamWaQ(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_actions,
            history_dim,
            self.env.cfg.env.num_latent_dims,
            self.env.cfg.env.num_explicit_dims,
            **self.policy_cfg,
        ).to(self.device)

        amp_data = Lite3AMPLoader(
            motion_files=self.cfg["amp_motion_files"],
            transition_dt=self.env.dt,
            device=self.device,
        )
        if amp_data.observation_dim != self.env.amp_observation_dim:
            raise ValueError(
                f"Expert AMP width {amp_data.observation_dim} does not match environment "
                f"width {self.env.amp_observation_dim}"
            )
        self.alg = PPO_DreamWaQAMP(
            actor_critic, amp_data=amp_data, device=self.device, **self.alg_cfg
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [num_critic_obs],
            [history_dim],
            [self.env.cfg.env.num_explicit_dims],
            [self.env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.amp_training_iteration = 0
        self.env.reset()

        bootstrap_checkpoint = self.cfg.get("amp_bootstrap_checkpoint")
        if bootstrap_checkpoint and not self.cfg.get("resume", False):
            bootstrap_checkpoint = os.path.abspath(os.path.expanduser(bootstrap_checkpoint))
            print(f"Bootstrapping Lite3 AMP from: {bootstrap_checkpoint}")
            self.load(bootstrap_checkpoint, load_optimizer=False)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, obs_hist, estimation = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        amp_obs = self.env.get_amp_observations()

        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)
        obs_hist = obs_hist.to(self.device)
        estimation = estimation.to(self.device)
        amp_obs = amp_obs.to(self.device)
        
        self.alg.actor_critic.train()
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)

        total_iterations = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, total_iterations):
            amp_reward_scale = self.alg.set_amp_reward_scale(self.amp_training_iteration)
            rollout_amp_reward = 0.0
            rollout_task_reward = 0.0
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, obs_hist, estimation, amp_obs)
                    (
                        obs,
                        privileged_obs,
                        obs_hist,
                        estimation,
                        task_rewards,
                        dones,
                        infos,
                    ) = self.env.step(actions)
                    next_amp_obs = self.env.get_amp_observations()
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    obs_hist = obs_hist.to(self.device)
                    estimation = estimation.to(self.device)
                    task_rewards = task_rewards.to(self.device)
                    dones = dones.to(self.device)
                    next_amp_obs = next_amp_obs.to(self.device)

                    rewards, amp_rewards = self.alg.process_env_step(
                        task_rewards, dones, infos, next_amp_obs
                    )
                    amp_obs = next_amp_obs
                    rollout_amp_reward += amp_rewards.mean().item()
                    rollout_task_reward += task_rewards.mean().item()

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                collection_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(critic_obs)

            update_metrics = self.alg.update()
            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_vae_loss,
                mean_vel_estimation_loss,
                mean_vel_estimation_mae,
                mean_vel_estimation_mae_x,
                mean_vel_estimation_mae_y,
                mean_vel_estimation_mae_z,
                mean_recon_loss,
                mean_kl_loss,
                mean_ppo_kl,
                mean_symmetry_loss,
                mean_amp_loss,
                mean_amp_gradient_penalty,
                mean_amp_policy_prediction,
                mean_amp_expert_prediction,
            ) = update_metrics
            learn_time = time.time() - start
            mean_amp_reward = rollout_amp_reward / self.num_steps_per_env
            mean_task_reward = rollout_task_reward / self.num_steps_per_env
            self.amp_training_iteration += 1

            if self.log_dir is not None:
                self.log(locals())
            if self.log_dir is not None and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"), iteration=it)
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        if self.log_dir is not None:
            self.save(
                os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
            )

    def log(self, locs, width=80, pad=35):
        if self.writer is not None:
            iteration = locs["it"]
            self.writer.add_scalar("AMP/reward_scale", locs["amp_reward_scale"], iteration)
            self.writer.add_scalar("AMP/raw_reward", locs["mean_amp_reward"], iteration)
            self.writer.add_scalar("AMP/task_reward", locs["mean_task_reward"], iteration)
            self.writer.add_scalar("AMP/discriminator_loss", locs["mean_amp_loss"], iteration)
            self.writer.add_scalar(
                "AMP/gradient_penalty", locs["mean_amp_gradient_penalty"], iteration
            )
            self.writer.add_scalar(
                "AMP/policy_prediction", locs["mean_amp_policy_prediction"], iteration
            )
            self.writer.add_scalar(
                "AMP/expert_prediction", locs["mean_amp_expert_prediction"], iteration
            )
        super().log(locs, width=width, pad=pad)
        weighted_amp_reward = locs["amp_reward_scale"] * locs["mean_amp_reward"]
        print(
            f"{'AMP reward scale:':>{pad}} {locs['amp_reward_scale']:.6f}\n"
            f"{'Mean raw AMP reward:':>{pad}} {locs['mean_amp_reward']:.4f}\n"
            f"{'Mean weighted AMP reward:':>{pad}} {weighted_amp_reward:.6f}\n"
            f"{'AMP discriminator loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"
            f"{'AMP gradient penalty:':>{pad}} {locs['mean_amp_gradient_penalty']:.4f}\n"
            f"{'AMP policy prediction:':>{pad}} {locs['mean_amp_policy_prediction']:.4f}\n"
            f"{'AMP expert prediction:':>{pad}} {locs['mean_amp_expert_prediction']:.4f}"
        )

    def save(self, path, infos=None, iteration=None):
        if iteration is None:
            iteration = self.current_learning_iteration
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "vae_optimizer_state_dict": self.alg.vae_optimizer.state_dict(),
                "learning_rate": self.alg.learning_rate,
                "amp_discriminator_state_dict": self.alg.discriminator.state_dict(),
                "amp_discriminator_optimizer_state_dict": self.alg.discriminator_optimizer.state_dict(),
                "amp_normalizer_state_dict": self.alg.amp_normalizer.state_dict(),
                "amp_training_iteration": self.amp_training_iteration,
                "iter": iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        if "model_state_dict" not in loaded_dict:
            raise KeyError(f"Checkpoint does not contain model_state_dict: {path}")

        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            if "optimizer_state_dict" in loaded_dict:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "vae_optimizer_state_dict" in loaded_dict:
                self.alg.vae_optimizer.load_state_dict(
                    loaded_dict["vae_optimizer_state_dict"]
                )
            if "learning_rate" in loaded_dict:
                self.alg.learning_rate = loaded_dict["learning_rate"]
                for parameter_group in self.alg.optimizer.param_groups:
                    parameter_group["lr"] = self.alg.learning_rate

        has_amp_state = "amp_discriminator_state_dict" in loaded_dict
        if has_amp_state:
            self.alg.discriminator.load_state_dict(
                loaded_dict["amp_discriminator_state_dict"]
            )
            if load_optimizer and "amp_discriminator_optimizer_state_dict" in loaded_dict:
                self.alg.discriminator_optimizer.load_state_dict(
                    loaded_dict["amp_discriminator_optimizer_state_dict"]
                )
            if "amp_normalizer_state_dict" in loaded_dict:
                self.alg.amp_normalizer.load_state_dict(
                    loaded_dict["amp_normalizer_state_dict"]
                )
            self.amp_training_iteration = int(
                loaded_dict.get("amp_training_iteration", 0)
            )
        else:
            self.amp_training_iteration = 0
            print("Loaded DreamWaQ baseline checkpoint; AMP discriminator will warm up from zero.")

        self.current_learning_iteration = self._resolve_loaded_iteration(
            path, loaded_dict.get("iter", 0)
        )
        return loaded_dict.get("infos")
