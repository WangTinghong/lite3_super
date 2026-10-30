import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo_dreamwaq import PPO_DreamWaQ
from rsl_rl.modules import AMPDiscriminator, AMPObservationNormalizer
from rsl_rl.storage import AMPReplayBuffer


class PPO_DreamWaQAMP(PPO_DreamWaQ):
    """DreamWaQ PPO with a separately optimized, low-weight AMP reward."""

    def __init__(
        self,
        actor_critic,
        amp_data,
        amp_reward_coef=0.005,
        amp_warmup_iterations=100,
        amp_ramp_iterations=900,
        amp_replay_buffer_size=200000,
        amp_discriminator_hidden_dims=(256, 128),
        amp_discriminator_learning_rate=1e-4,
        amp_num_discriminator_updates=4,
        amp_discriminator_batch_size=4096,
        amp_gradient_penalty=10.0,
        device="cpu",
        **kwargs,
    ):
        super().__init__(actor_critic, device=device, **kwargs)
        self.amp_data = amp_data
        self.amp_reward_coef = float(amp_reward_coef)
        self.amp_warmup_iterations = int(amp_warmup_iterations)
        self.amp_ramp_iterations = int(amp_ramp_iterations)
        self.amp_num_discriminator_updates = int(amp_num_discriminator_updates)
        self.amp_discriminator_batch_size = int(amp_discriminator_batch_size)
        self.amp_gradient_penalty = float(amp_gradient_penalty)
        self.amp_reward_scale = 0.0

        self.discriminator = AMPDiscriminator(
            amp_data.observation_dim, amp_discriminator_hidden_dims
        ).to(self.device)
        self.amp_normalizer = AMPObservationNormalizer(amp_data.observation_dim).to(self.device)
        self.amp_normalizer.update(torch.cat((amp_data.states, amp_data.next_states), dim=0))
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=amp_discriminator_learning_rate,
            weight_decay=1e-4,
        )
        self.amp_storage = AMPReplayBuffer(
            amp_data.observation_dim, amp_replay_buffer_size, self.device
        )
        self.current_amp_observations = None
        self.last_amp_reward_mean = 0.0
        self.last_task_reward_mean = 0.0

    def set_amp_reward_scale(self, amp_training_iteration: int) -> float:
        if amp_training_iteration < self.amp_warmup_iterations:
            scale = 0.0
        elif self.amp_ramp_iterations <= 0:
            scale = self.amp_reward_coef
        else:
            ramp_step = amp_training_iteration - self.amp_warmup_iterations + 1
            ramp_fraction = min(max(ramp_step / self.amp_ramp_iterations, 0.0), 1.0)
            scale = self.amp_reward_coef * ramp_fraction
        self.amp_reward_scale = scale
        return scale

    def act(self, obs, critic_obs, obs_history, estimation, amp_observations):
        self.current_amp_observations = amp_observations.detach()
        return super().act(obs, critic_obs, obs_history, estimation)

    def process_env_step(self, rewards, dones, infos, next_amp_observations):
        if self.current_amp_observations is None:
            raise RuntimeError("AMP observations must be supplied to act() before process_env_step()")

        next_amp_observations = next_amp_observations.detach()
        valid_transition = ~dones.bool().reshape(-1)
        normalized_state = self.amp_normalizer.normalize(self.current_amp_observations)
        normalized_next_state = self.amp_normalizer.normalize(next_amp_observations)
        amp_rewards = self.discriminator.predict_reward(normalized_state, normalized_next_state)
        amp_rewards = amp_rewards * valid_transition.float()

        if torch.any(valid_transition):
            self.amp_storage.insert(
                self.current_amp_observations[valid_transition],
                next_amp_observations[valid_transition],
            )

        combined_rewards = rewards + self.amp_reward_scale * amp_rewards
        self.last_task_reward_mean = rewards.mean().item()
        self.last_amp_reward_mean = amp_rewards.mean().item()
        super().process_env_step(combined_rewards, dones, infos)
        self.current_amp_observations = None
        return combined_rewards, amp_rewards

    def update(self):
        amp_metrics = self._update_discriminator()
        ppo_metrics = super().update()
        return ppo_metrics + amp_metrics

    def _update_discriminator(self):
        if self.amp_storage.num_samples == 0 or self.amp_num_discriminator_updates == 0:
            return 0.0, 0.0, 0.0, 0.0

        mean_loss = 0.0
        mean_gradient_penalty = 0.0
        mean_policy_prediction = 0.0
        mean_expert_prediction = 0.0

        for _ in range(self.amp_num_discriminator_updates):
            policy_state, policy_next_state = self.amp_storage.sample(
                self.amp_discriminator_batch_size
            )
            expert_state, expert_next_state = self.amp_data.sample(
                self.amp_discriminator_batch_size
            )
            self.amp_normalizer.update(
                torch.cat(
                    (policy_state, policy_next_state, expert_state, expert_next_state), dim=0
                )
            )

            policy_state = self.amp_normalizer.normalize(policy_state)
            policy_next_state = self.amp_normalizer.normalize(policy_next_state)
            expert_state = self.amp_normalizer.normalize(expert_state)
            expert_next_state = self.amp_normalizer.normalize(expert_next_state)

            policy_prediction = self.discriminator(policy_state, policy_next_state)
            expert_prediction = self.discriminator(expert_state, expert_next_state)
            policy_loss = torch.mean((policy_prediction + 1.0).square())
            expert_loss = torch.mean((expert_prediction - 1.0).square())
            discriminator_loss = 0.5 * (policy_loss + expert_loss)
            gradient_penalty = self.discriminator.gradient_penalty(
                expert_state, expert_next_state, self.amp_gradient_penalty
            )
            loss = discriminator_loss + gradient_penalty

            self.discriminator_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.discriminator_optimizer.step()

            mean_loss += discriminator_loss.item()
            mean_gradient_penalty += gradient_penalty.item()
            mean_policy_prediction += policy_prediction.mean().item()
            mean_expert_prediction += expert_prediction.mean().item()

        count = self.amp_num_discriminator_updates
        return (
            mean_loss / count,
            mean_gradient_penalty / count,
            mean_policy_prediction / count,
            mean_expert_prediction / count,
        )

