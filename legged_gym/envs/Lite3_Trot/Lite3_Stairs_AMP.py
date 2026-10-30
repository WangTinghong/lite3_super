import torch

from legged_gym.envs.Lite3_Trot.Lite3_Stairs import Lite3_Stairs_Robot
from legged_gym.envs.Lite3_Trot.Lite3_Stairs_AMP_config import Lite3_Stairs_AMP_Cfg


class Lite3_Stairs_AMP_Robot(Lite3_Stairs_Robot):
    amp_observation_dim = 30

    def __init__(
        self,
        cfg: Lite3_Stairs_AMP_Cfg,
        sim_params,
        physics_engine,
        sim_device,
        headless,
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        stand_ratio = float(self.cfg.commands.stand_env_ratio)
        if not 0.0 <= stand_ratio <= 1.0:
            raise ValueError(f"stand_env_ratio must be in [0, 1], got {stand_ratio}")

        stand_count = int(round(self.num_envs * stand_ratio))
        if stand_ratio > 0.0 and stand_count == 0:
            stand_count = 1
        self.stand_env_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if stand_count > 0:
            stand_ids = torch.randperm(self.num_envs, device=self.device)[:stand_count]
            self.stand_env_mask[stand_ids] = True
            self.commands[self.stand_env_mask] = 0.0

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        # Heading mode rewrites yaw every step, so clear all command dimensions
        # after the parent callback for the dedicated stationary cohort.
        self.commands[self.stand_env_mask] = 0.0

    def get_amp_observations(self):
        return torch.cat(
            (self.dof_pos, self.base_lin_vel, self.base_ang_vel, self.dof_vel), dim=-1
        )

    def _stationary_mask(self):
        return (
            torch.linalg.vector_norm(self.commands[:, :3], dim=-1)
            < self.cfg.rewards.stand_command_threshold
        )

    def _feet_contact_mask(self):
        return (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.stand_contact_force_threshold
        )
    
        ## 接触奖励
    def _reward_stand_contact(self):
        all_feet_contact = torch.all(self._feet_contact_mask(), dim=-1)
        return all_feet_contact.float() * self._stationary_mask().float()
        ## 缺少接触惩罚
    def _reward_stand_missing_contact(self):
        missing_contact = torch.any(~self._feet_contact_mask(), dim=-1)
        return missing_contact.float() * self._stationary_mask().float()
        ## 关节对称性奖励
    def _reward_stand_joint_symmetry(self):
        dof_pos = self.dof_pos.view(self.num_envs, 4, 3)
        left = dof_pos[:, (0, 2)]
        right = dof_pos[:, (1, 3)]
        symmetry_error = torch.cat(
            (
                left[:, :, 0] + right[:, :, 0],
                left[:, :, 1] - right[:, :, 1],
                left[:, :, 2] - right[:, :, 2],
            ),
            dim=-1,
        )
        near_level = (
            torch.linalg.vector_norm(self.projected_gravity[:, :2], dim=-1)
            < self.cfg.rewards.stand_symmetry_gravity_threshold
        )
        active = self._stationary_mask() & near_level
        return torch.sum(torch.abs(symmetry_error), dim=-1) * active.float()
