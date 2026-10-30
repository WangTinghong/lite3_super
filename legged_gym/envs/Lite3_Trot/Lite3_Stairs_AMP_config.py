import os

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.Lite3_Trot.Lite3_Stairs_config import (
    Lite3_Stairs_Cfg_Yu,
    Lite3_Stairs_PPO_Yu,
)


LITE3_MOTION_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "lite3_motions")
LITE3_BASELINE_CHECKPOINT = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", "Jul26_06-36-24_", "model_23000.pt"
)
LITE3_AMP_MOTIONS = [
    os.path.join(LITE3_MOTION_DIR, name)
    for name in (
        "lite3_右移.json",
        "lite3_右转.json",
        "lite3_后退.json",
        "lite3_左移.json",
        "lite3_左转.json",
        "lite3_直行快速.json",
        "lite3_直行慢速.json",
        "lite3_静止站立.json",
    )
]


class Lite3_Stairs_AMP_Cfg(Lite3_Stairs_Cfg_Yu):
    class commands(Lite3_Stairs_Cfg_Yu.commands):
        # Keep a fixed subset stationary so zero-command behavior is trained
        # continuously without reducing the locomotion distribution too much.
        stand_env_ratio = 0.10

    class rewards(Lite3_Stairs_Cfg_Yu.rewards):
        class scales(Lite3_Stairs_Cfg_Yu.rewards.scales):
            stand_contact = 0.5
            stand_missing_contact = -1.0
            stand_joint_symmetry = -0.25

        stand_command_threshold = 0.1
        stand_contact_force_threshold = 1.0
        stand_symmetry_gravity_threshold = 0.15


class Lite3_Stairs_AMP_PPO(Lite3_Stairs_PPO_Yu):
    runner_class_name = "DreamWaQAMPRunner"

    class algorithm(Lite3_Stairs_PPO_Yu.algorithm):
        # Conservative fine-tuning settings for the already stable baseline.
        learning_rate = 1e-4
        vae_learning_rate = 1e-4
        schedule = "fixed"
        clip_param = 0.1
        entropy_coef = 0.005

        # Task rewards remain dominant: unlike environment rewards, AMP is not dt-scaled.
        amp_reward_coef = 0.005
        amp_warmup_iterations = 100
        amp_ramp_iterations = 900
        amp_replay_buffer_size = 200000
        amp_discriminator_hidden_dims = [256, 128]
        amp_discriminator_learning_rate = 1e-4
        amp_num_discriminator_updates = 4
        amp_discriminator_batch_size = 4096
        amp_gradient_penalty = 10.0

    class runner(Lite3_Stairs_PPO_Yu.runner):
        policy_class_name = "ActorCriticDreamWaQ"
        algorithm_class_name = "PPO_DreamWaQAMP"
        experiment_name = "lite3_stairs_amp"
        run_name = ""
        max_iterations = 3000
        save_interval = 100
        resume = False
        amp_motion_files = LITE3_AMP_MOTIONS
        # This full checkpoint exactly matches the currently deployed actor and VAE.
        amp_bootstrap_checkpoint = LITE3_BASELINE_CHECKPOINT
