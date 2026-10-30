from legged_gym.envs.GO2_Trot.GO2_Stairs_config import GO2_Stairs_Cfg_Yu, GO2_Stairs_PPO_Yu


class Lite3_Stairs_Cfg_Yu(GO2_Stairs_Cfg_Yu):
    class init_state(GO2_Stairs_Cfg_Yu.init_state):
        pos = [0.0, 0.0, 0.32]
        default_joint_angles = {
            "FL_HipX_joint": 0.0,
            "FL_HipY_joint": -1.0,
            "FL_Knee_joint": 1.8,

            "FR_HipX_joint": 0.0,
            "FR_HipY_joint": -1.0,
            "FR_Knee_joint": 1.8,

            "HL_HipX_joint": 0.0,
            "HL_HipY_joint": -1.0,
            "HL_Knee_joint": 1.8,

            "HR_HipX_joint": 0.0,
            "HR_HipY_joint": -1.0,
            "HR_Knee_joint": 1.8,
        }

    class control(GO2_Stairs_Cfg_Yu.control):
        stiffness = {"joint": 20.0}
        damping = {"joint": 0.7}
        action_scale = 0.25
        decimation = 4

    class asset(GO2_Stairs_Cfg_Yu.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/lite3/urdf/Lite3.urdf"
        name = "lite3"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "SHANK"]
        terminate_after_contacts_on = ["TORSO"]
        self_collisions = 1

    class rewards(GO2_Stairs_Cfg_Yu.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.36
        clearance_height_target = -0.27
        still_all = True
        only_positive_rewards = True
        pitch_roll_factor = [1, 1]

    class normalization(GO2_Stairs_Cfg_Yu.normalization):
        dof_history_interval = 1
        clip_angles = [[-0.523, 0.523], [-0.314, 3.6], [-2.792, -0.524]]

    class noise(GO2_Stairs_Cfg_Yu.noise):
        heights_uniform_noise = False
        heights_gaussian_mean_mutable = True
        heights_downgrade_frequency = False

    class commands(GO2_Stairs_Cfg_Yu.commands):
        fixed_commands = None
        resampling_time = 6.0

    class domain_rand(GO2_Stairs_Cfg_Yu.domain_rand):
        friction_range = [0.1, 1.25]
        added_base_mass_range = [-1.0, 3.0]
        added_base_com_range = [-0.03, 0.03]


class Lite3_Stairs_PPO_Yu(GO2_Stairs_PPO_Yu):
    class algorithm(GO2_Stairs_PPO_Yu.algorithm):
        symmetry_loss_coef = 0.0

    class runner(GO2_Stairs_PPO_Yu.runner):
        experiment_name = "lite3_stairs_raw"
        resume = False
