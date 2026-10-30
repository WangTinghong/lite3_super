from legged_gym.envs.GO2_Trot.GO2_Stairs import GO2_Stairs_Robot
from legged_gym.envs.Lite3_Trot.Lite3_Stairs_config import Lite3_Stairs_Cfg_Yu


class Lite3_Stairs_Robot(GO2_Stairs_Robot):
    def __init__(self, cfg: Lite3_Stairs_Cfg_Yu, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
