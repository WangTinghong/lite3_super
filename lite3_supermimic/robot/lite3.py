"""Native Lite3 MJCF entity and PD control for mjlab.

Inertials/axes/limits come from the existing deployment MJCF. Gains and action
scale match Lite3_Stairs_config.py. Armature and latency are simulation assumptions,
not identified Lite3 motor parameters. The DC torque-speed envelope is approximate.
"""
from pathlib import Path
import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from lite3_supermimic.robot.constants import (
    ACTION_SCALE, DAMPING, DEFAULT_JOINT_POS, EFFORT_LIMIT, FOOT_GEOM_NAMES,
    STIFFNESS, VELOCITY_LIMIT,
)

LITE3_XML = Path(__file__).with_name("lite3.xml")
LITE3_ACTION_SCALE = {".*_joint": ACTION_SCALE}


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(LITE3_XML))


def get_lite3_robot_cfg() -> EntityCfg:
    foot = r"[FH][LR]_FOOT_collision"
    collision = r".*_collision"
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.32), joint_pos=DEFAULT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(DcMotorActuatorCfg(
                target_names_expr=(".*_joint",), stiffness=STIFFNESS,
                damping=DAMPING, effort_limit=EFFORT_LIMIT,
                saturation_effort=EFFORT_LIMIT, velocity_limit=VELOCITY_LIMIT,
                armature=0.01, delay_max_lag=2,
            ),),
            soft_joint_pos_limit_factor=0.9,
        ),
        collisions=(CollisionCfg(
            geom_names_expr=(collision,), contype=1, conaffinity=1,
            solref=(0.01, 1), condim={foot: 6, collision: 1},
            priority={foot: 1, ".*": 0}, friction={foot: (1, 5e-3, 5e-4)},
        ),),
    )
