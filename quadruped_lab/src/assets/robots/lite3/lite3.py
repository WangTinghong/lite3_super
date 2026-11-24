"""Native Lite3 MJCF entity and PD control for mjlab.

Inertials/axes/limits come from the existing deployment MJCF. Gains and action
scale match Lite3_Stairs_config.py. Armature and latency are simulation assumptions,
not identified Lite3 motor parameters. The DC torque-speed envelope is approximate.
"""
from functools import partial
from pathlib import Path
import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from .constants import (
    ACTION_SCALE, DAMPING, DEFAULT_JOINT_POS, EFFORT_LIMIT, FOOT_GEOM_NAMES,
    LEGS, LEGACY_SHANK_COLLISION_RADIUS_M, STIFFNESS, VELOCITY_LIMIT,
)

LITE3_XML = Path(__file__).with_name("lite3.xml")
LITE3_ACTION_SCALE = {".*_joint": ACTION_SCALE}


def get_spec(*, shank_radius_m=LEGACY_SHANK_COLLISION_RADIUS_M) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(LITE3_XML))
    if shank_radius_m != LEGACY_SHANK_COLLISION_RADIUS_M:
        for leg in LEGS:
            spec.geom(f"{leg}_SHANK_collision").size[0] = shank_radius_m
    return spec


def get_lite3_robot_cfg(cfg) -> EntityCfg:
    foot = r"[FH][LR]_FOOT_collision"
    collision = r".*_collision"
    shank_radius = cfg.experiment.lite3_shank_collision_radius_m
    return EntityCfg(
        spec_fn=(get_spec if shank_radius == LEGACY_SHANK_COLLISION_RADIUS_M else
                 partial(get_spec, shank_radius_m=shank_radius)),
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(cfg.init_state.pos), joint_pos=dict(cfg.init_state.default_joint),
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
