"""Four training stages. Speed limits are experiment settings, not validated capabilities."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Stage:
    terrain: bool
    forward: float
    lateral: float
    yaw: float
    interval: tuple[float, float] = (2.0, 5.0)
    backward: float = -0.5
    stand_fraction: float = 0.15
    initial_forward: float = 0.5
    terrain_speed_reduction: float = 0.0


STAGES = {
    "flat": Stage(False, 1.0, 0.3, 1.0),
    "terrain": Stage(True, 1.5, 0.3, 1.0),
    "fast": Stage(False, 3.0, 0.4, 1.0, initial_forward=1.0),
    "adaptive": Stage(True, 3.0, 0.4, 1.0, interval=(1.0, 3.0),
                      initial_forward=1.0, terrain_speed_reduction=0.15),
}
TASKS = {name: "Lite3-Locomotion-" + name.capitalize() for name in STAGES}
SCHEMA = "lite3-locomotion-v1"
HISTORY = 5


@dataclass(frozen=True)
class Rewards:
    track_linear: float = 2.0
    track_yaw: float = 1.0
    upright: float = 0.5
    base_height: float = -2.0
    roll_pitch_rate: float = -0.03
    vertical_velocity: float = -0.1
    diagonal_contact: float = 0.0
    all_airborne: float = 0.0
    hipx_swing_outward: float = 0.0
    action_rate: float = -0.02
    joint_limits: float = -1.0
    joint_torques: float = -2e-5
    joint_acceleration: float = -2.5e-7
    foot_slip: float = -0.05
    foot_clearance: float = -0.5
    stand_still: float = -0.1
    terminated: float = -200.0


WEIGHTS = Rewards()


def gait_refine_weights() -> Rewards:
    """Flat-ground, moderate-speed experiment; original task keeps WEIGHTS."""
    return replace(WEIGHTS, vertical_velocity=-0.8,
                   diagonal_contact=-1.5, all_airborne=-0.8)


def hipx_refine_weights() -> Rewards:
    """Keep diagonal timing while discouraging excessive outward swing."""
    return replace(gait_refine_weights(), hipx_swing_outward=-4.0)
