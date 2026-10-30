"""Lite3 native conventions shared by assets, learner and checkpoint checks."""
LEGS = ("FL", "FR", "HL", "HR")
JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in LEGS
                    for joint in ("HipX", "HipY", "Knee"))
DEFAULT_JOINT_POS = {".*_HipX_joint": 0.0, ".*_HipY_joint": -1.0, ".*_Knee_joint": 1.8}
STIFFNESS = 20.0
DAMPING = 0.7
EFFORT_LIMIT = 30.0  # Nm, local Lite3 URDF
VELOCITY_LIMIT = 20.0  # rad/s, local Lite3 URDF
ACTION_SCALE = 0.25
FOOT_GEOM_NAMES = tuple(f"{leg}_FOOT_collision" for leg in ("FR", "FL", "HR", "HL"))
