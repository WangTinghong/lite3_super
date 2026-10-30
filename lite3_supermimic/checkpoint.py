"""Reject shape-compatible checkpoints trained with different robot conventions."""
from lite3_supermimic.robot.constants import JOINT_NAMES


def validate_checkpoint(checkpoint):
    if checkpoint.get("robot") != "lite3":
        raise ValueError("Expected a Lite3 SuperMimic checkpoint; Go2/PPO weights are incompatible.")
    if tuple(checkpoint.get("joint_names", ())) != JOINT_NAMES:
        raise ValueError("Lite3 checkpoint joint order does not match FL/FR/HL/HR, HipX/HipY/Knee.")
