"""Bounds for Lite3 velocity-task joint position targets."""
from lite3_supermimic.robot.constants import HIP_X_JOINT_INDICES


def action_bounds(default_q, scale, soft_limits, hipx_target_limit):
  """Convert joint position bounds in radians to normalized action bounds."""
  low = soft_limits[..., 0] - default_q
  high = soft_limits[..., 1] - default_q
  low[..., HIP_X_JOINT_INDICES] = low[..., HIP_X_JOINT_INDICES].clamp_min(-hipx_target_limit)
  high[..., HIP_X_JOINT_INDICES] = high[..., HIP_X_JOINT_INDICES].clamp_max(hipx_target_limit)
  return low / scale, high / scale
