"""Shared simulator playback and metrics for Lite3 velocity policies."""
import math

import torch

from lite3_supermimic.locomotion.actions import action_bounds
from lite3_supermimic.locomotion.footfalls import summarize_footfalls, summarize_swings
from lite3_supermimic.locomotion.gait import GAITS, foot_schedule
from lite3_supermimic.robot.constants import HIP_X_JOINT_INDICES, JOINT_NAMES, LEGS
from lite3_supermimic.supermimic import state as st


class FrontTurnAssist:
  """Load-gated left-front swing assistance for evaluation playback."""

  def __init__(self, num_envs, device):
    self.used = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.engaged = torch.zeros_like(self.used)
    self.airborne = torch.zeros_like(self.used)
    self.elapsed = torch.zeros(num_envs, device=device)

  def reset(self, mask):
    self.used[mask] = False
    self.engaged[mask] = False
    self.airborne[mask] = False
    self.elapsed[mask] = 0.

  def gain(self, progress, turning, contact, vertical_load, dt):
    # Reserve the last part of swing for an unassisted touchdown.
    swing = turning & (progress > 0.) & (progress < .85)
    unloaded = vertical_load < 18.
    self.used &= swing
    self.engaged &= swing & unloaded & ~(self.airborne & contact)
    start = swing & (progress < .55) & unloaded & ~self.used
    self.used |= start
    self.engaged |= start
    self.airborne = self.engaged & (self.airborne | ~contact)
    self.elapsed = torch.where(self.engaged, self.elapsed + dt, 0.)
    ramp = (self.elapsed / .06).clamp(max=1.)
    return ramp * torch.sin(math.pi * (progress / .85).clamp(0., 1.)).square()


class VelocityEvaluation:
  """Share command playback and metrics between the batch loop and mjlab viewer."""

  def __init__(self, env, *, vx=None, vy=0.0, yaw_rate=0.0, hipx_target_limit=0.0,
               vx_sequence=None, segment_seconds=5.0, acceleration_steps=None, total_steps=None,
               record_footfalls=False, turn_front_assist=False):
    self.unwrapped = env
    self.cfg, self.num_envs, self.device = env.cfg, env.num_envs, env.device
    self.term = env.command_manager.get_term("velocity")
    self.action_bounds = None
    if hipx_target_limit:
      consts = st.robot_consts(env)
      self.action_bounds = action_bounds(consts.default_q, consts.action_scale,
                                         consts.soft_limits, hipx_target_limit)
    self.vx_sequence = tuple(vx_sequence) if vx_sequence is not None else None
    if self.vx_sequence is not None:
      speeds = self.vx_sequence
    elif vx is not None:
      speeds = (vx,)
    else:
      speeds = (0.0, 0.2, 0.8, 1.8, 0.8, 0.0)
    self.command_cycle = torch.tensor(
      [(speed, vy, yaw_rate) for speed in speeds],
      device=self.device, dtype=self.term.state.dtype,
    )
    self.segment_steps = max(1, round(segment_seconds / env.step_dt))
    self.writer = None
    self.video_stride = max(1, round(1 / (25 * env.step_dt)))
    self.step_index = 0
    self.metrics = {name: self._empty_metrics() for name in ("stand", *GAITS)}
    self.segment_metrics = [self._empty_metrics() for _ in self.vx_sequence or ()]
    self.acceleration_steps = acceleration_steps
    self.total_steps = total_steps
    self.turn_front_assist = turn_front_assist
    if turn_front_assist or record_footfalls:
      self.contact_sensor = env.scene["footfall_ground_contact"]
      names = list(self.contact_sensor.primary_names)
      self.contact_order = [names.index(f"{leg}_{part}_collision")
                            for leg in LEGS for part in ("FOOT", "SHANK")]
    if turn_front_assist:
      self.assist = FrontTurnAssist(self.num_envs, self.device)
      self.assist_action = self.term.state.new_zeros(len(JOINT_NAMES))
      self.assist_action[[JOINT_NAMES.index("FL_HipY_joint"), JOINT_NAMES.index("FL_Knee_joint")]] = 1.
      self.assisted_samples = torch.zeros((), device=self.device, dtype=torch.long)
    self.foot_contacts = None
    if record_footfalls:
      if acceleration_steps is None:
        raise ValueError("footfall recording requires a steady-state window")
      self.foot_contacts = []
      self.foot_positions = []
      self.foot_site_order = [st.FOOT_SITES.index(leg) for leg in LEGS]
    if acceleration_steps is not None:
      self.window_metrics = {name: self._empty_metrics() for name in ("acceleration", "steady_state")}
      self.heading_change = self.term.state.new_zeros(self.num_envs)
      self.heading_valid = torch.ones_like(self.heading_change, dtype=torch.bool)

  def _empty_metrics(self):
    return {"samples": 0, "velocity_squared_error": 0., "yaw_squared_error": 0.,
            "vx_squared_error": 0., "vy_squared_error": 0., "falls": 0,
            "foot_phase_agreement": 0., "vx_sum": 0., "command_vx_sum": 0.,
            "vy_sum": 0., "command_vy_sum": 0.,
            "hipx_sum": self.term.state.new_zeros(len(LEGS)),
            "hipx_squared_sum": self.term.state.new_zeros(len(LEGS))}

  @staticmethod
  def _accumulate(values, valid, fell, state, command, agree, hipx):
    values["samples"] += int(valid.sum())
    values["falls"] += int(fell.sum())
    velocity = state[valid, st.V_YAW][:, :2]
    desired_velocity = command[valid, 3:5]
    vx_error, vy_error = (velocity - desired_velocity).square().sum(dim=0).tolist()
    values["vx_squared_error"] += vx_error
    values["vy_squared_error"] += vy_error
    values["velocity_squared_error"] += vx_error + vy_error
    values["yaw_squared_error"] += float((state[valid, 9] - command[valid, 5]).square().sum())
    values["foot_phase_agreement"] += float(agree[valid].sum())
    vx, vy = velocity.sum(dim=0).tolist()
    desired_vx, desired_vy = desired_velocity.sum(dim=0).tolist()
    values["vx_sum"] += vx
    values["vy_sum"] += vy
    values["command_vx_sum"] += desired_vx
    values["command_vy_sum"] += desired_vy
    valid_hipx = hipx[valid]
    values["hipx_sum"] += valid_hipx.sum(dim=0)
    values["hipx_squared_sum"] += valid_hipx.square().sum(dim=0)

  def reset(self):
    self.obs, info = self.unwrapped.reset()
    if self.turn_front_assist:
      self.assist.reset(slice(None))
    return self.obs, info

  def get_observations(self):
    return self.obs

  def step(self, action):
    env = self.unwrapped
    segment = self.step_index // self.segment_steps
    if self.segment_metrics:
      segment = min(segment, len(self.segment_metrics) - 1)
    else:
      segment %= len(self.command_cycle)
    self.term.state[:, :3] = self.command_cycle[segment]
    command = self.term.state.clone()
    if self.turn_front_assist:
      phase, duty, _, _ = foot_schedule(command, self.term.cfg.gait)
      index = st.LEGS.index("FL")
      progress = (phase[:, index] - duty[:, 0]) / (1. - duty[:, 0])
      left_front = self.contact_order[:2]
      sensor = self.contact_sensor.data
      contact = (sensor.found[:, left_front] > 0).any(dim=-1)
      vertical_load = sensor.force[:, left_front, 2].abs().sum(dim=-1)
      turning = (command[:, 5] >= .30) & (command[:, 3:5].norm(dim=-1) < .03)
      gain = self.assist.gain(progress, turning, contact, vertical_load, env.step_dt)
      action = action + gain[:, None] * self.assist_action
      self.assisted_samples += (gain > 0).sum()
    if self.action_bounds is not None:
      action = action.clamp(*self.action_bounds)
    steady = self.acceleration_steps is not None and self.step_index >= self.acceleration_steps
    if steady:
      heading_before = st.planar_pose(env)[:, 2]
    result = env.step(action)
    self.obs, _, terminated, truncated, _ = result
    # Auto-reset states must be excluded from kinematic errors.
    keep = ~(terminated | truncated)
    if self.turn_front_assist:
      self.assist.reset(~keep)
    s = st.robot_state(env)
    hipx = s[:, st.Q][:, HIP_X_JOINT_INDICES]
    gait_cfg = self.term.cfg.gait
    _, _, stance, _ = foot_schedule(command, gait_cfg)
    actual_contact = st.feet_world_z(s) - st.FOOT_RADIUS < 0.02
    agree = (actual_contact == (stance > 0.5)).float().mean(-1)
    moving = command[:, 3:5].norm(dim=-1) + command[:, 5].abs() > gait_cfg.stand_speed
    selected = torch.where(moving, command[:, 10].long() + 1, 0)
    for index, values in enumerate(self.metrics.values()):
      chosen = selected == index
      self._accumulate(values, chosen & keep, terminated & chosen, s, command, agree, hipx)
    if self.segment_metrics:
      self._accumulate(self.segment_metrics[segment], keep, terminated, s, command, agree, hipx)
    if self.acceleration_steps is not None:
      window = "steady_state" if steady else "acceleration"
      self._accumulate(self.window_metrics[window], keep, terminated, s, command, agree, hipx)
      if steady:
        delta = st.planar_pose(env)[:, 2] - heading_before
        self.heading_valid &= keep & torch.isfinite(delta)
        self.heading_change += torch.where(self.heading_valid, torch.atan2(delta.sin(), delta.cos()), 0.)
        if self.foot_contacts is not None:
          found = self.contact_sensor.data.found[:, self.contact_order]
          self.foot_contacts.append((found > 0).reshape(self.num_envs, len(LEGS), 2).any(dim=-1))
          feet = st.limb_positions_w(env.scene["robot"])["feet"]
          self.foot_positions.append(feet[:, self.foot_site_order])
    if self.writer and self.step_index % self.video_stride == 0:
      self.writer.add_image(env.render())
    self.step_index += 1
    return result

  def close(self):
    self.unwrapped.close()

  def footfall_summary(self):
    if self.foot_contacts is None:
      raise ValueError("footfall recording is disabled")
    keep = self.heading_valid & (self.step_index == self.total_steps)
    env_ids = keep.nonzero().flatten().tolist()
    dt = self.unwrapped.step_dt
    start_s = self.acceleration_steps * dt
    result = {"start_s": start_s, "end_s": self.step_index * dt, "sample_dt_s": dt,
              "contact_geometries": ["foot", "shank"], "env_ids": env_ids}
    if self.foot_contacts and env_ids:
      contacts = torch.stack(self.foot_contacts, dim=0)[:, keep].cpu()
      result.update(summarize_footfalls(contacts, dt, start_s))
      feet = torch.stack(self.foot_positions, dim=0)[:, keep].cpu()
      stance = contacts[:-1] & contacts[1:]
      speed = (feet[1:, ..., :2] - feet[:-1, ..., :2]).norm(dim=-1) / dt
      totals = torch.where(stance, speed, 0.).sum(dim=(0, 1))
      counts = stance.sum(dim=(0, 1))
      result["stance_foot_speed_m_s_by_leg"] = {
        leg: float(total / count) if count else None
        for leg, total, count in zip(LEGS, totals, counts)
      }
      result["swing_by_leg"] = summarize_swings(contacts, feet[..., 2], dt)
    return result

  def _summary(self, values):
    n = values["samples"]
    result = {"falls": values["falls"], "valid_seconds": n * self.unwrapped.step_dt}
    if n:
      result.update({
        "velocity_rmse_m_s": math.sqrt(values["velocity_squared_error"] / n),
        "vx_rmse_m_s": math.sqrt(values["vx_squared_error"] / n),
        "vy_rmse_m_s": math.sqrt(values["vy_squared_error"] / n),
        "yaw_rmse_rad_s": math.sqrt(values["yaw_squared_error"] / n),
        "vx_mean_m_s": values["vx_sum"] / n,
        "vy_mean_m_s": values["vy_sum"] / n,
        "command_vx_mean_m_s": values["command_vx_sum"] / n,
        "command_vy_mean_m_s": values["command_vy_sum"] / n,
        "foot_phase_agreement": values["foot_phase_agreement"] / n,
        "hipx_mean_deg": dict(zip(LEGS, torch.rad2deg(values["hipx_sum"] / n).tolist())),
        "hipx_rms_deg": dict(zip(LEGS, torch.rad2deg((values["hipx_squared_sum"] / n).sqrt()).tolist())),
      })
    return result

  def summary(self):
    result = {name: self._summary(values) for name, values in self.metrics.items() if values["samples"]}
    if self.turn_front_assist:
      result["turn_front_assist"] = {"assisted_env_steps": int(self.assisted_samples)}
    if self.segment_metrics:
      segments = []
      for index, (speed, values) in enumerate(zip(self.vx_sequence, self.segment_metrics)):
        start = index * self.segment_steps
        if start >= self.step_index:
          break
        stop = min(start + self.segment_steps, self.step_index)
        if index == len(self.segment_metrics) - 1:
          stop = self.step_index
        dt = self.unwrapped.step_dt
        segments.append({"target_vx_m_s": speed, "start_s": start * dt, "end_s": stop * dt,
                         **self._summary(values)})
      result["segments"] = segments
    if self.acceleration_steps is not None:
      dt = self.unwrapped.step_dt
      split = self.acceleration_steps
      acceleration = self._summary(self.window_metrics["acceleration"])
      steady = self._summary(self.window_metrics["steady_state"])
      complete = self.step_index == self.total_steps
      valid = (self.heading_valid & complete).tolist()
      heading = [value if keep else None
                 for value, keep in zip(torch.rad2deg(self.heading_change).tolist(), valid)]
      measured = [value for value in heading if value is not None]
      steady.update({
        "complete_heading_envs": len(measured),
        "heading_change_deg_by_env": heading,
        "heading_signed_mean_deg": sum(measured) / len(measured) if measured else None,
        "heading_abs_mean_deg": sum(map(abs, measured)) / len(measured) if measured else None,
      })
      result["windows"] = {
        "complete": complete,
        "acceleration": {"start_s": 0., "end_s": min(split, self.step_index) * dt, **acceleration},
        "steady_state": {"start_s": split * dt, "end_s": max(split, self.step_index) * dt, **steady},
      }
    return result


def velocity_play_cfg(gait_cfg, num_envs, seed=None, *, record_footfalls=False, step_height=None,
                      turn_front_assist=False):
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Lite3-Velocity-Flat", play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  cmd_cfg = cfg.commands["velocity"]
  cmd_cfg.gait = gait_cfg
  cmd_cfg.resampling_time_range = (1e9, 1e9)
  cmd_cfg.standing_fraction = 1.0
  if step_height is not None:
    import mujoco
    from mjlab.entity import EntityCfg

    def step_spec():
      spec = mujoco.MjSpec()
      # The front edge is 0.75 m ahead, with a long landing after the riser.
      spec.worldbody.add_geom(
        name="platform", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(4.75, 0., step_height / 2), size=(4., 1.25, step_height / 2),
        rgba=(0.36, 0.49, 0.59, 1.), friction=(1., 0.005, 0.0001),
      )
      return spec

    cfg.scene.entities["test_step"] = EntityCfg(spec_fn=step_spec)
    pose_range = cfg.events["reset_base"].params["pose_range"]
    pose_range.update(x=(0., 0.), y=(0., 0.), yaw=(0., 0.))
    cfg.events.pop("push_robot", None)
  if record_footfalls or turn_front_assist:
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    # The 25 mm shank capsule extends past the 22 mm foot sphere at rest.
    # Count either collision geometry so the supporting leg still registers.
    cfg.scene.sensors += (ContactSensorCfg(
      name="footfall_ground_contact",
      primary=ContactMatch(mode="geom", pattern=r"[FH][LR]_(FOOT|SHANK)_collision", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force") if turn_front_assist else ("found",),
      reduce="netforce" if turn_front_assist else "maxforce",
      num_slots=1,
    ),)
  return cfg
