"""Straight uphill stair course and its matching world-model height field."""
import torch

from lite3_supermimic.target.terrain import ground

STEP_HEIGHTS = (.02, .04, .06, .08, .10)
RISERS = (.65, 1.05, 1.45, 1.85, 2.25)
COURSE_END = 12.
HALF_WIDTH = 1.25
EDGE_PROXIMITY_TERM = ("stair_edge_proximity", 4)
# The shank capsule extends farther down than the 22 mm foot sphere.
SHANK_RADIUS = .025


class StairField:
  def __init__(self, level, *, edges=RISERS, end=COURSE_END, height=None):
    self.height = STEP_HEIGHTS[level - 1] if height is None else height
    self.edges = edges
    self.end = end

  def sample(self, xy):
    x, y = xy.unbind(dim=-1)
    edges = xy.new_tensor(self.edges)
    # Smooth 3 cm around edges for pose derivatives during imagination.
    steps = ((x[..., None] - edges) / .03 + .5).clamp(0., 1.).sum(dim=-1)
    return self.height * steps * (y.abs() <= HALF_WIDTH) * (x <= self.end)

  def edge_profile(self, feet_xy):
    """Per-foot approach weight and top height of the nearby uphill riser."""
    x, y = feet_xy.unbind(dim=-1)
    edges = x.new_tensor(self.edges)
    distance = edges - x[..., None]
    approach = ((.18 - distance) / .12).clamp(0., 1.)
    crossing = ((distance + SHANK_RADIUS) / SHANK_RADIUS).clamp(0., 1.)
    proximity, edge = (approach * crossing).max(dim=-1)
    return proximity * (y.abs() < HALF_WIDTH - SHANK_RADIUS), (edge + 1) * self.height

  def edge_clearance(self, feet):
    proximity, top = self.edge_profile(feet[..., :2].detach())
    return proximity, feet[..., 2] - SHANK_RADIUS - top


def configure_stair_course(cfg, level, *, training=False):
  import mujoco
  from mjlab.entity import EntityCfg

  height = STEP_HEIGHTS[level - 1]

  def course_spec():
    spec = mujoco.MjSpec()
    for index, start in enumerate(RISERS, 1):
      top = index * height
      end = RISERS[index] if index < len(RISERS) else COURSE_END
      spec.worldbody.add_geom(
        name=f"tread_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=((start + end) / 2, 0., top / 2),
        size=((end - start) / 2, HALF_WIDTH, top / 2),
        rgba=(.36, .49, .59, 1.), friction=(1., .005, .0001),
      )
    return spec

  cfg.scene.entities["stair_course"] = EntityCfg(spec_fn=course_spec)
  # MuJoCo batches independent worlds. Each robot approaches the same local course.
  cfg.scene.env_spacing = 0.
  span = .06 if training else 0.
  pose = cfg.events["reset_base"].params["pose_range"]
  pose.update(x=(-span, span), y=(-span, span), yaw=(-span, span))
  cfg.events.pop("push_robot", None)
  cfg.commands["velocity"].stair_level = level


def local_foot_heights(state, pose, field):
  from lite3_supermimic.supermimic import state as st

  feet_xy = st.limb_points_world(state, pose)["feet"][..., :2]
  return (st.feet_world_z(state) + ground(pose[:, :2], field)[:, None]
          - ground(feet_xy, field) - st.FOOT_RADIUS)


def front_rise_ahead(front_xy, heading, field):
  """Height gain over 12 cm in the forward direction at each front foot, metres."""
  if field is None:
    return torch.zeros_like(front_xy[..., 0])
  offset = .12 * torch.stack((heading.cos(), heading.sin()), dim=-1)
  return (field.sample(front_xy + offset[:, None]) - field.sample(front_xy)).clamp_min(0.)


def front_rise_observation(env, platform=None):
  from lite3_supermimic.supermimic import state as st

  term = env.command_manager.get_term("velocity")
  field = term.height_field if platform is None else platform
  if field is None:
    return term.state.new_zeros((env.num_envs, 2))
  front = st.limb_positions_w(term.robot)["feet"][:, :2, :2]
  return front_rise_ahead(front, term.robot.data.heading_w, field) / .02


def edge_proximity_observation(env, platform=None):
  from lite3_supermimic.supermimic import state as st

  term = env.command_manager.get_term("velocity")
  field = term.height_field if platform is None else platform
  if field is None:
    return term.state.new_zeros((env.num_envs, EDGE_PROXIMITY_TERM[1]))
  feet_xy = st.limb_positions_w(term.robot)["feet"][..., :2]
  return field.edge_profile(feet_xy)[0]


def add_front_rise_observation(cfg, step_height=None, *, edge_preview=False):
  from mjlab.managers.observation_manager import ObservationTermCfg

  # The isolated evaluation platform starts at 0.75 m and ends at 8.75 m.
  platform = None if step_height is None else StairField(1, edges=(.75,), end=8.75, height=step_height)
  cfg.observations["actor"].terms["front_rise"] = ObservationTermCfg(
    func=front_rise_observation, params={"platform": platform})
  if edge_preview:
    cfg.observations["actor"].terms[EDGE_PROXIMITY_TERM[0]] = ObservationTermCfg(
      func=edge_proximity_observation, params={"platform": platform})


def riser_crossing_clearance(previous, current, edges, step_height):
  """Interpolate the foot sphere's bottom height above the tread at each riser."""
  from lite3_supermimic.supermimic import state as st

  x0, y0, z0 = previous.unsqueeze(-2).unbind(-1)
  x1, y1, z1 = current.unsqueeze(-2).unbind(-1)
  fraction = ((edges - x0) / (x1 - x0).clamp_min(1e-8)).clamp(0., 1.)
  y_at_edge = torch.lerp(y0, y1, fraction)
  z_at_edge = torch.lerp(z0, z1, fraction)
  crossing = (x0 < edges) & (x1 >= edges) & (y_at_edge.abs() < HALF_WIDTH - st.FOOT_RADIUS)
  top = torch.arange(1, edges.numel() + 1, device=edges.device, dtype=edges.dtype) * step_height
  return crossing, z_at_edge - st.FOOT_RADIUS - top
