"""Registers the chase / dodge tasks with mjlab's task registry on import."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

from lite3_supermimic.env.env_cfg import lite3_chase_env_cfg

for task_id, kw in (
  ("Lite3-Chase-Flat", {}),
  ("Lite3-Chase-Obstacles", {"obstacles": True}),
  ("Lite3-Chase-Terrain", {"terrain": True}),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=lite3_chase_env_cfg(**kw),
    play_env_cfg=lite3_chase_env_cfg(play=True, **kw),
    rl_cfg=RslRlOnPolicyRunnerCfg(),  # unused: training is SuperMimic (scripts/train.py)
  )

from lite3_supermimic.locomotion.env_cfg import flat_velocity_env_cfg

register_mjlab_task(
  task_id="Lite3-Velocity-Flat",
  env_cfg=flat_velocity_env_cfg(),
  play_env_cfg=flat_velocity_env_cfg(play=True),
  rl_cfg=RslRlOnPolicyRunnerCfg(),
)
