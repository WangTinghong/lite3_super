"""Phase-independent footfall diagnostics from measured ground contacts."""
from statistics import median

import torch

from lite3_supermimic.robot.constants import LEGS


def summarize_swings(contacts, foot_z, dt):
  """Measure complete airborne intervals relative to the preceding stance height."""
  starts = contacts[:-1] & ~contacts[1:]
  ends = ~contacts[:-1] & contacts[1:]
  result = {}
  for leg_index, leg in enumerate(LEGS):
    durations, peaks = [], []
    for env in range(contacts.shape[1]):
      lifts = starts[:, env, leg_index].nonzero().flatten().tolist()
      lands = ends[:, env, leg_index].nonzero().flatten().tolist()
      if not lifts:
        continue
      if lands and lands[0] < lifts[0]:
        lands = lands[1:]
      for lift, land in zip(lifts, lands):
        height = foot_z[lift:land + 1, env, leg_index]
        durations.append((land - lift) * dt)
        peaks.append(float(height.max() - height[0]))
    result[leg] = {
      "complete_swings": len(peaks),
      "one_sample_swings": sum(duration <= dt for duration in durations),
      "median_air_time_s": median(durations) if durations else None,
      "median_peak_rise_m": median(peaks) if peaks else None,
    }
  return result


def summarize_footfalls(contacts, dt, start_s):
  """Summarize one uninterrupted window: [time, env, FL/FR/HL/HR]."""
  fl, fr, hl, hr = contacts.unbind(-1)
  front_only = fl & fr & ~hl & ~hr
  hind_only = hl & hr & ~fl & ~fr
  diagonal_only = (fl & hr & ~fr & ~hl) | (fr & hl & ~fl & ~hr)
  support = contacts.any(dim=-1)
  support_count = int(support.sum())
  touchdown = ~contacts[:-1] & contacts[1:]
  liftoff = contacts[:-1] & ~contacts[1:]
  env_events = []
  alternations = transitions = 0
  for env in range(contacts.shape[1]):
    # Count new exclusive-support bouts. Repeated bouts of the same pair
    # separated by flight or a mixed stance still count as separate bouts.
    pair = front_only[:, env].int() - hind_only[:, env].int()
    bout_starts = torch.cat((pair[:1] != 0, (pair[1:] != 0) & (pair[1:] != pair[:-1])))
    bouts = pair[bout_starts]
    transitions += max(0, bouts.numel() - 1)
    alternations += int((bouts[1:] != bouts[:-1]).sum())
    env_events.append({
      leg: {
        name: [round(start_s + (i + 2) * dt, 6) for i in event[:, env, index].nonzero().flatten().tolist()]
        for name, event in (("touchdown_s", touchdown), ("liftoff_s", liftoff))
      }
      for index, leg in enumerate(LEGS)
    })
  return {
    "front_pair_mismatch": float((fl ^ fr).float().mean()),
    "hind_pair_mismatch": float((hl ^ hr).float().mean()),
    "front_only_fraction_of_support": int(front_only.sum()) / support_count if support_count else None,
    "hind_only_fraction_of_support": int(hind_only.sum()) / support_count if support_count else None,
    "diagonal_fraction_of_support": int(diagonal_only.sum()) / support_count if support_count else None,
    "pair_bout_transitions": transitions,
    "pair_bout_alternation_fraction": alternations / transitions if transitions else None,
    "stance_fraction_by_leg": dict(zip(LEGS, contacts.float().mean(dim=(0, 1)).tolist())),
    "events_by_env": env_events,
  }
