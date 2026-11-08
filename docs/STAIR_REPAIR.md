# Restricted stair-gait repair pilot

The level-8 ascent policy `extended_stairs_gait/model_15000.pt` still strikes
the riser with its FR shank and swings its HR leg outward. A paired evaluation
of an FR HipY target change of -0.1 rad reduced FR shank-only contact while
preserving forward progress. A hard HR HipX cap at 0.25 rad removed the visible
outward swing but reduced forward speed and increased stalling. These are
different findings; the cap is not used in this pilot.

`--stair-repair` starts a new run from the **original 15000 checkpoint**. It
freezes the actor observation normalizer, feature extractor and ten output
rows. Only FR HipY and HR HipX output rows can move. On replay states near an
ascending edge, FR HipY learns the previously tested -0.1 rad target change.
The HR output gets a weak soft cost only above 0.5 rad during forward swing;
there is no hard cap or global outward-swing penalty. The ordinary world-model
imagination objective remains active, and `--policy-anchor-weight` keeps the
two changing outputs near the source policy. Stair rows 4–8 and focused
forward commands are sampled. Reward weights and the Chase task are unchanged.
Saving checks that all frozen actor weights and normalizer buffers still match
the source exactly. The source checkpoint is never overwritten.

On the GPU server, after pulling the branch:

```bash
python -m pytest tests/supermimic/test_stair_repair.py tests/supermimic/test_terrain_gait_retention.py -q
```

Use `SOURCE` for the full path to the original `model_15000.pt` and `MOTIONS`
for the directory with the two straight-walk AMP motion JSON files. First run
a 30-iteration smoke test with small batches, then a **200-iteration pilot**
from `SOURCE`, not from the smoke checkpoint. Use a new run name and save at
iterations 100 and 200. Do not continue to a long run automatically.

For each saved policy, compare against `SOURCE` on ascending stairs, level 8,
1.0 m/s, seeds 13–15, 128 environments, 40 seconds. Inspect FR shank-only
fraction, clean FR touchdown, HR HipX > 0.5 fraction, forward velocity, stall,
tracking errors and falls together. Accept only a checkpoint with better FR
contact and reduced HR extreme swing **without** materially losing forward
progress or stability. The MuJoCo viewer is still required to judge gait
quality; these numerical checks cannot certify natural gait. A failed pilot
is discarded, and the unchanged 15000 checkpoint remains the baseline.
