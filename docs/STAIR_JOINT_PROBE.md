# Right-front stair joint intervention

The baseline audit still shows FR shank-only contact for 10.83% of samples,
and only 57.16% of FR touchdowns have no preceding shank-only contact. The
later reward pilot did not improve either measure. Previously supplied joint
logs also show deep Knee flexion combined with HipY near zero during these
contacts. A joint-angle correlation does not establish that an angle bias
will help, so test small interventions in real simulation before training.

This evaluation-only probe compares the identical baseline checkpoint with
FR HipY -0.1 rad, FR Knee -0.1 rad, and both. Offsets are in native joint target
coordinates; the FR action indices are 4 and 5. Each offset is multiplied by
a gate that requires forward straight motion, a rising edge ahead of the
foot, an airborne FR foot from the real contact sensor, and foot clearance.
The gate uses the existing height map, so its edge location is approximate.
All other actor outputs are retained. The actor observes the executed actions
on subsequent steps. No weights, gains, joint limits, or training rewards are
updated. The resulting controller is an intervention, not a new trained model.

```bash
cd ~/gpufree-data/lite3_stair_recovery
timeout 60s git -c http.version=HTTP/1.1 fetch --no-tags origin \
  refs/heads/codex/lite3-stair-recovery:refs/remotes/origin/codex/lite3-stair-recovery
git merge --ff-only origin/codex/lite3-stair-recovery
conda activate lite3_supermimic
python -m pytest tests/supermimic/test_stair_probe.py -q
BASE="$HOME/gpufree-data/lite3_extended_stairs/logs/lite3_locomotion/2026-11-01_11-16-12-355742_extended_stairs_gait/model_15000.pt"
python scripts/probe_stair_joints.py --checkpoint "$BASE" \
  --terrain-level 8 --vx 1.0 --num-envs 64 --seconds 20 \
  --output-dir audit/joint_probe
```

Send `audit/joint_probe/comparison.json`. Inspect intervention activity first:
if the gate rarely activates, a null result is inconclusive. A useful case
must reduce FR shank-only contact and improve clean touchdowns while retaining
forward motion and stability. Do not choose a case that merely stops moving.
The 20-second, single-seed test screens directions, not robustness; any useful
case then needs other seeds, lower speed and a visual check before training.
If all cases worsen, retain the baseline and investigate foot placement and
body height instead of forcing these joint offsets into a reward.
