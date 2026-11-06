# Phase 1: terrain/contact world-model validation

This branch extends `codex/lite3-stair-recovery` at `3b78963`. It does not
change the robot, actuator model, existing training rewards, actor observations,
or Chase task. The collection actor is the original checkpoint, with no joint
probe. No PPO is used and no policy training is performed in this stage.

## What is implemented

1. `eval_locomotion.py --contact-data NEW_DIRECTORY` records continuous
   transitions, actions, previous actions, context, poses, terrain scans, foot
   and shank contacts in FR/FL/HR/HL order. Contacts are sampled after the
   control step, before explicit reset. They describe endpoint contacts, not
   every possible collision inside the four physics substeps. Terminal windows
   are conservatively excluded from model fitting and validation.
2. `TerrainAudit` compares the current bilinear height map to CPU MuJoCo rays
   against the same static geometry at each foot and 5/12/25 cm ahead, every
   25 control steps. It reports MAE, P95, missed height-change paths and ray
   misses. The policy still sees its original map. A height-change path is not
   a measurement of exact horizontal edge-position error. The audit cannot
   by itself prove that the map caused the bad gait.
3. `ContactWorldModel` initializes continuous dynamics from the baseline and
   adds 12 sigmoid logits sharing its MLP trunk: four foot, four shank, and
   four joint shank-without-foot events. Contact truth is supervised output,
   never a required model/policy input. Joint events are supervised directly.
4. Offline autoregressive dynamics + BCE training uses baseline normalization
   and a small learning rate. Half of each training batch samples contact
   switches/shank-only windows, half samples all windows; this is intentional
   event emphasis, not an unbiased sampling estimator. Validation/test use
   uniformly selected, fixed windows and report calibration (Brier), precision,
   recall, false-positive rate, and actual positive sample counts. Metrics with
   undefined denominators are null, not fabricated zeros or perfect scores.
5. Seeds 0/1/2 separate train/validation/test simulations. Selected weights
   minimize validation five-step normalized dynamics MSE + 0.2 BCE. The test
   set is used only for the final baseline/candidate comparison. Five-step
   validation propagates predicted pose and recomputes terrain scans; training
   uses recorded scans, matching the original model training convention.

The saved `contact_world_model.pt` has a separate offline schema and no actor.
Do NOT pass it to `train_locomotion.py --resume` or the actor viewer.
It is a candidate world model, not a new walking policy. Contact-head accuracy
does not yet establish the correctness of policy gradients through that head.
A matched-state MuJoCo action-response/gradient audit and actual policy update
remain subsequent validation steps; this branch does not claim those pass.

If recall at threshold 0.5 is near zero, reuse the saved model/data without any
updates: `python scripts/audit_contact_probabilities.py`. This writes
`audit/contact_phase1/fit/probability_audit.json` with average precision (tied
scores grouped), prevalence, positive/negative score distributions, and Brier
comparison against a constant-prevalence predictor. It reuses the original
validation/test window selections and does not tune thresholds on the test set.

If probability ranking is weak, run `python scripts/refine_contact_head.py`.
This uses the existing offline artifact and datasets: only its linear contact
head is optimized (default LR 1e-3, 2000 updates), while every dynamics weight
and normalization buffer is frozen and checked bit-for-bit. Uniform training
windows and unweighted BCE avoid treating class-weighted scores as calibrated
probabilities. Training uses predicted pose and resampled terrain, including
multi-step prediction error, matching validation. Selection maximizes validation
mean AP over the four shank-only heads and one/five-step horizons; update 0 is
retained if none improve. Results go to `audit/contact_phase1/head_refine`.
Threshold remains 0.5. This tests whether a separately optimized head can read
the frozen features; failure may require new features/model structure, not
further arbitrary threshold changes. The test seed is now a repeated regression
set; use fresh data before claiming final generalization or training the actor.

## Server workflow

Use the existing `lite3_supermimic` Python 3.12 environment. The wrapper adds
the NVRTC builtins directory matching `torch.version.cuda` to child processes
when installed; it does not install or replace CUDA packages.

```bash
cd ~/gpufree-data/lite3_super
git -c http.version=HTTP/1.1 fetch --no-tags origin codex/lite3-contact-world-model
git worktree add -b codex/lite3-contact-world-model ../lite3_contact_wm origin/codex/lite3-contact-world-model
cd ../lite3_contact_wm
conda activate lite3_supermimic
python -m pytest tests/supermimic/test_contact_phase1.py -q

BASE="$HOME/gpufree-data/lite3_extended_stairs/logs/lite3_locomotion/2026-11-01_11-16-12-355742_extended_stairs_gait/model_15000.pt"
python scripts/contact_phase1.py --checkpoint "$BASE" --mode smoke
```

Smoke: 16 environments, 3 simulated seconds, three seeds, 10 model updates.
It checks plumbing, not prediction quality. If successful:

```bash
python scripts/contact_phase1.py --checkpoint "$BASE" --mode collect
python scripts/contact_phase1.py --checkpoint "$BASE" --mode fit
cat audit/contact_phase1/fit/summary.json
```

Full collection: four scenes per seed (flat 0.5 m/s; stairs_inv row 5 at
0.5 m/s; stairs_inv row 8 at 1.0 m/s; stairs row 5 at 0.5 m/s), each 64
environments for 20 simulated seconds. Terrain row is stored along with the
serialized physical heightfield. This samples specific fixed-map scenarios;
it does not establish generalization to new stair dimensions or maps.

The fit performs 1500 optimizer updates, NOT 1500 RL iterations. The actor and
source checkpoint are untouched. Progress prints during recording and fitting.
Environment/heightfield initialization may still take time before recording.

Results:

- `audit/contact_phase1/fit/summary.json`: concise baseline/candidate comparison.
- `audit/contact_phase1/fit/report.json`: full held-out metrics and geometry
  audits, provenance, package versions, and dataset manifests.
- `audit/contact_phase1/metrics/`: actual frozen-policy locomotion metrics.
- `audit/contact_phase1/data/`: chunked data; no checkpoint overwrite.

The collection wrapper skips already completed matching manifests. Interrupted
collections deliberately have no manifest; preserve that directory or choose a
new `--root` for a clean rerun. Fitting refuses a nonempty output directory.

For an ablation, call `fit_contact_world_model.py` with the same datasets and
`--contact-weight 0 --output-dir NEW_PATH`. This isolates contact supervision
from ordinary additional world-model fitting. The unused random contact head
in that ablation must not be interpreted as a trained classifier.

## Interpretation before any policy update

- Examine geometric errors on edge paths, not just all-terrain average error.
- Check counts/recall/precision for all legs, especially FR shank-only and
  contact switches. A classifier that always predicts no shank contact fails.
- Compare both one/five-step continuous predictions to the unchanged baseline.
  A lower contact loss with materially worse dynamics is not an accepted upgrade.
- This phase does not change geometry resolution automatically; do so only as
  a separately measured experiment after reviewing the geometry audit.
- No automatic "passed" claim: sparse data, unseen geometry, and model action
  gradients require further verification. Do not restart long policy training
  solely because fitting completed.

Local validation: CPU unit tests and a tiny synthetic end-to-end offline fit.
The development Mac has Python 3.9; local test execution used a temporary
dataclass keyword compatibility shim. Repository targets Python 3.12. No local
NVIDIA/mjlab simulation was available; the server smoke command is required.

## Fresh-seed audit after contact-head refinement

Keep the frozen actor source in `BASE`. Collect a new seed without fitting:

```bash
python scripts/contact_phase1.py --checkpoint "$BASE" --mode holdout --seed 3 \
  --root audit/contact_holdout_seed3
```

Then evaluate the already selected contact model, separately for each scene:

```bash
python scripts/audit_contact_probabilities.py \
  --model audit/contact_phase1/head_refine/contact_world_model.pt \
  --fresh-data audit/contact_holdout_seed3/data/holdout/flat_v05 \
    audit/contact_holdout_seed3/data/holdout/up5_v05 \
    audit/contact_holdout_seed3/data/holdout/up8_v10 \
    audit/contact_holdout_seed3/data/holdout/down5_v05 \
  --output audit/contact_holdout_seed3/contact_audit.json
```

This refuses seeds found in the model's original train/validation/test manifests
and data collected with another source checkpoint. Retain those original
manifests at their stored paths. The report contains contact metrics for all
legs, one/five-step dynamics metrics, source hashes, collection settings and
terrain geometry audits. The console prints FR shank-only metrics per scene.
Threshold remains 0.5; no parameter or policy update occurs. Existing output
files are refused. The synthetic CPU test checks fresh-seed acceptance, reused
seed rejection, checkpoint mismatch rejection and artifact preservation.

Fresh-seed performance is a replication within these scenarios, not proof of
unseen-terrain generalization or correct action gradients. If it holds up,
inspect geometry errors and validate predicted responses to action changes
against simulation before using the contact model to optimize the actor.

## Locate geometry errors before changing terrain inputs

Use a completed fresh-seed high-stair collection. This reconstructs the physical
terrain from its source checkpoint/configuration and samples recorded robot
states; it does not roll out or train a policy:

```bash
python scripts/audit_terrain_samples.py --checkpoint "$BASE" \
  --data audit/contact_holdout_seed3/data/holdout/up8_v10 \
  --output-dir audit/terrain_geometry_up8_seed3
```

Default: 128 recorded robot states, four feet, four path offsets per foot
(0/5/12/25 cm). `summary.json` reports signed/absolute height errors for the
existing cached map and hypothetical ray-cast grids at the original resolution,
5 cm and 2 cm. `samples.csv` preserves coordinates and chunk/time/environment
indices. It checks cached grid nodes against reconstructed physical geometry;
a mismatch must be investigated before blaming interpolation. The finer grids
are diagnostic only and never replace any policy observation or world-model
input. They need not improve every sampled edge. Full-map memory/runtime costs,
new observation distributions and action gradients are not validated here.

Matching source checkpoint hash and mjlab/MuJoCo/mujoco-warp versions are
required. Geometry reconstruction also relies on the current terrain config;
node mismatch can expose changed configuration or cache inconsistencies.
CPU tests cover planar interpolation, step-edge smearing and missing ray hits.
The native GPU environment must still be tested on the training server.

Geometry audit version 2 corrects the comparison protocol: `common_support`
uses the same finite query points for every resolution. Per-method statistics
remain available but cannot alone establish improvement when missingness differs.
`cached_node_checks` counts unique nodes and reconstructs the original builder's
`np.arange` coordinates, compares nominal origin+index*resolution coordinates,
and probes +/-1 micrometre in eight directions to identify discontinuity or
ray-miss sensitivity. `nodes.csv` records those checks. The explicit
`original_builder_zero_fallback` statistic reproduces the historical no-ray-hit
zero fill solely to diagnose cache agreement; zero fill is not ground truth.
Run to a new output directory, for example `audit/terrain_geometry_up8_seed3_v2`.
No policy, height-map cache, model or physics setting is changed by this audit.

## Paired 2 cm offline pilot (high-stair scene only)

After geometry audit v2, prepare a *separate* representation of the existing
train/validation/test `up8_v10` trajectories (seeds 0/1/2). No new actor rollout
is required; the original files and the actor's 10 cm observations are retained.

```bash
python scripts/prepare_terrain_ablation.py --checkpoint "$BASE"
```

Each split reconstructs its physical terrain with matching software versions,
then ray-casts a cropped 2 cm grid enclosing its recorded poses with a 2 m
margin. This is CPU ray work and can be slow despite low GPU utilization;
progress prints every 100 grid rows. The grid is limited to five million nodes.
A fixed +1 micrometre origin shift defines edge sampling. Missing rays retry
+x,+y,-x,-y by 1 micrometre, and unresolved misses fail instead of filling zero.
This changes both resolution and boundary sampling, not resolution alone.
The fine grid is saved only inside `audit/terrain_2cm_pilot/data`.

Conversion updates both s0/s1 base height using
`z_new = z_old + old_ground(base_xy) - new_ground(base_xy)` and recomputes relief.
World height, body-relative limb coordinates, velocities, actions, context,
contact labels, terminal flags and original near-riser grouping are preserved.
Completed output manifests contain source hashes and transformation metadata.
Outputs must be empty; preserve partial output and use a new output directory
if preparation is interrupted.

Run the matched-budget pilot:

```bash
python scripts/fit_terrain_ablation.py --updates 500
```

Both arms start with the same refined contact-world-model weights and frozen
normalization, use identical uniform window IDs, Adam learning rate 1e-5 and
five-step predicted-pose rollouts. Validation selects independently using
normalized MSE + 0.2 BCE and includes the unchanged initial model. Test is never
used to select an update. The test set is repeated regression data, not a new
claim of generalization. Cropped terrain queries outside the grid fail explicitly.

The report includes world-frame base/foot height RMSE to avoid mistaking a
change of height reference for a physical improvement. Other physical velocity
and contact metrics must also be checked, not just normalized MSE. The two
saved models use a distinct offline schema and cannot be resumed as actors.
No AMP/actor training, reward change or controller update occurs here. General
terrain regression, fresh-seed evaluation, and action-response validation must
precede policy optimization. This pilot cannot establish gait improvement.

CPU tests cover re-encoding physical invariance, missing-ray handling, grid
size bounds, and a synthetic end-to-end two-arm fit. GPU/mjlab terrain building
requires execution on the server.
