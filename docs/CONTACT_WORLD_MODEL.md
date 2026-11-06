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
