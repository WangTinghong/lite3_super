# Action-conditioned stair contact model

This phase trains the **world model only**. The source policy is frozen, and no actor optimizer is created. It records the actions actually sent to MuJoCo under five cases: baseline, FR HipY ±0.05 rad, and FR Knee ±0.05 rad. The offsets are gated by the existing stair approach probe. Train seeds 6 and 9, validation seed 7, and test seed 8 are disjoint. All runs use ascending stairs at level 8 and a 1.0 m/s forward command.

From the repository root, set `SOURCE` to the 15000-step policy checkpoint and `CONTACT` to `audit/contact_phase1/head_refine/contact_world_model.pt`.

```bash
python -u scripts/contact_action_phase.py --checkpoint "$SOURCE" --initialize-contact-model "$CONTACT" --mode smoke
python -u scripts/contact_action_phase.py --checkpoint "$SOURCE" --mode collect
python -u scripts/contact_action_phase.py --checkpoint "$SOURCE" --initialize-contact-model "$CONTACT" --mode fit --updates 1500
```

The production model and held-out summary are saved under `audit/contact_action_phase/fit/`. Collection resumes completed cases but refuses an incomplete or mismatched directory, preventing silent data mixing. Fitting requires an empty output directory.

The held-out report tests prediction on actions from distinct seeds. This is not yet a paired counterfactual test or a walking improvement result; keep the actor frozen until those checks and a gated actor pilot pass.

## Short actor pilot

If a one-step paired audit has no actual contact changes, it cannot establish contact-head action sensitivity. A separate opt-in training flag, `--contact-wm-init`, can still initialize the ordinary SuperMimic dynamics from the contact-supervised model. The contact head stays offline: imagined reward, AMP, and gait retention remain the actor objectives. This isolates whether the improved dynamics itself helps control, without granting the unvalidated contact head authority over the actor.

Use the same source policy checkpoint that was recorded in the contact artifact. The loader verifies its SHA-256, installs only the continuous dynamics, clears stale world-model optimizer moments, and resets the critic. This starts a new run and cannot be combined with `--continue-run`. Compare the saved pilot policy against the unchanged source checkpoint with identical MuJoCo evaluation seeds and terrain settings before extending training.

If the pilot regresses, run a matched control from the same source checkpoint and the same training arguments, omitting `--contact-wm-init` and adding `--reset-critic --reset-wm-optimizer`. This keeps the critic/optimizer reset conditions identical; the contact-supervised dynamics initialization is the intended experimental difference. Do not select checkpoints by imagined return alone. Reject a pilot that loses real forward progress or becomes more prone to stalling, even if its shank-only fraction falls.
