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
