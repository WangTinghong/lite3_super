# Lite3 uphill HipX pilot

This optional experiment keeps the Chase task and existing checkpoints intact.
It resumes the safer extended-stairs checkpoint, retains AMP and the gait
anchor, and adds a direct actor penalty on excessive outward front-leg HipX
targets only while a forward command approaches a rising step. The hinge starts
at 0.35 rad; 0.523 rad is the XML joint limit. Turning and flat ground are not
penalized. `--stair-hipx-weight` is zero by default and requires `--stair-refine`.

The evaluator now reports HipX outward mean, fraction of actual angles above
0.50 rad, and fraction of targets beyond 0.523 rad (leg order FR, FL, HR, HL).
It also reports maximum forward progress and maximum terrain-height gain per
completed episode, plus the fraction of forward-command samples below 0.1 m/s.
These are diagnostic proxies: tile-boundary truncation is counted as a
non-fall completion and is not itself proof of successful stair ascent.

On the GPU server, use a new worktree for this branch:

```bash
cd ~/gpufree-data/lite3_super
timeout 60s git -c http.version=HTTP/1.1 fetch --no-tags origin \
  refs/heads/codex/lite3-stair-hipx-pilot:refs/remotes/origin/codex/lite3-stair-hipx-pilot
git worktree add -b codex/lite3-stair-hipx-pilot ../lite3_stair_hipx \
  origin/codex/lite3-stair-hipx-pilot
cd ../lite3_stair_hipx
conda activate lite3_supermimic

SOURCE="$HOME/gpufree-data/lite3_extended_stairs/logs/lite3_locomotion/2026-11-01_11-16-12-355742_extended_stairs_gait/model_15000.pt"
MOTIONS="$HOME/gpufree-data/lite3_amp_motions"
ls -lh "$SOURCE" "$MOTIONS"/lite3_直行慢速.json "$MOTIONS"/lite3_直行快速.json
python -m pytest tests/supermimic/test_locomotion.py tests/supermimic/test_extended_stairs.py -q

python scripts/train_locomotion.py --stage terrain --harder-stairs --stair-refine \
  --stair-hipx-weight 2.0 --resume "$SOURCE" \
  --amp-motion-dir "$MOTIONS" --gait-anchor-weight 0.2 \
  --num-envs 256 --iterations 100 --warmup-iterations 20 \
  --critic-warmup-iterations 20 --actor-ramp-iterations 20 \
  --amp-start-iteration 1 --amp-ramp-iterations 10 \
  --wm-batch 128 --imag-batch 128 --amp-batch 128 --amp-updates 1 \
  --archive-size 2000 --no-compile --save-interval 100 \
  --run-name smoke_stair_hipx --log-interval 20

# Start from SOURCE again. Do not resume from the smoke checkpoint.
python scripts/train_locomotion.py --stage terrain --harder-stairs --stair-refine \
  --stair-hipx-weight 2.0 --resume "$SOURCE" \
  --amp-motion-dir "$MOTIONS" --gait-anchor-weight 0.2 \
  --num-envs 2048 --iterations 1500 --warmup-iterations 200 \
  --critic-warmup-iterations 200 --actor-ramp-iterations 300 \
  --speed-curriculum-iterations 2500 --actor-lr-scale 0.2 \
  --save-interval 500 --run-name stair_hipx_pilot --log-interval 10
```

After the pilot, compare the new checkpoint against the original source and
the previous ascent-refine checkpoint with the **same** terrain, command,
environment count, duration and seeds. At minimum check uphill levels 5 and 8
at 0.5 m/s, level 8 at 1.0 m/s, then flat-ground gait. Do not select a model
only because the shank-only fraction falls: forward progress, real terrain
gain, falls, peak contact forces and near-limit HipX targets must also hold up.
