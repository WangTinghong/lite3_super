# Stair foothold progress correction

The first foothold pilot (`fcfc681`) should not be resumed. Its reward was
positive at every stationary timestep while a front foot rested on a higher
tread, so waiting at a step could accumulate return. This branch multiplies
that reward by measured forward base progress divided by the forward command.
A stationary robot now gets exactly zero foothold reward. The previous HipX
penalty stays disabled by passing `--stair-hipx-weight 0`.

This corrects a reward-design bug; it does not establish that the new policy
will climb better. Start again from the earlier ascent-refine checkpoint and
run only a short pilot before comparing real contact and progress metrics.

```bash
cd ~/gpufree-data/lite3_super
timeout 60s git -c http.version=HTTP/1.1 fetch --no-tags origin \
  refs/heads/codex/lite3-stair-foothold-progress:refs/remotes/origin/codex/lite3-stair-foothold-progress
git worktree add -b codex/lite3-stair-foothold-progress ../lite3_stair_progress \
  origin/codex/lite3-stair-foothold-progress
cd ../lite3_stair_progress
conda activate lite3_supermimic

SOURCE="$HOME/gpufree-data/lite3_stair_ascent/logs/lite3_locomotion/2026-11-01_14-54-26-900512_stair_ascent_refine/model_4000.pt"
MOTIONS="$HOME/gpufree-data/lite3_amp_motions"
ls -lh "$SOURCE" "$MOTIONS"/lite3_直行慢速.json "$MOTIONS"/lite3_直行快速.json
python -m pytest tests/supermimic/test_locomotion.py tests/supermimic/test_extended_stairs.py -q

python scripts/train_locomotion.py --stage terrain --harder-stairs --stair-refine \
  --stair-hipx-weight 0 --stair-foothold-weight 0.5 \
  --resume "$SOURCE" --amp-motion-dir "$MOTIONS" --gait-anchor-weight 0.2 \
  --num-envs 256 --iterations 100 --warmup-iterations 20 \
  --critic-warmup-iterations 20 --actor-ramp-iterations 20 \
  --amp-start-iteration 1 --amp-ramp-iterations 10 \
  --wm-batch 128 --imag-batch 128 --amp-batch 128 --amp-updates 1 \
  --archive-size 2000 --no-compile --save-interval 100 \
  --run-name smoke_stair_progress --log-interval 20

# Fresh 800-iteration pilot from SOURCE, not from either foothold pilot.
python scripts/train_locomotion.py --stage terrain --harder-stairs --stair-refine \
  --stair-hipx-weight 0 --stair-foothold-weight 0.5 \
  --resume "$SOURCE" --amp-motion-dir "$MOTIONS" --gait-anchor-weight 0.2 \
  --num-envs 2048 --iterations 800 --warmup-iterations 100 \
  --critic-warmup-iterations 100 --actor-ramp-iterations 200 \
  --speed-curriculum-iterations 2500 --actor-lr-scale 0.2 \
  --save-interval 200 --run-name stair_progress_pilot --log-interval 10
```

Compare against `SOURCE` on level-8 ascent at 0.5 and 1.0 m/s with identical
seeds and rollout duration. Stop the experiment if front-foot duty, shank-only
contact, forward progress or fall rate are worse. The new clean-touchdown
metric counts all strides, including flat portions of a tile, so inspect
contact timelines and MuJoCo video before treating it as stair-specific proof.
