# Lite3 upper-tread foothold pilot

The previous `--stair-hipx-weight 2.0` pilot is rejected. At level 8,
1.0 m/s and seed 0, its 20-second single-robot rollout slowed from 0.889 to
0.631 m/s, increased low-speed samples from 1.9% to 9.3%, and raised FR
shank-only contact from 70 to 86 frames. FL foot duty fell from 40.4% to
28.2%, while its swing clearance increased from 0.059 to 0.101 m.
This suggests that simply limiting HipX does not solve the missing front-foot
support. These observations do not isolate the HipX term from the other
training changes, so the new experiment starts from the existing
`stair_ascent_refine/model_4000.pt` checkpoint and leaves HipX weight at zero.

`--stair-foothold-weight` is an optional positive reward for a front foot
that settles near a higher tread while the base is still on lower ground.
The differentiable proxy uses foot height, foot horizontal speed, and the
same height field in simulation and imagined rollouts. It is disabled on
flat ground and turning. The evaluation adds
`touchdown_without_prior_shank_only_fraction` for each leg: a touchdown is
counted clean when that leg's previous swing did not include a shank-only
contact. This is a diagnostic proxy, not proof of foot-first contact with a
specific riser.

Run on the GPU server in a new worktree:

```bash
cd ~/gpufree-data/lite3_super
timeout 60s git -c http.version=HTTP/1.1 fetch --no-tags origin \
  refs/heads/codex/lite3-stair-foothold-pilot:refs/remotes/origin/codex/lite3-stair-foothold-pilot
git worktree add -b codex/lite3-stair-foothold-pilot ../lite3_stair_foothold \
  origin/codex/lite3-stair-foothold-pilot
cd ../lite3_stair_foothold
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
  --run-name smoke_stair_foothold --log-interval 20

# Start a fresh run from SOURCE, not the smoke checkpoint.
python scripts/train_locomotion.py --stage terrain --harder-stairs --stair-refine \
  --stair-hipx-weight 0 --stair-foothold-weight 0.5 \
  --resume "$SOURCE" --amp-motion-dir "$MOTIONS" --gait-anchor-weight 0.2 \
  --num-envs 2048 --iterations 1500 --warmup-iterations 200 \
  --critic-warmup-iterations 200 --actor-ramp-iterations 300 \
  --speed-curriculum-iterations 2500 --actor-lr-scale 0.2 \
  --save-interval 500 --run-name stair_foothold_pilot --log-interval 10
```

Evaluate the source and the new checkpoints under the same level, speed,
number of environments, duration and seed. Require improved or equal forward
velocity and fall rate, fewer shank-only contacts, and more clean front-foot
touchdowns before any longer training. Review video as well: the proxy can be
exploited by a world model that predicts an unrealistic foot position.
