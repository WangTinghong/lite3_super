# Stair recovery and read-only audit

This branch starts at `44b6958`, before the stair HipX and foothold reward
experiments. It preserves the earlier locomotion and Chase tasks. The only
changes here add evaluation metrics and a world-model prediction audit. No
training reward, action, observation, terrain, or network definition changes.

Use the previously trained `extended_stairs_gait/model_15000.pt` as the
reference checkpoint. Do not resume from `stair_hipx_pilot`,
`stair_progress_pilot`, or the other later pilots until they beat this reference
under the same evaluation conditions.

`--wm-audit` reports one-step and five-step open-loop prediction errors for
forward body velocity, yaw rate, planar base position, front-foot position,
front-knee height, and front-foot terrain clearance. It separates states where a front
foot is near an uphill edge from other states. It also counts one-step errors
when a front shank contacts terrain without its foot. The near-edge flag is a
geometric proxy. Terminal transitions are excluded. The audit does not update
the world model or actor.

On the training server:

```bash
cd ~/gpufree-data/lite3_super
timeout 60s git -c http.version=HTTP/1.1 fetch --no-tags origin \
  refs/heads/codex/lite3-stair-recovery:refs/remotes/origin/codex/lite3-stair-recovery
git worktree add -b codex/lite3-stair-recovery ../lite3_stair_recovery \
  origin/codex/lite3-stair-recovery
cd ../lite3_stair_recovery
conda activate lite3_supermimic
python -m pytest tests/supermimic/test_locomotion_wm_audit.py -q

BASE="$HOME/gpufree-data/lite3_extended_stairs/logs/lite3_locomotion/2026-11-01_11-16-12-355742_extended_stairs_gait/model_15000.pt"
LATEST="$(find "$HOME/gpufree-data" -type f -path '*stair_progress_pilot/model_800.pt' -print -quit)"
ls -lh "$BASE" "$LATEST"
mkdir -p audit
python scripts/eval_locomotion.py --checkpoint "$BASE" --stage terrain \
  --terrain stairs_inv --terrain-level 8 --vx 1.0 \
  --num-envs 64 --seconds 20 --seed 0 --wm-audit --output audit/baseline_up8_v10.json
python scripts/eval_locomotion.py --checkpoint "$LATEST" --stage terrain \
  --terrain stairs_inv --terrain-level 8 --vx 1.0 \
  --num-envs 64 --seconds 20 --seed 0 --wm-audit --output audit/latest_up8_v10.json
```

Compare completion, tracking RMSE, forward progress, stall fraction, foot and
shank contacts, HipX outward angle, and near-edge world-model errors. These
are descriptive comparisons because each checkpoint uses its own actor and
therefore visits different states. A low aggregate `wm_loss` alone does not
prove accurate foot prediction at an uphill stair edge.
