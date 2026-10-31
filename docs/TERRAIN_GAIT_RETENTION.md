# 地形训练中的平地步态保留实验

`codex/lite3-terrain-gait-retention` 在地形阶段启用两项可选约束：

- AMP 判别器继续使用慢速、快速直行动作，但只从真实仿真中局部高度差较小的前进样本学习，并只在想象轨迹中相同条件下给步态奖励。
- 冻结平地 AMP 检查点的策略作为参考，在真实平坦区域的前进观测上约束地形策略输出；遇到台阶和明显起伏时不应用该约束。

两者旨在减少从平地转到地形时的步态遗忘。地形样本本身仍由速度跟踪、平衡、足端和地形奖励学习；这是实验实现，需通过固定条件评估确认效果。

服务器在现有仓库中创建新工作目录：

```bash
cd ~/gpufree-data/lite3_super
git fetch origin codex/lite3-terrain-gait-retention
git worktree add -b codex/lite3-terrain-gait-retention \
  ../lite3_terrain_retain origin/codex/lite3-terrain-gait-retention
cd ../lite3_terrain_retain
conda activate lite3_supermimic
python -m pytest tests/supermimic/test_terrain_gait_retention.py -q
```

从**步态良好的平地 AMP 4000 轮模型**重新开始，不从已经侧摆的地形 5000 轮模型继续：

```bash
SOURCE="$HOME/gpufree-data/lite3_gait/logs/lite3_locomotion/2026-10-31_18-41-34-258084_flat_amp_gait/model_4000.pt"
MOTIONS="$HOME/gpufree-data/lite3_amp_motions"
ls -lh "$SOURCE" "$MOTIONS"/lite3_直行*.json

python scripts/train_locomotion.py --stage terrain --resume "$SOURCE" \
  --amp-motion-dir "$MOTIONS" --gait-anchor-weight 0.2 \
  --num-envs 2048 --iterations 5000 --warmup-iterations 1000 \
  --actor-lr-scale 0.2 --save-interval 1000 \
  --run-name terrain_gait_retention --log-interval 10
```

观察 `train/amp_batches`、`train/amp_imagined_reward`、`train/anchor_batch`、`train/anchor_mse`，同时关注速度跟踪和跌倒率。若 `amp_batches` 或 `anchor_batch` 长期为零，说明地形采样中没有足够的平坦前进样本，保留机制实际上未生效。

在 2000、5000 轮分别以相同种子评估平地 1.0 m/s 和固定等级台阶 0.5 m/s，对比原平地 AMP 模型及无保留约束的地形 5000 轮模型。重点看 FL、HR 的 HipX 摆幅、对角足接触时序、跌倒率、速度误差和视频。高等级地形无需模仿平地关节轨迹；对高难地形的稳定性仍须单独验证。
