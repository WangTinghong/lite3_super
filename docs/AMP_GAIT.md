# Lite3 平地直行 AMP 步态实验

此实验位于 `codex/lite3-amp-gait`，在现有 `codex/lite3-hipx-target-refinement` 上增加可选的 AMP 步态奖励。原追逐任务和原运动控制检查点保持可用。AMP 用示范与真实仿真过渡训练判别器，再在世界模型的想象轨迹里为策略提供可微的步态奖励；速度跟踪与防摔等奖励仍然保留。这是待 GPU 验证的实验，不能仅凭 AMP 奖励宣布步态已改善。

## 动作数据

原项目动作目录实际为 `lite3_gym/lite3_motions/lite3_motions-main/`，其中有 `lite3_直行慢速.json` 和 `lite3_直行快速.json`。两段各约 10 秒，帧间隔 0.02 秒，示范速度约 0.50 和 0.79 m/s。关节顺序 LF/RF/LH/RH 对应新项目 FL/FR/HL/HR；关节角和角速度按数据说明都要取反。快速示范以 1.0、1.15、1.25 倍速采样，扩展到约 1.0 m/s；超过这个速度的高质量步态没有示范支持。

为避免复制来源不明的动作文件到代码仓库，代码只接收动作目录路径。先把上述两段 JSON 上传到服务器的 `~/gpufree-data/lite3_amp_motions/`，然后确认：

```bash
ls -lh ~/gpufree-data/lite3_amp_motions/lite3_直行慢速.json \
       ~/gpufree-data/lite3_amp_motions/lite3_直行快速.json
```

## 训练

在服务器现有 `lite3_super` 仓库中获取已推送的分支并建立独立工作目录。若已经有同名工作目录，进入该目录即可。

```bash
cd ~/gpufree-data/lite3_super
git fetch origin codex/lite3-amp-gait
git worktree add -b codex/lite3-amp-gait ../lite3_amp_gait origin/codex/lite3-amp-gait
cd ~/gpufree-data/lite3_amp_gait
conda activate lite3_supermimic
python -m pytest tests/supermimic/test_locomotion_amp.py -q
```

建议从已经训练的 `flat_hipx_target_refine/model_4000.pt` 开始一个**新**训练实验，保留原检查点用于对照：

```bash
SOURCE="$HOME/gpufree-data/lite3_hipx_target/logs/lite3_locomotion/2026-10-31_16-00-56-897021_flat_hipx_target_refine/model_4000.pt"
MOTIONS="$HOME/gpufree-data/lite3_amp_motions"
test -f "$SOURCE" && test -f "$MOTIONS/lite3_直行慢速.json" && test -f "$MOTIONS/lite3_直行快速.json"
```

先运行一个短时冒烟测试，确认 GPU 上的 AMP 判别器训练、想象奖励、策略更新和存盘都能完成。这个小模型只用于检查流程，不作为正式策略：

```bash
python scripts/train_locomotion.py --stage flat --gait-refine \
  --hipx-refine --hipx-target-refine --amp-motion-dir "$MOTIONS" \
  --resume "$SOURCE" --num-envs 64 --iterations 30 \
  --warmup-iterations 5 --critic-warmup-iterations 5 \
  --actor-ramp-iterations 5 --speed-curriculum-iterations 10 \
  --amp-start-iteration 1 --amp-ramp-iterations 5 \
  --amp-batch 64 --amp-updates 1 --wm-batch 64 --imag-batch 64 \
  --archive-size 1000 --no-compile --save-interval 30 \
  --run-name smoke_amp_gait --log-interval 5
```

通过后，从原始 `SOURCE` 重新开始正式实验：

```bash
python scripts/train_locomotion.py --stage flat --gait-refine \
  --hipx-refine --hipx-target-refine --amp-motion-dir "$MOTIONS" \
  --resume "$SOURCE" --num-envs 2048 --iterations 4000 \
  --warmup-iterations 1000 --actor-lr-scale 0.2 \
  --speed-curriculum-iterations 1500 \
  --run-name flat_amp_gait --log-interval 10
```

启动后判别器会先学习；AMP 权重从第 500 次起缓慢增加，到第 1500 次达到默认值 0.5。速度课程也在第 1500 次达到平地阶段的 1.0 m/s 上限，使 4000 次实验能覆盖目标速度。载入未使用 AMP 的检查点时，价值网络重置并热身，策略权重及世界模型沿用。新实验会写入新的日志目录。TensorBoard 查看 `train/amp_d_expert`、`train/amp_d_policy`、`train/amp_scale`、`train/amp_imagined_reward`，同时关注速度误差和跌倒率。判别器分数只供诊断，不能代替真实评估。

## 对照评估

分别用原模型和新模型在固定种子下测试 0.5、0.8、1.0 m/s，每个速度至少 30 秒：

```bash
MODEL='logs/lite3_locomotion/实际AMP训练目录/model_4000.pt'
python scripts/eval_locomotion.py --checkpoint "$MODEL" --stage flat \
  --vx 1.0 --seconds 30 --num-envs 1 --seed 0 \
  --video videos/amp_flat_v1.mp4 \
  --gait-csv eval/amp_flat_v1_gait.csv \
  --joint-csv eval/amp_flat_v1_joint.csv \
  --output eval/amp_flat_v1.json
```

比较 FL、HR 的 HipX 摆幅与 FR、HL 的对称性，以及四腿接触相位、机身上下速度、速度误差、足端打滑和跌倒。若新策略只是让判别器奖励上升，却让这些指标或视频恶化，应降低 `--amp-weight` 或回到原检查点。当前示范只包含平地直行，本实验暂不把 AMP 奖励施加到转弯、横移及复杂地形。

若转入地形训练后平地步态退化，可使用 [地形步态保留实验](TERRAIN_GAIT_RETENTION.md) 在平坦区域延续 AMP 与原策略约束。
