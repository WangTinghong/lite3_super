# Lite3 高台阶登阶改进实验

`codex/lite3-stair-ascent-refine` 从小腿接触诊断分支派生。已有追逐任务、默认运动控制任务和 `model_15000.pt` 不变。新增 `--stair-refine` 是可选训练实验，要求 `--stage terrain --harder-stairs --resume`。

现有第 8 级（18 cm/阶）在 1 m/s 的 128 环境测试中，FR 小腿接触地形的时间占比为 14.7%，其中足端未触地时为 10.9%；平均前向速度 0.73 m/s，线速度 RMSE 0.46 m/s。第 8 级 0.3 m/s 的单机 20 秒轨迹在 8～10 秒几乎停住，FR 曾连续 2.38 秒未触地。尽管零跌倒，这些数据不能证明登阶成功。

实验改动：

- 台阶前方 12 cm 和 25 cm 取最高地面，要求摆动足端相对前方落脚面留出净空；原净空项只看足下地面且最多要求约 10 cm。
- 沿小腿碰撞胶囊的轴线取三个点，对低于台阶面的部分加可微惩罚。训练同时仍保留真实小腿接触传感器作为评估指标；该传感器不作为世界模型无法预测的奖励输入。
- 用身前/身后与左右地面高差，惩罚斜向接近明显台阶边缘；不依赖固定世界朝向。
- 65% 的随机速度指令为 0.25～1.0 m/s 的正向直行。真实环境与想象轨迹使用同一个采样器。
- 起始地形等级分布为 4～8，跌倒降级下界为 4；其余地形类型仍保留。继续使用 AMP 和冻结的原步态策略锚定。

这些奖励是几何代理，是否真的减少小腿碰撞必须通过原传感器复测。高台阶本身可能超过当前模型、执行器和 10 cm 高度地图的有效工作范围；不得仅凭训练回报认定可实机登阶。

在服务器先做代码和 100 轮冒烟验证：

```bash
cd ~/gpufree-data/lite3_stair_ascent
conda activate lite3_supermimic
python -m pytest tests/supermimic/test_locomotion.py tests/supermimic/test_extended_stairs.py -q

SOURCE="$HOME/gpufree-data/lite3_extended_stairs/logs/lite3_locomotion/2026-11-01_11-16-12-355742_extended_stairs_gait/model_15000.pt"
MOTIONS="$HOME/gpufree-data/lite3_amp_motions"
python scripts/train_locomotion.py --stage terrain --harder-stairs --stair-refine \
  --resume "$SOURCE" --amp-motion-dir "$MOTIONS" --gait-anchor-weight 0.2 \
  --num-envs 256 --iterations 100 --warmup-iterations 20 \
  --critic-warmup-iterations 20 --actor-ramp-iterations 20 \
  --amp-start-iteration 1 --amp-ramp-iterations 10 \
  --wm-batch 128 --imag-batch 128 --amp-batch 128 --amp-updates 1 \
  --archive-size 2000 --no-compile --save-interval 100 \
  --run-name smoke_stair_ascent --log-interval 20
```

如冒烟测试通过，从原 `SOURCE` 新开正式实验，不接着冒烟检查点训练。先跑 4000 轮并每 500 轮存一次，以便在 12、18 cm 上台阶和平地 0.5/1.0 m/s 比较小腿接触率、前向速度、跌倒、足力和 AMP 步态：

```bash
python scripts/train_locomotion.py --stage terrain --harder-stairs --stair-refine \
  --resume "$SOURCE" --amp-motion-dir "$MOTIONS" --gait-anchor-weight 0.2 \
  --num-envs 2048 --iterations 4000 --warmup-iterations 500 \
  --speed-curriculum-iterations 2500 --actor-lr-scale 0.2 --save-interval 500 \
  --run-name stair_ascent_refine --log-interval 10
```

需要记录同一检查点、同一种子、同一等级和速度下的前后对照。尤其关注 `shank_without_foot_fraction` 是否下降，以及 `mean_body_forward_vx_mps`、跌倒和步态是否同时保持；单独降低小腿接触率可能只是学会绕开或停在台阶前。
