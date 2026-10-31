# 平地步态修正实验

这是 `codex/lite3-gait-refinement` 独立分支中的可选训练模式。原 `codex/lite3-locomotion` 分支及其正在运行的 40000 次训练不受影响。原追逐任务也不受影响。

## 为什么做这个实验

在 seed 0、平地、30 秒、固定 1.0 m/s 的记录中，约 19% 的采样时刻四足均未触地；FR 比同组的 HL 提前约 0.06 秒落地。0.5 m/s 下该错位约 0.10 秒。机器人可以稳定追踪速度，但视频呈现明显跳跃感。四足腾空本身并非错误；这里是针对用户希望更平顺、对角同步的中速运动做实验。

`--gait-refine` 仅用于 `--stage flat`。它保留原有速度与姿态奖励，另加入可微的对角腿接触错位和四足腾空惩罚，并提高垂直速度惩罚。新增惩罚只对近似直行且前进速度达到约 0.3 m/s 以上的命令逐渐生效；站立、转向和横移不会被要求保持 trot。地形、fast、adaptive 阶段仍使用原奖励。真实仿真与世界模型想象轨迹调用同一奖励函数和同一权重。

这只是一个可检验的奖励修正，并不能预先保证真实仿真会得到更平顺的步态。接触奖励使用可微的足端高度代理，需用真实接触传感器的 CSV/视频复核，以排除策略利用代理误差。

## 服务器隔离工作目录

建议当前平地训练结束后执行。服务器已有 `lite3_locomotion` 工作目录时，直接从现有仓库增加新工作目录：

```bash
cd ~/gpufree-data/lite3_super
git fetch origin codex/lite3-gait-refinement
git worktree add -b codex/lite3-gait-refinement ../lite3_gait origin/codex/lite3-gait-refinement
cd ~/gpufree-data/lite3_gait
conda activate lite3_supermimic
python scripts/train_locomotion.py --help | grep gait-refine
```

如果新工作目录已存在，只需进入 `~/gpufree-data/lite3_gait`，不要再次创建。新终端的 NVRTC `LD_LIBRARY_PATH` 若未持久化，应照之前已验证的命令恢复。

## 小规模验证与正式微调

先验证奖励和环境：

```bash
python -m pytest tests/supermimic -q
python scripts/check_locomotion.py --stage flat --num-envs 8
```

`check_locomotion.py` 验证基础任务；`--gait-refine` 的完整训练路径已在本地 CPU 小规模运行通过。服务器可用下列命令从原平地检查点开始一个新的实验 run。把 `SOURCE` 换成真实文件路径；即使从另一个工作目录训练，也可以用原工作目录的绝对路径：

```bash
SOURCE="$HOME/gpufree-data/lite3_locomotion/logs/lite3_locomotion/实际目录/model_20000.pt"
python scripts/train_locomotion.py --stage flat --gait-refine --resume "$SOURCE" \
  --num-envs 2048 --iterations 8000 --speed-curriculum-iterations 1000 \
  --warmup-iterations 1000 --actor-lr-scale 0.3 \
  --save-interval 1000 --log-interval 10 --run-name flat_gait_refine
```

不要加 `--continue-run`：新奖励需要新的 run 和新的价值网络。程序检测到检查点的奖励模式变化时会自动重置价值网络，在热身期冻结策略；世界模型和策略权重仍从原检查点继承。原检查点和训练日志不会被覆盖。

## 对照评估

分别用原检查点和本实验检查点，在相同 seed、相同速度下记录 `--gait-csv` 和 MP4。至少比较 0.5 与 1.0 m/s，检查：速度误差、跌倒、四足均腾空的时间比例、FR–HL/FL–HR 落地时差、四足支撑比例。若腾空减少但开始拖脚、速度误差或跌倒增加，则这组奖励不合格；不要只看训练损失或想象回报。

```bash
MODEL='logs/lite3_locomotion/新实验目录/model_8000.pt'
python scripts/eval_locomotion.py --checkpoint "$MODEL" --stage flat --vx 1.0 \
  --num-envs 1 --seconds 30 --gait-csv eval/gait_refine_v10.csv \
  --output eval/gait_refine_v10.json
python scripts/eval_locomotion.py --checkpoint "$MODEL" --stage flat --vx 1.0 \
  --num-envs 1 --seconds 30 --video videos/gait_refine_v10.mp4
```
