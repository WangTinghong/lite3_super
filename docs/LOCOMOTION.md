# Lite3 运动控制：独立分支与四阶段训练

平地直行步态的 AMP 实验见 [AMP 步态实验](AMP_GAIT.md)。该实验需要单独分支和本机原项目中的两段动作 JSON，不会修改追逐任务。

本功能在 `codex/lite3-locomotion` 分支。原有 Chase 源码、任务名称、训练脚本和模型格式均未修改。新任务单独缩短四条小腿的简化碰撞胶囊 3 cm，使足底球及接触传感器真实接触地面；原机器人 MJCF 文件不变。新模块为 `lite3_supermimic/locomotion`，新入口为 `scripts/train_locomotion.py`、`scripts/eval_locomotion.py`。

这是一套待训练、待 GPU 验证的实验实现；创建四个任务不等于机器人已经具备高速稳定或步态自适应能力。已有的追逐检查点不能直接加载到这些任务。

## 在服务器上隔离工作目录

先将本地新分支推到 GitHub；然后在服务器现有仓库里仅获取新分支，创建另一个工作目录。以下命令不会切换原目录分支，也不会移动或覆盖现有训练日志、视频及未提交修改。

```bash
cd ~/gpufree-data/lite3_super
git fetch origin codex/lite3-locomotion
git worktree add -b codex/lite3-locomotion ../lite3_locomotion origin/codex/lite3-locomotion
cd ~/gpufree-data/lite3_locomotion
conda activate lite3_supermimic
```

如果该分支或工作目录已经存在，直接进入已创建的 `lite3_locomotion` 目录，不要重复创建。新脚本会优先导入自己的项目目录，无需重新执行 `pip install -e .` 改写原目录的安装指向；依赖仍是现有的 mjlab 1.6.0 环境。

Linux 新终端如果仍遇到 NVRTC 动态库问题，在当前 shell 恢复之前已验证的 `LD_LIBRARY_PATH`，不必重装 PyTorch。

## 训练前验证

```bash
python -m pytest tests/supermimic -q
python scripts/check_locomotion.py --stage flat --num-envs 8
python scripts/check_locomotion.py --stage terrain --num-envs 8
```

检查内容包括实际环境步进、观测历史顺序、有限奖励与想象轨迹的策略梯度。初次生成高度图、编译 Warp 内核会耗时。检查脚本通过后，再进行小规模完整训练：

```bash
python scripts/train_locomotion.py --stage flat \
  --num-envs 64 --iterations 30 --warmup-iterations 10 \
  --wm-batch 64 --imag-batch 64 --archive-size 1000 \
  --wm-hidden 128 128 --actor-hidden 64 64 --critic-hidden 64 64 \
  --no-compile --log-interval 5 --save-interval 30 --run-name smoke_flat
```

这个小模型只验证完整采集、世界模型更新、策略更新、保存流程。正式训练重新开始，勿加载它。随后可用同样的小规模配置去掉 `--no-compile` 验证服务器 PyTorch/Triton 编译路径。

## 四个阶段

| 阶段参数 | 任务名 | 地形 | 前向速度上限课程 |
|---|---|---|---|
| `flat` | `Lite3-Locomotion-Flat` | 平地 | 0.5 → 1.0 m/s |
| `terrain` | `Lite3-Locomotion-Terrain` | 平地、上下台阶、凸块、粗糙地面、上下坡 | 0.5 → 1.5 m/s |
| `fast` | `Lite3-Locomotion-Fast` | 平地 | 1.0 → 3.0 m/s |
| `adaptive` | `Lite3-Locomotion-Adaptive` | 混合地形 | 简单地形最高 3.0 m/s，困难地形降低指令范围 |

这些上限是**仿真实验设置，不是 Lite3 已验证的安全能力**。每阶段也采样站立、后退、横移、转向；统一使用机器人朝向的水平坐标系速度指令。`adaptive` 缩短指令保持时间，并随机化初始线速度和角速度以练习恢复；所有阶段沿用摩擦、质心和编码器偏差随机化。当前没有训练未建模的连续外力冲击，不能据此宣称抗任意推力。

地形等级根据每个环境的实际表现调整：完成足够路程、跟踪误差较低且未跌倒时升级，跌倒时降级。地形边界作为截断重置，防止跑到其他类型地块污染测试。速度上限在默认 15000 次迭代内逐渐扩展。**是否进入下一阶段仍由固定条件下的评估决定，不按训练次数自动宣布达标。**

正式阶段一：

```bash
python scripts/train_locomotion.py --stage flat --num-envs 2048 \
  --iterations 40000 --log-interval 10
```

阶段一通过速度跟踪和启停评估后，将 `saved ...` 输出的实际文件路径填入 `FLAT`：

```bash
FLAT='logs/lite3_locomotion/实际阶段一目录/model_40000.pt'
python scripts/train_locomotion.py --stage terrain --resume "$FLAT" \
  --num-envs 2048 --iterations 40000 --log-interval 10
```

阶段二通过低中速多地形评估后：

```bash
TERRAIN='logs/lite3_locomotion/实际阶段二目录/model_40000.pt'
python scripts/train_locomotion.py --stage fast --resume "$TERRAIN" \
  --num-envs 2048 --iterations 40000 --log-interval 10
```

阶段三通过高速平地评估后：

```bash
FAST='logs/lite3_locomotion/实际阶段三目录/model_40000.pt'
python scripts/train_locomotion.py --stage adaptive --resume "$FAST" \
  --num-envs 2048 --iterations 40000 --log-interval 10
```

四阶段的 610 维观测（5 帧历史）保持一致。转阶段时继承策略和世界模型，重新训练价值网络，并在热身期冻结策略，以适应新的数据分布。阶段三的平地训练可能遗忘地形能力，阶段四应从简单地形重新开始并持续评估；保留阶段二检查点用于对照。

`--resume` 默认重新开始当前阶段的迭代计数和课程。只有同阶段中断后续训才加 `--continue-run`；此时 `--iterations` 表示目标总迭代数。检查点不包含回放缓存或完整模拟器状态，续训会重新收集并热身，不能保证逐位复现中断前轨迹。

## 评估、MuJoCo 窗口和视频

先以明确的速度指令测平地。`MODEL` 必须是新任务的检查点：

```bash
MODEL='logs/lite3_locomotion/实际训练目录/model_40000.pt'
python scripts/eval_locomotion.py --checkpoint "$MODEL" --stage flat \
  --vx 0.5 --num-envs 256 --seconds 60 --seed 0 --output eval/flat_v05.json
```

指定台阶类型与固定等级，所有环境和重置后均保持该条件：

```bash
python scripts/eval_locomotion.py --checkpoint "$MODEL" --stage terrain \
  --terrain stairs --terrain-level 2 --vx 0.5 \
  --num-envs 256 --seconds 60 --seed 0 --output eval/stairs_l2_v05.json
```

改变 `--vx`、`--vy`、`--yaw-rate` 分别测直行、横移、转向。先在平地验证更高速度，再测低等级地形。至少使用种子 0、1、2 重复评估。`--random-commands` 则测指令切换；默认是恒定指令。

远程桌面终端直接打开 MuJoCo 窗口：

```bash
python scripts/eval_locomotion.py --checkpoint "$MODEL" --stage terrain \
  --terrain stairs --terrain-level 1 --vx 0.5 --viewer --seconds 120
```

在窗口中用键盘实时调整速度指令（从静止开始，点击 MuJoCo 窗口使其获得焦点）：

```bash
python scripts/eval_locomotion.py --checkpoint "$MODEL" --stage flat \
  --viewer --keyboard --seconds 300
```

每次按键改变一个速度档：`W/S` 前进/后退各 0.1 m/s，`J/L` 左/右横移各 0.1 m/s，`Q/E` 左/右转各 0.15 rad/s，`X` 将三个指令归零。终端会打印当前速度指令。这里控制的是机身坐标系目标速度，策略仍负责生成 12 个关节动作；不会直接用键盘操纵关节。可将 `--stage flat` 改为 `terrain` 并选择 `--terrain stairs --terrain-level 1`。键盘设定的速度限制为对应阶段训练的范围；在高等级地形上宜从低速开始。

视频与接触时间线：

```bash
python scripts/eval_locomotion.py --checkpoint "$MODEL" --stage adaptive \
  --terrain rough --terrain-level 1 --vx 1.0 --num-envs 1 --seconds 30 \
  --video videos/rough_v1.mp4 --gait-csv eval/rough_v1_gait.csv --output eval/rough_v1.json
```

输出包括线速度矢量 RMSE、角速度 RMSE、每机器人分钟跌倒数、完成回合数与非跌倒结束比例、平均速度、动作变化、支撑期间足端打滑速度、足端最大接触力、各腿支撑比例、落脚频率、相邻有效落脚的步长、摆动时平均离地高度。回合比例受边界截断影响，**不能直接视为标准障碍通过率**；严格通过率需另外定义固定起终点路线。

姿态与速度统计包括终止前状态；手动重置发生在测量之后。步长计算排除跨回合瞬移。CSV 的腿顺序为 FR、FL、HR、HL，带终止标记，可用于绘制接触时序图。报告同时保存地形、等级、速度、种子，避免混合条件下的均值掩盖问题。

## 步态自适应的范围与局限

当前策略自由输出关节目标，没有预设固定步频或强制对角相位。它通过速度跟踪、平衡、动作平顺、足端滑动和地面净空奖励学习运动。是否形成有用的步频、步长、抬脚高度和接触时序调整，需要对比不同地形和速度下的 CSV/JSON。

奖励计算在真实环境与想象训练中共享；终止在真实环境中为硬判定，在想象中为平滑概率。足端滑动/净空奖励使用可微的几何接触近似；评估的接触数据来自仿真传感器。mjlab 的奖励与最终观测存在一个物理子步的运动学更新时间差，检查代码需按其接口理解。

高度扫描目前由仿真真值高度图提供。实机部署还需要对应感知/状态估计输入与部署链路验证。**本版本不保证自动形成指定的 walk/trot/gallop 离散步态，也不以降低命令速度的课程机制证明策略自主安全减速。**

TensorBoard：

```bash
tensorboard --logdir logs/lite3_locomotion --host 0.0.0.0 --port 6006
```

优先观察 `locomotion/linear_error_mps`、`locomotion/yaw_error_radps`、`locomotion/falls_per_minute`、`Curriculum/terrain_levels/mean_level`；世界模型损失和想象回报只作为训练诊断。正式能力验收使用独立固定条件评估。
