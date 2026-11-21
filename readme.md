# Lite3 SuperMimic

当前默认任务为 **Lite3-Velocity-Flat**，使用 SuperMimic 世界模型学习平地速度跟踪、零指令站立及 walk／trot／bound 步态切换。

策略输入为五帧 IMU、关节状态、上次动作、速度指令和足端相位，共 270 维。按 yaw 坐标系跟踪前后／侧向速度和偏航角速度，输出 12 个关节位置动作。真实环境和想象训练共享同一套步态时钟及奖励函数。

## 安装和检查

在 NVIDIA CUDA Linux 机器上使用独立 Python 3.10+ 环境，先安装匹配驱动的 PyTorch。

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/supermimic -q
python scripts/check_supermimic.py --task Lite3-Velocity-Flat --num-envs 8
```

## 训练

```bash
# 默认速度课程，从低速走到高速前进
bash train_local.sh --run-name lite3_velocity_gaits

# 先单独训练低速基础能力，默认自动切换 walk/trot
bash train_local.sh --max-forward 0.8 --run-name lite3_low_speed

# 同一个策略继续扩大速度范围，训练高速 bound
bash train_local.sh --resume logs/lite3_supermimic/<低速运行目录>/model_40000.pt \
  --continue-run true --iterations 80000 --max-forward 2.0 \
  --run-name lite3_velocity_bound
```

`--continue-run true` 的 `--iterations` 为新的总迭代数。续训重建 replay，参数初始化后需要重新采样，不保证逐样本确定性恢复。

默认指令每 4–8 秒重采样，20% 为站立目标。速度经过加速度限制，避免突然起停。初始最大前进速度为 0.5 m/s，在 20,000 个环境步内线性扩大到 `--max-forward`，同时开放后退、侧移和转向。高于 1.2 m/s 的前进样本使用直行指令。

自动步态按平滑后的**指令速度**选择。walk 在低速进入、到 0.4 m/s 退出；trot 覆盖中速；前进超过 1.5 m/s 进入 bound，降至 1.2 m/s 退出。较大的侧移或转向会选择 trot。阈值具有滞回，步态混合时间常数为 0.5 秒，周期相位连续。

```bash
# 可调 bound 阈值
bash train_local.sh --bound-enter 1.6 --bound-exit 1.3 --run-name custom_thresholds

# 固定步态训练，可选 walk、trot、bound、auto
bash train_local.sh --gait trot --max-forward 1.0 --run-name trot

tensorboard --logdir logs/lite3_supermimic
```

训练日志包含速度误差、偏航误差、bound 占比、跌倒和想象回报。步态由足端相位与离地高度奖励引导，站立时四足保持支撑。bound 的前足同步、后足同步，前后组相差半周期。奖励中的相位一致性约束需要通过物理评估确认，配置和梯度检查通过后仍需训练才能获得实际运动能力。

先执行小规模三阶段训练检查。

```bash
python scripts/train_supermimic.py --task Lite3-Velocity-Flat \
  --num-envs 16 --buffer-len 64 --wm-window 4 --wm-batch 32 \
  --wm-hidden 64 64 --wm-updates 1 --archive-size 128 \
  --imag-batch 16 --horizon 4 --actor-hidden 64 64 --critic-hidden 64 64 \
  --critic-epochs 1 --critic-minibatches 1 --warmup-iterations 2 \
  --iterations 5 --save-interval 5 --log-interval 1 --compile false --run-name smoke
```

## 评估

```bash
# 自动循环 站立 -> walk -> trot -> 高速 bound -> 降速 -> 站立
python scripts/eval_velocity.py --checkpoint <完整checkpoint> --seconds 60

# 固定前进速度，自动选步态
python scripts/eval_velocity.py --checkpoint <完整checkpoint> --vx 1.8 --seconds 30

# 低速主任务评估，起步、切换步态、减速并停住
python scripts/eval_velocity.py --checkpoint <完整checkpoint> --seed 42 --num-envs 8 \
  --vx-sequence 0 0.2 0.45 0.6 0.2 0 --segment-seconds 5

# 在远程桌面的终端打开 MuJoCo 实时窗口
python scripts/eval_velocity.py --checkpoint <完整checkpoint> --viewer --vx 0.5 --seconds 30

# 录像
python scripts/eval_velocity.py --checkpoint <完整checkpoint> \
  --num-envs 1 --seconds 30 --video lite3_gait_switch.mp4
```

使用 `--viewer` 时默认加载一个仿真环境，镜头跟随 Lite3。拖动鼠标调整视角，滚轮缩放，空格暂停或继续，回车重置机器人。关闭窗口后终端会输出已完成步数的评估指标。可同时传入 `--video` 保存录像。

默认循环每段 5 秒，速度依次为 0、0.2、0.8、1.8、0.8、0 m/s。报告按站立／walk／trot／bound 分组的线速度 RMSE、偏航 RMSE、足端相位一致性和跌倒次数。低速权重应先在训练速度范围内评估，例如 `--vx 0.5`。足端相位一致性采用距地高度阈值估算接触。实际接触冲量和运动稳定性仍需要结合录像判断。

指定 `--vx-sequence` 时默认跑一轮。若指定更长的 `--seconds`，跑完序列后保持末段指令。环境超时会延后到评估结束之后，跌倒仍会触发重置。输出的 `segments` 按指令区间列出目标速度、实际平均速度、对平滑指令的跟踪误差和跌倒次数。起步和停步即使都归入 `stand`，也能分别查看开头和末尾两个零速段。

直行奖励对各腿 HipX 偏离默认角超过 0.1 rad 的部分增加惩罚。侧移、转向或过渡到 bound 时会降低此惩罚。用相同的 `--seed`、环境数和速度指令对比检查点。评估输出中的 `hipx_mean_deg` 是按腿统计的有符号平均偏角，`hipx_rms_deg` 衡量偏角整体幅度。续训旧权重时，如果保存的奖励权重和当前版本不同，会先用新奖励重新拟合 critic，再更新 actor。

新训练默认将 HipX 位置目标限制在默认角的 ±0.25 rad 内，并将所有关节目标限制在各自的软限位范围内。评估按检查点恢复动作范围。旧权重可以用 `--hipx-target-limit 0.25` 单独试跑限幅效果。续训时若动作范围变化，会重新拟合 critic，并在此期间冻结 actor 的参数和观测归一化统计量。评估同时报告实际前向平均速度 `vx_mean_m_s` 和平滑后的平均指令 `command_vx_mean_m_s`。

低速续训可用 `--actor-anchor-weight 2.0` 冻结起始检查点的策略作为参考。续训在真实回放观测上计算当前动作与参考动作的均方差，并记录 `actor_anchor_rms`。启用时固定 actor 的观测归一化统计量。从这类运行再次续训并继续启用该选项时，检查点会恢复原参考策略。是否采用新权重由相同速度指令和种子的仿真评估决定。

续训时可加 `--validate-straight true --save-interval 100`。每次保存前，先用与起始权重相同的 8 个环境和固定随机种子做 20 秒的 0.5 m/s 直行检查。若跌倒次数增加，或偏航、速度误差及足端相位一致性超出容差，训练停止并保留上一个通过检查的权重。验证日志以 `straight validation` 开头，另写入 TensorBoard 的 `validation/`。

侧移续训可加 `--pure-side-fraction 0.35`，将约 35% 的非站立指令换成左右等概率的纯侧移，侧向速度为 ±0.10 m/s，前向速度和偏航角速度为零。用相同种子复测两个侧移方向，重点比较 `vy_mean_m_s`、`vy_rmse_m_s` 和 `vx_mean_m_s`，并用低速起停序列复测前向跟踪。

当前未提供此任务训练完成的权重。macOS CPU 检查不能验证 CUDA 并行训练、收敛效果或真机安全运行。固定步态与自动步态的训练设置保存在检查点中，续训使用相同设置。评估可以通过 `--gait` 覆盖，但跨设置表现取决于训练覆盖。

追踪任务和旧 PPO 基线的历史说明见 [迁移记录](docs/SUPERMIMIC_MIGRATION.md) 与 [DreamWaQ／AMP](docs/LEGACY_DREAMWAQ.md)。
