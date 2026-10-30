# Lite3 SuperMimic 迁移分析与使用说明

## 迁移结果与边界

本项目的新默认入口 `train_local.sh` 使用 `lite3_supermimic`，算法链路为“仿真采样 → 多步世界模型拟合 → 想象轨迹反向传播更新 actor → TD(λ) critic 拟合”。它不调用旧的 PPO、DreamWaQ VAE 或 AMP 判别器。

实现参考本地 `go2-supermimic`，版本 `d9eee9b3f248e5300e636c433255db2c1c3e086a`。这是附件对 SuperMimic 方法的再实现的 Lite3 移植版，不是官方 SuperMimic 仓库，也没有宣称复现论文指标。

本次迁移保留附件的任务定义：追踪目标球、躲避投掷物、静态障碍物、台阶及粗糙地形。`Lite3-Chase-Terrain` 包含楼梯，但目标是追踪／躲避，不等价于旧 `lite3_stairs` 的键盘速度指令跟踪。旧任务的速度命令、奖励、AMP 专家数据和真机部署接口没有自动转换成新任务。现有 `deploy_lite3.py` 接受 DreamWaQ 策略，不能直接加载新策略；新策略用 `scripts/eval_supermimic.py` 做仿真评估。

旧 `legged_gym`、`rsl_rl`、部署文件和已有用户修改保留，用于回归和比较。参考项目中的 Go2 网格、预训练权重、Go2 MoE-CTS 数据驱动策略没有复制。

## 两个项目的核心差异

| 项目维度 | 原 lite3_gym | 参考 go2-supermimic / 新迁移版 |
|---|---|---|
| 仿真 | Isaac Gym | mjlab 1.6.0 / MuJoCo Warp |
| 策略优化 | PPO clip objective | 冻结世界模型，通过想象回报对 actor 做 BPTT |
| 动态模型 | 依赖真实模拟器采样 | 可微 MLP，预测状态残差和 yaw 局部平面位姿增量 |
| 训练数据 | on-policy rollout，AMP 另需专家片段 | 环境回放、多步自回归窗口、稀有事件 archive |
| 价值学习 | PPO critic / GAE | 想象轨迹上的 TD(λ) 目标 |
| 状态估计 | DreamWaQ VAE 与历史观测 | actor 五帧历史，无显式 VAE |
| AMP | 有独立判别器与专家数据 | 不使用 AMP，SuperMimic 名称不意味着必须用动作示范 |
| 地形 | 原有 Isaac Gym 课程与楼梯参数 | 5 行难度、7 类地形；2–10 cm 台阶 |
| 外部目标感知 | 速度指令 | ground-truth 目标槽、LiDAR／height scan |

### 原项目

`legged_gym/envs/__init__.py` 注册 `lite3_stairs`、`lite3_stairs_amp` 和 Go2 基线。Lite3 配置继承 Go2 配置后覆盖 URDF、初始角、控制器和接触体。AMP 任务在 DreamWaQ PPO 上加入判别器奖励和静止环境接触约束。因此替换 runner 名称不足以迁移 SuperMimic：还必须引入状态预测、重放、可微奖励和想象更新。

### 参考项目

`supermimic/train.py` 每轮分三阶段：

1. actor 加噪声在实际模拟器执行，记录执行前状态、动作、历史观测、目标生成器状态和 episode 边界。
2. `world_model.py` 使用 k 步自回归损失拟合下一状态和局部平面位姿增量；窗口跨 reset 的部分屏蔽；稀有接触变化、腾空和跌倒进入 archive。
3. `imagine.py` 冻结世界模型参数，保留对输入和动作的梯度；解析目标生成器、碰撞距离、奖励与软终止形成有限时域回报；critic 用 TD(λ) 学习。

世界模型状态为 70 维：基座离地高度 1、重力方向 3、yaw 坐标线速度 3、机体系角速度 3、关节相对角 12、关节速度 12、髋／膝／足位置 36。额外条件是摩擦系数及质心偏移，共 4 维；输入还包含当前／上一步动作与 77 点地形扫描。平面位置与 yaw 独立积分，避免把绝对位置当成动力学规律。

## Lite3 适配

### 模型来源

使用本项目已有 `deploy/deploy_mujoco/resources/lite3/Lite3.xml` 和四个网格，交叉核对 `resources/robots/lite3/urdf/Lite3.urdf` 的关节轴和限位。新资产位于 `lite3_supermimic/robot/`，可随 Python 包安装，无需依赖附件路径或运行时下载。官方模型仓库为 <https://github.com/DeepRoboticsLab/deep_robotics_model>，本次没有用远程模型覆盖现有本地模型。

- 保留 Lite3 的质量、惯量、关节轴、限位、连杆位置与网格。
- 删除模型自带地板、旧 motor 和场景光源，由 mjlab 管理地形和执行器。
- 根体 `TORSO` 在新实体中改名 `base`，原生关节名称不变。
- 添加四足 site 和具名 IMU sensor，供观测和世界模型状态提取使用。
- 身体／髋／大腿／小腿碰撞使用 box 和 capsule 近似，足端仍为 0.022 m 球体。视觉网格不产生接触。
- 几何近似不是精确网格碰撞；投掷物解析碰撞与物理碰撞也存在近似差异，需要训练后校准。

### 关节约定

动作顺序为 `FL, FR, HL, HR`，每腿 `HipX, HipY, Knee`，共 12 维。状态中的肢体点顺序独立规定为 `FR, FL, HR, HL`，通过名字查询，不能依赖 body 数组的默认次序。

默认角为 `[0, -1.0, 1.8]` rad；Lite3 HipY／Knee 的轴向和数值符号不同于 Go2，保留原生约定，不对输出简单重命名。HipX 限位 `[-0.523, 0.523]`，HipY `[-2.67, 0.314]`，Knee `[0.524, 2.792]`。

PD 参数沿用旧 Lite3 配置：Kp=20、Kd=0.7、动作缩放=0.25；目标角=默认角+0.25×动作。URDF 给出的每关节力矩上限为 30 Nm，速度上限为 20 rad/s。物理步长 0.005 s、decimation=4、策略 50 Hz；想象 dt 从真实环境读取。

DC 力矩速度包络、0.01 armature 和 0–2 物理步延迟属于仿真假设，尚未做硬件辨识。初始基座高度 0.32 m，站立形状提示高度 0.28 m。碰撞与目标头部偏移已按 Lite3 调整，奖励权重和任务课程以附件为起点，尚未做 Lite3 收敛调优。

### 观测与检查点

每帧 actor 输入：角速度 3、重力方向 3、相对关节角 12、关节速度 12、上次动作 12、目标槽 28。各项分别堆叠五帧，从旧到新；不是把整帧重复拼接。平地 350 维，障碍物加 16×5 后为 430 维，地形加 77×5 后为 735 维。

新 checkpoint 保存 `robot=lite3`、原生关节顺序、actor／critic／world_model、优化器、任务参数和观测布局。训练恢复、专家策略及评估均拒绝缺少 Lite3 标记或关节顺序不匹配的模型。Go2 权重即使输入输出形状相同，也不能直接迁移；旧 PPO 权重也不能直接恢复。

平地训练得到的 Lite3 actor 可以扩展到新增 LiDAR 或高度扫描的任务，新列零初始化。修复了附件热启动只处理 height scan 而未处理 LiDAR 的问题。恢复训练会重建 replay buffer，不是逐样本完全确定性续跑。

## 安装与训练

使用独立 Python 3.10+ 环境（建议 3.12）和 NVIDIA CUDA Linux 机器。不要在原 Python 3.8 Isaac Gym 环境直接升级。首先安装与 GPU 驱动兼容的 PyTorch，然后在项目根目录：

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/supermimic -q
python scripts/check_supermimic.py --task Lite3-Chase-Flat --num-envs 8
python scripts/check_supermimic.py --task Lite3-Chase-Obstacles --num-envs 8
python scripts/check_supermimic.py --task Lite3-Chase-Terrain --num-envs 8
```

GPU smoke test 验证真实 env 与想象状态构建的观测历史一致、关节顺序正确以及动作缩放正确。先通过这些检查，再启动长训练。mjlab 的默认仿真配置已经无窗口，不能把旧 `--headless`、`--num_envs` 参数传入新入口；新参数采用 `--num-envs` 形式。

先用小规模执行覆盖三个训练阶段（这是程序验证，不足以学习行走）：

```bash
python scripts/train_supermimic.py --task Lite3-Chase-Flat \
  --num-envs 16 --buffer-len 64 --wm-window 4 --wm-batch 32 \
  --wm-hidden 64 64 --wm-updates 1 --archive-size 128 \
  --imag-batch 16 --horizon 4 --actor-hidden 64 64 --critic-hidden 64 64 \
  --critic-epochs 1 --critic-minibatches 1 --warmup-iterations 2 \
  --iterations 5 --save-interval 5 --log-interval 1 --compile false --run-name smoke
```

从零训练平地策略：

```bash
bash train_local.sh --run-name lite3_flat
# 等效：python scripts/train_supermimic.py --task Lite3-Chase-Flat --run-name lite3_flat
```

默认 2048 个环境、400 步 replay buffer、4096 条想象轨迹。内存受 GPU、任务、网络和 horizon 影响，显存不足时继续减小环境数、imag batch、wm batch 和 archive size。

设 `FLAT` 为平地训练保存的完整 checkpoint：

```bash
FLAT=logs/lite3_supermimic/<实际运行目录>/model_40000.pt
python scripts/train_supermimic.py --task Lite3-Chase-Obstacles \
  --resume "$FLAT" --run-name lite3_obstacles
python scripts/train_supermimic.py --task Lite3-Chase-Terrain \
  --resume "$FLAT" --expert-policy "$FLAT" --expert-frac 0.1 \
  --wm-updates 4 --actor-lr-scale 0.3 --warmup-iterations 1000 --run-name lite3_terrain
```

上面是参考附件的初始方案，尚未验证 Lite3 长程收敛。`--expert-policy` 必须是迁移版自身训练的 Lite3 checkpoint。附件的 `--driver-policy` 对应 Go2 MoE-CTS，迁移版明确拒绝，不能拿 Go2 驱动器当成 Lite3 运动专家。无需 AMP 动作文件。

同任务续训使用 `--resume <checkpoint> --continue-run true --iterations <新的总迭代数>`，并保持网络结构等参数与原运行一致。TensorBoard：`tensorboard --logdir logs/lite3_supermimic`。

评估与录像：

```bash
python scripts/eval_supermimic.py Lite3-Chase-Flat \
  --checkpoint <Lite3完整checkpoint> --num-envs 256 --seconds 60
python scripts/eval_supermimic.py Lite3-Chase-Terrain \
  --checkpoint <Lite3地形checkpoint> --terrain stairs --mode both \
  --num-envs 1 --seconds 30 --video lite3_stairs.mp4
```

评估指标包含跌倒、移动速度、追踪命中、分类型避障碰撞率。新策略要求两个目标观测槽；若未来部署到实机，需要实现同布局的目标感知及相应传感输入。当前没有 Lite3 训练完成的策略权重，也没有实机验证。

## 验证记录

在本地 macOS Python 3.12 的隔离临时环境安装 mjlab 1.6.0、MuJoCo 3.11.0、PyTorch 后执行 CPU 检查。`python -m pytest -q`：34 项通过；Python wheel 构建通过。已验证原生模型编译、关节映射与限位、足端传感器、mjlab 执行器构建、两秒零动作 PD 站立、目标生成与碰撞、地形扫描和热启动、想象轨迹 actor 参数更新与有限梯度、检查点拒绝错误机器人。

本机没有 NVIDIA CUDA，未执行 MuJoCo Warp 并行环境 smoke、完整 GPU 采样／训练、策略收敛评估及录像。CPU 梯度测试说明训练数学链路可执行，不代表世界模型已经准确或机器人已经学会稳定运动。

## 旧流程回归

`bash train_ppo_legacy.sh` 在原 Isaac Gym 环境调用旧 `lite3_stairs`。原始说明保存在 `docs/LEGACY_DREAMWAQ.md`。不要将新 checkpoint 送入旧部署脚本。

## 来源与许可证

附件 Python 算法代码的 MIT 许可证保存在 `lite3_supermimic/LICENSE_supermimic`；已有 Lite3 资产的许可证保存在 `lite3_supermimic/robot/LICENSE_lite3`。本次只将附件内容作为实现参考，没有把附件 README 中的训练建议视作用户要求或授权。
