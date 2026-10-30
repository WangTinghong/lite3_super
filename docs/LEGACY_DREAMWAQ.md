# Lite3 DreamWaQ

本项目基于 Legged Gym、Isaac Gym 和 RSL-RL，实现 Lite3 四足机器人的 DreamWaQ 强化学习训练、AMP 步态优化以及 MuJoCo 仿真部署。原有 `lite3_stairs` 任务保留作为基线；新增的 `lite3_stairs_amp` 用于在不改变基线训练链路的前提下改善步态自然度。

## 环境安装

建议使用 Python 3.8，并先安装 Isaac Gym：

```bash
pip install -e ~/isaacgym/python
pip install -e ./rsl_rl
export PYTHONPATH="$PWD:$PWD/rsl_rl:${PYTHONPATH:-}"
```

需要设置 Isaac Gym 和项目依赖能够被当前 Python 环境找到。训练和播放命令均在项目根目录执行。

## Lite3 AMP 训练

AMP 专家数据位于 [`lite3_motions/`](../lite3_motions/)。训练任务会读取其中的 Lite3 运动片段，并将 AMP 判别器奖励以较小权重混入原任务奖励。

为修正零速度时右后腿可能腾空的问题，`lite3_stairs_amp` 额外保留 10% 的固定静止环境。在这些环境中命令始终为零，并启用四足同时接触约束和仅静止、近水平姿态生效的左右关节对称约束。运动环境仍使用原有命令采样和奖励；基线任务 `lite3_stairs` 不包含这些约束。

从当前基线开始新的 AMP 训练：

```bash
python legged_gym/scripts/train.py \
  --task=lite3_stairs_amp \
  --num_envs=8192 \
  --headless \
  --run_name=amp_safe_v1
```

从已有 AMP checkpoint 继续训练：

```bash
python legged_gym/scripts/train.py \
  --task=lite3_stairs_amp \
  --num_envs=8192 \
  --headless \
  --resume \
  --load_run=Aug09_14-36-29_stand_fix_v1 \
  --checkpoint=28000 \
  --max_iterations=5000 \
  --run_name=stand_fix_v2
```

将 `--load_run` 和 `--checkpoint` 替换为实际存在的日志目录和模型编号。训练日志及 checkpoint 默认写入 `logs/`，该目录被 Git 忽略。

## 播放与导出策略

播放训练好的策略，同时会将当前 actor/VAE 导出为 MuJoCo 使用的 `policy_dwaq.pt`：

```bash
python legged_gym/scripts/play.py \
  --task=lite3_stairs_amp \
  --load_run=Aug09_14-36-29_stand_fix_v1 \
  --checkpoint=31000 \
  --num_envs=1
```

导出文件位置为：

```text
logs/lite3_stairs_amp/exported/policies/policy_dwaq.pt
```

## Lite3 MuJoCo 仿真

MuJoCo 配置和地形位于 [`deploy/deploy_mujoco/`](../deploy/deploy_mujoco/)。配置文件中默认启用 `cross_stairs.xml`；也可以在 [`deploy/deploy_mujoco/configs/lite3.yaml`](../deploy/deploy_mujoco/configs/lite3.yaml) 中切换为 `flat.xml`、`cross_slope.xml`、`race_track.xml` 或 `stairs.xml`。

使用项目中已准备好的部署策略：

```bash
python deploy/deploy_mujoco/deploy_lite3.py \
  --config deploy/deploy_mujoco/configs/lite3.yaml
```

使用刚导出的策略时传入绝对路径。部署脚本的相对路径是相对于 `deploy/` 目录解析的，因此从项目根目录直接写 `logs/...` 会被解析为 `deploy/logs/...`：

```bash
python deploy/deploy_mujoco/deploy_lite3.py \
  --config deploy/deploy_mujoco/configs/lite3.yaml \
  --policy-path "$PWD/logs/lite3_stairs_amp/exported/policies/policy_dwaq.pt"
```

只检查配置、XML 和策略是否可加载，不启动窗口：

```bash
python deploy/deploy_mujoco/deploy_lite3.py \
  --config deploy/deploy_mujoco/configs/lite3.yaml \
  --policy-path "$PWD/logs/lite3_stairs_amp/exported/policies/policy_dwaq.pt" \
  --check-only
```

仿真运行时使用键盘实时发送速度命令；松开按键后命令通过平滑器回到零，策略本身负责保持站立，不在部署代码中硬覆盖默认关节角。

## 目录说明

- `legged_gym/`：Isaac Gym 环境、任务配置和训练脚本。
- `rsl_rl/`：DreamWaQ、AMP 判别器、专家数据加载和 runner。
- `resources/robots/lite3/`：Lite3 URDF、网格和 Isaac Gym 资源。
- `deploy/deploy_mujoco/`：Lite3 MuJoCo 模型、地形和键盘控制部署脚本。
- `lite3_motions/`：AMP 专家运动数据。

`AMP_for_hardware/` 是参考项目的独立第三方仓库，不是本项目运行时依赖，不随本仓库上传。
## 参考项目：
https://github.com/yusongmin1/Dreamwaq；
https://github.com/escontra/AMP_for_hardware；
https://github.com/wzx8848/lite3_motions。
