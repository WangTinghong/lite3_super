# Lite3 SuperMimic

新增独立的速度指令运动控制实验：平地、多地形、高速平地、联合适应四阶段，见 [运动控制训练与评估](docs/LOCOMOTION.md)。入口为 `scripts/train_locomotion.py`，日志为 `logs/lite3_locomotion`。原追逐/躲避入口保持兼容；新旧检查点不可混用。四阶段运动能力尚待 GPU 训练和评估验证。

Lite3 四足机器人的世界模型强化学习项目，新默认训练链路使用多步动力学学习和想象轨迹反向传播，旧 PPO／DreamWaQ／AMP 代码保留作基线。

完整的[两项目分析、机器人适配、安装训练及验证说明](docs/SUPERMIMIC_MIGRATION.md)。

在 NVIDIA CUDA Linux 机器、独立 Python 3.10+ 环境中安装好 PyTorch 后：

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/supermimic -q
python scripts/check_supermimic.py --task Lite3-Chase-Flat --num-envs 8
bash train_local.sh --run-name lite3_flat
```

任务：`Lite3-Chase-Flat`、`Lite3-Chase-Obstacles`、`Lite3-Chase-Terrain`。地形任务包含楼梯，沿用附件的追踪／躲避目标；与原 `lite3_stairs` 速度指令任务不同。完整 GPU 训练和 Lite3 收敛尚未验证，没有预训练 Lite3 权重。

CPU 检查覆盖模型、动作映射、PD 站立、世界模型及完整想象策略梯度。新策略不能加载 Go2 或旧 PPO 权重，也不能直接用于旧真机部署接口。

旧流程见 [DreamWaQ／AMP 说明](docs/LEGACY_DREAMWAQ.md)，在原环境中运行 `bash train_ppo_legacy.sh`。
