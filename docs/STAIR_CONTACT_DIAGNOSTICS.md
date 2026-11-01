# 台阶接触诊断

这个分支只为 Lite3 运动控制环境增加小腿与地形的接触传感器，以及评估输出。追逐任务、策略输入、奖励、训练参数和已有模型权重不变。旧 `model_15000.pt` 可直接评估。

`mean_speed_mps` 是机身平面速度的模；新增 `mean_body_forward_vx_mps` 与 `mean_body_lateral_vy_mps` 用来区分前进和侧移。`shank_contact_fraction` 按 FR、FL、HR、HL 顺序统计小腿碰地形的时间占比；`shank_without_foot_fraction` 只计同侧足端未触地时的小腿接触，更接近“用小腿顶台阶”的现象。`max_sampled_shank_force_N` 是所有采样中的最大值，不是平均受力。CSV 末尾新增四列 `shank_contact_*`，可与足端接触和净空逐帧对照。

在服务器原工作区更新这个分支后，先执行：

```bash
python -m pytest tests/supermimic/test_locomotion.py -q

MODEL="$HOME/gpufree-data/lite3_extended_stairs/logs/lite3_locomotion/2026-11-01_11-16-12-355742_extended_stairs_gait/model_15000.pt"
mkdir -p eval/stair_contact
python scripts/eval_locomotion.py --checkpoint "$MODEL" \
  --stage terrain --terrain stairs_inv --terrain-level 8 --vx 1.0 \
  --num-envs 1 --seconds 20 --seed 0 \
  --output eval/stair_contact/up_l8_v1.json \
  --gait-csv eval/stair_contact/up_l8_v1.csv
```

再在第 5 级、0.5 m/s 上做同样测试作对照。`falls=0` 和回合完成率不能证明已登阶；应结合视频看是否真正跨过台阶。若小腿接触率确实偏高，再针对性调整预见性足端净空、台阶朝向和小腿碰撞约束，单独开训练实验。
