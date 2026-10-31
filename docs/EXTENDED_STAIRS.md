# Lite3 扩展台阶实验

`codex/lite3-extended-stairs` 给运动控制地形阶段增加可选的 `--harder-stairs`。原追逐任务及默认五级地形不变。

台阶高度从原先五级的每级 2、4、6、8、10 cm，扩为九级的每级 2、4、6、8、10、12、14、16、18 cm，`stairs` 与 `stairs_inv` 同步扩展。新增等级是仿真实验范围，不代表 Lite3 已能稳定攀爬或具备实机安全性。金字塔地形包含连续多级台阶，最高处的总爬升远大于单级高度。

九级地形的高度扫描改用 10 cm 地图网格；每块台阶踏面深 40 cm，仍有约四个地图采样点。原 2 cm 网格在九级布局下首次启动需两千多万次 CPU 射线采样。新网格减少初始化成本，但台阶边缘的地面高度会比原分辨率更粗，须在仿真中验证接触和速度表现。

从已有的、步态稳定的地形检查点**新开一次训练**，不要使用 `--continue-run`：

```bash
cd ~/gpufree-data/lite3_super
git fetch origin codex/lite3-extended-stairs
git worktree add -b codex/lite3-extended-stairs \
  ../lite3_extended_stairs origin/codex/lite3-extended-stairs
cd ../lite3_extended_stairs
conda activate lite3_supermimic
python -m pytest tests/supermimic/test_extended_stairs.py -q

SOURCE="$HOME/gpufree-data/lite3_terrain_retain/logs/lite3_locomotion/2026-10-31_21-41-05-601121_terrain_gait_retention/model_5000.pt"
MOTIONS="$HOME/gpufree-data/lite3_amp_motions"
```

先用较小规模确认扩展地形、AMP、策略保留及存盘在服务器 GPU 上能完整运行：

```bash
python scripts/train_locomotion.py --stage terrain --harder-stairs \
  --resume "$SOURCE" --amp-motion-dir "$MOTIONS" --gait-anchor-weight 0.2 \
  --num-envs 64 --iterations 30 --warmup-iterations 5 \
  --critic-warmup-iterations 5 --actor-ramp-iterations 5 \
  --amp-start-iteration 1 --amp-ramp-iterations 5 \
  --amp-batch 64 --amp-updates 1 --wm-batch 64 --imag-batch 64 \
  --archive-size 1000 --no-compile --save-interval 30 \
  --run-name smoke_extended_stairs --log-interval 5
```

成功保存后，从原 `SOURCE` 重新开始正式实验：

```bash
python scripts/train_locomotion.py --stage terrain --harder-stairs \
  --resume "$SOURCE" --amp-motion-dir "$MOTIONS" --gait-anchor-weight 0.2 \
  --num-envs 2048 --iterations 15000 --warmup-iterations 1000 \
  --actor-lr-scale 0.2 --save-interval 1000 \
  --run-name extended_stairs_gait --log-interval 10
```

若已有后续表现更好的保留步态地形检查点，可把 `SOURCE` 换为其实际路径。新地形布局会重置价值网络并从旧五级范围内（0～4 级）分配初始地形，之后按成功移动与跟踪逐级升级；同时，速度课程在新 run 重新开始。不能仅凭训练次数判断模型接触过 5～8 级，应查看 `Curriculum/terrain_levels/mean_level` 和检查点中的 `terrain_levels` 分布。

评估脚本自动从检查点配置恢复九级地形。先复测 0～4 级，确认未失去原能力，再逐级检查 5～8 级；对上台阶用 `stairs_inv`，下台阶用 `stairs`：

```bash
MODEL='logs/lite3_locomotion/实际训练目录/model_15000.pt'
python scripts/eval_locomotion.py --checkpoint "$MODEL" --stage terrain \
  --terrain stairs_inv --terrain-level 5 --vx 0.3 \
  --num-envs 128 --seconds 60 --seed 0 --output eval/up_l5_v03.json
```

从低速开始，并分别检查跌倒、速度误差、足力峰值、视频中的上/下台阶过程和平地 trot。更高等级的非跌倒回合完成比例仍不等于严格路线通过率。
