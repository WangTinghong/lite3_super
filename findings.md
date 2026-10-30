# 研究发现

- 当前仓库 `deploy/deploy_mujoco/deploy_lite3.py` 使用 `pygame.event.pump()` + `pygame.key.get_pressed()` 在控制周期读取按键。
- `get_keyboard_command()` 将 W/S、A/D、Q/E 映射到三维速度指令。
- `cmd_target` 在没有键盘输入且没有手柄时回到 `cmd_init`，随后通过 `cmd_smoothing` 更新实际 `cmd`。
- `/home/abc/rl_gym_lite3_2/deploy/deploy_mujoco/deploy_go2.py` 当前仅在有手柄时更新 `cmd`，没有键盘映射函数；其 `cmd_init` 为 `[1.0, 0.0, 0.0]`。
- 目标脚本的观测布局与当前 Lite3 脚本不同，不能直接复制整段策略循环；只移植输入、指令平滑和配置读取，保留目标脚本的观测与动作布局。
- 目标脚本已有 `pygame` 依赖和 Xbox 读取函数，因此键盘和手柄可以共存：有键盘指令时优先键盘，否则回退到手柄，再否则回到零指令。
- 目标脚本的控制周期为 `simulation_dt * control_decimation = 0.002 * 10 = 0.02s`，键盘状态按 50Hz 读取。
- `python -c compile(...)` 通过；YAML 可解析并包含 `cmd_init=[0,0,0]`、`cmd_smoothing=0.15`。
- `/home/abc/rl_gym_lite3/deploy/deploy_mujoco/deploy_go2.py` 原先通过 MuJoCo viewer callback 累加命令；该入口通过 `--config lite3.yaml` 运行 Lite3。
- `/home/abc/rl_gym_lite3` 的 Lite3 XML 和 `logs/lite3_moe_cts/exported/policies/policy.pt` 均存在。
- 目标 3 仓库的脚本语法、YAML 解析和键盘函数静态行为测试通过；当前环境仍未安装 `pygame`/`mujoco`，未启动窗口仿真。

## Lite3 AMP 初步发现

- 当前 Lite3 使用 `ActorCriticDreamWaQ`、`PPO_DreamWaQ` 和 `DreamWaQRunner`，不能直接替换为参考项目面向标准 ActorCritic 的 AMP PPO。
- 当前策略单帧 observation 为 45 维，AMP 风格状态应作为判别器的独立输入，不能改变 actor observation contract。
- `lite3_motions` 文件每帧为 61 维、帧间隔为 0.02 秒；字段映射和腿顺序仍需用生成脚本/模型几何核实。
- 为降低退化风险，AMP 应通过新任务启用，task reward 保持主导，第一阶段关闭 reference-state initialization。
- 零指令右后腿腾空需要独立的 Lite3 站立接触/足端一致性奖励，且只能在静止命令时启用。
- `lite3_motions/README.md` 明确原始腿序为 `LF, RF, LH, RH`，与当前 Lite3 的 `FL, FR, HL, HR` 语义一致，无需套用参考 A1 loader 的 PyBullet 腿重排。
- 专家数据的 joint position/joint velocity 轴方向与训练 URDF 相反，Lite3 loader 必须只对这两段取负；足端、机身速度不能随之取负。
- 工作区内 `ppo_dreamwaq.py` 和 `actor_critic_dreamwaq.py` 已有既存修改；AMP 应优先新增派生算法/runner，避免覆盖稳定基线和用户改动。
- 现有 `lite3_stairs` 任务注册保持不变，后续新注册名暂定 `lite3_stairs_amp`。
- 静止专家帧的 12 个关节角为四腿一致的 `[0, 0.8, -1.5]`（取反后匹配训练轴为 `[0, -0.8, 1.5]`），但 toe local z 分别约为 `[-0.234, -0.285, -0.287, -0.237]`，右后足比左后足高约 5 cm。
- 上述静止足端不对称与用户报告的右后腿腾空方向一致；第一阶段判别器不得包含数据中的 toe position/velocity，否则可能强化该问题。
- Lite3 AMP state 决定采用 30 维：joint pos 12 + base linear velocity 3 + base angular velocity 3 + joint velocity 12。actor 的 45 维 observation 完全不变。
- `lite3_motions/README.md` 声称 quaternion 为 `(w,x,y,z)`，但静止 identity 实际写成 `[0,0,0,1]`，格式自相矛盾。由于第一阶段不做 reference initialization 且 AMP state 不含 quaternion，不让这个不确定字段进入训练。
- 参考 AMPPPO 将判别器和策略放在同一 optimizer；本项目为降低对现有 DreamWaQ/VAE 的干扰，计划使用独立 discriminator optimizer，并在 PPO 更新后单独训练判别器。
- 当前 `GO2_Stairs_Robot.step()` 在返回 observation 前已对 `done` 环境执行 reset；AMP replay 必须过滤终止 transition，否则 next state 实际是新 episode 的初始状态。
- 参考 AMPPPO 虽计算了 `not_done_idxs` 却没有使用，仍插入了全部 transition；Lite3 实现会修正这一点，并屏蔽终止步 AMP reward。
- `TaskRegistry` 通过模块级 `eval(runner_class_name)` 选择 runner，且 `DreamWaQRunner` 内部硬编码 `PPO_DreamWaQ`；接入需新增并显式导入 `DreamWaQAMPRunner`，不能只改配置字符串。
- 环境 reward scale 会乘控制周期 `dt=0.02`。站立接触项计划返回四脚接触比例，使用较小正 scale，并以 `norm(command[:3]) < 0.1` 门控；不会影响运动中的正常摆腿。
- AMP reward 不经过环境的 `dt` 缩放，因此不能照搬参考值 `2.0`，也不宜直接用先前粗略设想的 `0.05~0.2`。为了让 task reward 始终占主导，默认系数将按本项目每步 reward 量级设为 `0.005`，并支持 warmup/ramp。
- AMP warmup 必须按“AMP 已训练迭代数”计数，而不是沿用 baseline checkpoint 的全局迭代号；从成熟 checkpoint 微调时判别器仍需从零预热。
- 参考 normalizer 每次在 CPU/Numpy 与 GPU 间转换；Lite3 版本计划实现 torch buffer normalizer，并把统计量存入 AMP checkpoint。
- 动作统计发现速度坐标不统一：`直行快速` 平均 lin vel 约 `[0.786,-0.023,0]`，`直行慢速` 约 `[-0.091,-0.492,0]`，左右移动主要落在 x 轴。这表明专家速度很可能是世界系且各片段初始朝向不同，不能直接与环境机体系速度拼入 discriminator。
- 在完成 quaternion/坐标变换核对前，最保守的 AMP state 应进一步收紧为 joint position 12 + joint velocity 12（24 维）；这两类字段的轴方向和腿序已有 README 明确定义。
- Lite3 URDF 的实际路径为 `resources/robots/lite3/urdf/Lite3.urdf`，DOF/FOOT 声明顺序为 FL、FR、HL、HR，与专家 README 顺序一致。
- root position 的帧差与记录的 base linear velocity 一致，证明该速度是世界系；root quaternion 实际为 `[x,y,z,w]`（README 此处写错）。
- 使用 quaternion inverse rotation 后，机体系平均速度与动作语义一致：右移 y=-0.280、左移 y=+0.281、后退 x=-0.287、快速前进 x=+0.786、慢速前进 x=+0.500 m/s。
- 最终 Lite3 AMP state 恢复为 30 维：absolute joint pos 12 + body-frame linear velocity 3 + body-frame angular velocity 3 + joint velocity 12；expert q/dq 取负，world velocities 按每帧 xyzw quaternion 转到 body frame。
- Lite3 URDF 四条腿的关节轴、连杆长度、足端球碰撞和质量声明左右/前后对称，未发现 HR 独有的 URDF 几何异常；右后腿腾空更可能来自策略动作/奖励局部最优，而非模型结构差异。
- `logs/Jul26_06-36-24_/model_23000.pt` 的 actor/VAE 26 个参数张量与当前部署 `deploy/pre_train/lite3/policy_dwaq.pt` 逐元素完全一致（max diff 0），且完整 checkpoint 的 actor/critic/VAE 维度匹配当前 DreamWaQ。
- `lite3_stairs_amp` 默认 bootstrap 上述 `model_23000.pt`，因此从当前实际部署策略开始；只恢复网络权重，不继承旧 PPO optimizer，AMP 判别器/normalizer 从专家数据初始化并重新 warmup。
- AMP 任务采用保守微调参数：fixed PPO/VAE lr `1e-4`、clip `0.1`、entropy `0.005`、默认 3000 iterations、每 100 iterations 保存；这些参数不影响原 `lite3_stairs`。

## Lite3 MuJoCo 地形迁移

- 用户指定的参考配置位于 `/home/abc/rl_gym_lite3/deploy/deploy_mujoco/configs/lite3.yaml`，目标只涉及 MuJoCo 部署资源。
- 源地形集合为 flat、cross_slope、cross_stairs、race_track、stairs；所有 scene 原本 include 源仓库专用 `lite3_body.xml`。
- 源仓库有完整 `resources/robots/lite3/mujoco_terrains/`：五个 scene XML、`lite3_body.xml`、四张 label 图片和 `gen_terrains.py`。
- 本项目当前 MuJoCo Lite3 资源位于 `deploy/deploy_mujoco/resources/lite3/`，只有 `cross_stairs.xml`、`lite3_stairs.xml`、`Lite3.xml` 和 mesh；尚无完整五地形集合。
- 本项目 YAML 还包含 deterministic VAE、历史输入维度、扭矩限制、heading/line hold、trim 和 stair pulse，不能用参考 YAML 覆盖。
- 迁移只应改变 `xml_path` 并添加 terrain 资源；保留本项目 `simulation_dt=0.005`、`control_decimation=4` 和全部现有策略/控制配置。
- 当前 `deploy/deploy_mujoco/configs/lite3.yaml` 已有既存修改，`cross_stairs.xml` 也是未跟踪文件；后续需在此基础上增量适配，不能回退。
- 源 `lite3_body.xml` 与本项目 `Lite3.xml` 的机器人参数基本一致，但源 base 初始 z 为 `1.0 m`、mesh 路径指向源 `mjcf`，本项目为 `0.28 m` 且已有部署 mesh；因此不能复制 source body。
- 两边四个 STL 的 SHA256 完全一致，无需重复迁移 mesh。
- 本项目现有 `cross_stairs.xml` 已采用正确适配：include `Lite3.xml`，场景比源少一个重复 floor，并将 label texture 指向现有部署资源。
- 推荐把 flat、cross_slope、race_track、stairs 与 cross_stairs 并列放入 `deploy/deploy_mujoco/resources/lite3/`，全部 include 当前 `Lite3.xml` 并去掉重复 floor；这不会改变 Lite3 动力学参数。
- 源四张 label 图片与本项目 `deploy/deploy_mujoco/resources/go2/imgs/` 对应文件哈希完全一致，可安全复用现有纹理。
- 五个迁移后 scene 的直接 geom 数分别为 0、16、68、235、168，均比参考对应场景少且只少一个 floor；实际 terrain 障碍没有丢失。
- MuJoCo 3.2.3 实际加载结果：五个模型均为 `nq=19`、`nv=18`、`nu=12`，总 geom 数依次为 35、51、103、270、203。
- `deploy_lite3.py --check-only` 已使用当前 `policy_dwaq.pt` 对全部五个 terrain 验证成功，说明配置、模型和 270 维策略输入契约兼容。

## GitHub 上传审计

- 当前 Git 分支为 `master`，仓库仅有一个本地提交 `667ec9e`，没有配置任何 remote。
- 工作区包含多项已跟踪修改与未跟踪 AMP/地形实现；`logs/` 已被 `.gitignore` 排除。
- 未跟踪项还包括完整参考目录 `AMP_for_hardware/`、专家数据 `lite3_motions/`、规划记录和 `play1.py`，上传前需按大小、许可证和实际运行依赖分类。
- 系统没有 `gh` 命令，不能依赖 GitHub CLI 创建仓库或检查登录；需检查 Git credential/SSH 或使用 GitHub API。
- `AMP_for_hardware/` 是带独立 `.git` 和许可证的第三方参考仓库，约 58 MB，并包含约 12 MB 的训练 checkpoint；不能作为普通目录直接提交到本仓库，且不是当前 AMP 实现的运行依赖。
- `lite3_motions/` 约 8 MB，是当前 Lite3 AMP 配置直接引用的专家数据，大小适合普通 Git，属于应上传的运行资源。
- 项目（排除 `logs/` 和 `.git/`）没有大于 50 MB 的单文件；现有 Git 对象约 48 MiB，未触及 GitHub 单文件 100 MB 限制。
- 明显 GitHub token、AWS key、私钥和常见明文凭据模式扫描无命中。
- 当前 repo-local/global Git 提交身份解析为 `Codex <codex@local>`，上传前应改成用户身份，避免新提交归属错误。

## AMP 静止站立修正设计

- 用户明确取消 GitHub 上传，后续不创建 remote、不提交、不推送。
- `reset_idx()` 实际不会直接重采样命令；命令由 `_post_physics_step_callback()` 按周期重采样，随后 heading 模式每步重算 yaw。因此纯静止环境必须在父 callback 完成后清零全部四维命令。
- 采用随机但固定的 10% AMP 环境作为静止 cohort，可确保训练中持续存在足量零命令样本，同时保留 90% 并行环境的原运动训练分布。
- 现有 `stand_contact` 返回四脚接触比例，缺一脚仍保留 75% 奖励；将其改为 `all()` 全接触奖励，并新增二值 `stand_missing_contact` 负奖励。
- 左右关节镜像关系为：HipX 左右取和应接近 0，HipY/Knee 左右取差应接近 0；分别约束 FL/FR 与 HL/HR。
- 横坡可能需要左右关节不对称，因此 symmetry 仅在零命令且 `projected_gravity` 的 roll/pitch 分量低于阈值时启用；缺足惩罚仍对所有静止地形生效。
- 保守 scale 为全接触 `+0.5`、缺任意足 `-1.0`、静止左右对称 `-0.25`；乘 `dt=0.02` 后量级分别为 `+0.01`、`-0.02` 和小于约 `-0.003`。
- 奖励边界单测通过：全接触静止时 contact=1；缺一足时 contact=0/missing=1；运动命令时三项为 0；近水平不对称样本 symmetry=0.4，横坡门控后为 0。
- 基线 `Lite3_Stairs_Cfg_Yu` 不包含静止 cohort 或新 reward scale，配置继承隔离成立。
- 16-env Isaac Gym 检查固定选中 2 个 stand env，父 heading callback 后其四维命令全为零；非 stand 环境仍有非零随机命令。
- 从现有完整 AMP `model_26000.pt` 恢复的 16-env/1-iteration 训练通过，PPO、VAE、AMP discriminator 均更新成功，三个新 reward 名称已注册。
- 正式修正应从当前 `model_26000.pt` 继续小步微调，而非重新 bootstrap baseline；建议先训练 1000 iterations，并每 100 iteration 在 MuJoCo 对照静止足高和运动能力。
