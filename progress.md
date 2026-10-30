# 进度记录

## 2026-08-08

- 已读取规划技能说明。
- 已检查目标仓库状态：存在大量既有修改，不能回退；当前分支为 `codex/lite3-isaacgym`。
- 已定位目标 MuJoCo 文件为 `deploy/deploy_mujoco/deploy_go2.py`，后续仅修改其输入控制相关代码。
- 已确认需要同步修改目标 `deploy/deploy_mujoco/configs/go2.yaml` 的 `cmd_init`，否则无键盘时仍会默认以 `Vx=1` 行走。
- 已修改目标 `deploy/deploy_mujoco/deploy_go2.py`：新增键盘映射、pygame 键盘窗口、50Hz 状态轮询、键盘/手柄/默认指令优先级、指令平滑和退出时 `pygame.quit()`。
- 已修改目标 `deploy/deploy_mujoco/configs/go2.yaml`：默认指令设为零，增加 `cmd_smoothing: 0.15`。
- 验证：Python 语法和 YAML 解析通过；运行时检查因当前环境缺少 `pygame`、`mujoco` 未执行。
- 本次改动完成。

## 追加任务：同步到 `/home/abc/rl_gym_lite3`

- 已修改 `deploy/deploy_mujoco/deploy_go2.py`：增加 pygame 键盘窗口和实时按键轮询，移除 Lite3 的 viewer 按键累加逻辑，保留手柄和 heading/lateral hold。
- 已修改 `deploy/deploy_mujoco/configs/lite3.yaml`：`cmd_init` 设为零，增加 `cmd_smoothing: 0.15`。
- 验证通过：脚本编译、YAML 解析、配置 XML/策略路径检查、键盘映射（按住/松开）测试。

## 2026-08-08：Lite3 AMP 适配

- 已恢复上下文并读取 `planning-with-files-zh` 技能规则。
- 已确认本次训练对象仅为 Lite3，参考 A1 AMP 只用于理解算法结构。
- 已建立独立 AMP 任务方案：保留现有 `lite3_stairs` 基线，不覆盖现有 checkpoint。
- 当前阶段：核对 Lite3 DreamWaQ 数据流、实际奖励开关和 `lite3_motions` 字段顺序。
- 已检查工作区状态：存在多处既有修改和未跟踪参考数据，后续只做任务所需的增量文件/局部编辑，不回退任何内容。
- 已否决照搬参考 43 维 AMP observation：静止专家足端数据本身包含约 5 cm 的后足高度不对称。
- 已确定首版 Lite3 AMP observation 为 30 维，不包含足端位置、root height 和不确定的 quaternion。
- 已梳理 DreamWaQ rollout 接口，确定新增 AMP runner/algorithm，不改变现有 runner。
- 已确定过滤 done transition，并将站立修正限制在零指令环境。
- 已确定 AMP reward 默认上限系数 `0.005`，增加独立 warmup/ramp，且从 baseline 恢复时重新预热判别器。
- 已统计八段专家动作，发现 base velocity 坐标与当前机体系不一致，正在通过 root rotation/位移继续核对；未确认前不让该字段进入判别器。
- 已完成坐标核对：专家速度为世界系，四元数实际为 xyzw；逆旋转后的速度与所有动作名称一致。
- 阶段 1 完成，进入 AMP observation、奖励混合和 checkpoint 兼容设计。
- 已完成设计：30 维 Lite3 AMP state、独立判别器 optimizer、done 过滤、100 iteration warmup + 900 iteration ramp、默认 AMP 系数 0.005。
- 开始实现新增 loader、discriminator/replay、DreamWaQ AMP algorithm/runner 和独立 Lite3 AMP task。
- 已新增 `Lite3AMPLoader`、`AMPReplayBuffer`、`AMPObservationNormalizer` 和 `AMPDiscriminator`，并加入 rsl_rl 包导出。
- 已新增 `PPO_DreamWaQAMP` 与 `DreamWaQAMPRunner`：独立判别器优化、done 过滤、AMP reward schedule、baseline/AMP checkpoint 兼容和 TensorBoard AMP 指标。
- 初步验证通过：新增文件编译、8 个动作/3987 个 transition 加载、30 维采样、有限值检查、判别器反向传播和 replay buffer。
- 配置隔离测试首次因命令未包含 Isaac Gym Python 路径而失败，准备按项目环境重试。
- 补路径后确认系统 `python3` 为 3.10，与 Isaac Gym 的 Python 3.8 binding 不兼容；后续 Isaac 导入测试改用 conda `isaacgym` 环境解释器。
- Python 3.8 Isaac 导入在沙箱内被 `/home/abc/.cache/torch_extensions/.../lock` 阻断；沙箱外尝试触发了扩展构建但未输出最终注册结果，准备改用 `/tmp` 扩展目录。
- DreamWaQ AMP 小批量端到端 smoke test 通过：2 步 rollout、PPO/VAE、AMP discriminator 更新，16 项指标均有限且 replay 有效。
- 使用 `/tmp/lite3_gym_torch_extensions` 后完成 Isaac Gym 短导入/注册检查：`lite3_stairs_amp` 已注册，baseline 没有 `stand_contact`，AMP runner/config 名称解析正确。
- 1-env 集成 smoke 首次因测试脚本导入顺序错误（torch 间接早于 isaacgym）失败，按正式入口顺序重试。
- 第二次 smoke 确认项目入口必须先导入 `legged_gym.envs`，再导入 `legged_gym.utils`；按正式 train.py 顺序重试。
- 1-env PhysX/Lite3 创建成功，真实 DOF 顺序确认为 FL/FR/HL/HR；后续仅因 smoke 脚本误解包 3 项 observation 接口而停止，修正后完成剩余检查。
- 修正后的 1-env 集成检查通过：actor obs `(1,45)`、history `(1,225)`、estimation `(1,3)`、AMP `(1,30)`，数值有限，stand-contact 可计算且单步后维度稳定。
- 完整 1-env/1-iteration AMP 训练 smoke 通过：24 步 rollout、PPO/VAE、discriminator 更新成功；AMP replay 24 条，warmup iteration 1 的 reward scale 正确为 0。
- AMP 实现和独立任务注册阶段完成，进入 checkpoint 路径与最终差异检查。
- 已确认当前部署 JIT 的 actor/VAE 与 `logs/Jul26_06-36-24_/model_23000.pt` 完全一致，AMP 默认 bootstrap 该完整 checkpoint；resume AMP checkpoint 时跳过 baseline bootstrap。
- 默认 bootstrap 集成验证通过：网络权重精确相等，runner iteration=23000、AMP iteration=0、初始 AMP scale=0。
- 已将 AMP 独立配置收紧为成熟策略微调参数，并提高 checkpoint 保存频率；baseline 配置不变。
- 最终验证通过：Python 3.8 编译、diff check、motion loader、纯 PyTorch 端到端更新、Isaac task 注册、1-env observation/step、1-env 完整训练 iteration、baseline bootstrap 精确加载。
- 训练命令：`python legged_gym/scripts/train.py --task lite3_stairs_amp --headless --run_name amp_safe_v1`。默认训练 3000 iterations，每 100 次保存。
- 播放命令：`python legged_gym/scripts/play.py --task lite3_stairs_amp --load_run <AMP_RUN_DIR> --checkpoint <N>`；`play.py` 会自动进入 resume 模式。
- 本次 AMP 适配任务完成。实际步态自然度和 HR 静止接触率仍需在正式并行训练后按 checkpoint 对照 baseline 评估。

## 2026-08-09：Lite3 MuJoCo 地形迁移

- 已重新读取 `planning-with-files-zh` 技能并恢复现有规划。
- 新任务边界：只迁移 `/home/abc/rl_gym_lite3` 的 Lite3 MuJoCo 地形到本项目部署层，不改任何训练核心框架和代码。
- 当前阶段：对比源/目标配置、地形 XML include/mesh 依赖和生成脚本。
- 已确认不复制参考 YAML 的策略和控制项，只迁移五套 MuJoCo terrain scene 与必要资源，并保留本项目所有现有部署控制参数。
- 已核对 robot body/mesh：源 body 初始高度不适用于本项目，mesh 内容已完全存在；决定复用当前 `Lite3.xml`，仅适配 scene XML。
- 四张地形等级标签图片与本项目 `resources/go2/imgs` 中现有文件 SHA256 完全一致，直接复用，未复制重复二进制资源。
- 已新增 `flat.xml`、`cross_slope.xml`、`race_track.xml`、`stairs.xml`；与现有 `cross_stairs.xml` 一起构成五地形集合。
- 所有 terrain scene 均 include 当前 `Lite3.xml`，删除源 scene 的重复 floor/groundplane，并适配 label 路径；未复制源 `lite3_body.xml` 或 mesh。
- 已在 `deploy/deploy_mujoco/configs/lite3.yaml` 增加五个 `xml_path` 注释切换项，默认保持 `cross_stairs.xml`；其他配置未改。
- 验证通过：五个 XML 可解析，include/纹理无残留错误路径；每个 scene 的几何数量恰好比参考少一个重复 floor。
- 使用 dreamwaq 环境的 MuJoCo 3.2.3 编译五个模型均成功；全部保持 `nq=19`、`nv=18`、`nu=12`。
- 使用正式部署脚本和当前预训练策略对五个场景逐个执行 `--check-only`，全部通过策略输入、确定性输出和模型执行器校验。
- 本次地形迁移完成，改动范围仅为 Lite3 MuJoCo terrain XML 与部署 YAML。

## 2026-08-09：上传当前项目到 GitHub

- 用户授权将当前项目上传到 `https://github.com/WangTinghong`。
- 使用 `planning-with-files-zh` 管理上传审计；当前阶段为检查 Git remote、GitHub CLI 登录、工作区文件范围、超大文件和明显敏感信息。
- 暂定目标仓库名为 `lite3_gym`，在确认远端状态前不创建、不覆盖、不推送。
- 本地审计：当前 `master` 只有提交 `667ec9e`、没有 remote；工作区有大量待提交源码和资源。
- `gh auth status` 因 GitHub CLI 未安装失败，已切换为检查标准 Git 凭据/SSH/API 的方案。
- 文件审计完成第一轮：无大于 50 MB 的上传文件，无明显凭据命中；`lite3_motions` 可纳入，嵌套的第三方 `AMP_for_hardware` 应排除。

## 2026-08-09：修正 AMP 零指令站立不对称

- 用户中止 GitHub 上传，要求先实现此前建议的训练侧修正；上传阶段保留为 deferred，未创建或推送任何远端。
- 修正边界：仅修改 `lite3_stairs_amp` 环境/配置，不改变 baseline、MuJoCo 部署和已有 checkpoint。
- 当前阶段：设计固定比例纯静止环境，确认 command resample/heading callback 生命周期，并确定缺足与左右对称奖励的保守量级。
- 用户再次明确“不用上传到 Git”，上传任务已取消，未发生任何 GitHub 远端写入。
- 设计完成：10% 固定纯静止 cohort；父 heading callback 后强制四维零命令；全四足接触奖励 + 缺足二值惩罚；近水平静止状态下的小权重左右镜像约束。
- 已修改 AMP 环境：初始化随机固定的静止 cohort，并在每次父 callback/heading 更新后清零其全部命令。
- 已将 `stand_contact` 改为仅四足全部接触才得分，新增缺任意足二值惩罚和静止近水平左右关节镜像误差。
- 已修改 AMP 配置：静止比例 10%，奖励 scale 为 `+0.5/-1.0/-0.25`，基线配置未改；进入静态、数值和 Isaac Gym 集成验证。
- Python 编译与 diff whitespace 检查通过；首次纯奖励单测因临时脚本先导入 torch 触发 Isaac Gym 已知限制，已修正测试导入顺序。
- 修正导入顺序后测试进入项目导入阶段，但临时 `PYTHONPATH` 缺少 `rsl_rl/`；已定位为测试路径问题并补齐。
- 奖励边界与配置隔离单测通过；16-env 环境和 AMP runner 创建成功，但默认 bootstrap 指向的旧 baseline 文件本机已不存在，训练尚未开始。
- 已定位现有完整 AMP checkpoint `logs/lite3_stairs_amp/Aug_09-23-36_/model_26000.pt`，改为从该 checkpoint 恢复完成集成 smoke。
- 16-env/1-iteration AMP 恢复训练通过：从 `model_26000.pt` 加载，PPO/VAE/discriminator 完成更新，AMP scale 保持 0.005，三个新 reward 均已注册。
- 环境级 cohort 检查通过：16 个环境固定选择 2 个，heading callback 后其四维命令全零，非静止环境仍保留随机命令。
- 本次静止站立修正实现与验证完成；建议从 26000 继续 1000 iterations，每 100 次保存并做 MuJoCo checkpoint 对照。
