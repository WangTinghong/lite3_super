# Lite3 AMP 适配计划

目标：在不改变现有 `lite3_stairs` 稳定训练链路的前提下，为 Lite3 新增可选 AMP 训练任务，使用 `lite3_motions` 改善步态自然度，并针对零速度站立时右后腿腾空增加 Lite3 专用约束。

## 阶段

### Phase 1：核对环境与动作数据
**Status:** complete

### Phase 2：设计 Lite3 AMP 状态与低风险奖励混合
**Status:** complete

### Phase 3：实现独立环境、DreamWaQ AMP 算法与 runner
**Status:** complete

### Phase 4：增加静止四足接触约束并注册独立任务
**Status:** complete

### Phase 5：完成格式、张量维度和最小训练验证
**Status:** complete

### Phase 6：汇总训练、基线对照和调参策略
**Status:** complete

### Phase 7：对比两个仓库的 Lite3 MuJoCo 地形依赖
**Status:** complete

### Phase 8：迁移部署地形资源并调整本项目 MuJoCo 配置
**Status:** complete

### Phase 9：验证全部地形 XML、mesh 路径和 Lite3 模型加载
**Status:** complete

### Phase 10：汇总地形切换与仿真命令
**Status:** complete

### Phase 11：审计本地 Git 状态、远端、文件大小与敏感信息
**Status:** complete

### Phase 12：确定并整理适合上传的项目文件范围
**Status:** complete（用户明确取消 GitHub 上传，无远端操作）

### Phase 13：在 WangTinghong 账号创建仓库并提交、推送当前项目
**Status:** complete（用户明确取消 GitHub 上传，无远端操作）

### Phase 14：校验 GitHub 远端分支与提交内容
**Status:** complete（用户明确取消 GitHub 上传，无远端操作）

### Phase 15：设计 AMP 专用固定静止环境与安全奖励
**Status:** complete

### Phase 16：实现纯静止命令、四足缺失惩罚和静止左右对称约束
**Status:** complete

### Phase 17：完成配置隔离、奖励数值和 Isaac Gym 集成验证
**Status:** complete

### Phase 18：汇总重新训练、导出和 MuJoCo 对照流程
**Status:** complete

### Phase 19：补充项目 Markdown 文档
**Status:** in_progress

### Phase 20：审计 GitHub 上传范围和认证状态
**Status:** pending

### Phase 21：提交并推送到 WangTinghong/lite3_gym
**Status:** pending

### Phase 22：校验远端提交和文档内容
**Status:** pending

## 约束与决策

- 训练对象仅为 Lite3；不直接复用 A1 的关节顺序、足端顺序、默认姿态或网络参数。
- 保留原有 `lite3_stairs` 配置、算法和任务注册行为，新功能通过独立任务启用。
- AMP 初始权重保持较低，任务奖励继续占主导；默认不启用 reference-state initialization。
- 专家数据必须先验证 61 维字段、四元数、关节和足端顺序，再进入判别器。
- 右后腿腾空不能只依赖 AMP，零指令站立接触约束应仅在静止命令下生效，避免妨碍正常摆腿。
- 不覆盖现有 checkpoint；优先支持从现有 Lite3 checkpoint 恢复后微调。
- MuJoCo 地形迁移只修改 `deploy/deploy_mujoco` 及其部署资源，不改训练环境、AMP、`rsl_rl`、奖励、任务注册或 checkpoint。
- 源仓库 `/home/abc/rl_gym_lite3` 只作为只读参考；迁移时按本项目的路径变量和 Lite3 MJCF 结构适配。
- GitHub 上传前必须排除明显凭据、缓存、运行日志和超出 GitHub 普通 Git 限制的大文件；不能用未经审计的 `git add .`。
- 目标账号为 `WangTinghong`，默认候选仓库名为当前目录名 `lite3_gym`；创建或覆盖远端前先确认登录身份和同名仓库状态。
- 静止站立修正只加入 `lite3_stairs_amp`，基线任务和 MuJoCo 部署不做零指令动作硬覆盖。
- 固定静止环境必须持续输出四维零命令，不能被 heading-command 回路重新写入 yaw 指令。
- 接触约束应对“缺任意一足”产生明确损失；左右对称只在静止命令门控下生效，避免限制运动步态。
- AMP 固定静止 cohort 比例采用 10%，随机分散在并行环境中并在整个训练期间保持固定；其余 90% 环境的命令采样不变。
- 静止左右对称使用较小权重，并增加机身近水平门控；横坡需要左右腿主动不对称时不施加该项。
- GitHub 上传时只提交当前项目和所需资源；第三方参考仓库、运行日志、缓存和本地规划记录不纳入提交。

## 遇到的错误

| 错误 | 尝试次数 | 处理 |
|------|---------:|------|
| 直接按 `legged_gym/resources/robots/lite3/...` 查找 URDF 不存在 | 1 | 核对 `LEGGED_GYM_ROOT_DIR` 定义后使用实际资源根目录，不修改配置 |
| 独立配置导入缺少 `isaacgym` module | 1 | 测试命令补充项目安装位置 `/home/abc/isaacgym/python` |
| 系统 Python 3.10 无法加载 Isaac Gym 的 `gym_38.so` | 1 | 改用 `/home/abc/anaconda3/envs/isaacgym/bin/python` 重试，不再使用系统 `python3` |
| Isaac Gym Python 3.8 导入遇到沙箱外 torch extension cache 锁文件 | 1 | 使用 `/tmp/lite3_gym_torch_extensions` 编译缓存后完成导入/注册检查 |
| 1-env smoke 脚本先导入 `torch` 间接触发 Isaac Gym 依赖错误 | 1 | 按正式入口先 `import isaacgym`，再导入 `legged_gym.utils` |
| 1-env smoke 先导入 `legged_gym.utils` 触发项目原有 task registry 循环导入 | 1 | 完全按 `train.py` 顺序先导入 `legged_gym.envs`，再导入 utils |
| smoke 脚本将 `get_observations()` 误解包为 4 项 | 1 | 按当前 DreamWaQ 接口改为 obs/history/estimation 3 项 |
| 默认 Python 环境缺少 `mujoco`，无法进行 terrain 编译检查 | 1 | 改用实际部署使用的 `/home/abc/anaconda3/envs/dreamwaq/bin/python`，五个场景全部通过 |
| 系统未安装 GitHub CLI（`gh: 未找到命令`） | 1 | 不重复调用 `gh`；后续检查可用 Git 凭据并选择 HTTPS/API 方案 |
| 更新规划文件时补丁上下文与实际文本不一致 | 1 | 重新读取文件尾部并按实际上下文拆分补丁 |
| 静止奖励临时单测先导入 `torch`，触发 Isaac Gym 导入顺序错误 | 1 | 按正式入口先 `import isaacgym`，再导入 `torch` 后重试 |
| 静止奖励单测的 `PYTHONPATH` 缺少仓库内 `rsl_rl/` 包根目录 | 1 | 将 `/home/abc/lite3_gym/rsl_rl` 加入测试路径后重试 |
| 真实训练 smoke 的默认 baseline bootstrap 文件已不在本机 | 1 | 改用现有完整 AMP `model_26000.pt` 恢复测试，同时验证 checkpoint 兼容性 |
