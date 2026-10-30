import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const workspaceDir = '/Users/wangtinghong/Desktop/工作/lite3_gym';
const buildDir = path.join(workspaceDir, '.ppt_build_lite3_20261011');
const previewDir = path.join(buildDir, 'previews');
const outputDir = path.join(workspaceDir, 'output');
const finalPath = path.join(outputDir, 'Lite3户外巡检鲁棒行走路线图_20261011_修订版.pptx');
const skillDir = '/Users/wangtinghong/.codex/plugins/cache/openai-primary-runtime/presentations/26.1009.11318/skills/presentations';
const runtimeModules = '/Users/wangtinghong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const runtimePython = '/Users/wangtinghong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';

await fs.mkdir(previewDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
await fs.symlink(runtimeModules, path.join(buildDir, 'node_modules'), 'dir').catch((err) => {
  if (err.code !== 'EEXIST') throw err;
});

const C = {
  navy: '#10283A',
  navy2: '#173B52',
  ink: '#183343',
  muted: '#587080',
  teal: '#168C86',
  tealDark: '#0C6E69',
  aqua: '#DDF1EF',
  blue: '#DCE8F1',
  amber: '#E2A33B',
  amberPale: '#FFF2D9',
  red: '#C9574B',
  redPale: '#FBE7E3',
  bg: '#F5F8F9',
  white: '#FFFFFF',
  line: '#D4E0E5',
  light: '#EAF0F2',
};
const FONT = 'PingFang SC';
const W = 1280;
const H = 720;
const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function text(slide, value, x, y, w, h, opts = {}) {
  const shape = slide.shapes.add({
    geometry: 'textbox',
    name: opts.name,
    position: { left: x, top: y, width: w, height: h },
    fill: 'none',
    line: { style: 'solid', fill: 'none', width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    typeface: FONT,
    fontSize: opts.size ?? 24,
    bold: opts.bold ?? false,
    italic: opts.italic ?? false,
    color: opts.color ?? C.ink,
    alignment: opts.align ?? 'left',
    verticalAlignment: opts.valign ?? 'top',
    lineSpacing: opts.lineSpacing ?? 1.12,
    wrap: 'square',
    autoFit: opts.autoFit ?? 'shrinkText',
    insets: opts.insets ?? 0,
  };
  return shape;
}

function rect(slide, x, y, w, h, fill, opts = {}) {
  return slide.shapes.add({
    geometry: opts.geometry ?? 'rect',
    name: opts.name,
    position: { left: x, top: y, width: w, height: h, rotation: opts.rotation ?? 0 },
    fill,
    line: { style: 'solid', fill: opts.line ?? 'none', width: opts.lineWidth ?? 0 },
    borderRadius: opts.radius,
  });
}

function rule(slide, x, y, w, h = 2, color = C.line) {
  return rect(slide, x, y, w, h, color);
}

function header(slide, title, n, subtitle = '') {
  slide.background.fill = C.bg;
  text(slide, title, 64, 34, 1080, 56, { size: 40, bold: true, color: C.navy, valign: 'middle' });
  text(slide, `${String(n).padStart(2, '0')} / 11`, 1120, 44, 96, 28, { size: 16, color: C.muted, align: 'right', valign: 'middle' });
  rule(slide, 64, 103, 1152, 2, C.line);
  rule(slide, 64, 103, 150, 4, C.teal);
  if (subtitle) text(slide, subtitle, 64, 116, 1120, 32, { size: 21, color: C.muted, valign: 'middle' });
}

function footer(slide, page) {
  text(slide, 'Lite3  ·  户外巡检鲁棒运动控制', 64, 682, 720, 20, { size: 14, color: C.muted, valign: 'middle' });
  text(slide, String(page).padStart(2, '0'), 1176, 682, 40, 20, { size: 14, color: C.muted, align: 'right', valign: 'middle' });
}

function newSlide(title, n, subtitle = '') {
  const slide = presentation.slides.add();
  header(slide, title, n, subtitle);
  footer(slide, n);
  return slide;
}

function table(slide, values, x, y, width, height, colWidths, opts = {}) {
  const tbl = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: x,
    top: y,
    width,
    height,
    columnWidths: colWidths,
    values,
  });
  tbl.borders.assign({ style: 'solid', fill: C.line, width: 1 });
  tbl.styleOptions = { headerRow: false, bandedRows: false };
  for (let r = 0; r < values.length; r += 1) {
    tbl.rows[r].height = height / values.length;
    for (let c = 0; c < values[r].length; c += 1) {
      const cell = tbl.getCell(r, c);
      cell.fill = r === 0 ? C.navy : (r % 2 === 0 ? '#F0F5F6' : C.white);
      cell.text.style = {
        typeface: FONT,
        fontSize: r === 0 ? (opts.headerSize ?? 22) : (opts.bodySize ?? 20),
        bold: r === 0 || (opts.boldFirstCol !== false && c === 0),
        color: r === 0 ? C.white : (c === 0 && opts.boldFirstCol !== false ? C.tealDark : C.ink),
        verticalAlignment: 'middle',
        alignment: c === 0 ? 'center' : 'left',
        autoFit: 'shrinkText',
        wrap: 'square',
        insets: { top: 9, right: 10, bottom: 8, left: 10 },
      };
    }
  }
  return tbl;
}

function note(slide, ...lines) {
  slide.speakerNotes.text = lines.join('\n');
}

// 1. Cover
{
  const s = presentation.slides.add();
  s.background.fill = C.navy;
  rect(s, 64, 90, 8, 486, C.teal);
  text(s, '研究路线讨论  /  2026.10', 104, 92, 520, 38, { size: 20, bold: true, color: '#9AD8D1', valign: 'middle' });
  text(s, 'Lite3 户外巡检\n鲁棒行走控制', 104, 171, 810, 188, { size: 64, bold: true, color: C.white, lineSpacing: 1.02 });
  text(s, '从湿滑适应到多场景自主运动', 106, 390, 720, 48, { size: 30, color: '#D9E7EC', valign: 'middle' });
  rule(s, 106, 470, 580, 2, '#3B5A6A');
  text(s, '项目现状  ·  研究创新  ·  工程验证', 106, 498, 640, 38, { size: 21, color: '#B7C9D1', valign: 'middle' });
  text(s, 'Lite3', 106, 630, 180, 30, { size: 18, bold: true, color: '#9AD8D1', valign: 'middle' });
  note(s, '本汇报结合 Lite3 项目当前训练结构和用户提供的工程目标，比较多步态切换与湿滑适应，并给出户外巡检的扩展路线。');
}

// 2. Engineering objective and operational domain
{
  const s = newSlide('工程目标与运行范围', 2, '目标是在地面条件未知或变化时，保持巡检路线上的连续、安全运动');
  text(s, '任务要求', 72, 173, 300, 34, { size: 25, bold: true, color: C.tealDark, valign: 'middle' });
  text(s, '雨后路面、湿坡和局部积水会改变足地抓地能力。机器人需要继续行走、转弯和通过缓坡，出现滑移后能够恢复。', 72, 220, 455, 142, { size: 25, lineSpacing: 1.2 });
  rule(s, 562, 177, 2, 360, C.line);
  text(s, '巡检路线中的典型变化', 604, 173, 550, 34, { size: 25, bold: true, color: C.tealDark, valign: 'middle' });
  const routeY = 292;
  rect(s, 620, routeY, 170, 98, '#AFC3CB');
  rect(s, 790, routeY, 166, 98, '#74B9B0');
  rect(s, 956, routeY, 192, 98, '#9EB2C0');
  text(s, '干燥硬路面', 620, routeY + 110, 170, 36, { size: 20, bold: true, align: 'center', color: C.ink, valign: 'middle' });
  text(s, '局部湿滑', 790, routeY + 110, 166, 36, { size: 20, bold: true, align: 'center', color: C.tealDark, valign: 'middle' });
  text(s, '坡面 / 转弯', 956, routeY + 110, 192, 36, { size: 20, bold: true, align: 'center', color: C.ink, valign: 'middle' });
  rect(s, 620, routeY + 73, 528, 7, C.navy2);
  text(s, '运行范围需要明确材料、坡度、速度、载荷和可接受的湿滑程度。', 72, 423, 458, 82, { size: 21, bold: true, color: C.navy, lineSpacing: 1.16 });
  rect(s, 72, 554, 1136, 76, C.amberPale, { radius: 14 });
  text(s, '“稳定”要用任务完成率、滑移、跌倒与恢复时间定义，并限定可验证的户外运行范围。', 96, 569, 1088, 46, { size: 22, bold: true, color: C.ink, valign: 'middle' });
  note(s, '工程场景定义建议：先聚焦雨后硬质路面、湿坡和干湿交界。草地、泥地和松散碎石同时涉及沉陷与材料形变，需要后续单独建模。');
}

// 3. Current project baseline
{
  const s = newSlide('项目现有基础与主要缺口', 3, '已具备鲁棒运动底座，下一步需要补足空间变化摩擦与显式滑移适应');
  text(s, '可复用的能力', 72, 172, 410, 34, { size: 25, bold: true, color: C.tealDark, valign: 'middle' });
  text(s, '稳定上楼策略\n45 维本体观测与 5 帧历史\n多地形训练和 PPO / DreamWaQ\n全局摩擦随机化', 72, 222, 398, 228, { size: 25, lineSpacing: 1.34 });
  rect(s, 72, 498, 400, 94, C.aqua, { radius: 14 });
  text(s, 'Lite3 摩擦范围：0.1 – 1.25', 96, 515, 350, 32, { size: 24, bold: true, color: C.tealDark, valign: 'middle' });
  text(s, '仿真中按环境采样整体摩擦参数', 96, 550, 348, 26, { size: 18, color: C.ink, valign: 'middle' });
  rule(s, 516, 176, 2, 410, C.line);
  text(s, '训练时的非对称观测结构', 558, 172, 550, 34, { size: 25, bold: true, color: C.tealDark, valign: 'middle' });
  rect(s, 558, 232, 260, 115, C.white, { line: C.line, lineWidth: 1, radius: 14 });
  text(s, 'Actor 策略', 582, 251, 210, 30, { size: 23, bold: true, color: C.navy, valign: 'middle' });
  text(s, '本体感觉与历史信息', 582, 291, 210, 30, { size: 20, color: C.muted, valign: 'middle' });
  rect(s, 918, 232, 260, 115, C.blue, { radius: 14 });
  text(s, 'Critic 价值网络', 942, 251, 218, 30, { size: 23, bold: true, color: C.navy, valign: 'middle' });
  text(s, '可见特权摩擦等信息', 942, 291, 218, 30, { size: 20, color: C.muted, valign: 'middle' });
  rect(s, 834, 273, 66, 18, C.teal);
  text(s, '策略部署时不直接知道真实摩擦系数', 558, 380, 620, 38, { size: 22, bold: true, color: C.ink, valign: 'middle' });
  text(s, '当前奖励中已有 stumble、足端清障等项，尚未形成支撑相滑移检测与恢复闭环。', 558, 436, 620, 86, { size: 21, color: C.ink, lineSpacing: 1.2 });
  rect(s, 558, 552, 620, 54, C.amberPale, { radius: 12 });
  text(s, '研究缺口：局部湿滑区域、风险估计、策略恢复', 580, 561, 578, 36, { size: 22, bold: true, color: C.navy, valign: 'middle' });
  note(s,
    '项目代码依据：legged_gym/envs/Lite3_Trot/Lite3_Stairs_config.py（friction_range=[0.1, 1.25]）；legged_gym/envs/GO2_Trot/GO2_Stairs_config.py（45 维单帧观测、5 帧历史）；legged_gym/envs/GO2_Trot/GO2_Stairs.py（critic 特权观测含摩擦系数，按环境随机摩擦；含 stumble 与 foot-clearance 奖励）。',
    '用户提供的工程事实：Lite3 已能稳定上楼。代码仍需在后续实现前确认部署分支和训练 checkpoint。');
}

// 4. Difficulty comparison table
{
  const s = newSlide('多步态切换与湿滑适应：实现难度比较', 4, '按当前项目的复用程度、研究新增量与户外巡检收益综合判断');
  const values = [
    ['比较维度', '多步态控制切换', '湿滑场景适应'],
    ['已有基础', '已有多方向与快慢运动命令；尚无显式步态标签和相位控制。', '已有整体摩擦随机化、特权 critic 和稳定爬楼策略。'],
    ['新增模块', '步态条件、各步态策略或专家、切换选择器、过渡稳定约束。', '局部摩擦地图、支撑相滑移估计、低附着恢复与风险自适应。'],
    ['实现难点', '技能数据与相位定义；策略切换时保持连续性和稳定性。', '真实路面标定、干湿突变泛化、仿真结果迁移到真机。'],
    ['工程价值', '提高速度、能耗或特定地形下的运动选择能力。', '直接降低雨后路面、湿坡和湿台阶的巡检失稳风险。'],
    ['项目建议', '作为第二阶段的运动能力扩展。', '作为近期主线，先做感知与恢复闭环。'],
  ];
  table(s, values, 64, 168, 1152, 458, [200, 476, 476], { bodySize: 20, headerSize: 22 });
  note(s,
    '难度是基于当前 Lite3 仓库和目标范围的工程估计，不是论文结果。局部湿滑的真实表面验证成本高，但算法可以从现有策略与摩擦训练结构渐进扩展。',
    '多步态研究对照：Stamatopoulou et al., eGAIT, IEEE Transactions on Automation Science and Engineering, 2025, DOI 10.1109/TASE.2025.3636224。');
}

// 5. Recent journal evidence
{
  const s = newSlide('近年期刊工作给出的研究参照', 5, '两类工作都能复用现有项目基础，但面向的工程问题不同');
  rect(s, 72, 174, 536, 382, C.white, { line: C.line, lineWidth: 1, radius: 16 });
  rect(s, 72, 174, 536, 12, C.teal);
  text(s, '滑移检测与恢复', 98, 205, 470, 38, { size: 28, bold: true, color: C.navy, valign: 'middle' });
  text(s, 'Slip Detection and Recovery for Quadruped Robots via Orthogonal Decomposition', 98, 257, 470, 76, { size: 22, bold: true, color: C.ink, lineSpacing: 1.18 });
  text(s, 'IEEE Transactions on Industrial Electronics\n2025 年卷期，Unitree A1 实机验证', 98, 345, 470, 55, { size: 19, color: C.muted, lineSpacing: 1.2 });
  text(s, '本体感知滑移检测，结合位置 / 力控制恢复，并在线估计接触摩擦。', 98, 420, 470, 70, { size: 21, color: C.ink, lineSpacing: 1.2 });
  text(s, 'DOI 10.1109/TIE.2024.3488360', 98, 511, 470, 26, { size: 17, bold: true, color: C.tealDark, valign: 'middle' });

  rect(s, 672, 174, 536, 382, C.white, { line: C.line, lineWidth: 1, radius: 16 });
  rect(s, 672, 174, 536, 12, C.amber);
  text(s, '多技能步态切换', 698, 205, 470, 38, { size: 28, bold: true, color: C.navy, valign: 'middle' });
  text(s, 'eGAIT: Multi-Skilled Policy for Energy-Efficient Gait Transitions', 698, 257, 470, 76, { size: 22, bold: true, color: C.ink, lineSpacing: 1.18 });
  text(s, 'IEEE Transactions on Automation Science and Engineering\n2025 年，Unitree Go1 实机验证', 698, 345, 470, 55, { size: 19, color: C.muted, lineSpacing: 1.2 });
  text(s, '九种速度优化步态，结合 MPC 轨迹、PPO / AMP 策略和层级切换选择器。', 698, 420, 470, 70, { size: 21, color: C.ink, lineSpacing: 1.2 });
  text(s, 'DOI 10.1109/TASE.2025.3636224', 698, 511, 470, 26, { size: 17, bold: true, color: '#986416', valign: 'middle' });
  rect(s, 72, 580, 1136, 54, C.blue, { radius: 12 });
  text(s, '对当前项目的启示：滑移方向的闭环更直接服务户外安全；多步态是后续的技能与效率扩展。', 92, 589, 1096, 36, { size: 20, bold: true, color: C.navy, valign: 'middle' });
  note(s,
    'Yan et al., “Slip Detection and Recovery for Quadruped Robots via Orthogonal Decomposition,” IEEE Transactions on Industrial Electronics, Vol. 72, Issue 6, June 2025, pp. 6166–6174. DOI: https://doi.org/10.1109/TIE.2024.3488360. IEEE Xplore abstract reports a proprioceptive slip detector, hybrid position/force recovery controller, online friction estimation, and Unitree A1 experiment.',
    'Stamatopoulou et al., “eGAIT: Multi-Skilled Policy for Energy-Efficient Gait Transitions,” IEEE Transactions on Automation Science and Engineering, Vol. 23, pp. 1575–1586, published 3 Dec 2025. DOI: https://doi.org/10.1109/TASE.2025.3636224. IEEE Xplore abstract reports nine velocity-optimized gaits, MPC-generated references, PPO/AMP-style training, HRL selector, and Unitree Go1 validation.');
}

// 6. Proposed architecture
{
  const s = newSlide('推荐控制结构：运动策略加滑移风险闭环', 6, '保留已稳定的 DreamWaQ 策略，让风险估计器与恢复监督器处理低附着变化');
  const nodes = [
    { x: 72, title: '可部署观测', body: 'IMU、关节位置 / 速度\n动作历史、接触估计', fill: C.blue },
    { x: 360, title: '抓地风险估计', body: '滑移概率 / 置信度\n历史编码辅助判断', fill: C.aqua },
    { x: 648, title: '名义运动策略', body: '当前 DreamWaQ\n维持行走与爬楼能力', fill: C.white },
    { x: 936, title: '恢复监督器', body: '限速、减小加速度\n调整支撑与落足目标', fill: C.amberPale },
  ];
  nodes.forEach((n) => {
    rect(s, n.x, 230, 240, 168, n.fill, { line: n.fill === C.white ? C.line : 'none', lineWidth: 1, radius: 16 });
    text(s, n.title, n.x + 20, 251, 200, 35, { size: 23, bold: true, color: C.navy, align: 'center', valign: 'middle' });
    text(s, n.body, n.x + 18, 305, 204, 68, { size: 20, color: C.ink, align: 'center', valign: 'middle', lineSpacing: 1.18 });
  });
  [320, 608, 896].forEach((x) => rect(s, x, 298, 30, 22, C.teal, { geometry: 'rightArrow' }));
  text(s, '训练时的特权信息', 72, 452, 260, 32, { size: 22, bold: true, color: C.tealDark, valign: 'middle' });
  rect(s, 72, 496, 1136, 86, C.navy, { radius: 14 });
  text(s, '仿真 critic 可使用摩擦系数、接触真值与地形参数训练；部署 actor 只使用实机能获得的观测。', 98, 515, 1086, 50, { size: 22, bold: true, color: C.white, valign: 'middle' });
  text(s, '第一版用命令限速 / 加速度做恢复；验证有效后，再扩展到足端轨迹和支撑时间调节。', 72, 606, 1136, 36, { size: 20, color: C.ink, valign: 'middle' });
  note(s, '建议将风险估计器作为辅助头或轻量监督器，先判断“滑移风险是否升高”，再决定是否估计有效摩擦。仿真摩擦系数无法在真实户外直接读取，不能作为部署 actor 的必需输入。');
}

// 7. Scenario expansion
{
  const s = newSlide('场景扩展：从硬质湿路面逐步到复杂户外地面', 7, '按物理建模和真机验证成本递增，避免把所有恶劣环境压成一个摩擦参数');
  const stageY = 190;
  const stageW = 340;
  const stages = [
    { x: 72, title: '阶段 A  硬质路面', color: C.blue, items: '干 / 湿水泥\n石板、沥青、金属板\n平地与固定坡度', result: '建立低附着基准' },
    { x: 470, title: '阶段 B  空间变化', color: C.aqua, items: '局部湿滑斑块\n干湿边界与小水洼\n斜坡、转弯、加减速', result: '验证在线适应能力' },
    { x: 868, title: '阶段 C  复杂地面', color: C.amberPale, items: '草地、碎石、落叶\n泥地与松散土壤\n不平整及接触形变', result: '补充可变形地面模型' },
  ];
  stages.forEach((st) => {
    rect(s, st.x, stageY, stageW, 336, st.color, { radius: 18 });
    text(s, st.title, st.x + 24, stageY + 24, stageW - 48, 38, { size: 25, bold: true, color: C.navy, valign: 'middle' });
    rule(s, st.x + 24, stageY + 78, stageW - 48, 2, C.white);
    text(s, st.items, st.x + 24, stageY + 104, stageW - 48, 126, { size: 22, color: C.ink, lineSpacing: 1.32 });
    text(s, st.result, st.x + 24, stageY + 274, stageW - 48, 34, { size: 21, bold: true, color: st.x === 868 ? '#986416' : C.tealDark, valign: 'middle' });
  });
  text(s, '泥地、草地与碎石还涉及沉陷和地面形变，应在硬质低附着验证完成后单独扩展。', 72, 565, 1136, 58, { size: 21, bold: true, color: C.navy, valign: 'middle' });
  note(s, '场景扩展建议：先把可控、可重复的硬质材料做透，再增加局部摩擦突变；对可变形地面采用独立接触模型和独立评测，不把所有失败归因于摩擦系数。');
}

// 8. Training method and innovation
{
  const s = newSlide('训练创新：从摩擦随机化走向可诊断的滑移适应', 8, '让每次滑移都有标签、可量化，也能对应策略调整');
  text(s, '环境随机化', 72, 176, 280, 34, { size: 24, bold: true, color: C.tealDark, valign: 'middle' });
  text(s, '将单一环境摩擦扩展为局部摩擦地图，随机湿滑位置、区域长度和干湿边界。保留干燥路面与现有楼梯任务。', 72, 220, 310, 176, { size: 21, lineSpacing: 1.22 });
  rule(s, 408, 180, 2, 370, C.line);
  text(s, '支撑相滑移量', 446, 176, 300, 34, { size: 24, bold: true, color: C.tealDark, valign: 'middle' });
  text(s, '只在足端接触地面时，计算足端相对地面的切向速度或滑移距离。', 446, 220, 322, 92, { size: 21, lineSpacing: 1.22 });
  rect(s, 446, 330, 322, 102, C.aqua, { radius: 14 });
  text(s, '滑移惩罚 = 接触门控 × 切向滑移速度²', 466, 350, 282, 60, { size: 20, bold: true, color: C.tealDark, align: 'center', valign: 'middle' });
  text(s, '同时保留速度跟踪与前进奖励，防止策略通过停住来规避滑移。', 446, 458, 322, 72, { size: 19, color: C.ink, lineSpacing: 1.2 });
  rule(s, 800, 180, 2, 370, C.line);
  text(s, '风险自适应', 838, 176, 300, 34, { size: 24, bold: true, color: C.tealDark, valign: 'middle' });
  text(s, '用滑移事件训练风险估计器。高风险时降低速度与加速度，恢复后再逐步回到目标速度。', 838, 220, 330, 142, { size: 21, lineSpacing: 1.22 });
  text(s, '推荐消融', 838, 390, 270, 30, { size: 20, bold: true, color: C.navy, valign: 'middle' });
  text(s, '基线策略\n整体摩擦随机化\n局部湿滑训练\n局部湿滑 + 风险适应', 838, 430, 330, 118, { size: 19, color: C.ink, lineSpacing: 1.23 });
  rect(s, 72, 581, 1136, 54, C.navy, { radius: 12 });
  text(s, '核心创新：在未见过的局部低附着变化下，检测滑移并恢复，而不是只提高训练时的摩擦覆盖。', 94, 590, 1092, 36, { size: 20, bold: true, color: C.white, valign: 'middle' });
  note(s, '仿真里可由足端接触力和足端刚体速度计算监督标签。真机部署时用已存在的本体感觉历史，必要时增加可估计的接触状态；先验证观测可用性，再确定网络接口。');
}

// 9. Evaluation plan
{
  const s = newSlide('评价指标与真机验证', 9, '把“稳定”转成任务表现、滑移损失与恢复能力三类证据');
  const values = [
    ['指标', '测量方式', '工程意义'],
    ['任务完成率 / 跌倒率', '固定路线、速度与载荷重复测试', '巡检路线能否连续完成'],
    ['支撑相滑移', '滑移距离、滑移时间占比、单步最大滑移', '直接反映足地失稳程度'],
    ['路径与速度误差', '路线偏差、目标速度跟踪误差', '判断适应是否以牺牲任务效率换稳定'],
    ['滑移后恢复时间', '首次滑移到恢复稳定行走的时间', '体现闭环恢复能力'],
    ['能耗与基线保持', '单位距离能耗；干地和楼梯非劣化', '适合长时间巡检且不丢已有能力'],
  ];
  table(s, values, 64, 169, 1152, 376, [230, 470, 452], { bodySize: 19, headerSize: 21 });
  const labels = [
    ['仿真', '已知摩擦 / 已知地形'],
    ['泛化', '未见位置 / 干湿突变'],
    ['实验场', '可控湿路面与缓坡'],
    ['户外', '限定路线巡检验证'],
  ];
  labels.forEach((a, i) => {
    const x = 72 + i * 286;
    rect(s, x, 576, 250, 58, i < 2 ? C.blue : (i === 2 ? C.aqua : C.amberPale), { radius: 12 });
    text(s, a[0], x + 12, 582, 68, 24, { size: 17, bold: true, color: C.tealDark, valign: 'middle' });
    text(s, a[1], x + 14, 604, 224, 24, { size: 17, color: C.ink, valign: 'middle' });
  });
  note(s, '建议每个表面和路线设置重复次数、速度、载荷与环境状态；报告均值和离散度。真机验证从防摔保护下的可控湿地面开始，再进入限定户外路线。');
}

// 10. Execution roadmap
{
  const s = newSlide('实施路线与阶段性交付', 10, '从当前稳定策略出发，用逐阶段对照实验降低工程风险');
  const steps = [
    { x: 72, n: '01', title: '摸清失稳边界', body: '跑当前 checkpoint\n复现湿滑失败\n交付：基线数据集', color: C.blue },
    { x: 356, n: '02', title: '搭建场景与指标', body: '局部摩擦地图\n接触门控滑移指标\n交付：仿真测试集', color: C.aqua },
    { x: 640, n: '03', title: '训练风险适应', body: '时序风险估计\n限速与恢复监督\n交付：策略消融结果', color: C.amberPale },
    { x: 924, n: '04', title: '真机巡检验证', body: '材料与坡度分级\n湿干切换路线测试\n交付：工程能力边界', color: '#E8ECEE' },
  ];
  rule(s, 180, 260, 820, 4, C.line);
  steps.forEach((st) => {
    rect(s, st.x, 198, 236, 74, st.color, { radius: 16 });
    text(s, st.n, st.x + 16, 213, 48, 42, { size: 25, bold: true, color: C.tealDark, align: 'center', valign: 'middle' });
    text(s, st.title, st.x + 72, 211, 148, 46, { size: 21, bold: true, color: C.navy, valign: 'middle' });
    rect(s, st.x + 104, 251, 26, 26, C.teal, { geometry: 'ellipse' });
    text(s, st.body, st.x + 4, 306, 228, 115, { size: 20, color: C.ink, align: 'center', lineSpacing: 1.3 });
  });
  rect(s, 72, 482, 1136, 126, C.navy, { radius: 16 });
  text(s, '阶段验收', 100, 503, 160, 32, { size: 23, bold: true, color: '#9AD8D1', valign: 'middle' });
  text(s, '湿滑成功率与恢复能力提升，同时不降低干地及楼梯性能；需与当前策略和仅摩擦随机化策略对照。', 100, 543, 1060, 46, { size: 21, bold: true, color: C.white, valign: 'middle' });
  note(s, '阶段顺序建议：先获得当前策略的失稳基线，再定义可重复的滑移标签和局部低摩擦场景；只有消融证明风险估计有收益后，再增加传感器或复杂地形。');
}

// 11. Additional expandable tasks
{
  const s = newSlide('后续可扩展任务与实现方法', 11, '低附着适应形成主线后，可按巡检需求扩展运动能力、感知与任务自治');
  const values = [
    ['扩展任务', '实现方法', '工程用途'],
    ['多步态切换', '加入步态条件与相位；训练技能策略和过渡选择器；评估切换稳定性。', '速度、能耗与地形适应'],
    ['前视落足规划', '融合深度 / LiDAR 地形图与湿滑风险，筛选安全落足区域。', '绕开积水、裂缝和障碍'],
    ['坡地与非结构地面', '高度图课程训练；草地、碎石、泥地采用独立接触与形变模型。', '扩大可巡检路面范围'],
    ['载荷与外部扰动', '随机化质量 / 重心、阵风和碰撞；用风险估计调整速度与姿态。', '适配巡检传感器并抗扰'],
    ['任务级巡检自治', '导航目标、路径覆盖、巡检点停靠与运动控制分层联动。', '从稳定行走走向自动巡检'],
  ];
  table(s, values, 64, 168, 1152, 408, [230, 570, 352], { bodySize: 19, headerSize: 21 });
  rect(s, 72, 601, 1136, 44, C.aqua, { radius: 10 });
  text(s, '推荐顺序：先低附着稳定行走，再加入步态选择和前视感知，最后扩展巡检任务自治。', 92, 605, 1096, 34, { size: 19, bold: true, color: C.tealDark, valign: 'middle' });
  note(s,
    '多步态期刊参照：Stamatopoulou et al., IEEE Transactions on Automation Science and Engineering, 2025, https://doi.org/10.1109/TASE.2025.3636224。',
    '场景扩展中的“前视落足规划”和任务自治是结合户外巡检目标提出的后续工程路线，属于建议，不是当前 Lite3 已实现能力。');
}

const expectedSlideSizeEmu = '12192000,6858000';
const draftPath = path.join(buildDir, 'candidate.pptx');
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(draftPath);

for (const [index, slide] of presentation.slides.items.entries()) {
  const blob = await presentation.export({ slide, format: 'png', scale: 1.4 });
  await fs.writeFile(path.join(previewDir, `slide-${String(index + 1).padStart(2, '0')}.png`), new Uint8Array(await blob.arrayBuffer()));
}
const montage = await presentation.export({ format: 'webp', montage: true, scale: 0.5 });
await fs.writeFile(path.join(previewDir, 'montage.webp'), new Uint8Array(await montage.arrayBuffer()));

const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
const result = await finalizePresentation({
  explicitTotalSlideCount: 11,
  requiredNativeTableOwnerSlides: [4, 9, 11],
  workspaceDir,
  candidatePath: draftPath,
  finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: [
    '--expected-slide-size-emu', expectedSlideSizeEmu,
    '--validate-heading-fit',
    '--require-native-table-slide', '4',
    '--require-native-table-slide', '9',
    '--require-native-table-slide', '11',
  ],
  fontPolicy: { basis: 'design', families: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, 'validation_revision.json'),
});
console.log(JSON.stringify({ finalPath, result }, null, 2));
