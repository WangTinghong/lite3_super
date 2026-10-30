import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
import { pathToFileURL } from 'node:url';

const sourcePath = '/Users/wangtinghong/Desktop/工作/副本重点突击到全面开花(1).pptx';
const buildDir = '/Users/wangtinghong/Desktop/工作/lite3_gym/.ppt_build';
const previewDir = path.join(buildDir, 'preview');
const skillDir = '/Users/wangtinghong/.codex/plugins/cache/openai-primary-runtime/presentations/26.915.20218/skills/presentations';
const { applyPresentationChartFont } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
await fs.mkdir(previewDir, { recursive: true });

const p = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const slides = p.slides.items;
const FONT = 'PingFang SC';
const C = {
  bg: '#F6F9FC',
  ink: '#102A43',
  muted: '#53657A',
  accent: '#2F6BCA',
  accentDark: '#1F4F9A',
  sky: '#DCEBFA',
  sky2: '#EAF2FB',
  teal: '#2E9C9A',
  gold: '#E39A22',
  red: '#D95D39',
  white: '#FFFFFF',
  line: '#C9D8E7',
  grid: '#D9E5F0',
};

function clearSlide(slide) {
  slide.shapes.deleteAll();
  for (const c of [...slide.charts.items]) slide.charts.deleteById(c.id);
  for (const t of [...slide.tables.items]) slide.tables.deleteById(t.id);
  for (const i of [...slide.images.items]) slide.images.deleteById(i.id);
  slide.background.fill = C.bg;
}

function addText(slide, text, left, top, width, height, style = {}) {
  const s = slide.shapes.add({
    geometry: 'textbox',
    position: { left, top, width, height },
    fill: 'none',
    line: { style: 'solid', fill: 'none', width: 0 },
  });
  s.text = text;
  s.text.style = {
    typeface: FONT,
    fontSize: style.fontSize ?? 18,
    bold: style.bold ?? false,
    italic: style.italic ?? false,
    color: style.color ?? C.ink,
    alignment: style.alignment ?? 'left',
  };
  s.text.autoFit = style.autoFit ?? 'shrinkText';
  s.text.wrap = 'square';
  s.text.verticalAlignment = style.verticalAlignment ?? 'top';
  s.text.lineSpacing = style.lineSpacing ?? 1.15;
  s.text.insets = style.insets ?? { top: 0, right: 0, bottom: 0, left: 0 };
  return s;
}

function addBox(slide, left, top, width, height, fill = C.white, line = C.line, radius = 16) {
  const s = slide.shapes.add({
    geometry: 'roundRect',
    position: { left, top, width, height },
    fill,
    line: { style: 'solid', fill: line, width: line === 'none' ? 0 : 1 },
    borderRadius: radius,
  });
  return s;
}

function addRule(slide, left, top, width, color = C.grid, height = 2) {
  return slide.shapes.add({
    geometry: 'rect',
    position: { left, top, width, height },
    fill: color,
    line: { style: 'solid', fill: 'none', width: 0 },
  });
}

function addPageHeader(slide, title, page, subtitle = '') {
  addText(slide, title, 72, 36, 900, 54, { fontSize: 34, bold: true, color: C.ink, verticalAlignment: 'middle' });
  addText(slide, `${String(page).padStart(2, '0')} / 06`, 1110, 42, 98, 28, { fontSize: 13, color: C.muted, alignment: 'right', verticalAlignment: 'middle' });
  addRule(slide, 72, 104, 1136, C.grid, 2);
  addRule(slide, 72, 104, 132, C.accent, 4);
  if (subtitle) addText(slide, subtitle, 72, 116, 900, 24, { fontSize: 14, color: C.muted, verticalAlignment: 'middle' });
}

function addMetric(slide, x, y, label, value, body, valueColor = C.accent) {
  addText(slide, label, x, y, 250, 22, { fontSize: 14, bold: true, color: C.muted, verticalAlignment: 'middle' });
  addText(slide, value, x, y + 22, 250, 52, { fontSize: 34, bold: true, color: valueColor, verticalAlignment: 'middle' });
  addText(slide, body, x, y + 82, 250, 84, { fontSize: 15, color: C.ink, lineSpacing: 1.2 });
}

// Preserve the source notes; slide-local objects are rebuilt to make spacing and hierarchy consistent.
const sourceNotes = slides.map((s) => s.speakerNotes.text);
slides.forEach(clearSlide);

// Slide 1: cover
{
  const s = slides[0];
  addText(s, '区域市场汇报', 108, 164, 240, 26, { fontSize: 15, bold: true, color: C.accent, verticalAlignment: 'middle' });
  addRule(s, 72, 200, 8, C.accent, 124);
  addText(s, '重点突击，\n全面开花', 108, 188, 700, 150, { fontSize: 50, bold: true, color: C.ink, lineSpacing: 1.0 });
  addText(s, '西京医院突破案例与区域市场进展', 112, 356, 560, 34, { fontSize: 20, color: C.muted, verticalAlignment: 'middle' });
  addRule(s, 112, 408, 360, C.line, 2);
  addText(s, '汇报人：郑阳阳\nDM：北京-赵国兴\nRM：北京-李萍', 112, 460, 360, 120, { fontSize: 17, color: C.muted, lineSpacing: 1.35 });
  addText(s, '01 / 06', 1110, 42, 98, 28, { fontSize: 13, color: C.muted, alignment: 'right', verticalAlignment: 'middle' });
}

// Slide 2: background and pain points
{
  const s = slides[1];
  addPageHeader(s, '01 背景与痛点｜西京医院进入壁垒', 2, '制度、认知与替代方案共同构成进入壁垒');
  addText(s, '目标客户', 72, 162, 280, 30, { fontSize: 22, bold: true, color: C.accentDark, verticalAlignment: 'middle' });
  addText(s, '西京医院', 72, 193, 280, 38, { fontSize: 28, bold: true, color: C.ink, verticalAlignment: 'middle' });
  addText(s, '专家：王*春教授（主任医师，西京毛发门诊专家）\n竞品：托法 / 巴瑞院内准入\n月门诊量：1,000 人次\n核心特征：部队医院、军事化管理，不处方院外产品', 72, 252, 300, 214, { fontSize: 17, color: C.ink, lineSpacing: 1.25 });
  addRule(s, 392, 164, 2, C.line, 330);
  addText(s, '三重进入壁垒', 432, 162, 360, 30, { fontSize: 22, bold: true, color: C.accentDark, verticalAlignment: 'middle' });
  const barriers = [
    ['01', '制度壁垒', '医院管理严格，医生以院内用药为主，不处方院外产品。'],
    ['02', '认知壁垒', '医生对乐复诺暂无使用经验，难以形成认可。'],
    ['03', '替代方案', '现有方案已能满足疗效与费用需求，医生不愿承担额外风险。'],
  ];
  barriers.forEach((b, i) => {
    const y = 214 + i * 108;
    const circ = s.shapes.add({ geometry: 'ellipse', position: { left: 432, top: y + 2, width: 34, height: 34 }, fill: i === 2 ? C.teal : C.accent, line: { style: 'solid', fill: 'none', width: 0 } });
    addText(s, b[0], 432, y + 4, 34, 28, { fontSize: 13, bold: true, color: C.white, alignment: 'center', verticalAlignment: 'middle' });
    addText(s, b[1], 480, y, 170, 28, { fontSize: 18, bold: true, color: C.accentDark, verticalAlignment: 'middle' });
    addText(s, b[2], 480, y + 32, 350, 54, { fontSize: 16, color: C.ink, lineSpacing: 1.2 });
  });
  addRule(s, 868, 164, 2, C.line, 330);
  addText(s, '医生原话', 910, 162, 180, 26, { fontSize: 14, bold: true, color: C.accent, verticalAlignment: 'middle' });
  addText(s, '“现有疗法效果挺好，\n费用还便宜还在院内，\n我为什么要冒险给病人用\n一个院外的新药？”', 910, 218, 282, 174, { fontSize: 20, italic: true, color: C.ink, lineSpacing: 1.25 });
  const band = addBox(s, 72, 596, 1136, 58, C.accentDark, C.accentDark, 10);
  addText(s, '当前状态', 96, 609, 120, 30, { fontSize: 15, bold: true, color: '#CFE1F6', verticalAlignment: 'middle' });
  addText(s, '零准入  ·  零使用  ·  零认知', 220, 605, 560, 38, { fontSize: 24, bold: true, color: C.white, verticalAlignment: 'middle' });
  s.speakerNotes.setText(sourceNotes[1]);
}

// Slide 3: four-step method
{
  const s = slides[2];
  addPageHeader(s, '02 四步破局法', 3, '从需求识别到观念升级，逐步降低进入风险');
  const centers = [180, 455, 730, 1005];
  addRule(s, 180, 226, 825, C.line, 3);
  const steps = [
    ['01', '需求洞察', '识别客户对安全与无风险的深层诉求，以乐复诺唯一的青少年适应症切入。'],
    ['02', '精准切入', '锁定青少年患者与竞品应答不佳患者，用一句话术打开对话。'],
    ['03', '建立信任', '共同管理患者，及时回顾疗效，用真实成功案例改变医生判断。'],
    ['04', '观念升级', '将青少年应用场景扩展至更广泛的斑秃患者，推动首选方案与长期处方。'],
  ];
  steps.forEach((step, i) => {
    const cx = centers[i];
    s.shapes.add({ geometry: 'ellipse', position: { left: cx - 26, top: 200, width: 52, height: 52 }, fill: i === 3 ? C.teal : C.accent, line: { style: 'solid', fill: 'none', width: 0 } });
    addText(s, step[0], cx - 26, 211, 52, 28, { fontSize: 16, bold: true, color: C.white, alignment: 'center', verticalAlignment: 'middle' });
    addText(s, step[1], cx - 105, 284, 210, 32, { fontSize: 22, bold: true, color: C.accentDark, alignment: 'center', verticalAlignment: 'middle' });
    addRule(s, cx - 34, 332, 68, i === 3 ? C.teal : C.accent, 4);
    addText(s, step[2], cx - 104, 360, 208, 170, { fontSize: 17, color: C.ink, lineSpacing: 1.28 });
    if (i < 3) addRule(s, cx + 136, 286, 2, C.line, 260);
  });
  const band = addBox(s, 72, 610, 1136, 54, C.sky2, C.sky2, 10);
  addText(s, '破局逻辑', 96, 622, 120, 28, { fontSize: 15, bold: true, color: C.accentDark, verticalAlignment: 'middle' });
  addText(s, '先解决风险顾虑，再用病例推动处方习惯升级。', 224, 618, 800, 32, { fontSize: 20, bold: true, color: C.ink, verticalAlignment: 'middle' });
  s.speakerNotes.setText(sourceNotes[2]);
}

// Slide 4: results and repeatable method
{
  const s = slides[3];
  addPageHeader(s, '03 结果｜从破冰到引领', 4, '月均处方量增长，医生观念与处方周期同步升级');
  addText(s, '月均处方量（盒）', 72, 144, 300, 28, { fontSize: 16, bold: true, color: C.muted, verticalAlignment: 'middle' });
  const bar = s.charts.add('bar', {
    position: { left: 72, top: 178, width: 300, height: 338 },
    categories: ['起始', '3个月', '6个月'],
    series: [{ name: '处方量', values: [2, 20, 150], fill: C.accent }],
    hasLegend: false,
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 70 },
    chartFill: C.white,
    plotAreaFill: C.white,
    chartLine: { style: 'solid', fill: C.line, width: 1 },
    plotAreaLine: { style: 'solid', fill: 'none', width: 0 },
    xAxis: { textStyle: { typeface: FONT, fontSize: 13, fill: C.muted }, line: { style: 'solid', fill: C.line, width: 1 }, majorGridlines: null },
    yAxis: { min: 0, max: 160, majorUnit: 40, textStyle: { typeface: FONT, fontSize: 12, fill: C.muted }, line: { style: 'solid', fill: C.line, width: 1 }, majorGridlines: { style: 'solid', fill: C.grid, width: 1 } },
    dataLabels: { showValue: true, position: 'outEnd', textStyle: { typeface: FONT, fontSize: 13, fill: C.ink, bold: true } },
  });
  applyPresentationChartFont(bar, { fontFamily: FONT });
  addRule(s, 392, 144, 2, C.line, 372);
  const values = [
    ['观念维度', '起始状态', '6个月后状态'],
    ['扩重度', '偶尔尝试青少年患者', '优势患者首选乐复诺'],
    ['早启动', '传统升阶梯治疗', '认可早起始的临床获益'],
    ['赢首选', '首选巴瑞替尼，偶尔尝试青少年', '优势患者首选乐复诺，并主动告知优势'],
    ['久治疗', '处方周期短（DOT 1）', '认可长期方案，延长至 DOT 6'],
  ];
  const table = s.tables.add({ rows: 5, columns: 3, left: 420, top: 144, width: 548, height: 372, values });
  table.borders.assign({ style: 'solid', fill: C.line, width: 1 });
  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 3; c++) {
      const cell = table.getCell(r, c);
      cell.fill = r === 0 ? C.accent : c === 0 ? C.sky2 : C.white;
      cell.text.style = { typeface: FONT, fontSize: r === 0 ? 13 : 12, bold: r === 0 || c === 0, color: r === 0 ? C.white : C.ink };
      cell.text.verticalAlignment = 'middle';
      cell.text.autoFit = 'shrinkText';
      cell.text.wrap = 'square';
      cell.text.insets = { top: 6, right: 8, bottom: 6, left: 8 };
      cell.text.lineSpacing = 1.08;
    }
  }
  addBox(s, 994, 144, 214, 372, C.sky2, C.sky2, 12);
  addText(s, '可复制方法论', 1018, 174, 170, 30, { fontSize: 20, bold: true, color: C.accentDark, verticalAlignment: 'middle' });
  addRule(s, 1018, 218, 58, C.accent, 4);
  addText(s, '1  锁定关键医生\n2  识别核心需求\n3  找到突破类型\n4  传递针对性信息\n5  巩固并推动升级', 1018, 244, 170, 220, { fontSize: 16, color: C.ink, lineSpacing: 1.42 });
  addBox(s, 72, 590, 1136, 58, C.accent, C.accent, 10);
  addText(s, '一句话复制', 96, 603, 120, 30, { fontSize: 15, bold: true, color: '#CFE1F6', verticalAlignment: 'middle' });
  addText(s, '选对核心 KOL，洞察顾虑，用真实病例建立信任，让观念与业绩同步突破。', 224, 600, 930, 36, { fontSize: 18, bold: true, color: C.white, verticalAlignment: 'middle' });
  s.speakerNotes.setText(sourceNotes[3]);
}

// Slide 5: market review
{
  const s = slides[4];
  addPageHeader(s, '市场回顾｜西北五省', 5, '市场覆盖、处方客户与医院准入进展');
  const cards = [
    { x: 72, label: '市场覆盖', value: '5省', sub: '19 家医院', body: '陕西 9 家｜破冰 9 家（7 家处方）\n新疆 3 家｜破冰 2 家\n宁夏 2 家｜破冰 1 家\n兰州 3 家｜破冰 2 家\n青海 2 家｜破冰 1 家', accent: C.accent },
    { x: 452, label: '处方客户', value: '53位', sub: '+7 位较去年新增', body: '超 A 客户：3 位\nA 客户：3 位\nB 客户：5 位\n其他 C / D 客户', accent: C.teal },
    { x: 832, label: '医院准入', value: '2家正式', sub: '2024–2025', body: '2024｜延安大学咸阳医院\n2025｜唐都医院（正式）\n2026 目标｜西安医学院第二附属医院\n（临采 2 次）', accent: C.gold },
  ];
  cards.forEach((card) => {
    addBox(s, card.x, 164, 328, 416, C.white, C.line, 14);
    addRule(s, card.x, 164, 328, card.accent, 8);
    addText(s, card.label, card.x + 24, 196, 190, 26, { fontSize: 18, bold: true, color: C.accentDark, verticalAlignment: 'middle' });
    addText(s, card.value, card.x + 24, 230, 220, 54, { fontSize: 36, bold: true, color: card.accent, verticalAlignment: 'middle' });
    addText(s, card.sub, card.x + 24, 286, 260, 26, { fontSize: 15, bold: true, color: C.muted, verticalAlignment: 'middle' });
    addRule(s, card.x + 24, 326, 80, card.accent, 4);
    addText(s, card.body, card.x + 24, 356, 278, 190, { fontSize: 16, color: C.ink, lineSpacing: 1.3 });
  });
  s.speakerNotes.setText(sourceNotes[4]);
}

// Slide 6: data analysis
{
  const s = slides[5];
  addPageHeader(s, '数据分析｜达成与增长', 6, '季度指标、销售与达成率的变化');
  addText(s, '达成率', 72, 140, 240, 24, { fontSize: 16, bold: true, color: C.muted, verticalAlignment: 'middle' });
  const quarters = ['24Q1', '24Q2', '24Q3', '24Q4', '25Q1', '25Q2', '25Q3', '25Q4', '26Q1'];
  const completion = [1.99186991869919, 1.40243902439024, 1.1096256684492, 1.05325795894079, 1.15850302696753, 1.000231000231, 0.714285714285714, 1.14864864864865, 0.520833333333333];
  const line = s.charts.add('line', {
    position: { left: 72, top: 162, width: 690, height: 156 },
    categories: quarters,
    series: [{ name: '达成率', values: completion, line: { style: 'solid', fill: C.accent, width: 3 }, marker: { symbol: 'circle', size: 7 } }],
    hasLegend: false,
    lineOptions: { smooth: true },
    chartFill: C.white,
    plotAreaFill: C.white,
    chartLine: { style: 'solid', fill: C.line, width: 1 },
    plotAreaLine: { style: 'solid', fill: 'none', width: 0 },
    xAxis: { textStyle: { typeface: FONT, fontSize: 11, fill: C.muted }, line: { style: 'solid', fill: C.line, width: 1 }, majorGridlines: null },
    yAxis: { min: 0, max: 2.2, majorUnit: 0.5, numberFormatCode: '0%', textStyle: { typeface: FONT, fontSize: 11, fill: C.muted }, line: { style: 'solid', fill: C.line, width: 1 }, majorGridlines: { style: 'solid', fill: C.grid, width: 1 } },
    dataLabels: { showValue: false },
  });
  applyPresentationChartFont(line, { fontFamily: FONT });
  addText(s, '指标与销售', 72, 326, 240, 24, { fontSize: 16, bold: true, color: C.muted, verticalAlignment: 'middle' });
  const bars = s.charts.add('bar', {
    position: { left: 72, top: 348, width: 690, height: 270 },
    categories: quarters,
    series: [
      { name: '指标', values: [49.2, 196.8, 299.2, 336.1, 363.4, 432.9, 450.8, 370, 480], fill: '#9CB7E5' },
      { name: '销售', values: [98, 276, 332, 354, 421, 433, 322, 425, 250], fill: C.accent },
    ],
    hasLegend: true,
    legend: { position: 'bottom', overlay: false, textStyle: { typeface: FONT, fontSize: 12, fill: C.muted } },
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 35 },
    chartFill: C.white,
    plotAreaFill: C.white,
    chartLine: { style: 'solid', fill: C.line, width: 1 },
    plotAreaLine: { style: 'solid', fill: 'none', width: 0 },
    xAxis: { textStyle: { typeface: FONT, fontSize: 11, fill: C.muted }, line: { style: 'solid', fill: C.line, width: 1 }, majorGridlines: null },
    yAxis: { min: 0, max: 600, majorUnit: 100, textStyle: { typeface: FONT, fontSize: 11, fill: C.muted }, line: { style: 'solid', fill: C.line, width: 1 }, majorGridlines: { style: 'solid', fill: C.grid, width: 1 } },
    dataLabels: { showValue: true, position: 'outEnd', textStyle: { typeface: FONT, fontSize: 9, fill: C.muted } },
  });
  applyPresentationChartFont(bars, { fontFamily: FONT });
  addRule(s, 790, 140, 2, C.line, 478);
  addMetric(s, 824, 144, '2024 实际', '139%', '全年增长 700%\n全年初诊 200+｜DOT 6.2 / 人\n跑单率 11.3%', C.accent);
  addRule(s, 824, 304, 340, C.grid, 1);
  addMetric(s, 824, 326, '2025 预计', '100%', '较去年增长 50%\n上半年初诊 200+｜DOT 4.3 / 人\n跑单率 45%', C.teal);
  addRule(s, 824, 486, 340, C.grid, 1);
  addMetric(s, 824, 508, '非核心市场', '10.7%', '2025 全年销售占比\n增长来自核心市场深耕、区域扩面\n以及处方客户与 DOT 管理', C.gold);
  s.speakerNotes.setText(sourceNotes[5]);
}

// Export draft and individual previews for visual QA.
const candidatePath = path.join(buildDir, 'candidate.pptx');
const exported = await PresentationFile.exportPptx(p);
await exported.save(candidatePath);
for (let i = 0; i < slides.length; i++) {
  const png = await slides[i].export({ format: 'png', scale: 1.2 });
  await fs.writeFile(path.join(previewDir, `slide-${i + 1}.png`), new Uint8Array(await png.arrayBuffer()));
}
console.log(`Wrote ${candidatePath}`);
