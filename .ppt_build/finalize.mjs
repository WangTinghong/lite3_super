import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const workspaceDir = '/Users/wangtinghong/Desktop/工作/lite3_gym';
const skillDir = '/Users/wangtinghong/.codex/plugins/cache/openai-primary-runtime/presentations/26.915.20218/skills/presentations';
const runtimePython = '/Users/wangtinghong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const candidatePath = path.join(workspaceDir, '.ppt_build', 'candidate.pptx');
const finalPath = path.join(workspaceDir, 'output', '副本重点突击到全面开花_美化版.pptx');
const stagingDir = path.join(workspaceDir, '.codex-finalizer');
await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });
try { await fs.unlink(finalPath); } catch {}
const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
const result = await finalizePresentation({
  explicitTotalSlideCount: 6,
  requiredNativeTableOwnerSlides: [4],
  requiredNativeChartOwnerSlides: [4, 6],
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: [
    '--expected-slide-size-emu', '12192000,6858000',
    '--validate-bullet-geometry',
    '--validate-heading-fit',
    '--require-native-table-slide', '4',
  ],
  requiredNativeTableOwnerSlides: [4],
  fontPolicy: { basis: 'design', families: ['PingFang SC'] },
  verifyArtifactToolImport: true,
  materializeLiteralChartWorkbooks: true,
  receiptPath: path.join(stagingDir, '副本重点突击到全面开花_美化版.validation.json'),
});
console.log(JSON.stringify(result, null, 2));
