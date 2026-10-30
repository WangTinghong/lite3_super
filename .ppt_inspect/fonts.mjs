import { pathToFileURL } from 'node:url'; import path from 'node:path';
const u=await import(pathToFileURL(path.join('/Users/wangtinghong/.codex/plugins/cache/openai-primary-runtime/presentations/26.915.20218/skills/presentations/container_tools/artifact_tool_utils.mjs')).href);
console.log(Object.keys(u));
for (const fam of [undefined,'Microsoft YaHei','PingFang SC','Noto Sans CJK SC','Source Han Sans SC','Arial']) { try { console.log(fam, u.resolvePresentationFont(fam?{fontFamily:fam}:undefined)); } catch(e){ console.log(fam,'ERR',e.message); } }
