import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const src = '/Users/wangtinghong/Desktop/工作/副本重点突击到全面开花(1).pptx';
const p = await PresentationFile.importPptx(await FileBlob.load(src));
const snap = await p.inspect({kind:'slide,textbox,shape,image,table,chart,notes,layout', maxChars:50000});
await fs.writeFile('.ppt_inspect/snapshot.ndjson', snap.ndjson);
console.log(snap.ndjson);
console.log('slides', p.slides.items.length, 'size', p.slideSize);
for (let i=0;i<p.slides.items.length;i++) {
 const s=p.slides.items[i];
 const out = await s.export({format:'png', scale:1});
 await fs.writeFile(`.ppt_inspect/slide-${i+1}.png`, new Uint8Array(await out.arrayBuffer()));
}
