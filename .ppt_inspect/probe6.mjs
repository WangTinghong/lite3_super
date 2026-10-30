import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('/Users/wangtinghong/Desktop/工作/副本重点突击到全面开花(1).pptx'));
console.log('slides keys',Object.keys(p.slides),Object.getOwnPropertyNames(Object.getPrototypeOf(p.slides)));
const s=p.slides.items[1]; console.log('slide keys',Object.keys(s),Object.getOwnPropertyNames(Object.getPrototypeOf(s)));
for(const k of ['shapes','charts','tables','images','placeholders']){try{const c=s[k]; console.log(k, c&&Object.keys(c), c&&Object.getOwnPropertyNames(Object.getPrototypeOf(c)), c?.items?.length)}catch(e){}}
