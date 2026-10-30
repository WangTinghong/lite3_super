import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const p = await PresentationFile.importPptx(await FileBlob.load('/Users/wangtinghong/Desktop/工作/副本重点突击到全面开花(1).pptx'));
const o=p.resolve('sh/ozy1ofad');
console.log('shape keys',Object.keys(o)); console.log('shape proto',Object.getOwnPropertyNames(Object.getPrototypeOf(o)));
console.log('text keys',Object.keys(o.text)); console.log('text proto',Object.getOwnPropertyNames(Object.getPrototypeOf(o.text)));
console.log('text value?', o.text.toString?.(), o.text.text, o.text.runs?.length);
console.log('runs',o.text.runs?.items?.map(x=>({keys:Object.keys(x),proto:Object.getOwnPropertyNames(Object.getPrototypeOf(x)),text:x.text,style:x.style})));
