import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('/Users/wangtinghong/Desktop/工作/副本重点突击到全面开花(1).pptx'));
const t=p.resolve('sh/ozy1ofad').text;
for(const k of ['fontSize','typeface','lineSpacing','alignment','verticalAlignment','autoFit','wrap','insets','style','className','defaultTextStyle','color','fill','bold','italic']){try{console.log(k,JSON.stringify(t[k]));}catch(e){console.log(k,'ERR',e.message)}}
console.log('get', t.get(0), t.getRange(0,5));
