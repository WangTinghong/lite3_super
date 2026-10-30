import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('/Users/wangtinghong/Desktop/工作/副本重点突击到全面开花(1).pptx'));
for(const id of ['sh/k3yl0zql','sh/7qp4be9c','sh/298ryl4v','sh/n6ls3alk','sh/w3i1sfa9','sh/cf2tcr61','sh/xg3ylgj2','sh/7m98ru9g','sh/ml07i9sv']){
 const o=p.resolve(id); const t=o.text; console.log(id,JSON.stringify({text:t.toString(),typeface:t.typeface,fontSize:t.fontSize,color:t.color,fill:t.fill,bold:t.bold,align:t.alignment,valign:t.verticalAlignment,autoFit:t.autoFit,wrap:t.wrap,insets:t.insets,style:t.style})); }
