import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const src = '/Users/wangtinghong/Desktop/工作/副本重点突击到全面开花(1).pptx';
const p = await PresentationFile.importPptx(await FileBlob.load(src));
for (const id of ['sl/hwbqtkby','sh/ozy1ofad','sh/x4r21kru','sh/w3i1sfa9','ch/65sfup0n','tb/1g3ex4zu']) {
  const o=p.resolve(id);
  console.log('\nID',id,'type',o?.constructor?.name);
  for(const k of ['name','text','position','frame','fill','line','borderRadius','shadow','geometry','title','xAxis','yAxis','legend','dataLabels']){
    try { const v=o?.[k]; if(v!==undefined) console.log(k,typeof v==='object'?JSON.stringify(v,null,2):v); } catch(e){console.log(k,'ERR',e.message)}
  }
  try { console.log('textstyle',JSON.stringify(o.text?.style)); } catch(e){}
}
