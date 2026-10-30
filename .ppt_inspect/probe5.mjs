import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('/Users/wangtinghong/Desktop/工作/副本重点突击到全面开花(1).pptx'));
for(const id of ['ch/pwrm5kvq','ch/65sfup0n','tb/1g3ex4zu']){
 const o=p.resolve(id); console.log('\n',id, o.constructor.name, Object.keys(o));
 for(const k of ['title','categories','series','xAxis','yAxis','legend','dataLabels','dataTable']){try{const v=o[k]; if(v!==undefined){if(k==='series') console.log('series',v.items?.map(s=>({name:s.name,values:s.values,categories:s.categories,fill:s.fill,line:s.line}))); else console.log(k,JSON.stringify(v));}}catch(e){console.log(k,'ERR',e.message)}}
 if(id.startsWith('tb')){
  console.log('table proto',Object.getOwnPropertyNames(Object.getPrototypeOf(o)));
  for(let r=0;r<5;r++){let row=[];for(let c=0;c<3;c++){try{const cell=o.getCell(r,c); row.push({text:cell.text.toString(),style:cell.text.style,fill:cell.fill,shape:cell.constructor.name});}catch(e){row.push({err:e.message});}} console.log(r,JSON.stringify(row));}
 }
}
