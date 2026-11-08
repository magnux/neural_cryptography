/** Offline dataset JSON: one current schema, without a file-sized JS string.
 * Sample metadata remains resident for indexing; image arrays stay on disk.
 * Limits apply to individual metadata values, not the total corpus size.
 */
import {open,mkdtemp,rename,rm} from 'node:fs/promises';
import {dirname,join,basename} from 'node:path';
import {readObjectJSON} from './recording-json.mjs';

export async function readDatasetDocument(path,options={}){
  return readObjectJSON(path,{maxBytes:Number.MAX_SAFE_INTEGER,maxArrayItems:Number.MAX_SAFE_INTEGER,
    maxTokenBytes:16*1024*1024,arrayKeys:['samples'],...options});
}
export async function readDatasetJSON(path,options={}){
  return (await readDatasetDocument(path,options)).metadata;
}
/** Compact, incremental and atomically replaced. Never stringify all samples. */
export async function writeDatasetJSON(path,value){
  if(!value||typeof value!=='object'||Array.isArray(value))throw Error('Dataset JSON must be an object.');
  const temporary=await mkdtemp(join(dirname(path),`.${basename(path)}-writing-`));
  const staging=join(temporary,'manifest.json');let file;
  try{
    file=await open(staging,'wx');let pending=[],bytes=0;
    const flush=async()=>{if(pending.length){await file.writeFile(pending.join(''));pending=[];bytes=0;}};
    const emit=async text=>{pending.push(text);bytes+=Buffer.byteLength(text);if(bytes>=256*1024)await flush();};
    await emit('{');let first=true;
    for(const [key,field] of Object.entries(value)){
      if(field===undefined)continue;
      await emit((first?'':',')+JSON.stringify(key)+':');first=false;
      if(key==='samples'&&Array.isArray(field)){
        await emit('[');let firstItem=true;
        for(const item of field){await emit((firstItem?'':',')+JSON.stringify(item));firstItem=false;}
        await emit(']');
      }else await emit(JSON.stringify(field));
    }
    await emit('}\n');await flush();await file.close();file=null;
    await rename(staging,path);
  }finally{await file?.close();await rm(temporary,{recursive:true,force:true});}
}
