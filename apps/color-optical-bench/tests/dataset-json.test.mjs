import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,readdir,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {readDatasetJSON,readDatasetDocument,writeDatasetJSON} from '../../../scripts/lib/dataset-json.mjs';
import {readRecordingJSON} from '../../../scripts/lib/recording-json.mjs';

async function fixture(body){
  const root=await mkdtemp(join(tmpdir(),'dataset-json-'));
  try{await body(join(root,'manifest.json'),root);}finally{await rm(root,{recursive:true,force:true});}
}
test('offline manifests parse samples separately across arbitrary UTF-8 boundaries',()=>fixture(async path=>{
  const value={samples:Array.from({length:1100},(_,i)=>({id:String(i),nested:{text:'á 🌈 \\" [] {}',array:[i,null,true]}})),format:'neural-qr-dataset',version:2};
  const text=JSON.stringify(value,null,2)+'\n';await writeFile(path,text);
  const parsed=await readDatasetDocument(path,{highWaterMark:7});
  assert.deepEqual(parsed.metadata,value);
  assert.equal(parsed.counts.samples,1100);
  assert.equal(parsed.sha256,createHash('sha256').update(text).digest('hex'));
  const visited=[];
  const streamed=await readDatasetDocument(path,{streamArrays:true,onItem:(key,row)=>visited.push([key,row.id])});
  assert.equal(streamed.metadata.samples.length,0);assert.equal(visited.length,1100);
}));
test('manifest writer roundtrips compact JSON including metadata after samples',()=>fixture(async path=>{
  const value={format:'neural-qr-dataset',version:2,samples:[{id:'a',hashes:{input:'f'.repeat(64)}}],generator:{seed:41}};
  await writeDatasetJSON(path,value);
  assert.deepEqual(await readDatasetJSON(path),value);
  assert.equal(await readFile(path,'utf8'),JSON.stringify(value)+'\n');
}));
test('malformed, truncated, duplicate-key and trailing JSON are rejected',()=>fixture(async path=>{
  for(const text of ['{"samples":[{},]}','{"samples":[{','{"samples":[],"samples":[]}','{"samples":[]}x','[]','{"samples":[],}']){
    await writeFile(path,text);
    await assert.rejects(readDatasetJSON(path,{highWaterMark:1}));
  }
  await writeFile(path,Buffer.from([123,34,120,34,58,34,0xff,34,125]));
  await assert.rejects(readDatasetJSON(path));
}));
test('offline size policy preserves per-value bounds and recording limits',()=>fixture(async path=>{
  await writeFile(path,JSON.stringify({samples:[{id:'x'.repeat(200)}]}));
  await assert.rejects(readDatasetJSON(path,{maxTokenBytes:64}),/per-value/);
  await writeFile(path,JSON.stringify({records:Array.from({length:1025},()=>({}))}));
  await assert.rejects(readRecordingJSON(path),/item limit/);
}));
test('failed manifest replacement preserves previous content and cleans staging',()=>fixture(async(path,root)=>{
  const value={samples:[],format:'original'};await writeDatasetJSON(path,value);
  await assert.rejects(writeDatasetJSON(path,{samples:[{notJSON:1n}]}));
  assert.deepEqual(await readDatasetJSON(path),value);
  assert.deepEqual(await readdir(root),['manifest.json']);
}));
