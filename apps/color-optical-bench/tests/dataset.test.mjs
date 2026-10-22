import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,mkdtemp,rm,readFile,writeFile,symlink} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {join} from 'node:path';
import {createSender,senderFrames,base64ToBytes,bytesToBase64,floatBytes,floatsFromBytes} from '../dist/js/dataset/contract.js';
import {captureFrame,CaptureBuffer,validateCapture} from '../dist/js/dataset/capture.js';
import {renderFrame,FORMAT,parseHeader} from '../dist/js/format.js';
import {gridLayout} from '../dist/js/grid.js';
import {generate,joinCaptures,evaluate} from '../../../scripts/decoder-dataset.mjs';

const base=fileURLToPath(new URL('../../../validation/',import.meta.url));
const sender=await createSender({seed:123,sessionId:'sender-test'});
const frame=sender.frames[1],image=renderFrame(frame,1024),record=captureFrame(image,{timestamp:1000});
const copy=value=>JSON.parse(JSON.stringify(value));
async function temp(fn){await mkdir(base,{recursive:true});const root=await mkdtemp(join(base,'dataset-'));try{await fn(root);}finally{await rm(root,{recursive:true,force:true});}}
function bundle(){const b=new CaptureBuffer();b.add(record);b.add({...record,timestamp:1600});return b.export({sessionId:'physical-session'});}

test('sender manifests regenerate exact physical phases, including fixed references and parity',async()=>{
  const restored=await senderFrames(sender.manifest),layout=gridLayout(64,16,8);
  assert.equal(restored.length,4);
  for(let i=0;i<4;i++)assert.deepEqual(restored[i].symbols,sender.frames[i].symbols);
  for(const cell of layout.pilots)assert.equal(restored[0].symbols[cell],restored[3].symbols[cell]);
  assert.ok(layout.cells.some(cell=>restored[0].symbols[cell]!==restored[3].symbols[cell]));
  const damaged=copy(sender.manifest);damaged.payloadBase64=bytesToBase64(new Uint8Array(base64ToBytes(damaged.payloadBase64).length));
  await assert.rejects(senderFrames(damaged),/hash/);
  const wrongInfo=copy(sender.manifest);wrongInfo.frames[0].info.bits=1;
  await assert.rejects(senderFrames(wrongInfo),/metadata/);
});

test('receiver produces normalized evidence without sender knowledge; failed payloads remain usable',()=>{
  assert.equal(record.ok,true);validateCapture(record);
  assert.equal(record.info.phase,1);assert.equal(record.inputBase64.length>0,true);
  const damaged={...image,data:image.data.slice()},p=FORMAT.payload,scale=image.width/256;
  for(let y=Math.ceil(p.y*scale);y<Math.floor((p.y+p.side)*scale);y++)for(let x=Math.ceil(p.x*scale);x<Math.floor((p.x+p.side)*scale);x++) {
    const i=(y*image.width+x)*4;damaged.data[i]=damaged.data[i+1]=damaged.data[i+2]=100;
  }
  const bad=captureFrame(damaged,{timestamp:2000});
  assert.equal(bad.ok,true);assert.equal(bad.baseline.accepted,false);validateCapture(bad);
  const malformed=copy(record);malformed.inputBase64='AAAA';assert.throws(()=>validateCapture(malformed),/length/);
});

test('settling guard resets on invalid headers and rejects duplicate captures; bounds preserve diagnostics',()=>{
  const b=new CaptureBuffer({maxRecords:1});
  assert.equal(b.add(record).accepted,false);
  assert.equal(b.add({...record,timestamp:1200}).accepted,false);
  b.add({ok:false,timestamp:1300,stage:'header',error:'bad',original:{mime:'image/png',base64:'AQID'}});
  assert.equal(b.add({...record,timestamp:1600}).accepted,false);
  assert.equal(b.add({...record,timestamp:2200}).accepted,true);
  assert.equal(b.add({...record,timestamp:2200}).stage,'duplicate');
  assert.equal(b.add({...record,timestamp:2800}).stage,'full');
  assert.equal(b.export({sessionId:'receiver'}).diagnostics.find(x=>x.stage==='header').original.base64,'AQID');
  assert.equal(b.summary().records,1);assert.ok(b.summary().bytes<32*1024*1024);
});

test('offline joining requires full header match and preserves receiver session and failed diagnostic records',async()=>temp(async root=>{
  const s=join(root,'sender.json'),r=join(root,'capture.json');
  await writeFile(s,JSON.stringify(sender.manifest));await writeFile(r,JSON.stringify(bundle()));
  const out=join(root,'joined');await joinCaptures({out,senders:s,captures:r,split:'validation'});
  const m=JSON.parse(await readFile(join(out,'manifest.json'),'utf8'));
  assert.equal(m.samples.length,1);assert.equal(m.samples[0].sessionId,'physical-session');assert.equal(m.samples[0].senderSessionId,'sender-test');
  const labels=await readFile(join(out,m.samples[0].labels));assert.deepEqual(new Uint8Array(labels),frame.symbols);
  const conflicting={...sender.manifest,sessionId:'another-session'},s2=join(root,'sender2.json');await writeFile(s2,JSON.stringify(conflicting));
  await assert.rejects(joinCaptures({out:join(root,'conflict'),senders:`${s},${s2}`,captures:r}),/Conflicting sender/);
  const broken=bundle();broken.records[0].info.bits=1;await writeFile(r,JSON.stringify(broken));
  await assert.rejects(joinCaptures({out:join(root,'bad'),senders:s,captures:r}),/metadata/);
  const invalid=bundle();invalid.records[0].stability.observations=null;await writeFile(r,JSON.stringify(invalid));
  await assert.rejects(joinCaptures({out:join(root,'bad-guard'),senders:s,captures:r}),/guard/);
}));

test('clean generation yields distinct dataset IDs and exact canonical recovery with independent predictions',async()=>temp(async root=>{
  const train=join(root,'train'),validation=join(root,'validation');
  await generate({out:train,count:'1',size:'64',colors:'4',phases:'1',seed:'1',split:'train'});
  await generate({out:validation,count:'1',size:'64',colors:'4',phases:'1',seed:'2',split:'validation'});
  const a=JSON.parse(await readFile(join(train,'manifest.json'),'utf8')),b=JSON.parse(await readFile(join(validation,'manifest.json'),'utf8'));
  assert.notEqual(a.samples[0].id,b.samples[0].id);
  const sample=a.samples[0],labels=new Uint8Array(await readFile(join(train,sample.labels))),logits=new Float32Array(16*64*64).fill(-20);
  labels.forEach((color,cell)=>{logits[color*64*64+cell]=20;});
  await writeFile(join(root,'prediction.f32'),floatBytes(logits));const prediction=join(root,'prediction.json');
  await writeFile(prediction,JSON.stringify({samples:[{id:sample.id,logits:'prediction.f32'}]}));
  const result=await evaluate({dataset:train,predictions:prediction});
  assert.equal(result.summary.classical.exact,1);assert.equal(result.summary.learned.exact,1);assert.equal(result.summary.learned.wrongAccepted,0);
  await assert.rejects(generate({out:train,count:'1'}),/empty/);
  await writeFile(join(root,'outside.f32'),floatBytes(new Float32Array(3*256*256)));
  await rm(join(train,sample.input));await symlink(join(root,'outside.f32'),join(train,sample.input));
  await assert.rejects(evaluate({dataset:train,predictions:prediction}),/symlink/);
}));

test('float/base64 contract rejects malformed, oversized, and non-finite tensors',()=>{
  const values=Float32Array.of(-1.5,.5,1.5);assert.deepEqual(floatsFromBytes(base64ToBytes(bytesToBase64(floatBytes(values))),3),values);
  assert.throws(()=>base64ToBytes('!!!='),/Invalid/);assert.throws(()=>base64ToBytes('AQID',2),/limit|oversized/);
  assert.throws(()=>floatBytes(Float32Array.of(NaN)),/Non-finite/);
});
