import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,mkdtemp,rm,readFile,writeFile,symlink} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {createSender,senderFrames,base64ToBytes,bytesToBase64,floatBytes,floatsFromBytes} from '../dist/js/dataset/contract.js';
import {captureFrame,CaptureBuffer,validateCapture} from '../dist/js/dataset/capture.js';
import {renderFrame,FORMAT,parseHeader} from '../dist/js/format.js';
import {gridLayout} from '../dist/js/grid.js';
import {generate,joinCaptures,targets,evaluate} from '../../../scripts/decoder-dataset.mjs';
import {extractLearnedInput,logitsToLogicalScores} from '../dist/js/learned-input.js';
import {decodeScores} from '../dist/js/vision.js';

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
  const larger=bundle(),first=larger.records[0];
  larger.records=Array.from({length:24},(_,i)=>({...first,timestamp:first.timestamp+i*600}));
  await writeFile(r,JSON.stringify(larger));
  const largeResult=await joinCaptures({out:join(root,'large-pool'),senders:s,captures:r});
  assert.equal(largeResult.samples,24,'current importer accepts pools larger than 20');
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
  await writeFile(prediction,JSON.stringify({framesPerInput:1,outputLabelSpace:'physical',samples:[{id:sample.id,memberIds:[sample.id],logits:'prediction.f32'}]}));
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

test('target reconstruction deduplicates captures and restores sender colors without reading camera pixels',async()=>temp(async root=>{
  const s=join(root,'sender.json'),r=join(root,'capture.json'),joined=join(root,'joined'),out=join(root,'targets');
  const b=new CaptureBuffer();let timestamp=1000;
  for(const phase of [1,3]) {
    const captured=captureFrame(renderFrame(sender.frames[phase],1024),{timestamp});
    b.add(captured);b.add({...captured,timestamp:timestamp+600});b.add({...captured,timestamp:timestamp+1200});timestamp+=2000;
  }
  await writeFile(s,JSON.stringify(sender.manifest));await writeFile(r,JSON.stringify(b.export({sessionId:'physical-target-session'})));
  await joinCaptures({out:joined,senders:s,captures:r,split:'validation'});
  const source=JSON.parse(await readFile(join(joined,'manifest.json'),'utf8'));
  assert.equal(source.samples.length,4);
  // The target path must depend on sender truth, never on a noisy input tensor.
  for(const sample of source.samples)await rm(join(joined,sample.input));
  const result=await targets({dataset:joined,out});
  assert.equal(result.samples,2);assert.equal(result.deduplicated,2);
  const clean=JSON.parse(await readFile(join(out,'manifest.json'),'utf8'));
  for(const sample of clean.samples) {
    const original=source.samples.find(x=>x.info.phase===sample.info.phase);
    for(const field of ['sessionId','senderSessionId','payloadId','split','size','colors','info','burstId'])assert.deepEqual(sample[field],original[field]);
    assert.equal(sample.source,'synthetic');assert.notEqual(sample.id,original.id);
    for(const field of ['labels','mask','expected'])assert.equal(sample.hashes[field],original.hashes[field]);
    const input=extractLearnedInput(renderFrame(sender.frames[sample.info.phase],1024),{transform:[4,0,0,0,4,0,0,0,1]}).input;
    assert.deepEqual(await readFile(join(out,sample.input)),Buffer.from(floatBytes(input)));
  }
  const again=join(root,'again');await targets({dataset:joined,out:again});
  const repeated=JSON.parse(await readFile(join(again,'manifest.json'),'utf8'));
  assert.deepEqual(clean.samples.map(s=>[s.id,s.hashes]),repeated.samples.map(s=>[s.id,s.hashes]));
}));

test('target reconstruction preserves G128/4 phases and refuses split leakage or inconsistent canonical truth',async()=>temp(async root=>{
  const source=join(root,'source'),out=join(root,'targets');
  await generate({out:source,count:'1',size:'128',colors:'4',phases:'2',seed:'456',split:'train'});
  const path=join(source,'manifest.json'),manifest=JSON.parse(await readFile(path,'utf8'));
  for(const sample of manifest.samples){sample.source='real';sample.senderSessionId='sender-target-large';}
  await writeFile(path,JSON.stringify(manifest));
  await targets({dataset:source,out});
  const clean=JSON.parse(await readFile(join(out,'manifest.json'),'utf8'));
  assert.equal(clean.samples.length,2);
  for(const sample of clean.samples) {
    const original=manifest.samples.find(x=>x.info.phase===sample.info.phase);
    assert.equal(sample.size,128);assert.equal(sample.colors,4);assert.equal(sample.split,'train');
    assert.equal(sample.hashes.input,original.hashes.input);assert.deepEqual(sample.info,original.info);
  }
  const wrongSplit=copy(manifest),conflict=join(root,'conflict');await mkdir(conflict);
  for(const sample of wrongSplit.samples){sample.id+='-other';sample.split='validation';}
  await writeFile(join(conflict,'manifest.json'),JSON.stringify(wrongSplit));
  await assert.rejects(targets({dataset:`${source},${conflict}`,out:join(root,'bad-split')}),/crosses dataset splits/);
  const wrongHeader=copy(manifest);wrongHeader.samples[0].info.chunkCRC^=1;
  await writeFile(path,JSON.stringify(wrongHeader));
  await assert.rejects(targets({dataset:source,out:join(root,'bad-header')}),/header/);
  await writeFile(path,JSON.stringify(manifest));
  const sample=manifest.samples[0],labelPath=join(source,sample.labels),labels=await readFile(labelPath);labels[0]=(labels[0]+1)%4;
  await writeFile(labelPath,labels);sample.hashes.labels=createHash('sha256').update(labels).digest('hex');
  await writeFile(path,JSON.stringify(manifest));
  await assert.rejects(targets({dataset:source,out:join(root,'bad-labels')}),/labels or mask/);
}));


for(const frames of [4,5])test(`burst evaluation fuses exactly ${frames} observations and rejects identity mixing or reuse`,async()=>temp(async root=>{
  const directory=join(root,'burst');
  await generate({out:directory,count:'1',size:'64',colors:'4',phases:'1',seed:'765',split:'validation'});
  const path=join(directory,'manifest.json'),manifest=JSON.parse(await readFile(path,'utf8')),original=manifest.samples[0];
  const labels=new Uint8Array(await readFile(join(directory,original.labels))),logits=new Float32Array(16*64*64).fill(-20);
  labels.forEach((color,cell)=>{logits[color*64*64+cell]=20;});
  const correct=logitsToLogicalScores(logits,original.info),wrong=correct.slice();
  for(let cell=0;cell<wrong.length;cell+=4)for(let color=0;color<4;color++)wrong[cell+color]=correct[cell+(color+1)%4];
  assert.equal(decodeScores(original.info,wrong,{}, {recovery:'erasures'}).ok,false);
  manifest.samples=Array.from({length:frames},(_,index)=>({...copy(original),id:`burst-${index}`,source:'real',
    sessionId:'real-fixture-session',senderSessionId:'sender-fixture-session',burstId:'one-burst',timestamp:index*100,
    baseline:`baseline-${index}.f32`}));
  for(const [index,row] of manifest.samples.entries()) {
    const bytes=floatBytes(index===0?wrong:correct);await writeFile(join(directory,row.baseline),bytes);
    row.hashes.baseline=createHash('sha256').update(bytes).digest('hex');
  }
  await writeFile(path,JSON.stringify(manifest));
  await writeFile(join(root,'prediction.f32'),floatBytes(logits));
  const predictionPath=join(root,'prediction.json'),prediction={framesPerInput:frames,outputLabelSpace:'physical',samples:[{
    id:'burst-0',memberIds:manifest.samples.map(row=>row.id),logits:'prediction.f32'}]};
  await writeFile(predictionPath,JSON.stringify(prediction));
  const result=await evaluate({dataset:directory,predictions:predictionPath});
  assert.equal(result.summary.samples,1);assert.equal(result.summary.inputCaptures,frames);
  assert.equal(result.summary.classical.exact,1,'agreeing captures overcome the wrong first capture');
  assert.equal(result.summary.learned.exact,1);
  for(const invalid of [0,65,2.5,'5']) {
    await writeFile(predictionPath,JSON.stringify({...prediction,framesPerInput:invalid}));
    await assert.rejects(evaluate({dataset:directory,predictions:predictionPath}),/Unsupported dataset\/prediction/);
  }
  await writeFile(predictionPath,JSON.stringify(prediction));
  for(const mutation of [row=>row.source='synthetic',row=>row.sessionId='other-session',
    row=>row.info.phase=1,row=>row.burstId='independent-synthetic-burst',row=>row.split='test']) {
    const mixed=copy(manifest);mutation(mixed.samples[3]);await writeFile(path,JSON.stringify(mixed));
    await assert.rejects(evaluate({dataset:directory,predictions:predictionPath}),/Burst members/);
  }
  await writeFile(path,JSON.stringify(manifest));
  const duplicated=copy(prediction);duplicated.samples[0].memberIds[3]='burst-0';
  await writeFile(predictionPath,JSON.stringify(duplicated));
  await assert.rejects(evaluate({dataset:directory,predictions:predictionPath}),/distinct member/);
  const reused=copy(prediction);reused.samples.push({...reused.samples[0],id:'burst-1',memberIds:['burst-1','burst-0',...prediction.samples[0].memberIds.slice(2)]});
  await writeFile(predictionPath,JSON.stringify(reused));
  await assert.rejects(evaluate({dataset:directory,predictions:predictionPath}),/reused across candidates/);
  await writeFile(predictionPath,JSON.stringify(prediction));
  // Ground truth affects the after-decode comparison only. Incorrect expected
  // bytes cannot be substituted into a successfully decoded observed payload.
  const expected=await readFile(join(directory,original.expected));expected[0]^=1;
  await writeFile(join(directory,original.expected),expected);
  for(const row of manifest.samples)row.hashes.expected=createHash('sha256').update(expected).digest('hex');
  await writeFile(path,JSON.stringify(manifest));
  const wrongTruth=await evaluate({dataset:directory,predictions:predictionPath});
  for(const method of ['classical','learned']) {
    assert.equal(wrongTruth.summary[method].accepted,1);
    assert.equal(wrongTruth.summary[method].exact,0);
    assert.equal(wrongTruth.summary[method].wrongAccepted,1);
  }
}));

test('four-phase evaluation canonicalizes each observation and never phase-inverts logical predictions twice',async()=>temp(async root=>{
  const directory=join(root,'phases');
  await generate({out:directory,count:'1',size:'64',colors:'16',phases:'4',seed:'441',split:'validation'});
  const path=join(directory,'manifest.json'),manifest=JSON.parse(await readFile(path,'utf8')),originals=manifest.samples;
  const labels=new Uint8Array(await readFile(join(directory,originals[0].labels))),logits=new Float32Array(16*64*64).fill(-20);
  labels.forEach((color,cell)=>{logits[color*64*64+cell]=20;});
  manifest.samples=originals.flatMap(original=>Array.from({length:2},(_,index)=>({...copy(original),
    id:`phase-${original.info.phase}-${index}`,burstId:`phase-burst-${original.info.phase}`,
    synthetic:{phaseGroupId:'shared-corruption'},timestamp:index*100})));
  await writeFile(path,JSON.stringify(manifest));await writeFile(join(root,'prediction.f32'),floatBytes(logits));
  const predictionPath=join(root,'prediction.json');
  // A nonzero anchor phase makes an accidental second palette reversal visible.
  const ids=['phase-1-0','phase-0-0','phase-2-0','phase-3-0','phase-1-1','phase-0-1','phase-2-1','phase-3-1'];
  const prediction=count=>({framesPerInput:count,outputLabelSpace:'logical',samples:[{
    id:ids[0],memberIds:ids.slice(0,count),logits:'prediction.f32'}]});
  for(const count of [4,5,6,7,8]) {
    await writeFile(predictionPath,JSON.stringify(prediction(count)));
    const result=await evaluate({dataset:directory,predictions:predictionPath});
    assert.equal(result.summary.inputCaptures,count);
    assert.equal(result.summary.learned.exact,1,'canonical logits must already be in logical label space');
    assert.equal(result.summary.classical.exact,1,'each synthetic RGB observation uses its own phase mapping');
  }
  await writeFile(predictionPath,JSON.stringify(prediction(4)));
  for(const mutate of [row=>row.synthetic.phaseGroupId='unrelated',row=>delete row.synthetic.phaseGroupId,
    row=>row.sessionId='different-session',row=>row.info.index=1,row=>row.info.phases=8]) {
    const bad=copy(manifest);mutate(bad.samples.find(row=>row.id==='phase-2-0'));
    await writeFile(path,JSON.stringify(bad));
    await assert.rejects(evaluate({dataset:directory,predictions:predictionPath}),/Phase-grouped|Synthetic phase groups/);
  }
  await writeFile(path,JSON.stringify(manifest));
  const missing=prediction(4);missing.samples[0].memberIds[3]='phase-1-1';
  await writeFile(predictionPath,JSON.stringify(missing));
  await assert.rejects(evaluate({dataset:directory,predictions:predictionPath}),/every phase/);
  const noSpace=prediction(4);delete noSpace.outputLabelSpace;
  await writeFile(predictionPath,JSON.stringify(noSpace));
  await assert.rejects(evaluate({dataset:directory,predictions:predictionPath}),/Unsupported dataset\/prediction/);
}));
