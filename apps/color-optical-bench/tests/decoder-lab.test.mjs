import test from 'node:test';
import assert from 'node:assert/strict';
import {validateManifest, validateOptions, compareLogits, accumulateParity, summarizeTimings, decodeFloat32, verifyArtifact, MAX_INPUT_FRAMES, MAX_INPUT_BYTES} from '../dist/js/decoder-lab/runtime.js';
import {CameraBurst} from '../dist/js/decoder-lab/burst.js';
import {DecoderClient} from '../dist/js/decoder-lab/client.js';

function manifest(frames=1,profile='normalized-tube') {
  const entry=(file,bytes)=>({file,bytes,sha256:'a'.repeat(64)});
  const [base,perFrame]={'normalized-tube':[140112,864],longtube:[275472,864]}[profile];
  const count=base+perFrame*frames;
  return {format:'optical-decoder',version:1,status:'untrained',modelProfile:profile,framesPerInput:frames,parameterCount:count,pixelsPerCell:4,channels:3*frames,classes:16,
    normalization:{divisor:127.5,offset:-1,clip:[-1.5,1.5]},networkInputClip:[-1,1],
    inputName:'pixels',outputName:'logits',seed:7,opset:17,versions:{torch:'test'},cases:[64,128].map(size=>({size,
      inputShape:[1,3*frames,4*size,4*size],outputShape:[1,16,size,size],model:entry(`model-${size}.onnx`,500000),
      input:entry(`input-${size}.f32`,3*frames*16*size*size*4),expected:entry(`expected-${size}.f32`,16*size*size*4)}))};
}
test('decoder manifest accepts only bounded current shape/artifact contracts',()=>{
  const valid=manifest(); assert.equal(validateManifest(valid),valid);
  for (const mutate of [m=>m.status='trained',m=>m.cases[0].size=512,m=>m.cases[1].size=64,m=>m.channels=15,
    m=>m.cases[0].inputShape[2]=512,m=>m.cases[0].model.bytes=8*1024*1024,m=>m.cases[0].expected.bytes--,
    m=>m.cases[0].model.file='../model.onnx',m=>m.cases[0].model.file='https://example.com/a',
    m=>m.cases[0].model.sha256='invalid',m=>m.cases[1].model.file=m.cases[0].model.file]) {
    const invalid=manifest(); mutate(invalid); assert.throws(()=>validateManifest(invalid));
  }
});
test('manifest distinguishes retained capture range from the mandatory in-graph input clip',()=>{
  for (const profile of ['normalized-tube','longtube']) {
    const valid=manifest(5,profile); assert.equal(validateManifest(valid),valid);
    for (const mutate of [m=>delete m.networkInputClip,m=>m.networkInputClip=[-1.5,1.5],
      m=>m.networkInputClip=[-1,1,0],m=>m.networkInputClip=['-1','1'],
      m=>delete m.normalization,m=>m.normalization.divisor=255,m=>m.normalization.offset=0,
      m=>m.normalization.clip=[-1,1],m=>m.normalization.clip=null]) {
      const invalid=manifest(5,profile);mutate(invalid);assert.throws(()=>validateManifest(invalid),/normalization\/clip/);
    }
  }
});
test('trained exports require checkpoint provenance and the current format',()=>{
  const trained={...manifest(),status:'trained',checkpointSha256:'b'.repeat(64),training:{steps:20}};
  assert.equal(validateManifest(trained),trained);
  assert.throws(()=>validateManifest({...trained,checkpointSha256:'bad'}),/provenance/);
  assert.throws(()=>validateManifest({...trained,training:null}),/provenance/);
  assert.throws(()=>validateManifest({...trained,format:'optical-decoder-m1'}),/manifest/);
});
test('bounded benchmark choices reject oversized workloads',()=>{
  assert.deepEqual(validateOptions(),{size:64,backend:'auto',iterations:10});
  for(const options of [{size:32},{backend:'webgl'},{iterations:31},{iterations:0},{iterations:NaN},{iterations:2.5}]) assert.throws(()=>validateOptions(options));
});
test('parity checks every logit including a late mismatch and non-finite values',()=>{
  const expected=new Float32Array(1000).fill(.25),actual=expected.slice();
  assert.equal(compareLogits(actual,expected).passed,true);
  actual[999]+=.001; const result=compareLogits(actual,expected);
  assert.equal(result.passed,false);assert.equal(result.failures,1);assert.equal(result.checked,1000);
  actual[3]=NaN; assert.equal(compareLogits(actual,expected).nonFinite,1);
  assert.throws(()=>compareLogits(actual,new Float32Array(3)));
});
test('float fixture format is explicitly little-endian and finite',()=>{
  const buffer=new ArrayBuffer(12),view=new DataView(buffer);
  [1.25,-2,.125].forEach((value,index)=>view.setFloat32(index*4,value,true));
  assert.deepEqual([...decodeFloat32(buffer)],[1.25,-2,.125]);
  view.setFloat32(8,Infinity,true);assert.throws(()=>decodeFloat32(buffer));
  assert.throws(()=>decodeFloat32(new ArrayBuffer(3)));
});
test('a later passing run cannot hide an earlier relative-tolerance failure',()=>{
  const expected=[1000,.01],failure=compareLogits([1000,.011],expected),passing=compareLogits([1000.5,.01],expected);
  assert.equal(failure.passed,false); assert.equal(passing.passed,true);
  const merged=accumulateParity(accumulateParity(null,failure),passing);
  assert.equal(merged.passed,false); assert.equal(merged.failedRuns,1); assert.equal(merged.runsChecked,2);
  assert.equal(merged.maxAbsoluteError,.5);
});
test('artifact length and SHA prevent mixing model and numerical fixtures',async()=>{
  const bytes=new TextEncoder().encode('fixture'),buffer=bytes.buffer;
  const digest=await crypto.subtle.digest('SHA-256',buffer);
  const sha256=Buffer.from(digest).toString('hex'),entry={file:'input.f32',bytes:bytes.length,sha256};
  assert.equal(await verifyArtifact(buffer,entry),buffer);
  await assert.rejects(()=>verifyArtifact(buffer,{...entry,bytes:99}),/byte length/);
  await assert.rejects(()=>verifyArtifact(buffer,{...entry,sha256:'a'.repeat(64)}),/SHA-256/);
});
test('timings report nearest-rank percentiles without mutating samples',()=>{
  const values=[10,2,3,4,5,6,7,8,9,1],copy=[...values];
  const result=summarizeTimings(values); assert.equal(result.p50Ms,5);assert.equal(result.p95Ms,10);
  assert.deepEqual(values,copy);assert.throws(()=>summarizeTimings([NaN]));
});
class FakeWorker {
  constructor(){this.messages=[];this.terminated=false;}
  postMessage(value){this.messages.push(value);}
  terminate(){this.terminated=true;}
  receive(value){this.onmessage?.({data:value});}
}
test('client rejects concurrent work and resolves one bounded request',async()=>{
  const worker=new FakeWorker(),progress=[]; const client=new DecoderClient({workerFactory:()=>worker,onProgress:p=>progress.push(p)});
  const promise=client.call('benchmark',{size:64});await assert.rejects(client.call('capture'),/already running/);
  const id=worker.messages[0].id;worker.receive({id,type:'progress',message:'working'});
  worker.receive({id,type:'result',result:{passed:true}});
  assert.deepEqual(await promise,{passed:true});assert.deepEqual(progress,['working']);client.cancel();
});
test('cancel terminates inference and late messages cannot resolve a new request',async()=>{
  const workers=[];const client=new DecoderClient({workerFactory:()=>{const w=new FakeWorker();workers.push(w);return w;}});
  const first=client.call('benchmark');const rejected=assert.rejects(first,{name:'AbortError'});client.cancel();await rejected;
  assert.equal(workers[0].terminated,true);
  const second=client.call('benchmark');const id=workers[1].messages[0].id;
  workers[0].receive({id,type:'result',result:'stale'});
  workers[1].receive({id,type:'result',result:'current'});assert.equal(await second,'current');client.cancel();
});

test('active model profiles validate exact parameter counts without changing tensor shapes',()=>{
  const tube=manifest(),long=manifest(1,'longtube');validateManifest(tube);validateManifest(long);
  for (const obsolete of ['compact','wide','normalized-compact',undefined]) {
    assert.throws(()=>validateManifest({...tube,modelProfile:obsolete}),/profile/);
  }
  assert.throws(()=>validateManifest({...long,modelProfile:'normalized-tube'}),/profile/);
  assert.throws(()=>validateManifest({...long,parameterCount:276337}),/profile/);
});


test('arbitrary frame stacks require exact channels/counts through the G128/64-frame bound',()=>{
  assert.equal(MAX_INPUT_FRAMES,64);assert.equal(MAX_INPUT_BYTES,192*1024*1024);
  for (const profile of ['normalized-tube','longtube']) for (const frames of [1,4,5,32,64]) {
    const value=manifest(frames,profile);
    assert.equal(validateManifest(value),value);
    assert.equal(value.cases[1].input.bytes,3*frames*512*512*4);
    for (const mutate of [m=>delete m.framesPerInput,m=>m.framesPerInput=0,m=>m.framesPerInput=65,
      m=>m.framesPerInput=2.5,m=>m.framesPerInput='5',m=>m.channels++,
      m=>m.cases[0].inputShape[1]++,m=>m.cases[1].input.bytes++,m=>m.parameterCount++]) {
      const bad=manifest(frames,profile);mutate(bad);assert.throws(()=>validateManifest(bad));
    }
  }
  assert.equal(manifest(5,'normalized-tube').parameterCount,144432);
  assert.throws(()=>validateManifest({...manifest(5,'normalized-tube'),parameterCount:125808}),/profile/);
  assert.equal(manifest(64,'normalized-tube').parameterCount,195408);
  assert.equal(manifest(1,'longtube').parameterCount,276336);
  assert.equal(manifest(5,'longtube').parameterCount,279792);
  assert.equal(manifest(64,'longtube').parameterCount,330768);
  assert.equal(manifest(64).cases[1].input.bytes,MAX_INPUT_BYTES);
});
test('camera assembly retains calibrated overshoot until the model graph clips it',()=>{
  const crop=cameraCrop(1.25);crop.input[0]=-1.5;
  const result=new CameraBurst(1).add(crop,{context:'normalized-model/run',mediaTime:1,presentedFrames:1});
  assert.equal(result.ready,true);
  assert.equal(result.input[0],-1.5);assert.equal(result.input[1],1.25);
  assert.equal(crop.input[0],-1.5);assert.equal(crop.input[1],1.25);
});
function cameraCrop(value=0,overrides={}) {
  return {input:new Float32Array(3*256*256).fill(value),info:{size:64,colors:16,
    id:'0000000000000001',phase:0,phases:4,index:0,count:1,...overrides},stats:{preprocessMs:2}};
}
test('camera burst stacks five distinct exposures in RGB channel order without a duplicate stack',()=>{
  const burst=new CameraBurst(5);let result;
  for(let i=0;i<5;i++) {
    result=burst.add(cameraCrop(i*.25),{context:'model/run1',mediaTime:i/30,presentedFrames:i+1});
    assert.equal(result.ready,i===4);assert.equal(result.collected,i+1);
    if(i<4)assert.equal(result.input,null);
    assert.equal(burst.rows[i].input,undefined);
  }
  assert.equal(result.input,burst.input);assert.equal(result.input.length,15*256*256);assert.equal(result.preprocessMs,10);
  for(let i=0;i<5;i++)assert.ok(result.input.subarray(i*3*256*256,(i+1)*3*256*256).every(x=>x===i*.25));
  assert.deepEqual(result.mediaTimes,[0,1/30,2/30,3/30,4/30]);
  assert.throws(()=>burst.add(cameraCrop(),{context:'model/run1',mediaTime:1,presentedFrames:10}),/complete/);
});
test('camera burst discards mixed headers/phases and resets on run/model/camera changes',()=>{
  const burst=new CameraBurst(4);
  burst.add(cameraCrop(),{context:'model/run1',mediaTime:1,presentedFrames:10});
  assert.equal(burst.add(cameraCrop(0,{phase:1}),{context:'model/run1',mediaTime:2,presentedFrames:20}).collected,1);
  assert.equal(burst.add(cameraCrop(0,{phase:1,id:'0000000000000002'}),{context:'model/run1',mediaTime:3,presentedFrames:30}).collected,1);
  assert.equal(burst.add(cameraCrop(),{context:'other-model/run2',mediaTime:0,presentedFrames:1}).collected,1);
  burst.reset();assert.equal(burst.rows.length,0);
  assert.equal(burst.add(cameraCrop(),{context:'other-model/run2',mediaTime:0,presentedFrames:1}).collected,1);
});
test('camera burst rejects duplicate timestamps and malformed inputs without fabricating observations',()=>{
  const burst=new CameraBurst(4),crop=cameraCrop();
  burst.add(crop,{context:'run',mediaTime:1,presentedFrames:10});
  for(const mediaTime of [1,.5])assert.throws(()=>burst.add(crop,{context:'run',mediaTime,presentedFrames:11}),/distinct/);
  assert.throws(()=>burst.add(crop,{context:'run',mediaTime:2,presentedFrames:10}),/distinct/);
  assert.equal(burst.rows.length,1);
  const bad=cameraCrop();bad.input[2]=NaN;
  assert.throws(()=>burst.add(bad,{context:'run',mediaTime:2,presentedFrames:20}),/Invalid/);
  assert.equal(burst.rows.length,0);
  for(const frames of [0,-1,65,2.5,NaN,Infinity,'5',undefined])assert.throws(()=>new CameraBurst(frames));
  const largest=new CameraBurst(64);assert.equal(largest.frames,64);assert.equal(largest.input,null);
  assert.throws(()=>burst.add({...crop,input:new Float32Array(12)},{context:'run',mediaTime:3,presentedFrames:30}),/Invalid/);
  const single=new CameraBurst(1).add(crop,{context:'single',mediaTime:1,presentedFrames:10});
  assert.equal(single.ready,true);assert.equal(single.input.length,crop.input.length);
});
