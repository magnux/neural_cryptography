import test from 'node:test';
import assert from 'node:assert/strict';
import {validateManifest, validateOptions, compareLogits, accumulateParity, summarizeTimings, decodeFloat32, verifyArtifact, MAX_INPUT_FRAMES, MAX_INPUT_BYTES, inputPresence, packFrameStack, clipNetworkInput} from '../dist/js/corrector-lab/runtime.js';
import {CameraBurst} from '../dist/js/corrector-lab/burst.js';
import {CorrectorClient} from '../dist/js/corrector-lab/client.js';

function manifest(frames=1,profile='U-tube') {
  const entry=(file,bytes)=>({file,bytes,sha256:'a'.repeat(64)});
  const neuralQR=['W-tube','VW'].includes(profile),attention=neuralQR||profile==='U-tube-2x-attn';
  const [base,perFrame]={'U-tube':[140112,864],'U-tube-2x':[275472,864],'U-tube-2x-attn':[432848,0],'W-tube':[487760,0],'VW':[696400,0]}[profile];
  const channels=attention?32:3*frames;
  const count=base+perFrame*frames;
  return {format:'neural-corrector',version:1,status:'untrained',modelProfile:profile,framesPerInput:frames,inputLayout:attention?'phase-paired-rgb-presence-planes':'stacked-rgb',
    minFramesPerInput:attention?4:frames,frameCounts:attention?[4,5,6,7,8]:[frames],outputLabelSpace:attention?'logical':'physical',phaseMasks:attention?[0,15,7,8]:[],parameterCount:count,pixelsPerCell:4,channels,classes:16,
    normalization:{divisor:127.5,offset:-1,clip:[-1.5,1.5]},networkInputClip:[-1,1],tolerance:{atol:1e-4,rtol:1e-3},
    ...(neuralQR?{datasetFormat:'neural-qr-dataset',datasetVersion:2,anchor:{representation:'dct-low-frequency',maxModes:16,normalization:'ortho',spatialScale:'grid-over-modes'},anchorDeployment:'baked-selected-spatial-grid',bottleneck:{stages:profile==='VW'?3:2,blocks:['conv','attention'],layersPerBlock:2,scales:[4,2,1,2,4],fusion:'normalized-residual'}}:{}),
    inputName:'pixels',outputName:'logits',seed:7,opset:17,versions:{torch:'test'},cases:(neuralQR?[16,32,64,128]:attention?[64]:[64,128]).map(size=>({size,framesAvailable:attention?4:frames,
      inputShape:[1,channels,4*size,4*size],outputShape:[1,16,size,size],model:entry(`model-${size}.onnx`,500000),
      input:entry(`input-${size}.f32`,channels*16*size*size*4),expected:entry(`expected-${size}.f32`,16*size*size*4)}))};
}
test('corrector manifest accepts only bounded current shape/artifact contracts',()=>{
  const valid=manifest(); assert.equal(validateManifest(valid),valid);
  for (const mutate of [m=>m.status='trained',m=>m.cases[0].size=512,m=>m.cases[1].size=64,m=>m.channels=15,
    m=>m.cases[0].inputShape[2]=512,m=>m.cases[0].model.bytes=8*1024*1024,m=>m.cases[0].expected.bytes--,
    m=>m.cases[0].model.file='../model.onnx',m=>m.cases[0].model.file='https://example.com/a',
    m=>m.cases[0].model.sha256='invalid',m=>m.cases[1].model.file=m.cases[0].model.file]) {
    const invalid=manifest(); mutate(invalid); assert.throws(()=>validateManifest(invalid));
  }
});
test('manifest requires the same current parity policy used by browser inference',()=>{
  const valid=manifest();assert.equal(validateManifest(valid),valid);
  for(const tolerance of [undefined,null,{},[],{atol:1e-4},{rtol:1e-3},
    {atol:Infinity,rtol:1e-3},{atol:1e-4,rtol:NaN},
    {atol:1e-3,rtol:1e-3},{atol:1e-4,rtol:1e-2},
    {atol:2e-5,rtol:2e-4},{atol:'0.0001',rtol:1e-3},
    {atol:1e-4,rtol:1e-3,ignoreFailures:true}]) {
    const invalid={...manifest(),tolerance};
    if(tolerance===undefined)delete invalid.tolerance;
    assert.throws(()=>validateManifest(invalid),/parity tolerance.*Regenerate/);
  }
});
test('manifest distinguishes retained capture range from the mandatory in-graph input clip',()=>{
  for (const profile of ['U-tube','U-tube-2x']) {
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
  assert.throws(()=>validateManifest({...trained,format:'neural-corrector-m1'}),/manifest/);
});
test('bounded benchmark choices reject oversized workloads',()=>{
  assert.deepEqual(validateOptions(),{size:64,backend:'auto',iterations:10,captureFrames:0});
  for(const options of [{size:8},{backend:'webgl'},{iterations:31},{iterations:0},{iterations:NaN},{iterations:2.5},{captureFrames:3},{captureFrames:9},{captureFrames:5.5}]) assert.throws(()=>validateOptions(options));
});
test('parity checks every logit including a late mismatch and non-finite values',()=>{
  const expected=new Float32Array(1000).fill(.25),actual=expected.slice();
  assert.equal(compareLogits(actual,expected).passed,true);
  actual[999]+=.001; const result=compareLogits(actual,expected);
  assert.equal(result.passed,false);assert.equal(result.failures,1);assert.equal(result.checked,1000);
  actual[3]=NaN; assert.equal(compareLogits(actual,expected).nonFinite,1);
  assert.throws(()=>compareLogits(actual,new Float32Array(3)));
});
test('reported-scale near-zero float32 drift passes while a single larger outlier fails',()=>{
  const expected=new Float32Array(65536),actual=expected.slice();
  actual[65535]=5.91278076171875e-5;
  const small=compareLogits(actual,expected);
  assert.equal(small.passed,true);assert.equal(small.failures,0);
  assert.equal(small.checked,65536);assert.equal(small.atol,1e-4);assert.equal(small.rtol,1e-3);
  actual[65535]=1.1e-4;
  const outlier=compareLogits(actual,expected);
  assert.equal(outlier.passed,false);assert.equal(outlier.failures,1);
  assert.equal(outlier.checked,65536);
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
  const worker=new FakeWorker(),progress=[]; const client=new CorrectorClient({workerFactory:()=>worker,onProgress:p=>progress.push(p)});
  const promise=client.call('benchmark',{size:64});await assert.rejects(client.call('capture'),/already running/);
  const id=worker.messages[0].id;worker.receive({id,type:'progress',message:'working'});
  worker.receive({id,type:'result',result:{passed:true}});
  assert.deepEqual(await promise,{passed:true});assert.deepEqual(progress,['working']);client.cancel();
});
test('cancel terminates inference and late messages cannot resolve a new request',async()=>{
  const workers=[];const client=new CorrectorClient({workerFactory:()=>{const w=new FakeWorker();workers.push(w);return w;}});
  const first=client.call('benchmark');const rejected=assert.rejects(first,{name:'AbortError'});client.cancel();await rejected;
  assert.equal(workers[0].terminated,true);
  const second=client.call('benchmark');const id=workers[1].messages[0].id;
  workers[0].receive({id,type:'result',result:'stale'});
  workers[1].receive({id,type:'result',result:'current'});assert.equal(await second,'current');client.cancel();
});

test('active model profiles validate exact parameter counts without changing tensor shapes',()=>{
  const tube=manifest(),long=manifest(1,'U-tube-2x');validateManifest(tube);validateManifest(long);
  for (const obsolete of ['compact','wide','normalized-compact',undefined]) {
    assert.throws(()=>validateManifest({...tube,modelProfile:obsolete}),/profile/);
  }
  assert.throws(()=>validateManifest({...long,modelProfile:'U-tube'}),/profile/);
  assert.throws(()=>validateManifest({...long,parameterCount:276337}),/profile/);
});


test('arbitrary frame stacks require exact channels/counts through the G128/64-frame bound',()=>{
  assert.equal(MAX_INPUT_FRAMES,64);assert.equal(MAX_INPUT_BYTES,192*1024*1024);
  for (const profile of ['U-tube','U-tube-2x']) for (const frames of [1,4,5,32,64]) {
    const value=manifest(frames,profile);
    assert.equal(validateManifest(value),value);
    assert.equal(value.cases[1].input.bytes,3*frames*512*512*4);
    for (const mutate of [m=>delete m.framesPerInput,m=>m.framesPerInput=0,m=>m.framesPerInput=65,
      m=>m.framesPerInput=2.5,m=>m.framesPerInput='5',m=>m.channels++,
      m=>m.cases[0].inputShape[1]++,m=>m.cases[1].input.bytes++,m=>m.parameterCount++]) {
      const bad=manifest(frames,profile);mutate(bad);assert.throws(()=>validateManifest(bad));
    }
  }
  assert.equal(manifest(5,'U-tube').parameterCount,144432);
  assert.throws(()=>validateManifest({...manifest(5,'U-tube'),parameterCount:125808}),/profile/);
  assert.equal(manifest(64,'U-tube').parameterCount,195408);
  assert.equal(manifest(1,'U-tube-2x').parameterCount,276336);
  assert.equal(manifest(5,'U-tube-2x').parameterCount,279792);
  assert.equal(manifest(64,'U-tube-2x').parameterCount,330768);
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


test('attention manifest describes fixed eight slots, four-to-eight phase-grouped observations and G64 only',()=>{
  const value=manifest(8,'U-tube-2x-attn');assert.equal(validateManifest(value),value);
  assert.equal(value.parameterCount,432848);assert.equal(value.channels,32);
  for(const mutate of [m=>m.framesPerInput=5,m=>m.channels=24,m=>m.inputLayout='stacked-rgb',
    m=>m.minFramesPerInput=1,m=>m.frameCounts=[5,6,8],m=>m.cases[0].size=128,
    m=>m.cases[0].framesAvailable=3,m=>m.outputLabelSpace='physical',m=>m.phaseMasks=[0,1,2,3],m=>m.parameterCount++,m=>delete m.inputLayout]) {
    const wrong=manifest(8,'U-tube-2x-attn');mutate(wrong);assert.throws(()=>validateManifest(wrong));
  }
  for(const mutate of [m=>delete m.frameCounts,m=>m.minFramesPerInput=4,m=>m.inputLayout='rgb-presence-planes']) {
    const wrong=manifest(5);mutate(wrong);assert.throws(()=>validateManifest(wrong));
  }
});

test('camera packing preserves phase pairs and distinguishes zero captures from padding',()=>{
  const model=manifest(8,'U-tube-2x-attn'),plane=256*256,one=3*plane,order=[6,1,4,2,7,0,5,3];
  for(const count of [4,5,6,7,8]) {
    const stack=new Float32Array(count*one);
    for(let i=0;i<count;i++)stack.fill(i*.2,i*one,(i+1)*one);
    const before=stack.slice(),slots=order.slice(0,count),phases=slots.map(slot=>Math.floor(slot/2)),packed=packFrameStack(stack,model,64,{slots,phases});
    assert.deepEqual(stack,before);assert.equal(packed.length,32*plane);
    assert.deepEqual(inputPresence(packed,model,64),slots.slice().sort((a,b)=>a-b));
    for(let slot=0;slot<8;slot++) {
      const capture=slots.indexOf(slot),values=packed.subarray(slot*one,(slot+1)*one);
      assert.ok(values.every(x=>x===(capture<0?0:Math.fround(capture*.2))));
      assert.ok(packed.subarray((24+slot)*plane,(25+slot)*plane).every(x=>x===Number(capture>=0)));
    }
  }
  assert.throws(()=>packFrameStack(new Float32Array(3*one),model,64),/capture count/);
  assert.throws(()=>packFrameStack(new Float32Array(4*one),model,64),/phase ID/);
  assert.throws(()=>packFrameStack(new Float32Array(4*one),model,64,{phases:[0,1,2,2]}),/every phase/);
  assert.throws(()=>packFrameStack(new Float32Array(5*one),model,64,{phases:[0,1,2,3,3],slots:[0,1,2,3,3]}),/slots/);
  assert.throws(()=>packFrameStack(new Float32Array(6*one),model,64,{phases:[0,0,0,1,2,3]}),/every phase/);
  const fixed=manifest(5),rgb=new Float32Array(5*one);
  assert.equal(packFrameStack(rgb,fixed,64),rgb);
  assert.deepEqual(inputPresence(rgb,fixed,64),[0,1,2,3,4]);
});

test('presence validation rejects missing phases, malformed masks and clips RGB only',()=>{
  const model=manifest(8,'U-tube-2x-attn'),plane=256*256;
  const packed=packFrameStack(new Float32Array(5*3*plane).fill(1.25),model,64,{phases:[0,1,2,3,0]});
  const clipped=clipNetworkInput(packed,model,64);
  assert.equal(clipped[0],1);assert.equal(packed[0],1.25);
  assert.deepEqual(clipped.subarray(24*plane),packed.subarray(24*plane));
  for(const mutate of [p=>p[24*plane]=.5,p=>p[24*plane+1]=0,
    p=>p.fill(0,24*plane),p=>p[0]=NaN,
    p=>{p.fill(0,30*plane,31*plane);p.fill(1,27*plane,28*plane); }]) {
    const bad=packed.slice();mutate(bad);assert.throws(()=>inputPresence(bad,model,64));
  }
});

test('phase camera burst can infer from first four distinct phases and ignores duplicate-phase excess',()=>{
  const burst=new CameraBurst(4,{phaseGrouped:true});let result;
  const phaseOrder=[2,2,2,0,0,3,1];
  phaseOrder.forEach((phase,i)=>{
    result=burst.add(cameraCrop(phase*.25,{phase,permute:true}),{context:'group/run',mediaTime:i,presentedFrames:i+1});
    assert.equal(result.ready,i===phaseOrder.length-1);
    if(!result.ready)assert.equal(result.input,null);
  });
  assert.deepEqual(result.phases,[0,1,2,3]);assert.deepEqual(result.slots,[0,2,4,6]);
  assert.deepEqual(result.phaseCounts,[1,1,1,1]);assert.equal(result.input.length,4*3*256*256);
  for(let i=0;i<4;i++)assert.equal(result.input[i*3*256*256],i*.25);
  assert.throws(()=>burst.add(cameraCrop(0,{permute:true}),{context:'group/run',mediaTime:9,presentedFrames:10}),/complete/);
});

test('phase camera burst keeps at most two captures per phase and needs coverage at all counts',()=>{
  for(const count of [5,6,7,8]) {
    const burst=new CameraBurst(count,{phaseGrouped:true});let result,frame=0;
    for(const phase of [0,0,0,1,1,2,2,3,3]) {
      result=burst.add(cameraCrop(phase*.25,{phase,permute:true}),{context:'run',mediaTime:frame++,presentedFrames:frame});
      assert.ok(burst.rows.length<=8);
      if(result.ready)break;
    }
    assert.equal(result.ready,true);assert.equal(result.input.length,count*3*256*256);
    assert.equal(result.phases.length,count);assert.ok(result.phaseCounts.every(c=>c>=1&&c<=2));
    const packed=packFrameStack(result.input,manifest(8,'U-tube-2x-attn'),64,{phases:result.phases,slots:result.slots});
    assert.equal(inputPresence(packed,manifest(8,'U-tube-2x-attn'),64).length,count);
  }
});

test('phase camera burst resets logical frame changes and rejects invalid phase profiles',()=>{
  const burst=new CameraBurst(4,{phaseGrouped:true});
  const add=(phase,t,extra={})=>burst.add(cameraCrop(0,{phase,permute:true,...extra}),{context:'run',mediaTime:t,presentedFrames:t});
  add(0,1);add(1,2);assert.equal(add(2,3,{index:1,count:2}).collected,1);
  assert.equal(add(3,4,{index:1,count:2}).collected,2);
  assert.throws(()=>add(0,5,{permute:false}),/four-phase/);
  assert.equal(burst.rows.length,0);
  assert.throws(()=>new CameraBurst(3,{phaseGrouped:true}),/four to eight/);
});


test('phase capture survives undetected transition frames but expires stale evidence and resets contexts',()=>{
  const burst=new CameraBurst(4,{phaseGrouped:true});
  const identity=t=>({context:'run',mediaTime:t,presentedFrames:t+1});
  burst.add(cameraCrop(0,{phase:0,permute:true}),identity(0));
  assert.deepEqual(burst.miss(identity(1)).phaseCounts,[1,0,0,0]);
  burst.add(cameraCrop(0,{phase:1,permute:true}),identity(2));
  assert.deepEqual(burst.miss(identity(3)).phaseCounts,[1,1,0,0]);
  burst.add(cameraCrop(0,{phase:2,permute:true}),identity(4));
  const ready=burst.add(cameraCrop(0,{phase:3,permute:true}),identity(5));
  assert.equal(ready.ready,true);assert.deepEqual(ready.mediaTimes,[0,2,4,5]);
  assert.deepEqual(burst.miss(identity(61)).phaseCounts,[0,0,0,0]);
  burst.add(cameraCrop(0,{phase:0,permute:true}),identity(62));
  assert.deepEqual(burst.miss({...identity(0),context:'newrun'}).phaseCounts,[0,0,0,0]);
  assert.throws(()=>burst.miss({...identity(0),context:'newrun'}),/distinct/);
});


test('VW and W-tube manifests admit only current four-pixel static Neural-QR geometries',()=>{
  for(const profile of ['W-tube','VW']){
    const valid=manifest(8,profile);valid.cases.at(-1).model.bytes=17500980;
    assert.equal(validateManifest(valid),valid);
    for(const mutate of [m=>m.datasetVersion=1,m=>m.pixelsPerCell=2,m=>m.parameterCount++,
      m=>m.anchor.maxModes=32,m=>m.bottleneck.stages=1,m=>m.cases[0].size=256,
      m=>m.cases.at(-1).model.bytes=32*1024*1024+1,m=>m.cases=[]]){
      const bad=structuredClone(valid);mutate(bad);assert.throws(()=>validateManifest(bad));
    }
    for(const size of [16,32,64,128]){
      assert.equal(validateOptions({size}).size,size);
      const plane=(4*size)**2,stack=new Float32Array(4*3*plane).fill(.25);
      const packed=packFrameStack(stack,valid,size,{phases:[3,1,0,2]});
      assert.deepEqual(inputPresence(packed,valid,size),[0,2,4,6]);
      assert.equal(packed.length,32*plane);
      assert.ok(packed.subarray(3*plane,6*plane).every(v=>v===0));
    }
  }
});
