import test from 'node:test';
import assert from 'node:assert/strict';
import {validateManifest, validateOptions, compareLogits, accumulateParity, summarizeTimings, decodeFloat32, verifyArtifact} from '../dist/js/decoder-lab/runtime.js';
import {DecoderClient} from '../dist/js/decoder-lab/client.js';

function manifest() {
  const entry=(file,bytes)=>({file,bytes,sha256:'a'.repeat(64)});
  return {format:'optical-decoder-m1',version:1,status:'untrained',parameterCount:121792,pixelsPerCell:4,channels:3,classes:16,
    inputName:'pixels',outputName:'logits',seed:7,opset:17,versions:{torch:'test'},cases:[64,128].map(size=>({size,
      inputShape:[1,3,4*size,4*size],outputShape:[1,16,size,size],model:entry(`model-${size}.onnx`,500000),
      input:entry(`input-${size}.f32`,3*16*size*size*4),expected:entry(`expected-${size}.f32`,16*size*size*4)}))};
}
test('M1 manifest accepts only bounded current shape/artifact contracts',()=>{
  const valid=manifest(); assert.equal(validateManifest(valid),valid);
  for (const mutate of [m=>m.status='trained',m=>m.cases[0].size=512,m=>m.cases[1].size=64,m=>m.channels=15,
    m=>m.cases[0].inputShape[2]=512,m=>m.cases[0].model.bytes=8*1024*1024,m=>m.cases[0].expected.bytes--,
    m=>m.cases[0].model.file='../model.onnx',m=>m.cases[0].model.file='https://example.com/a',
    m=>m.cases[0].model.sha256='invalid',m=>m.cases[1].model.file=m.cases[0].model.file]) {
    const invalid=manifest(); mutate(invalid); assert.throws(()=>validateManifest(invalid));
  }
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
