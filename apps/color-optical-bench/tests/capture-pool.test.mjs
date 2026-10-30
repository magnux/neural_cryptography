import test from 'node:test';
import assert from 'node:assert/strict';
import {CaptureBuffer,captureFrame} from '../dist/js/dataset/capture.js';
import {captureSignature,meanCaptureDistance,selectCaptureReplacement} from '../dist/js/dataset/diversity.js';
import {createSender,captureKey} from '../dist/js/dataset/contract.js';
import {renderFrame} from '../dist/js/format.js';
import {ScanLoop} from '../dist/js/dataset/scan-loop.js';

const sender=await createSender({seed:55,sessionId:'pool-test'});
const sample=captureFrame(renderFrame(sender.frames[0],1024),{timestamp:1000,signature:true});
function fill(buffer,count,record=sample,offset=0){
  buffer.add({...record,timestamp:1000+offset},{alreadyValidated:true,maxPerPhase:32});
  for(let i=1;i<=count;i++)assert.equal(buffer.add({...record,timestamp:1000+offset+i*600},{alreadyValidated:true,maxPerPhase:32}).accepted,true);
}

test('32-sample pools continue via guarded replacements and isolate other phases',()=>{
  const b=new CaptureBuffer();fill(b,32);
  assert.equal(b.add({...sample,timestamp:21000},{alreadyValidated:true,maxPerPhase:32}).stage,'kept');
  const other={...sample,info:{...sample.info,phase:1}};other.key=captureKey(other.info);
  fill(b,3,other,30000);
  const otherRows=b.phaseRecords(other.key),oldRows=b.phaseRecords(sample.key),oldRecord=oldRows[4];
  // Phase change must settle again even when a selected replacement is available.
  assert.equal(b.add({...sample,timestamp:60000},{alreadyValidated:true,maxPerPhase:32,replaceIndex:4}).stage,'settling');
  const fresh={...sample,timestamp:60600};
  const replacement=b.add(fresh,{alreadyValidated:true,maxPerPhase:32,replaceIndex:4});
  assert.equal(replacement.stage,'replaced');assert.equal(replacement.accepted,true);
  assert.deepEqual(b.phaseRecords(other.key),otherRows);
  assert.equal(b.phaseRecords(sample.key).length,32);
  assert.equal(b.phaseSignatures(sample.key).length,32);
  assert.equal(b.phaseRecords(sample.key)[4].timestamp,60600);
  assert.ok(!b.records.includes(oldRecord));
  assert.equal(b.add(fresh,{alreadyValidated:true,maxPerPhase:32,replaceIndex:4}).stage,'duplicate');
  b.add({ok:false,timestamp:61000,error:'lost header'});
  assert.equal(b.add({...sample,timestamp:61200},{alreadyValidated:true,maxPerPhase:32,replaceIndex:4}).stage,'settling');
  const exported=b.export({sessionId:'receiver'});
  assert.equal(exported.records.length,35);
  assert.ok(exported.records.every(x=>!('signature' in x)));
  assert.deepEqual(Object.keys(exported),['format','version','created','sessionId','notes','camera','limits','transitionPolicy','selection','records','diagnostics']);
  assert.equal(exported.format,'optical-decoder-captures');assert.equal(exported.version,1);
  assert.deepEqual(exported.selection,{method:'sampled-distance-replacement',recentThinning:[]});
});

test('net byte accounting allows full-count replacement and declines oversized new evidence',()=>{
  const b=new CaptureBuffer({maxRecords:2,maxBytes:3*1024*1024});
  fill(b,2);
  const count=b.records.length,bytes=b.bytes,diagnostics=b.diagnostics.length;
  assert.equal(b.add({...sample,timestamp:3000},{alreadyValidated:true,maxPerPhase:2,replaceIndex:0}).stage,'replaced');
  assert.equal(b.records.length,count);assert.ok(Math.abs(b.bytes-bytes)<100);
  const current=b.phaseRecords(sample.key)[0];
  const huge={...sample,timestamp:3600,original:{mime:'image/png',base64:'A'.repeat(1024*1024)}};
  assert.equal(b.add(huge,{alreadyValidated:true,maxPerPhase:2,replaceIndex:0}).stage,'kept');
  assert.equal(b.phaseRecords(sample.key)[0],current);assert.equal(b.originals,0);
  assert.equal(b.diagnostics.length,diagnostics);assert.ok(b.bytes<b.maxBytes);
  assert.throws(()=>b.add({...sample,timestamp:4200},{alreadyValidated:true,maxPerPhase:2,replaceIndex:2}),/replacement index/);
});

test('replacement correctly counts original snapshots, retaining at most two',()=>{
  const b=new CaptureBuffer({maxRecords:2}),original={mime:'image/png',base64:'AQID'};
  fill(b,2,{...sample,original});assert.equal(b.originals,2);
  b.add({...sample,timestamp:3000,original},{alreadyValidated:true,maxPerPhase:2,replaceIndex:0});
  assert.equal(b.originals,2);assert.ok(b.phaseRecords(sample.key)[0].original);
  b.add({...sample,timestamp:3600},{alreadyValidated:true,maxPerPhase:2,replaceIndex:0});
  assert.equal(b.originals,1);assert.equal(b.phaseRecords(sample.key)[0].original,undefined);
});

test('randomized crowding replacement rejects duplicates and isolated extremes, accepts useful variation',()=>{
  const a=new Float32Array(3072),pool=Array.from({length:32},()=>a);
  const options={random:()=>0},candidate=new Float32Array(3072).fill(.04);
  assert.equal(selectCaptureReplacement(pool,a,options).replaceIndex,null);
  assert.equal(selectCaptureReplacement(pool,new Float32Array(3072).fill(.0005),options).replaceIndex,null);
  const chosen=selectCaptureReplacement(pool,candidate,options);
  assert.equal(chosen.replaceIndex,0);assert.equal(chosen.reason,'greater-local-diversity');
  assert.ok(Math.abs(chosen.candidateDistance-2)<1e-5);
  assert.equal(selectCaptureReplacement(pool,new Float32Array(3072).fill(1.5),options).reason,'isolated-candidate');
  const randomized=selectCaptureReplacement(pool,candidate,{random:()=>.6});
  assert.notEqual(randomized.replaceIndex,chosen.replaceIndex);
  assert.throws(()=>selectCaptureReplacement(pool,Float32Array.of(NaN),options),/signature/);
  assert.throws(()=>selectCaptureReplacement(pool,candidate,{random:()=>1}),/randomness/);
});

test('distance measures same-coordinate RGB differences, with explicit insufficient-sample state',()=>{
  const a=new Float32Array(3072),b=new Float32Array(3072).fill(.2);
  assert.equal(meanCaptureDistance([]).mean,null);
  assert.equal(meanCaptureDistance([a]).mean,null);
  assert.equal(meanCaptureDistance([a,a]).mean,0);
  assert.ok(Math.abs(meanCaptureDistance([a,b]).mean-10)<1e-5);
  assert.equal(meanCaptureDistance([a,b,b]).pairs,3);
  assert.equal(meanCaptureDistance([a,b]).mean,meanCaptureDistance([b,a]).mean);
  assert.throws(()=>meanCaptureDistance([Float32Array.of(NaN)]),/signature/);
  for(const size of [64,128]){
    const tensor=new Float32Array(3*(4*size)**2).fill(.2);
    assert.deepEqual(captureSignature(tensor,size),b);
    assert.equal(captureSignature(tensor,size).length,3072);
  }
});

const flush=()=>new Promise(resolve=>setImmediate(resolve));
test('automatic scanner starts immediately, serializes work and drops late results after stop',async()=>{
  let resolve,started=0,canceled=0;const received=[],scheduled=[];
  const loop=new ScanLoop({sample:()=>{started++;return new Promise(r=>resolve=r);},receive:r=>received.push(r),
    cancel:()=>canceled++,schedule:fn=>{scheduled.push(fn);return fn;},unschedule:fn=>{const i=scheduled.indexOf(fn);if(i>=0)scheduled.splice(i,1);}});
  loop.start();loop.start();assert.equal(started,1);assert.equal(scheduled.length,0);
  resolve('first');await flush();assert.deepEqual(received,['first']);assert.equal(scheduled.length,1);
  scheduled.shift()();assert.equal(started,2);
  loop.stop();resolve('late');await flush();assert.deepEqual(received,['first']);assert.equal(scheduled.length,0);
  loop.start();resolve('resumed');await flush();assert.deepEqual(received,['first','resumed']);loop.stop();
  assert.equal(canceled,2);
});
test('scanner retries after extraction errors and waits for asynchronous metric work',async()=>{
  const scheduled=[],errors=[];let attempts=0,finish;
  const loop=new ScanLoop({sample:async()=>{if(++attempts===1)throw Error('No fresh frame');return {};},
    receive:()=>new Promise(r=>finish=r),onError:e=>errors.push(e.message),schedule:fn=>{scheduled.push(fn);return fn;},unschedule:()=>{}});
  loop.start();await flush();assert.deepEqual(errors,['No fresh frame']);scheduled.shift()();await flush();
  assert.equal(scheduled.length,0);finish();await flush();assert.equal(scheduled.length,1);loop.stop();
});
