import test from 'node:test';
import assert from 'node:assert/strict';
import {CaptureBuffer,captureFrame} from '../dist/js/dataset/capture.js';
import {captureSignature,meanCaptureDistance} from '../dist/js/dataset/diversity.js';
import {createSender,captureKey} from '../dist/js/dataset/contract.js';
import {renderFrame} from '../dist/js/format.js';
import {ScanLoop} from '../dist/js/dataset/scan-loop.js';

const sender=await createSender({seed:55,sessionId:'pool-test'});
const sample=captureFrame(renderFrame(sender.frames[0],1024),{timestamp:1000,signature:true});
function fill(buffer,count,record=sample,offset=0){
  buffer.add({...record,timestamp:1000+offset},{alreadyValidated:true,maxPerPhase:32});
  for(let i=1;i<=count;i++)assert.equal(buffer.add({...record,timestamp:1000+offset+i*600},{alreadyValidated:true,maxPerPhase:32}).accepted,true);
}

test('32-sample per-phase pools, random thinning, refill, and phase isolation',()=>{
  const b=new CaptureBuffer();fill(b,32);
  assert.equal(b.add({...sample,timestamp:21000},{alreadyValidated:true,maxPerPhase:32}).stage,'phase-full');
  const other={...sample,info:{...sample.info,phase:1}};other.key=captureKey(other.info);
  fill(b,3,other,30000);
  const before=b.bytes,otherRows=b.phaseRecords(other.key),ids=new Set(b.phaseRecords(sample.key).map(x=>x.timestamp));
  const thinning=b.thin(sample.key,16,()=>.37);
  assert.deepEqual(thinning,{before:32,retained:16,discarded:16});
  assert.deepEqual(b.phaseRecords(other.key),otherRows);
  assert.ok(b.bytes<before);assert.equal(b.phaseSignatures(sample.key).length,16);
  assert.ok(b.phaseRecords(sample.key).every(x=>ids.has(x.timestamp)));
  fill(b,16,sample,60000);
  assert.equal(b.phaseRecords(sample.key).length,32);
  const exported=b.export({sessionId:'receiver'});
  assert.equal(exported.records.length,35);
  assert.ok(exported.records.every(x=>!('signature' in x)));
  assert.equal(exported.selection.recentThinning[0].discarded,16);
});

test('byte budget blocks further samples and thinning frees space without losing diagnostics',()=>{
  const b=new CaptureBuffer({maxBytes:3*1024*1024});
  b.add(sample,{alreadyValidated:true});
  let full;
  for(let i=1;i<8;i++){
    const result=b.add({...sample,timestamp:1000+i*600},{alreadyValidated:true});
    if(result.stage==='full'){full=result;break;}
  }
  assert.equal(full.stage,'full');assert.ok(b.bytes<b.maxBytes);
  const n=b.diagnostics.length;b.thin(sample.key,1,()=>0);
  assert.equal(b.diagnostics.length,n);assert.equal(b.phaseRecords(sample.key).length,1);
  assert.equal(b.add({...sample,timestamp:10000},{alreadyValidated:true}).accepted,false);
  assert.equal(b.add({...sample,timestamp:10600},{alreadyValidated:true}).accepted,true);
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
