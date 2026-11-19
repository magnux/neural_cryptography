import test from 'node:test';
import assert from 'node:assert/strict';
import {AcceptedFrameGate,AcceptedProbeGate,payloadFingerprint} from '../dist/js/neural-auth/frame-gate.js';
import {StageProfiler} from '../dist/js/neural-auth/timing.js';

function solid(value,size=256){
  const data=new Uint8ClampedArray(size*size*4);
  for(let offset=0;offset<data.length;offset+=4)data.set([value,value,value,255],offset);
  return {width:size,height:size,data};
}
const identity=(mediaTime=1,extra={})=>({runId:3,expected:'data',mediaTime,presentedFrames:1,...extra});
const capture=(phase=0,size=64)=>({info:{phase,size},geometry:{width:256,height:256,transform:[1,0,0,0,1,0,0,0,1]}});
function acceptedGate(values=[40,90,150,220]){
  const gate=new AcceptedFrameGate(),images=values.map(value=>solid(value));
  const records=images.map((image,index)=>gate.record(image,capture(index%4),identity(1,{presentedFrames:index+1}),solid(values[index],32)));
  return {gate,images,records,state:gate.accept(records,identity())};
}

test('recording or recognizing pixels cannot gate work before explicit transport acceptance',()=>{
  const gate=new AcceptedFrameGate(),image=solid(90),record=gate.record(image,capture(),identity(),solid(90,32));
  assert.ok(record.fingerprint);
  assert.equal(gate.check(image,identity(1.1)),false);
  assert.equal(gate.matchesCapture(image,capture(),identity(1.1)),false);
  gate.accept([record],identity());
  assert.equal(gate.check(image,identity(1.1)),true);
  assert.equal(gate.check(solid(160),identity(1.2)),false,'a visibly changed payload must request decoding');
  gate.reset();
  assert.equal(gate.check(image,identity(1.3)),false);
});

test('accepted templates support every held phase but rectified matches require current phase and size',()=>{
  const {gate,images}=acceptedGate();
  for(let phase=0;phase<4;phase++){
    assert.equal(gate.check(images[phase],identity(1.1+phase*.1)),true);
    assert.equal(gate.matchesCapture(images[phase],capture(phase),identity(1.1)),true);
    assert.equal(gate.matchesCapture(images[phase],capture((phase+1)%4),identity(1.1)),false);
    assert.equal(gate.matchesCapture(images[phase],capture(phase,128),identity(1.1)),false);
  }
});

test('periodic acquisition, explicit full scans, context changes and hard expiry preserve liveness',()=>{
  const {gate,images}=acceptedGate(),image=images[0];
  assert.equal(gate.check(image,identity(1.999)),true);
  assert.equal(gate.check(image,identity(2)),false,'one second requests fresh acquisition');
  assert.equal(gate.check(image,identity(2.1)),true);
  assert.equal(gate.check(image,identity(2.2),{forceFullScan:true}),false);
  assert.equal(gate.check(image,identity(2.3,{runId:4})),false);
  assert.equal(gate.check(image,identity(2.3,{expected:'ack'})),false);
  assert.equal(gate.check(image,identity(.9)),false);
  assert.equal(gate.check(image,identity(5)),false,'four-second expiry prevents indefinite suppression');
  assert.equal(gate.matchesCapture(image,capture(),identity(5)),false,'expiry also bypasses the rectified gate');
});

test('a change between sparse sample points still cannot survive the acceptance TTL',()=>{
  const {gate,images}=acceptedGate([90]),changed={...images[0],data:images[0].data.slice()};
  changed.data.set([250,0,0,255],4*(100*256+100));
  assert.deepEqual(payloadFingerprint(changed,capture().geometry),payloadFingerprint(images[0],capture().geometry));
  assert.equal(gate.check(changed,identity(1.1)),true,'sparse collisions are possible and must be bounded');
  assert.equal(gate.check(changed,identity(5)),false);
  assert.equal(gate.matchesCapture(changed,capture(),identity(5)),false);
});

test('template and probe memory are bounded, and invalid imagery never suppresses work',()=>{
  const {gate,state}=acceptedGate(Array.from({length:12},(_,index)=>index*20));
  assert.equal(gate.templates.length,8);assert.equal(state.templates.length,8);
  assert.equal(gate.check(solid(0),identity(1.1)),false,'oldest templates must be evicted');
  assert.equal(gate.check(solid(100,128),identity(1.1)),false,'old camera geometry cannot gate a resized image');
  assert.equal(gate.check({width:256,height:256,data:new Uint8ClampedArray(3)},identity(1.1)),false);
  const probe=new AcceptedProbeGate(),image=solid(100,32);
  assert.equal(probe.shouldSkip(image,identity(1.1)),false);
  probe.accept(state);assert.equal(probe.shouldSkip(image,identity(1.1)),true);
  assert.equal(probe.shouldSkip(solid(100,64),identity(1.1)),false);
  assert.equal(probe.shouldSkip(solid(1,32),identity(1.2)),false);
  assert.equal(probe.shouldSkip(image,identity(2)),false,'main-thread readback cannot be skipped indefinitely');
  assert.equal(probe.shouldSkip(image,identity(2.1)),true);
  assert.equal(probe.shouldSkip(image,identity(2.2,{expected:'ack'})),false);
  assert.equal(probe.shouldSkip(image,identity(2.2,{runId:8})),false);
  assert.equal(probe.shouldSkip(image,identity(.9)),false);
  assert.equal(probe.shouldSkip(image,identity(5)),false);
  probe.reset();assert.equal(probe.shouldSkip(image,identity(1.1)),false);
});

test('video thumbnail readback runs only when it could skip an accepted duplicate',()=>{
  const {state}=acceptedGate(),gate=new AcceptedProbeGate();
  const never=()=>assert.fail('camera read that could not suppress a full capture');
  assert.equal(gate.shouldSkip(never,identity(1)),false);
  gate.accept(state);
  for(const phase of [40,90,150,220])assert.equal(gate.shouldSkip(()=>solid(phase,32),identity(1.1)),true);
  assert.equal(gate.shouldSkip(never,identity(1.15,{expected:'ack'})),false);
  assert.equal(gate.shouldSkip(never,identity(2)),false,'scheduled geometry refresh needs no camera probe');
  assert.equal(gate.shouldSkip(()=>solid(17,32),identity(2.1)),false,'unknown pixels still request acquisition');
  assert.equal(gate.shouldSkip(never,identity(5)),false,'expired acceptance cannot suppress a capture');
});

test('disabled profiling reads no clocks, retains no rows and preserves function results and errors',async()=>{
  const profiler=new StageProfiler(false,{now:()=>{throw Error('disabled profiler read the clock');}});
  assert.equal(profiler.measure('sync',()=>42),42);
  assert.equal(await profiler.measure('async',async()=>84),84);
  assert.throws(()=>profiler.measure('throws',()=>{throw Error('original failure');}),/original failure/);
  await assert.rejects(profiler.measure('rejects',async()=>{throw Error('original rejection');}),/original rejection/);
  profiler.record('manual',4);profiler.merge({manual:{count:1,totalMs:4,minMs:4,maxMs:4}});
  assert.equal(profiler.snapshot(),undefined);assert.equal(profiler.rows.size,0);
});

test('enabled profiling aggregates finite bounded stages, including asynchronous failures',async()=>{
  let now=0;const profiler=new StageProfiler(true,{now:()=>now,maxStages:2});
  assert.equal(profiler.measure('work',()=>{now+=2;return 1;}),1);
  await profiler.measure('work',async()=>{now+=4;});
  await assert.rejects(profiler.measure('failed',async()=>{now+=3;throw Error('failure');}),/failure/);
  profiler.record('ignored',100);profiler.record('work',NaN);profiler.record('work',-1);
  profiler.merge({work:{count:2,totalMs:10,minMs:3,maxMs:7}});
  const result=profiler.snapshot();
  assert.deepEqual(Object.keys(result),['work','failed']);
  assert.deepEqual(result.work,{count:4,totalMs:16,minMs:2,maxMs:7,avgMs:4});
  assert.deepEqual(result.failed,{count:1,totalMs:3,minMs:3,maxMs:3,avgMs:3});
  result.work.count=100;assert.equal(profiler.snapshot().work.count,4,'reports cannot mutate retained aggregate state');
});
