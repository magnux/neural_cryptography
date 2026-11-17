import test from 'node:test';
import assert from 'node:assert/strict';
import {ReceivePipeline} from '../dist/js/neural-auth/receive-pipeline.js';
import {TurnChannel} from '../dist/js/neural-auth/channel.js';
import {createTurnSender} from '../dist/js/neural-auth/transport.js';
import {packetFrames,unpackOptical} from '../dist/js/neural-auth/optical.js';
import {renderNeuralQR} from '../dist/js/neural-qr/layout.js';

const manifest={modelProfile:'VW',framesPerInput:8,frameCounts:[4,5,6,7,8],channels:32,inputLayout:'phase-paired-rgb-presence-planes'};
const logits=frame=>{
  const values=new Float32Array(16*frame.size**2);
  frame.symbols.forEach((color,index)=>{values[color*frame.size**2+index]=10;});return values;
};
async function fixture({optimized=true,statistics=true}={}) {
  const sender=await createTurnSender(Uint8Array.from({length:80},(_,i)=>i),{sessionId:'12'.repeat(16),messageId:'34'.repeat(8),now:0,chunkBytes:80});
  const bytes=sender.next(0),frames=packetFrames(bytes,{size:32});
  let images=frames.map(frame=>renderNeuralQR(frame,512)),output=logits(frames[0]),calls=0,frameNumber=0;
  const pipeline=new ReceivePipeline({loadContext:async()=>({manifest,effectiveBackend:'test-only'}),
    infer:async(_context,input)=>{assert.equal(input.length,32*128**2);calls++;return {values:output,ms:1};}});
  const scan=(phase,time,extra={})=>pipeline.scan({image:images[phase],runId:1,mediaTime:time,presentedFrames:++frameNumber,
    expected:'data',captureIntervalMs:100,maxSize:32,optimized,statistics,...extra});
  const cycle=async(start=0,extra={})=>{let result;for(let phase=0;phase<4;phase++)result=await scan(phase,start+phase*.1,extra);return result;};
  return {pipeline,sender,bytes,frames,scan,cycle,get calls(){return calls;},
    corrupt(){output=new Float32Array(output.length);},
    replace(next){images=next.map(frame=>renderNeuralQR(frame,512));output=logits(next[0]);}};
}

test('real acquisition and recovery gate repeated phases only after matching TurnChannel acceptance',async()=>{
  const h=await fixture(),channel=new TurnChannel({now:()=>0});
  const decoded=await h.cycle();assert.equal(decoded.ok,true);assert.deepEqual(decoded.chunk,h.bytes);assert.equal(h.calls,1);
  // A decoded result does not itself enable suppression; rejection keeps fresh captures flowing.
  const before=await h.scan(0,.4);assert.equal(before.stage,'collect');assert.equal(before.skipped,undefined);
  assert.equal(h.pipeline.accept({token:decoded.acceptToken,runId:1,expected:'data'}).accepted,false,'another scan invalidates a delayed token');
  const retry=await h.cycle(.5);assert.equal(retry.ok,true);
  const accepted=await channel.receive(unpackOptical(retry.chunk,'data'));assert.equal(accepted.accepted,true);
  assert.equal(h.pipeline.accept({token:retry.acceptToken,runId:1,expected:'data'}).accepted,true);
  const inferenceCount=h.calls;
  for(let phase=0;phase<4;phase++){
    const held=await h.scan(phase,.9+phase*.1);assert.equal(held.stage,'unchanged');assert.equal(held.skipped,true);
  }
  assert.equal(h.calls,inferenceCount);assert.equal(h.pipeline.snapshot().gated,4);
  const periodic=await h.scan(0,1.9,{forceAcquisition:true});assert.equal(periodic.stage,'unchanged');
  assert.equal(periodic.stats.geometrySource,'detected');assert.equal(periodic.timings.rectification,undefined);
  assert.ok(periodic.timings.orientation.count>0);assert.equal(h.calls,inferenceCount);
});

test('next packet requires fresh four-phase captures and goes through RS/CRC and the channel',async()=>{
  const h=await fixture(),channel=new TurnChannel({now:()=>0});
  const first=await h.cycle();const received=await channel.receive(unpackOptical(first.chunk,'data'));
  assert.equal(received.accepted,true);h.pipeline.accept({token:first.acceptToken,runId:1,expected:'data'});
  assert.equal(h.sender.receive(channel.current(),0).accepted,true);
  const next=h.sender.next(0),frames=packetFrames(next,{size:32});h.replace(frames);
  for(let phase=0;phase<3;phase++){
    const result=await h.scan(phase,.4+.1*phase);assert.equal(result.ok,false);assert.equal(result.stage,'collect');
    assert.equal(result.collected,phase+1);
  }
  const result=await h.scan(3,.7);assert.equal(result.ok,true);assert.deepEqual(result.chunk,next);
  assert.equal((await channel.receive(unpackOptical(result.chunk,'data'))).accepted,true);assert.equal(h.calls,2);
});

test('hard expiry and explicit full scans bypass every gate without treating phase cycling as new content',async()=>{
  const h=await fixture(),first=await h.cycle();h.pipeline.accept({token:first.acceptToken,runId:1,expected:'data'});
  for(let phase=0;phase<4;phase++)assert.equal((await h.scan(phase,.4+phase*.1)).stage,'unchanged');
  assert.equal((await h.scan(0,.8,{forceFullScan:true})).stage,'collect');
  const afterExpiry=await h.cycle(4.4);assert.equal(afterExpiry.ok,true);assert.equal(h.calls,2);
  assert.equal(afterExpiry.timings.inference.count,1);assert.ok(afterExpiry.timings['rs-crc-recovery']);
});

test('corrupt logits never arm a gate and failed attempts retain retry eligibility',async()=>{
  const h=await fixture();h.corrupt();const result=await h.cycle();
  assert.equal(result.ok,false);assert.equal(result.acceptToken,undefined);
  assert.equal(h.pipeline.accept({token:'invented',runId:1,expected:'data'}).accepted,false);
  const retry=await h.cycle(.4);assert.equal(retry.ok,false);assert.equal(retry.skipped,undefined);assert.equal(h.calls,2);
  h.replace(h.frames);assert.equal((await h.cycle(.8)).ok,true);assert.equal(h.calls,3);
});

test('new run and expected kind discard accepted fingerprints and old burst captures',async()=>{
  const h=await fixture(),first=await h.cycle();h.pipeline.accept({token:first.acceptToken,runId:1,expected:'data'});
  const nextRun=await h.scan(0,.4,{runId:2});assert.equal(nextRun.stage,'collect');assert.equal(nextRun.collected,1);
  const otherKind=await h.scan(1,.5,{runId:2,expected:'ack'});assert.equal(otherKind.stage,'collect');assert.equal(otherKind.collected,1);
});

test('statistics off avoids acquisition and stage clock reads and omits timing output',async()=>{
  const h=await fixture({statistics:false}),saved=Object.getOwnPropertyDescriptor(globalThis,'performance');let reads=0;
  Object.defineProperty(globalThis,'performance',{configurable:true,value:{now(){reads++;return 0;}}});
  try {
    const result=await h.cycle();assert.equal(result.ok,true);assert.equal(reads,0);
    assert.equal(result.timings,undefined);assert.equal(result.statistics,undefined);assert.equal(result.inferenceMs,undefined);
    assert.equal(result.preprocessMs,undefined);assert.equal(result.stats.ms,undefined);assert.equal(h.pipeline.snapshot().timings,undefined);
  } finally {Object.defineProperty(globalThis,'performance',saved);}
});

test('baseline collection retains normal repeated inference and existing total timing fields',async()=>{
  const h=await fixture({optimized:false}),first=await h.cycle(),second=await h.cycle(.4);
  assert.equal(first.ok,true);assert.equal(second.ok,true);assert.equal(first.acceptToken,undefined);assert.equal(h.calls,2);
  assert.equal(first.inferenceMs,1);assert.ok(first.preprocessMs>=0);
});
