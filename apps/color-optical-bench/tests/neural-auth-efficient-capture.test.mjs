import test from 'node:test';
import assert from 'node:assert/strict';
import {ReceivePipeline} from '../dist/js/neural-auth/receive-pipeline.js';
import {AcceptedProbeGate} from '../dist/js/neural-auth/frame-gate.js';
import {TurnChannel} from '../dist/js/neural-auth/channel.js';
import {createTurnSender} from '../dist/js/neural-auth/transport.js';
import {encodePacket,decodePacket} from '../dist/js/link/protocol.js';
import {packetFrames,unpackOptical} from '../dist/js/neural-auth/optical.js';
import {renderNeuralQR} from '../dist/js/neural-qr/layout.js';

const manifest={modelProfile:'VW',framesPerInput:8,frameCounts:[4,5,6,7,8],channels:32,inputLayout:'phase-paired-rgb-presence-planes'};
const logits=frame=>{
  const values=new Float32Array(16*frame.size**2);
  frame.symbols.forEach((color,index)=>{values[color*frame.size**2+index]=10;});return values;
};
// Reduce the actual rendered camera image to the main-thread probe dimensions.
// Area averaging keeps phase-dependent payload colors instead of inventing a
// uniform or phase-labelled fixture that could pass without real imagery.
function probeImage(image){
  const side=32,scale=image.width/side,data=new Uint8ClampedArray(side*side*4);
  assert.equal(image.width,image.height);assert.ok(Number.isInteger(scale));
  for(let y=0;y<side;y++)for(let x=0;x<side;x++)for(let channel=0;channel<4;channel++){
    let total=0;
    for(let dy=0;dy<scale;dy++)for(let dx=0;dx<scale;dx++)
      total+=image.data[4*((y*scale+dy)*image.width+x*scale+dx)+channel];
    data[4*(y*side+x)+channel]=Math.round(total/scale**2);
  }
  return {width:side,height:side,data};
}
async function fixture({optimized=true,statistics=true}={}) {
  const sender=await createTurnSender(Uint8Array.from({length:80},(_,i)=>i),{sessionId:'12'.repeat(16),messageId:'34'.repeat(8),now:0,chunkBytes:80});
  const bytes=sender.next(0),frames=packetFrames(bytes,{size:32});
  let images=frames.map(frame=>renderNeuralQR(frame,512)),probes=images.map(probeImage),output=logits(frames[0]),calls=0,frameNumber=0;
  const pipeline=new ReceivePipeline({loadContext:async()=>({manifest,effectiveBackend:'test-only'}),
    infer:async(_context,input)=>{assert.equal(input.length,32*128**2);calls++;return {values:output,ms:1};}});
  const scan=(phase,time,extra={})=>pipeline.scan({image:images[phase],probe:probes[phase],runId:1,mediaTime:time,presentedFrames:++frameNumber,
    expected:'data',captureIntervalMs:100,maxSize:32,optimized,statistics,...extra});
  const cycle=async(start=0,extra={})=>{let result;for(let phase=0;phase<4;phase++)result=await scan(phase,start+phase*.1,extra);return result;};
  return {pipeline,sender,bytes,frames,scan,cycle,get calls(){return calls;},get probes(){return probes;},
    corrupt(){output=new Float32Array(output.length);},
    replace(next){images=next.map(frame=>renderNeuralQR(frame,512));probes=images.map(probeImage);output=logits(next[0]);}};
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

test('accepted four-phase camera probes skip every phase, refresh geometry, and expire at four seconds',async()=>{
  const h=await fixture(),channel=new TurnChannel({now:()=>0}),probeGate=new AcceptedProbeGate();
  const identity=mediaTime=>({runId:1,expected:'data',mediaTime});
  const decoded=await h.cycle();assert.equal(decoded.ok,true);assert.equal(decoded.inputCaptures,4);
  for(const probe of h.probes)assert.equal(probeGate.shouldSkip(probe,identity(.3)),false);
  assert.equal((await channel.receive(unpackOptical(decoded.chunk,'data'))).accepted,true);
  const accepted=h.pipeline.accept({token:decoded.acceptToken,runId:1,expected:'data'});
  assert.equal(accepted.accepted,true);probeGate.accept(accepted.probeGate);
  assert.deepEqual(accepted.probeGate.templates.map(row=>row.phase),[0,1,2,3]);
  for(let phase=0;phase<4;phase++){
    assert.notDeepEqual(h.probes[phase].data,h.probes[(phase+1)%4].data,'real optical phases have distinct images');
    assert.equal(probeGate.shouldSkip(h.probes[phase],identity(.4+phase*.1)),true,`phase ${phase} avoids full readback`);
    const held=await h.scan(phase,.4+phase*.1);assert.equal(held.stage,'unchanged');
  }
  assert.equal(h.calls,1,'neither gate treats phase cycling as new logical content');
  for(const time of [1.3,2.3,3.3]){
    assert.equal(probeGate.shouldSkip(h.probes[0],identity(time)),false,'one-second refresh requests acquisition');
    const refreshed=await h.scan(0,time,{forceAcquisition:true});
    assert.equal(refreshed.stage,'unchanged');assert.equal(refreshed.stats.geometrySource,'detected');
    assert.equal(refreshed.timings.rectification,undefined);assert.equal(h.calls,1);
  }
  assert.equal(probeGate.shouldSkip(h.probes[0],identity(4.299)),true,'refreshes do not renew acceptance');
  assert.equal((await h.scan(0,4.299)).stage,'unchanged');
  assert.equal(probeGate.shouldSkip(h.probes[0],identity(4.3)),false,'exact four-second acceptance expiry');
  const expired=await h.cycle(4.3,{forceAcquisition:true});
  assert.equal(expired.ok,true);assert.deepEqual(expired.chunk,h.bytes);assert.equal(h.calls,2);
});

test('a changed logical packet misses every accepted camera probe and is decoded afresh',async()=>{
  const h=await fixture(),channel=new TurnChannel({now:()=>0}),probeGate=new AcceptedProbeGate();
  const first=await h.cycle();assert.equal((await channel.receive(unpackOptical(first.chunk,'data'))).accepted,true);
  probeGate.accept(h.pipeline.accept({token:first.acceptToken,runId:1,expected:'data'}).probeGate);
  assert.equal(h.sender.receive(channel.current(),0).accepted,true);
  const next=h.sender.next(0);h.replace(packetFrames(next,{size:32}));
  for(let phase=0;phase<4;phase++){
    assert.equal(probeGate.shouldSkip(h.probes[phase],{runId:1,expected:'data',mediaTime:.4+phase*.1}),false,
      `new packet phase ${phase} cannot match a previous phase template`);
  }
  const decoded=await h.cycle(.4,{forceAcquisition:true});
  assert.equal(decoded.ok,true);assert.deepEqual(decoded.chunk,next);assert.equal(h.calls,2);
  assert.equal((await channel.receive(unpackOptical(decoded.chunk,'data'))).accepted,true);
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

test('local presentation changes discard partial phases without forgetting the accepted packet or its geometry',async()=>{
  const h=await fixture(),channel=new TurnChannel({now:()=>0});
  const first=await h.cycle(0,{presentationId:0});
  assert.equal((await channel.receive(unpackOptical(first.chunk,'data'))).accepted,true);
  assert.equal(h.pipeline.accept({token:first.acceptToken,runId:1,expected:'data'}).accepted,true);
  assert.equal((await h.scan(0,.32,{presentationId:1})).stage,'unchanged','old packet is still on screen');
  assert.equal(h.sender.receive(channel.current(),0).accepted,true);
  const next=h.sender.next(0);h.replace(packetFrames(next,{size:32}));
  for(let p=0;p<3;p++)assert.equal((await h.scan(p,.34+p*.01,{presentationId:1})).collected,p+1);
  // Both presentations expect the same kind and geometry. Phase 3 alone
  // cannot complete inference with three samples taken before the switch.
  const fresh=await h.scan(3,.38,{presentationId:2});
  assert.deepEqual(fresh.phaseCounts,[0,0,0,1]);assert.equal(fresh.stage,'collect');assert.equal(h.calls,1);
  assert.equal(fresh.stats.geometrySource,'previous');
  let decoded;for(let p=0;p<3;p++)decoded=await h.scan(p,.40+p*.01,{presentationId:2});
  assert.equal(h.calls,2);assert.equal(decoded.ok,true);assert.deepEqual(decoded.chunk,next);
  await assert.rejects(h.scan(0,.45,{presentationId:-1}),/Invalid Neural-ZKBoo/);
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


test('REQUEST reception corrects G32 controls and rejects held G64 DATA before rasterization',async()=>{
  const h=await fixture(),start=decodePacket(h.bytes);
  const request=encodePacket({kind:'poll',messageType:'raw',sessionId:start.sessionId,messageId:start.messageId,base:0});
  h.replace(packetFrames(request,{size:32}));
  let result;
  for(let phase=0;phase<4;phase++)result=await h.scan(phase,phase*.1,{expected:'request'});
  assert.equal(result.ok,true);assert.deepEqual(unpackOptical(result.chunk,'request'),request);
  assert.equal(h.calls,1);
  h.replace(packetFrames(h.bytes,{size:64}));
  result=await h.scan(0,.5,{expected:'request',maxSize:64});
  assert.equal(result.ok,false);assert.equal(result.stage,'profile');
  assert.match(result.error,/G32 Neural-QR REQUEST/);
  assert.equal(result.timings.rectification,undefined);assert.equal(h.calls,1);
});
