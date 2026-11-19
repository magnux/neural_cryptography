import test from 'node:test';
import assert from 'node:assert/strict';
import {SweepReader} from '../dist/sweep-reader.js';
import {ReceivePipeline} from '../dist/js/neural-auth/receive-pipeline.js';
import {packetFrames} from '../dist/js/neural-auth/optical.js';
import {encodePacket} from '../dist/js/link/protocol.js';
import {renderNeuralQR,NQR_LAYOUT} from '../dist/js/neural-qr/layout.js';

const metadata={ready:true,modelStatus:'trained',modelProfile:'VW',effectiveBackend:'test',parity:{passed:true}};
const identity=n=>({presentedFrames:n,mediaTime:n*.05,captureIntervalMs:450});
test('rendered single exposures go through canonical four-phase acquisition, measured K4 inference and packet recovery',async()=>{
  const wire=encodePacket({kind:'data',messageType:'raw',sessionId:'11'.repeat(16),messageId:'22'.repeat(8),index:0,data:Uint8Array.of(3,4)});
  const frames=packetFrames(wire,{size:32,phaseMs:500});
  const manifest={modelProfile:'VW',pixelsPerCell:4,channels:32,framesPerInput:8,minFramesPerInput:4,frameCounts:[4,5,6,7,8],
    inputLayout:'phase-paired-rgb-presence-planes',outputLabelSpace:'logical',phaseMasks:[0,15,7,8]};
  // Inject a deliberately uninformative model, *not* the target symbols. This
  // exercises real marker/warp/burst/RS stages without a trained checkpoint.
  const pipeline=new ReceivePipeline({loadContext:async()=>({manifest,effectiveBackend:'test'}),
    infer:async()=>({values:new Float32Array(16*32*32),ms:0})});
  const client={call:async(op,o,transfers)=>{
    if(op==='transport-init')return metadata;
    assert.equal(op,'transport-scan');assert.deepEqual(transfers,[o.image.data.buffer]);
    return pipeline.scan({...o,maxSize:o.size});
  },cancel(){}};
  const reader=new SweepReader({role:'receiver',size:32,client});await reader.init();reader.beginStep();
  let result;
  for(let i=0;i<4;i++){
    result=await reader.scan(renderNeuralQR(frames[i],512),identity(i+1));
    if(i<3){assert.equal(result.stage,'collect',result.error);assert.equal(reader.snapshot().stages.inference,undefined);}
  }
  assert.equal(result.ok,false,'a failed neural output must never be accepted as an optical match');
  assert.equal(result.inputCaptures,4);
  const s=reader.snapshot();assert.deepEqual(s.phases,[1,1,1,1]);assert.deepEqual(s.inferences,{K4:1,K8:0});
  assert.equal(s.geometry.previous,3);assert.equal(s.geometry.detected,1);assert.equal(s.geometry.hintAttempts,3);
  assert.equal(s.stages['marker-detection'].measuredScans,1);assert.ok(s.stages.homography.count>=1);
  assert.equal(s.stages.rectification.count,4);assert.equal(s.stages.normalization.count,4);
  assert.equal(s.stages['burst-pack'].count,1);assert.equal(s.stages.inference.count,1);assert.equal(s.stages['rs-crc-recovery'].count,1);
  assert.ok(s.firstAllPhasesMs>=s.firstPhaseMs[0]);assert.equal(s.trace.length,4);
  reader.beginStep();const next=await reader.scan(renderNeuralQR(frames[2],512),identity(5));
  assert.equal(next.collected,1);assert.deepEqual(reader.snapshot().phases,[0,0,1,0]);assert.equal(reader.snapshot().geometry.detected,1);
  assert.deepEqual(s.phases,[1,1,1,1],'completed step report must remain independent of the new pool');reader.close();
});

test('sender uses the trained G32 ACK profile, rejects untrained assets, and bounds its capture trace',async()=>{
  let untrained=true,now=0;const image={data:new Uint8ClampedArray(0)};
  const client={call:(op,opts)=>{
    assert.equal(opts.size,32);
    if(op==='transport-init')return {...metadata,modelStatus:untrained?'untrained':'trained'};
    assert.equal(opts.expected,'ack');now+=2;return {ok:false,stage:'detect',timings:{},stats:{}};
  },cancel(){}};
  const reader=new SweepReader({role:'sender',size:128,client,now:()=>now});
  await assert.rejects(reader.init(),/trained/);untrained=false;
  await reader.init();reader.beginStep();
  for(let n=1;n<=80;n++)await reader.scan(image,identity(n));
  const s=reader.snapshot();assert.equal(s.scans,80);assert.equal(s.workerRoundTripMs.count,80);assert.equal(s.stages.inference,undefined);
  assert.equal(s.trace.length,64);assert.equal(s.omittedScans,16);assert.equal(s.outcomes.detect,80);assert.equal(s.trace[0].startMs,0);
  assert.equal(s.trace.at(-1).endMs,160);assert.deepEqual(s.firstPhaseMs,[null,null,null,null]);
});

test('an in-flight old scan cannot populate the next step',async()=>{
  let resolve;const client={call:async op=>op==='transport-init'?metadata:new Promise(r=>resolve=r),cancel(){}};
  const reader=new SweepReader({role:'receiver',size:64,client});await reader.init();reader.beginStep();
  const stale=reader.scan({data:new Uint8Array(0)},identity(1));assert.equal(reader.snapshot().withoutResult,1);
  reader.beginStep();resolve({ok:false,stage:'detect',timings:{}});await assert.rejects(stale,{name:'AbortError'});
  assert.equal(reader.snapshot().scans,0);assert.equal(reader.snapshot().withoutResult,0);
});

test('torn G32 ACK records opposite border votes but does not fill the sender burst',async()=>{
  const wire=encodePacket({kind:'ack',messageType:'raw',sessionId:'11'.repeat(16),messageId:'22'.repeat(8),
    base:0,bitmap:1,status:'complete',digest:'44'.repeat(32)});
  const frames=packetFrames(wire,{size:32,phaseMs:500});
  const image=renderNeuralQR(frames[3],256),other=renderNeuralQR(frames[1],256);
  for(const [x,y] of NQR_LAYOUT.phasePairs[1]){
    for(let dy=-10;dy<10;dy++){
      const i=4*((y+dy)*256+x-10);image.data.set(other.data.subarray(i,i+80),i);
    }
  }
  const nope=()=>assert.fail('rejected phase reached corrected inference');
  const pipeline=new ReceivePipeline({loadContext:nope,infer:nope});
  const client={call:async(op,o)=>op==='transport-init'?metadata:pipeline.scan({...o,maxSize:o.size}),cancel(){}};
  const reader=new SweepReader({role:'sender',size:64,client});await reader.init();reader.beginStep();
  const r=await reader.scan(image,identity(1)),s=reader.snapshot();
  assert.equal(r.ok,false);assert.equal(r.info,null);assert.equal(r.collected,0);
  assert.deepEqual(s.phaseRejections,{'opposite-disagree':1});assert.deepEqual(s.rejectedCueNearest,[[0,0,0,1],[0,1,0,0]]);
  assert.deepEqual(s.trace[0].nearestCuePhases,[3,1]);assert.deepEqual(s.trace[0].cuePassed,[true,true]);
  assert.equal(s.trace[0].phase,null);assert.deepEqual(s.phases,[0,0,0,0]);reader.close();
});
