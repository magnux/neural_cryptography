import test from 'node:test';
import assert from 'node:assert/strict';
import {TransportCapture} from '../dist/js/neural-auth/capture.js';
import {captureFeedback} from '../dist/js/neural-auth/feedback.js';
import {packetFrames} from '../dist/js/neural-auth/optical.js';
import {encodePacket} from '../dist/js/link/protocol.js';
import {NQR_LAYOUT,renderNeuralQR} from '../dist/js/neural-qr/layout.js';
import {geometryHintMaxAgeMs} from '../dist/js/neural-qr/vision.js';
import {PALETTE} from '../dist/js/format.js';

const common={messageType:'raw',sessionId:'12'.repeat(16),messageId:'34'.repeat(8)};
const packet=encodePacket({...common,kind:'data',index:0,data:Uint8Array.from({length:512},(_,i)=>i&255)});
const frames=packetFrames(packet,{size:64});
const images=frames.map(frame=>renderNeuralQR(frame,512));
const scan=(reader,time,phase=0,context='camera/1',captureIntervalMs=100)=>reader.scan(images[phase],{expected:'data',context,mediaTime:time,captureIntervalMs});
function paint(image,x,y,w,h,color){
  const scale=image.width/256;
  for(let yy=Math.floor(y*scale);yy<Math.ceil((y+h)*scale);yy++)for(let xx=Math.floor(x*scale);xx<Math.ceil((x+w)*scale);xx++)
    image.data.set([...color,255],4*(yy*image.width+xx));
}

test('geometry reuse reads every current phase and expires from the last full detection',()=>{
  const reader=new TransportCapture();
  const first=scan(reader,0);assert.equal(first.ok,true,first.error);assert.equal(first.stats.geometrySource,'detected');
  const next=scan(reader,.1,1);assert.equal(next.info.phase,1);assert.equal(next.stats.geometrySource,'previous');
  const third=scan(reader,(geometryHintMaxAgeMs(100)-1)/1000,2);assert.equal(third.info.phase,2);
  assert.equal(third.stats.geometrySource,'previous');assert.equal(third.geometry.acquiredAt,0);
  const expired=scan(reader,geometryHintMaxAgeMs(100)/1000,3);assert.equal(expired.info.phase,3);
  assert.equal(expired.stats.geometrySource,'detected');
  assert.equal(scan(reader,(geometryHintMaxAgeMs(100)+100)/1000,0).stats.geometrySource,'previous');
});

test('new runs, backwards timestamps, camera dimensions and explicit failures invalidate hints',()=>{
  const reader=new TransportCapture();scan(reader,1);
  assert.equal(scan(reader,1.1,1,'camera/2').stats.geometrySource,'detected');
  assert.equal(scan(reader,.5,1,'camera/2').stats.geometrySource,'detected');
  const resized=reader.scan(renderNeuralQR(frames[2],768),{expected:'data',context:'camera/2',mediaTime:.6});
  assert.equal(resized.stats.geometrySource,'detected');assert.equal(resized.info.phase,2);
  scan(reader,.7,0,'camera/2');reader.invalidate();
  assert.equal(scan(reader,.8,1,'camera/2').stats.geometrySource,'detected');
  reader.reset();assert.equal(scan(reader,.9,2,'camera/2').stats.geometrySource,'detected');
});

test('G32 ACKs use the same Neural-QR acquisition and turn changes start a fresh search',()=>{
  const reader=new TransportCapture();scan(reader,0);
  const ack=encodePacket({...common,kind:'ack',base:0,bitmap:1,status:'progress',digest:'56'.repeat(32)});
  const ackFrames=packetFrames(ack,{size:128});assert.equal(ackFrames[0].size,32);
  for(let phase=0;phase<4;phase++){
    const result=reader.scan(renderNeuralQR(ackFrames[phase],512),{expected:'ack',context:'camera/1',mediaTime:.1+phase*.05});
    assert.equal(result.ok,true,result.error);assert.equal(result.info.size,32);assert.equal(result.info.phase,phase);
    assert.equal(result.stats.geometrySource,phase===0?'detected':'previous');
    assert.deepEqual(result.dims,[1,3,128,128]);
    assert.equal(result.chunk,undefined,'acquisition alone must not claim packet recovery');
  }
  assert.equal(scan(reader,.3,1).stats.geometrySource,'detected');
});

test('geometry is read from current markers even when it changes within a cached outline',()=>{
  const reader=new TransportCapture();scan(reader,0);
  const larger=packetFrames(packet,{size:128})[2];
  const result=reader.scan(renderNeuralQR(larger,512),{expected:'data',context:'camera/1',mediaTime:.1});
  assert.equal(result.ok,true,result.error);assert.equal(result.stats.geometrySource,'previous');
  assert.equal(result.info.size,128);assert.equal(result.info.phase,2);assert.deepEqual(result.dims,[1,3,512,512]);
  // The worker applies the independently configured incoming cap after acquisition.
  const capped=captureFeedback({...result,ok:false,stage:'profile',error:'G128 exceeds this device receive limit G64.',expected:'data'});
  assert.equal(capped.validated,false);assert.equal(capped.corners.length,4);assert.match(capped.label,/receive limit G64/);
});

test('disagreeing phase cues give orange feedback; acquired phases provide green alignment counts',()=>{
  const reader=new TransportCapture(),transition={...images[0],data:images[0].data.slice()};
  for(const [i,[x,y]] of NQR_LAYOUT.phasePairs[1].entries())paint(transition,x-10,y-10,20,20,PALETTE[(1+i)%4]);
  const result=reader.scan(transition,{expected:'data',context:'camera/1',mediaTime:0});
  assert.equal(result.ok,false);assert.equal(result.stage,'phase');assert.equal(result.input,undefined);
  const feedback=captureFeedback({...result,expected:'data'});
  assert.equal(feedback.validated,false);assert.equal(feedback.corners.length,4);
  assert.match(feedback.label,/Phase cues uncertain.*disagree/);assert.ok(feedback.markers.length>=4);
  assert.ok(reader.geometry,'validated geometry survives a phase transition');
  const good=scan(reader,.1,1);assert.equal(good.stats.geometrySource,'previous');
  const accepted=captureFeedback({...good,ok:false,stage:'collect',expected:'data',phaseCounts:[0,1,0,0],collected:1});
  assert.equal(accepted.validated,true);assert.match(accepted.label,/G64.*0 \/ 1 \/ 0 \/ 0.*1 captures/);
  assert.doesNotMatch(accepted.label,/RS\+CRC passed/);
});

test('ACK feedback distinguishes acquisition, neural recovery and final packet validation',()=>{
  const miss=captureFeedback({expected:'ack',ok:false,stage:'detect',phaseCounts:[0,0,0,0]});
  assert.match(miss.label,/Neural-QR ACK/);assert.equal(miss.validated,false);
  const info={size:32},corners=[[0,0],[1,0],[1,1],[0,1]];
  const acquired=captureFeedback({expected:'ack',ok:false,info,corners,phaseCounts:[1,0,1,0],collected:2});
  assert.equal(acquired.validated,true);assert.match(acquired.label,/G32.*1 \/ 0 \/ 1 \/ 0/);
  const correcting=captureFeedback({expected:'ack',ok:false,info,corners,method:'neural-corrector'});
  assert.match(correcting.label,/collecting more samples/);assert.doesNotMatch(correcting.label,/passed/);
  const decoded=captureFeedback({expected:'ack',ok:true,info,corners});
  assert.equal(decoded.validated,true);assert.match(decoded.label,/ACK RS\+CRC passed.*checking transfer/);
});

test('capture context validation rejects invalid expected kinds and timestamps',()=>{
  const reader=new TransportCapture();
  for(const options of [{expected:'qr',context:'camera',mediaTime:0},{expected:'data',context:'',mediaTime:0},
    {expected:'data',context:'camera',mediaTime:-1},{expected:'data',context:'camera',mediaTime:NaN}])
    assert.throws(()=>reader.scan(images[0],options),/context/);
});


test('capture hint lifetime is exactly two configured reading intervals',()=>{
  for(const interval of [100,500]){
    const reader=new TransportCapture();scan(reader,0,0,'timing',interval);
    const within=scan(reader,(2*interval-1)/1000,1,'timing',interval);
    assert.equal(within.stats.geometrySource,'previous');
    assert.equal(within.stats.geometryMaxAgeMs,2*interval);
    assert.equal(within.stats.captureIntervalMs,interval);
    assert.equal(within.geometry.acquiredAt,0);
    const expired=scan(reader,2*interval/1000,2,'timing',interval);
    assert.equal(expired.stats.previousGeometryTried,false);
    assert.equal(expired.stats.geometrySource,'detected');
  }
  const reader=new TransportCapture();
  for(const interval of [0,49,5001,NaN,'500'])
    assert.throws(()=>scan(reader,0,0,'invalid',interval),/interval/i);
});

test('acquisition feedback preserves the specific failure and stays orange',()=>{
  for(const [stage,heading] of [['quad','arrangement'],['contrast','contrast'],['orientation','Orientation'],['size','Grid-size'],['phase','Phase']]){
    const feedback=captureFeedback({expected:'data',ok:false,stage,error:'Measured local contrast is too low.',
      tentativeCorners:[[0,0],[1,0],[1,1],[0,1]],markers:[{x:0,y:0}]});
    assert.equal(feedback.validated,false);assert.equal(feedback.corners.length,4);
    assert.ok(feedback.label.includes(heading));assert.match(feedback.label,/Measured local contrast/);
  }
});
