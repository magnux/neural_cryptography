import test from 'node:test';
import assert from 'node:assert/strict';
import {TransportCapture} from '../dist/js/neural-auth/capture.js';
import {captureFeedback} from '../dist/js/neural-auth/feedback.js';
import {renderACK} from '../dist/js/neural-auth/qr-ack.js';
import {encodePacket} from '../dist/js/link/protocol.js';
import {encodeSequence,phaseFrames,renderFrame} from '../dist/js/format.js';

const frames=phaseFrames(encodeSequence(Uint8Array.from({length:512},(_,i)=>i&255),{
  size:64,colors:16,phases:4,permute:true,parity:32,gridBlock:8,id:Uint8Array.of(1,2,3,4,5,6,7,8)}));
const images=frames.map(frame=>renderFrame(frame,512));
const scan=(reader,time,phase=0,context='camera/1')=>reader.scan(images[phase],{expected:'data',context,mediaTime:time});

test('geometry reuse expires from last full detection, not last successful reuse',()=>{
  const reader=new TransportCapture();
  assert.equal(scan(reader,0).stats.geometrySource,'detected');
  assert.equal(scan(reader,.1,1).stats.geometrySource,'previous');
  assert.equal(scan(reader,.4,2).info.phase,2);
  assert.equal(scan(reader,.5,3).stats.geometrySource,'detected');
  assert.equal(scan(reader,.6,0).stats.geometrySource,'previous');
});

test('new runs, backwards timestamps, camera dimensions and payload failures invalidate hints',()=>{
  const reader=new TransportCapture();scan(reader,1);
  assert.equal(scan(reader,1.1,1,'camera/2').stats.geometrySource,'detected');
  assert.equal(scan(reader,.5,1,'camera/2').stats.geometrySource,'detected');
  const resized=reader.scan(renderFrame(frames[2],768),{expected:'data',context:'camera/2',mediaTime:.6});
  assert.equal(resized.stats.geometrySource,'detected');
  scan(reader,.7,0,'camera/2');reader.invalidate();
  assert.equal(scan(reader,.8,1,'camera/2').stats.geometrySource,'detected');
  reader.reset();assert.equal(scan(reader,.9,2,'camera/2').stats.geometrySource,'detected');
});

test('ACK mode uses QR reception and returning to data starts a fresh geometry search',()=>{
  const reader=new TransportCapture();scan(reader,0);
  const ack=encodePacket({kind:'ack',messageType:'raw',sessionId:'12'.repeat(16),messageId:'34'.repeat(8),
    base:0,bitmap:1,status:'progress',digest:'56'.repeat(32)});
  const result=reader.scan(renderACK(ack),{expected:'ack',context:'camera/1',mediaTime:.1});
  assert.equal(result.ok,true,result.error);assert.equal(result.method,'qr-ack');assert.deepEqual(result.chunk,ack);
  assert.equal(reader.geometry,null);
  assert.equal(scan(reader,.2,1).stats.geometrySource,'detected');
});

test('unreadable headers get orange feedback without ever claiming a valid lock',()=>{
  const reader=new TransportCapture(),header=frames[0].header.slice();header[45]^=1;
  const result=reader.scan(renderFrame({...frames[0],header},512),{expected:'data',context:'camera/1',mediaTime:0});
  const feedback=captureFeedback({...result,expected:'data'});
  assert.equal(feedback.validated,false);assert.equal(feedback.corners.length,4);
  assert.match(feedback.label,/Markers found.*header unreadable/);
  assert.ok(feedback.markers.length>=4);assert.equal(reader.geometry,null);
  const good=scan(reader,.1,1),accepted=captureFeedback({...good,expected:'data',phaseCounts:[0,1,0,0],collected:1});
  assert.equal(accepted.validated,true);assert.match(accepted.label,/Header read.*0 \/ 1 \/ 0 \/ 0/);
});

test('ACK feedback never shows misleading palette phase counters',()=>{
  const miss=captureFeedback({expected:'ack',method:'qr-ack',ok:false,error:'Looking for a static ACK QR.',phaseCounts:[0,0,0,0]});
  assert.match(miss.label,/ACK QR/);assert.doesNotMatch(miss.label,/phase|captures/i);assert.equal(miss.validated,false);
  const decoded=captureFeedback({expected:'ack',ok:true,corners:[[0,0],[1,0],[1,1],[0,1]]});
  assert.equal(decoded.validated,true);assert.match(decoded.label,/checking transfer/);
});
