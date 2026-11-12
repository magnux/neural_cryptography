import test from 'node:test';
import assert from 'node:assert/strict';
import {encodePacket} from '../dist/js/link/protocol.js';
import {packetFrames} from '../dist/js/neural-auth/optical.js';
import {paletteMap} from '../dist/js/grid.js';
import {LivePhaseBurst,requireTransportModel,isTransportData,recoverCorrectedPayload} from '../dist/js/corrector-lab/live-burst.js';

const ID=Uint8Array.of(1,2,3,4,5,6,7,8),ONE=3*256*256;
const PAYLOAD=encodePacket({kind:'data',messageType:'raw',sessionId:'12'.repeat(16),messageId:'34'.repeat(8),index:0,data:Uint8Array.from({length:500},(_,i)=>(17*i+9)&255)});
function frames(payload=PAYLOAD) {
  return packetFrames(payload,{size:64});
}
function crop(frame,value=0) {return {info:{...frame},input:new Float32Array(ONE).fill(value),stats:{preprocessMs:1}};}
const identity=(frame,context='camera/1')=>({context,mediaTime:frame/10,presentedFrames:frame});
const goodModel={status:'trained',modelProfile:'VW',datasetFormat:'neural-qr-dataset',datasetVersion:2,pixelsPerCell:4,inputLayout:'phase-paired-rgb-presence-planes',
  outputLabelSpace:'logical',framesPerInput:8,minFramesPerInput:4,cases:[{size:32},{size:64},{size:128}]};

test('live authentication rejects untrained or incompatible model exports',()=>{
  assert.doesNotThrow(()=>requireTransportModel(goodModel));
  for(const patch of [{status:'untrained'},{modelProfile:'U-tube'},{inputLayout:'stacked-rgb'},
    {outputLabelSpace:'physical'},{framesPerInput:5},{minFramesPerInput:5},{cases:[{size:128}]},{cases:[]}]) {
    assert.throws(()=>requireTransportModel({...goodModel,...patch}),/trained W-tube or VW/);
  }
});

test('live pool begins with all four phases and grows through K8 without duplicating exposures',()=>{
  const pool=new LivePhaseBurst(),source=frames();
  for(let phase=0;phase<4;phase++) {
    const state=pool.add(crop(source[phase],phase/4),identity(phase+1));
    assert.equal(state.ready,phase===3);
    if(phase<3)assert.equal(pool.snapshot(),null);
  }
  let result=pool.snapshot();assert.equal(result.inputCaptures,4);
  assert.deepEqual(result.slots,[0,2,4,6]);assert.deepEqual(result.phases,[0,1,2,3]);
  for(let phase=0;phase<4;phase++) {
    const state=pool.add(crop(source[phase],1),identity(phase+5));
    assert.equal(state.collected,phase+5);
    result=pool.snapshot();assert.equal(result.inputCaptures,phase+5);
    assert.equal(new Set(result.presentedFrames).size,result.inputCaptures);
  }
  assert.deepEqual(result.phaseCounts,[2,2,2,2]);
  assert.equal(pool.add(crop(source[0]),identity(8)).duplicateExposure,true);
  assert.equal(pool.status().collected,8);
});

test('an early held phase cannot fill missing phase slots or skip the first K4 attempt',()=>{
  const pool=new LivePhaseBurst(),source=frames();
  for(let i=1;i<=20;i++)pool.add(crop(source[0]),identity(i));
  assert.deepEqual(pool.status().phaseCounts,[2,0,0,0]);assert.equal(pool.snapshot(),null);
  for(let phase=1;phase<4;phase++)pool.add(crop(source[phase]),identity(20+phase));
  const result=pool.snapshot();assert.equal(result.inputCaptures,4);assert.deepEqual(result.phaseCounts,[1,1,1,1]);
  assert.equal(result.presentedFrames[0],20);
});

test('full pools replace old evidence and remain bounded after repeated CRC failures',()=>{
  const pool=new LivePhaseBurst(),source=frames();
  for(let i=1;i<=36;i++) {
    const sourceCrop=crop(source[(i-1)%4],i/36);
    pool.add(sourceCrop,identity(i));sourceCrop.input.fill(-1);
    const result=pool.snapshot();
    if(result) {
      assert.ok(result.inputCaptures>=4&&result.inputCaptures<=8);
      assert.ok(result.input.every(value=>value>0));
    }
    assert.ok(pool.status().bufferBytes<=8*ONE*4);
  }
  assert.equal(pool.status().replacements,28);
  assert.equal(Math.min(...pool.snapshot().presentedFrames),29);
});

test('transient misses retain good phases, but age, geometry and run changes clear them',()=>{
  const pool=new LivePhaseBurst(),source=frames();
  pool.add(crop(source[0]),identity(1));pool.add(crop(source[1]),identity(2));
  assert.deepEqual(pool.miss(identity(3)).phaseCounts,[1,1,0,0]);
  assert.deepEqual(pool.miss({context:'camera/1',mediaTime:61,presentedFrames:100}).phaseCounts,[0,0,0,0]);
  pool.add(crop(source[2]),{context:'camera/1',mediaTime:62,presentedFrames:101});
  assert.equal(pool.add(crop(source[0]),identity(1,'camera/2')).collected,1);
  const other={...crop(source[1]),info:{...source[1],size:32},input:new Float32Array(3*128*128)};
  assert.equal(pool.add(other,identity(2,'camera/2')).collected,1);
  assert.deepEqual(pool.status().phaseCounts,[0,1,0,0]);
});

test('malformed captures, identities and unsupported transport profiles cannot enter the pool',()=>{
  const pool=new LivePhaseBurst(),source=frames(),item=crop(source[0]);
  for(const change of [{gridBlock:0},{pixelsPerCell:2},{phases:8},{permute:false},{size:256},{colors:4},{phase:4}]) {
    assert.equal(isTransportData({...item.info,...change}),false);
    assert.throws(()=>pool.add({...item,info:{...item.info,...change}},identity(1)),/Invalid/);
  }
  for(const value of [NaN,Infinity,1.6]) {
    const bad=crop(source[0]);bad.input[0]=value;
    assert.throws(()=>pool.add(bad,identity(1)),/Invalid/);
  }
  assert.throws(()=>pool.add(item,{context:'x',mediaTime:-1,presentedFrames:1}),/identity/);
  assert.throws(()=>pool.miss({context:'x',mediaTime:1,presentedFrames:0}),/identity/);
  assert.equal(pool.status().collected,0);
});

test('logical neural scores recover exact bytes for any phase, CRC failure never returns bytes',()=>{
  // Test-only ideal logits verify the model/codec interface, not model quality.
  for(const frame of frames()) {
    const cells=frame.size**2,logits=new Float32Array(16*cells).fill(-12);
    frame.symbols.forEach((physical,cell)=>{logits[paletteMap(physical,frame)*cells+cell]=12;});
    const result=recoverCorrectedPayload(logits,frame);
    assert.equal(result.ok,true,result.error);assert.deepEqual(result.chunk,PAYLOAD);
    assert.equal(result.stats.rawCRCValid,1);
  }
  const invalid=recoverCorrectedPayload(new Float32Array(16*64*64),frames()[0]);
  assert.equal(invalid.ok,false);assert.equal(invalid.chunk,undefined);
});




test('Neural-QR geometry alone never claims packet identity, while accepted-packet reset drops all old phases',()=>{
  const pool=new LivePhaseBurst(),a=frames(),b=frames(encodePacket({kind:'data',messageType:'raw',sessionId:'12'.repeat(16),messageId:'34'.repeat(8),index:1,data:new Uint8Array(500)}));
  pool.add(crop(a[0]),identity(1));pool.add(crop(b[1]),identity(2));
  assert.equal(pool.status().collected,2); // Neither payload is known before inference.
  pool.reset();assert.equal(pool.status().collected,0);
  assert.equal(pool.add(crop(b[2]),identity(3)).collected,1);
});

test('G128 pools remain bounded and lab capture budgets can request K8 directly',()=>{
  const pool=new LivePhaseBurst(),source=packetFrames(PAYLOAD,{size:128}),length=3*512*512;
  for(let i=1;i<=8;i++)pool.add({info:source[(i-1)%4],input:new Float32Array(length)},identity(i));
  assert.equal(pool.status().bufferBytes,24*1024*1024);
  assert.equal(pool.snapshot({frames:8}).inputCaptures,8);
  assert.equal(pool.snapshot({frames:4}).inputCaptures,4);
  assert.throws(()=>pool.snapshot({frames:9}));
});

test('transport attempts K4, K8, then waits for four genuine fresh captures per retry',()=>{
  const pool=new LivePhaseBurst(),source=frames(),attempts=[];
  for(let i=1;i<=16;i++) {
    pool.add(crop(source[(i-1)%4],i/16),identity(i));
    const snapshot=pool.nextTransportSnapshot();
    if(snapshot)attempts.push([i,snapshot.inputCaptures,snapshot.presentedFrames]);
    assert.equal(pool.add(crop(source[0]),identity(i)).duplicateExposure,true);
    assert.equal(pool.nextTransportSnapshot(),null,'duplicate exposure must not schedule inference');
    assert.ok(pool.status().bufferBytes<=8*ONE*4);
  }
  assert.deepEqual(attempts.map(([exposure,k])=>[exposure,k]),[[4,4],[8,8],[12,8],[16,8]]);
  assert.equal(Math.min(...attempts.at(-1)[2]),9);
  assert.equal(pool.status().replacements,8);
  for(let i=17;i<=21;i++)pool.miss(identity(i));
  assert.equal(pool.nextTransportSnapshot(),null,'misses are not fresh evidence');
  // Four new captures may come from one phase; each remains bounded to two.
  for(let i=22;i<=25;i++)pool.add(crop(source[0]),identity(i));
  assert.equal(pool.nextTransportSnapshot().inputCaptures,8);
  assert.deepEqual(pool.status().phaseCounts,[2,2,2,2]);
});
