import test from 'node:test';
import assert from 'node:assert/strict';
import {CAPTURE_SIZES,CAPTURE_PIXELS_PER_CELL,MAX_CAPTURE_BYTES,MAX_RECORDS,captureProfile,recommendedPhaseMaximum,
  createSender,senderFrames,base64ToBytes,floatsFromBytes} from '../dist/js/dataset/contract.js';
import {captureFrame,validateCapture,CaptureBuffer} from '../dist/js/dataset/capture.js';
import {captureSignature} from '../dist/js/dataset/diversity.js';
import {NQR_LAYOUT,renderNeuralQR} from '../dist/js/neural-qr/layout.js';
import {PALETTE} from '../dist/js/format.js';

test('header-free collection profiles cover six grids with one fixed color/phase contract',async()=>{
  assert.deepEqual(CAPTURE_SIZES,[8,16,32,64,128,256]);
  assert.equal(CAPTURE_PIXELS_PER_CELL,2);
  for(const size of CAPTURE_SIZES){
    const colors=16;
    const profile=captureProfile(size,colors);
    assert.equal(profile.parity,0);
    assert.equal(profile.pixelsPerCell,2);
    const sender=await createSender({...profile,seed:123,sessionId:`sender-${size}-${colors}`});
    assert.equal(sender.frames.length,4);
    const restored=await senderFrames(sender.manifest);
    assert.deepEqual(restored.map(f=>f.symbols),sender.frames.map(f=>f.symbols));
  }
  assert.throws(()=>captureProfile(12),/profile/);
  assert.throws(()=>captureProfile(64,8),/profile/);
  assert.throws(()=>captureProfile(64,4),/profile/);
  assert.throws(()=>captureProfile(64,16,8),/profile/);
});

test('pool recommendations leave room for every phase, metadata and two originals',()=>{
  for(const size of CAPTURE_SIZES)for(const phases of [1,2,4,8,16]){
    const n=recommendedPhaseMaximum(size,phases);
    const inputBytes=3*(2*size)**2*4,sampleBytes=(3*size*size+96)*4;
    const estimated=4*Math.ceil(inputBytes/3)+4*Math.ceil(sampleBytes/3)+8192;
    assert.ok(n>=1&&n<=32);
    assert.ok(n*phases<=MAX_RECORDS);
    assert.ok(n*phases*estimated+3*1024*1024<=MAX_CAPTURE_BYTES);
  }
  assert.equal(recommendedPhaseMaximum(64,4),32);
  assert.equal(recommendedPhaseMaximum(256,4),6);
  assert.throws(()=>recommendedPhaseMaximum(512,4),/profile/);
});

for(const size of CAPTURE_SIZES)test(`G${size} four-phase capture preserves calibrated 2x2-cell evidence`,async()=>{
  const sender=await createSender({...captureProfile(size),seed:456,sessionId:`multi-${size}`});
  for(const frame of sender.frames){
    const image=renderNeuralQR(frame,size>=128?2048:1024);
    const record=captureFrame(image,{profile:sender.manifest.profile,collectionCode:sender.manifest.collectionCode,signature:true,timestamp:1000+frame.phase*1000});
    assert.equal(record.ok,true,record.error);
    assert.equal(record.pixelsPerCell,2);
    assert.equal(record.info.size,size);assert.equal(record.info.phase,frame.phase);
    validateCapture(record);
    const side=2*size,input=floatsFromBytes(base64ToBytes(record.inputBase64),3*side*side);
    assert.ok(input.every(Number.isFinite));
    assert.equal(record.signature.length,3072);
    assert.ok(record.signature.every(Number.isFinite));
    // Known labels are used only for after-capture test comparisons.
    for(const [x,y]of [[0,0],[3,3],[size-3,size-3]]){
      const at=(2*y+1)*side+2*x+1,color=PALETTE[frame.symbols[y*size+x]];
      for(let channel=0;channel<3;channel++)assert.ok(
        Math.abs(input[channel*side*side+at]-(color[channel]/127.5-1))<.04,
        `G${size}, phase ${frame.phase}, cell ${x}/${y}, channel ${channel}`);
    }
  }
});

test('record validation requires current two-pixel acquisition evidence and coherent phase cues',async()=>{
  const sender=await createSender({seed:55,sessionId:'sampling'}),image=renderNeuralQR(sender.frames[0],1024);
  const options={profile:sender.manifest.profile,collectionCode:sender.manifest.collectionCode};
  const record=captureFrame(image,{...options,timestamp:1000});
  assert.equal(record.ok,true,record.error);assert.equal(record.pixelsPerCell,2);validateCapture(record);
  const {pixelsPerCell,...missingDensity}=record;
  assert.throws(()=>validateCapture(missingDensity),/record/);
  assert.throws(()=>validateCapture({...record,pixelsPerCell:4}),/record/);
  assert.throws(()=>validateCapture({...record,pixelsPerCell:3}),/record/);
  assert.throws(()=>validateCapture({...record,pixelsPerCell:null}),/record/);
  assert.throws(()=>captureFrame(image),/profile/);
  const b=new CaptureBuffer();b.add(record);b.add({...record,timestamp:1600});
  const saved=b.export({sessionId:'receiver'});
  assert.equal(saved.format,'neural-qr-captures');assert.equal(saved.records[0].pixelsPerCell,2);
  assert.equal(saved.collectionCode,sender.manifest.collectionCode);
  // A phase transition spanning the exposure must not yield a training crop.
  const phase1=renderNeuralQR(sender.frames[1],1024),scale=image.width/NQR_LAYOUT.units;
  for(const [cx,cy] of NQR_LAYOUT.phasePairs[1])
    for(let y=Math.round((cy-10)*scale);y<Math.round((cy+10)*scale);y++)
      for(let x=Math.round((cx-10)*scale);x<Math.round((cx+10)*scale);x++){
        const at=4*(y*image.width+x);image.data.set(phase1.data.subarray(at,at+4),at);
      }
  const damaged=captureFrame(image,options);
  assert.equal(damaged.ok,false);assert.equal(damaged.inputBase64,undefined);
});

test('tiny raster diversity signatures stay finite and retain RGB ordering',()=>{
  const input=new Float32Array(3*16*16);
  input.fill(-.4,0,256);input.fill(.2,256,512);input.fill(.6,512);
  const signature=captureSignature(input,8,2);
  for(let i=0;i<1024;i++)assert.deepEqual(signature.subarray(3*i,3*i+3),Float32Array.of(-.4,.2,.6));
  assert.throws(()=>captureSignature(input,8,4),/input/);
});
