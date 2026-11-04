import test from 'node:test';
import assert from 'node:assert/strict';
import {encodeSequence,phaseFrames,renderFrame} from '../dist/js/format.js';
import {extractCapture,decodeSamples} from '../dist/js/vision.js';

const payload=Uint8Array.from({length:512},(_,i)=>(i*73+19)&255);
const phases=phaseFrames(encodeSequence(payload,{size:64,colors:16,gridBlock:8,parity:32,
  phases:4,permute:true,holdMs:150,id:Uint8Array.of(1,2,3,4,5,6,7,8)}));
function scene(frame,{x=80,y=80,width=800,height=700}={}) {
  const source=renderFrame(frame,512),data=new Uint8ClampedArray(width*height*4).fill(255);
  for(let row=0;row<source.height;row++)data.set(source.data.subarray(row*source.width*4,(row+1)*source.width*4),((y+row)*width+x)*4);
  return {width,height,data};
}
const read=(image,previousGeometry=null)=>extractCapture(image,{preview:false,previousGeometry});
const recover=capture=>decodeSamples(capture.info,capture.samples,{recovery:'erasures'});

test('previous geometry reads the current phase, palette and payload without full detection',()=>{
  const first=read(scene(phases[0]));
  assert.equal(first.stats.geometrySource,'detected');
  assert.equal(first.info.phase,0);
  const next=read(scene(phases[2]),first.geometry);
  assert.equal(next.stats.geometrySource,'previous');
  assert.equal(next.stats.previousGeometryTried,true);
  assert.equal(next.info.phase,2);
  assert.equal(next.markers.length,0); // No fresh full-frame finder search.
  assert.notDeepEqual(next.samples,first.samples);
  assert.deepEqual(recover(next).chunk,payload);
  assert.deepEqual(next.tentativeCorners,[]);
  assert.notEqual(next.geometry.transform,first.geometry.transform);

  const tint=scene(phases[2]);
  for(let i=0;i<tint.data.length;i+=4)tint.data[i]=Math.round(tint.data[i]*.85+10);
  const fresh=read(tint,first.geometry);
  assert.equal(fresh.stats.geometrySource,'previous');
  assert.notDeepEqual(fresh.samples.subarray(64*64*3),next.samples.subarray(64*64*3));
  assert.deepEqual(recover(fresh).chunk,payload);
});

test('motion that invalidates cached header geometry falls back to current finders',()=>{
  const first=read(scene(phases[0]));
  const moved=read(scene(phases[1],{x:125,y:110}),first.geometry);
  assert.equal(moved.stats.previousGeometryTried,true);
  assert.equal(moved.stats.geometrySource,'detected');
  assert.equal(moved.info.phase,1);
  assert.notDeepEqual(moved.geometry.transform,first.geometry.transform);
  assert.deepEqual(recover(moved).chunk,payload);
});

test('an unreadable header gives orange-feedback geometry but never a validated crop',()=>{
  const first=read(scene(phases[0]));
  const badHeader=phases[0].header.slice();badHeader[45]^=1; // Deliberately do not update its CRC.
  const rejected=read(scene({...phases[0],header:badHeader}),first.geometry);
  assert.equal(rejected.stats.previousGeometryTried,true);
  assert.equal(rejected.stats.geometrySource,'none');
  assert.equal(rejected.stage,'header');
  assert.ok(rejected.markers.length>=4);
  assert.equal(rejected.tentativeCorners.length,4);
  assert.ok(rejected.tentativeCorners.every(p=>p.length===2&&p.every(Number.isFinite)));
  assert.equal(rejected.transform,undefined);
  assert.equal(rejected.geometry,undefined);
  assert.equal(rejected.info,undefined);
  assert.equal(rejected.samples,undefined);
  assert.equal(rejected.corners,undefined);
});

test('cached coordinates cannot create a lock when the current matrix is absent',()=>{
  const first=read(scene(phases[0]));
  const blank=scene(phases[0]);blank.data.fill(255);
  const rejected=read(blank,first.geometry);
  assert.equal(rejected.stats.previousGeometryTried,true);
  assert.equal(rejected.stage,'detect');
  assert.equal(rejected.geometry,undefined);
  assert.equal(rejected.info,undefined);
  assert.equal(rejected.samples,undefined);
  assert.deepEqual(rejected.tentativeCorners,[]);
});

test('dimension changes and invalid transform hints use fresh detection',()=>{
  const first=read(scene(phases[0]));
  for(const hint of [first.geometry,{...first.geometry,width:900,transform:Array(9).fill(NaN)}]) {
    const next=read(scene(phases[3],{width:900}),hint);
    assert.equal(next.stats.previousGeometryTried,false);
    assert.equal(next.stats.geometrySource,'detected');
    assert.equal(next.geometry.width,900);
    assert.equal(next.info.phase,3);
    assert.deepEqual(recover(next).chunk,payload);
  }
});
