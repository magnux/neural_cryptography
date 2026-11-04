import test from 'node:test';
import assert from 'node:assert/strict';
import {NQR_LAYOUT,renderNeuralQR} from '../dist/js/neural-qr/layout.js';
import {extractNeuralQR,GEOMETRY_MAX_AGE_MS} from '../dist/js/neural-qr/vision.js';
import {PALETTE,encodeSequence,renderFrame} from '../dist/js/format.js';
import {simulateCapture} from '../dist/js/vision.js';
import {gridLayout,paletteMap} from '../dist/js/grid.js';

const profile=size=>({size,colors:16,phases:4,gridBlock:8,pixelsPerCell:2});
function frame(size=64,phase=0){
  const info={...profile(size),phase,permute:true},layout=gridLayout(size,16,8);
  let state=18;const symbols=Uint8Array.from({length:size*size},(_,cell)=>{
    state=(Math.imul(state,1664525)+1013904223)>>>0;
    return layout.reference[cell]>=0?layout.reference[cell]:paletteMap(state>>>28,info);
  });
  return {...info,symbols};
}
function transform(source,rotation=0,mirror=false){
  const {width:n,height}=source;assert.equal(n,height);const data=new Uint8ClampedArray(source.data.length);
  for(let y=0;y<n;y++)for(let x=0;x<n;x++){
    let xx=mirror?n-1-x:x,yy=y;
    for(let k=0;k<rotation;k++)[xx,yy]=[n-1-yy,xx];
    data.set(source.data.subarray(4*(y*n+x),4*(y*n+x)+4),4*(yy*n+xx));
  }
  return {width:n,height:n,data};
}
function paint(image,x,y,w,h,color){
  const scale=image.width/256;
  for(let yy=Math.floor(y*scale);yy<Math.ceil((y+h)*scale);yy++)for(let xx=Math.floor(x*scale);xx<Math.ceil((x+w)*scale);xx++)
    image.data.set([...color,255],4*(yy*image.width+xx));
}
function copy(image){return {...image,data:image.data.slice()};}

test('all six profiles produce bounded, calibrated 2×2-per-cell tensors without a header',()=>{
  for(const size of NQR_LAYOUT.sizes){
    const source=frame(size,3),result=extractNeuralQR(renderNeuralQR(source,size>=128?2048:1024),{profile:profile(size),now:0});
    assert.equal(result.ok,true,`${size}: ${result.error}`);assert.equal(result.info.phase,3);
    assert.deepEqual(result.dims,[1,3,2*size,2*size]);assert.equal(result.samples.length,3*size*size+96);
    assert.equal(result.input.length,12*size*size);assert.ok(result.input.every(Number.isFinite));
    assert.ok(result.input.every(v=>v>=-1.5&&v<=1.5));assert.equal(result.info.id,undefined);assert.equal(result.info.header,undefined);
    for(const cell of [0,Math.floor(size*size*.3),size*size-1])
      for(let k=0;k<3;k++)assert.ok(Math.abs(result.samples[3*cell+k]-PALETTE[source.symbols[cell]][k])<1);
  }
});

test('all rotations and reflected rotations restore the same canonical raster and phase',()=>{
  const original=renderNeuralQR(frame(16,2),512),canonical=extractNeuralQR(original,{profile:profile(16),now:0});
  for(const mirror of [false,true])for(let rotation=0;rotation<4;rotation++){
    const result=extractNeuralQR(transform(original,rotation,mirror),{profile:profile(16),now:0});
    assert.equal(result.ok,true,`rotation ${rotation} mirror ${mirror}: ${result.error}`);
    assert.equal(result.info.phase,2);
    assert.ok(result.input.every((v,i)=>Math.abs(v-canonical.input[i])<1e-5));
  }
});

test('perspective and tint retain geometry and phase without relying on payload recovery',()=>{
  for(let phase=0;phase<4;phase++){
    const image=simulateCapture(renderNeuralQR(frame(64,phase)),{preset:'perspective',noise:3,blur:.3});
    const result=extractNeuralQR(image,{profile:profile(64),now:0});
    assert.equal(result.ok,true,result.error);assert.equal(result.info.phase,phase);
    assert.ok(result.stats.phaseMargin>30);assert.ok(result.stats.colorRMSE<5);
  }
});

test('phase transitions, occluded orientation, and collapsed color references never yield inputs',()=>{
  const image=renderNeuralQR(frame(16,0),512),transition=copy(image);
  for(const [i,[x,y]] of NQR_LAYOUT.phasePairs[1].entries())paint(transition,x-10,y-10,20,20,PALETTE[(1+i)%4]);
  const mixed=extractNeuralQR(transition,{profile:profile(16),now:0});
  assert.equal(mixed.ok,false);assert.equal(mixed.stage,'phase');assert.equal(mixed.input,undefined);
  assert.equal(mixed.tentativeCorners.length,4);
  const occluded=copy(image);paint(occluded,208,210,40,10,[248,248,248]);paint(occluded,210,220,8,28,[248,248,248]);
  const missing=extractNeuralQR(occluded,{profile:profile(16),now:0});
  assert.equal(missing.ok,false);assert.equal(missing.input,undefined);
  const grayscale=copy(image);
  for(let i=0;i<grayscale.data.length;i+=4){const v=.2126*grayscale.data[i]+.7152*grayscale.data[i+1]+.0722*grayscale.data[i+2];grayscale.data[i]=grayscale.data[i+1]=grayscale.data[i+2]=v;}
  const collapsed=extractNeuralQR(grayscale,{profile:profile(16),now:0});
  assert.equal(collapsed.ok,false);assert.equal(collapsed.input,undefined);
});

test('cached geometry reads fresh phases and expires from full detection, not successful reuse',()=>{
  const a=extractNeuralQR(renderNeuralQR(frame(16,0),512),{profile:profile(16),now:10});
  assert.equal(a.stats.geometrySource,'detected');
  const b=extractNeuralQR(renderNeuralQR(frame(16,1),512),{profile:profile(16),previousGeometry:a.geometry,now:100});
  assert.equal(b.info.phase,1);assert.equal(b.stats.geometrySource,'previous');assert.equal(b.geometry.acquiredAt,10);
  const c=extractNeuralQR(renderNeuralQR(frame(16,2),512),{profile:profile(16),previousGeometry:b.geometry,now:10+GEOMETRY_MAX_AGE_MS});
  assert.equal(c.info.phase,2);assert.equal(c.stats.geometrySource,'detected');
  const d=extractNeuralQR(transform(renderNeuralQR(frame(16,3),512),1),{profile:profile(16),previousGeometry:c.geometry,now:450});
  assert.equal(d.ok,true);assert.equal(d.info.phase,3);assert.equal(d.stats.geometrySource,'detected');
  const bad=copy(renderNeuralQR(frame(16,0),512));
  for(const [x,y] of NQR_LAYOUT.phasePairs[0])paint(bad,x-10,y-10,20,20,[128,128,128]);
  const e=extractNeuralQR(bad,{profile:profile(16),previousGeometry:d.geometry,now:460});
  assert.equal(e.ok,false);assert.equal(e.input,undefined);
});

test('probe skips tensor work; arbitrary old matrices are not accepted through a fallback reader',()=>{
  const result=extractNeuralQR(renderNeuralQR(frame(16,1),512),{profile:profile(16),dense:false,now:0});
  assert.equal(result.ok,true);assert.equal(result.input,undefined);assert.equal(result.samples,undefined);assert.equal(result.info.phase,1);
  const old=renderFrame(encodeSequence(new Uint8Array(10),{size:16,colors:16})[0],512);
  assert.equal(extractNeuralQR(old,{profile:profile(16),now:0}).ok,false);
  assert.equal(extractNeuralQR(old,{now:0}).stage,'profile');
});
