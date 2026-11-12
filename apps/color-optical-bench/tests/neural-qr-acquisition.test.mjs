import test from 'node:test';
import assert from 'node:assert/strict';
import {NQR_LAYOUT,renderNeuralQR} from '../dist/js/neural-qr/layout.js';
import {extractNeuralQR,geometryHintMaxAgeMs} from '../dist/js/neural-qr/vision.js';
import {PALETTE,encodeSequence,renderFrame} from '../dist/js/format.js';
import {simulateCapture} from '../dist/js/vision.js';
import {gridLayout,paletteMap} from '../dist/js/grid.js';

const profile=size=>({size,colors:16,phases:4,gridBlock:8,pixelsPerCell:4});
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

test('all four optical size codes produce calibrated 4×4-per-cell tensors without a configured grid',()=>{
  for(const size of NQR_LAYOUT.sizes){
    const source=frame(size,3),result=extractNeuralQR(renderNeuralQR(source,size>=128?2048:1024),{now:0});
    assert.equal(result.ok,true,`${size}: ${result.error}`);assert.equal(result.info.phase,3);
    assert.equal(result.info.size,size);assert.equal(result.stats.detectedSize,size);assert.equal(result.info.pixelsPerCell,4);
    assert.equal(result.geometry.size,size);assert.ok(result.stats.sizeMargin>.3);
    assert.deepEqual(result.dims,[1,3,4*size,4*size]);assert.equal(result.samples.length,3*size*size+96);
    assert.equal(result.input.length,48*size*size);assert.ok(result.input.every(Number.isFinite));
    assert.ok(result.input.every(v=>v>=-1.5&&v<=1.5));assert.equal(result.info.id,undefined);assert.equal(result.info.header,undefined);
    for(const cell of [0,Math.floor(size*size*.3),size*size-1])
      for(let k=0;k<3;k++)assert.ok(Math.abs(result.samples[3*cell+k]-PALETTE[source.symbols[cell]][k])<1);
  }
});

test('size patterns are balanced, well separated and stay outside other acquisition regions',()=>{
  assert.equal(NQR_LAYOUT.version,2);assert.deepEqual(NQR_LAYOUT.sizes,[16,32,64,128]);
  for(const a of NQR_LAYOUT.sizeCodes){
    assert.equal(a.length,8);assert.equal([...a].filter(bit=>bit==='1').length,4);
    for(const b of NQR_LAYOUT.sizeCodes)if(a!==b)assert.ok([...a].filter((bit,i)=>bit!==b[i]).length>=4);
  }
  const squares=[...NQR_LAYOUT.centers.map(([x,y])=>[x-14,y-14,28,28]),
    ...NQR_LAYOUT.orientation.map(([x,y])=>[x-3,y-3,6,6]),
    ...NQR_LAYOUT.phasePairs.flat().map(([x,y])=>[x-10,y-10,20,20]),
    ...NQR_LAYOUT.referenceX.flatMap(x=>Array.from({length:16},(_,i)=>[x-8,48+10*i,16,8])),
    [NQR_LAYOUT.payload.x,NQR_LAYOUT.payload.y,NQR_LAYOUT.payload.side,NQR_LAYOUT.payload.side]];
  const side=NQR_LAYOUT.sizePatchSide;
  for(const [x,y] of NQR_LAYOUT.sizeCopies.flat()){
    const box=[x-side/2,y-side/2,side,side];
    assert.ok(box[0]>=0&&box[1]>=0&&box[0]+side<=256&&box[1]+side<=256);
    for(const [a,b,w,h] of squares)assert.ok(box[0]+side<=a||a+w<=box[0]||box[1]+side<=b||b+h<=box[1]);
  }
});

test('size decoding rejects ambiguous patches, damaged patterns and disagreeing copies before raster work',()=>{
  const original=renderNeuralQR(frame(32,0),512);
  const unclear=copy(original),[x,y]=NQR_LAYOUT.sizeCopies[0][0];
  paint(unclear,x-4,y-4,8,8,[132,132,132]);
  const ambiguous=extractNeuralQR(unclear,{now:0});
  assert.equal(ambiguous.ok,false);assert.equal(ambiguous.stage,'size');assert.equal(ambiguous.input,undefined);
  assert.equal(ambiguous.tentativeCorners.length,4);
  const damaged=copy(original);paint(damaged,x-4,y-4,8,8,[248,248,248]);
  const invalid=extractNeuralQR(damaged,{now:0});
  assert.equal(invalid.ok,false);assert.equal(invalid.stage,'size');assert.equal(invalid.input,undefined);
  const conflict=copy(original),code=NQR_LAYOUT.sizeCodes[2];
  for(const [i,[xx,yy]] of NQR_LAYOUT.sizeCopies[1].entries())paint(conflict,xx-4,yy-4,8,8,code[i]==='1'?[16,16,16]:[248,248,248]);
  const mixed=extractNeuralQR(conflict,{now:0});
  assert.equal(mixed.ok,false);assert.equal(mixed.stage,'size');assert.match(mixed.error,/disagree/);
  assert.equal(mixed.input,undefined);
});

test('expected size is checked against optical evidence and cannot override it',()=>{
  const image=renderNeuralQR(frame(64,1),512);
  const wrong=extractNeuralQR(image,{profile:profile(32),now:0});
  assert.equal(wrong.ok,false);assert.equal(wrong.stage,'profile');assert.equal(wrong.stats.detectedSize,64);
  assert.equal(wrong.stats.expectedSize,32);assert.equal(wrong.tentativeCorners.length,4);assert.equal(wrong.input,undefined);
  assert.throws(()=>renderNeuralQR({...frame(64),pixelsPerCell:2}),/4×4/);
  assert.throws(()=>renderNeuralQR({...frame(64),size:256}),/G16/);
  const previous=extractNeuralQR(image,{now:10}).geometry;
  const next=extractNeuralQR(renderNeuralQR(frame(128,2),512),{previousGeometry:previous,now:100});
  assert.equal(next.ok,true,next.error);assert.equal(next.info.size,128);assert.equal(next.info.phase,2);
  assert.equal(next.stats.geometrySource,'previous');assert.equal(next.geometry.acquiredAt,10);
  assert.deepEqual(next.dims,[1,3,512,512]);
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

test('every optical size is identified across rotations and reflections without a profile',()=>{
  for(const size of NQR_LAYOUT.sizes){
    const image=renderNeuralQR(frame(size,1),512);
    for(const mirror of [false,true])for(let rotation=0;rotation<4;rotation++){
      const result=extractNeuralQR(transform(image,rotation,mirror),{dense:false,now:0});
      assert.equal(result.ok,true,`G${size}, rotation ${rotation}, mirror ${mirror}: ${result.error}`);
      assert.equal(result.info.size,size);assert.equal(result.info.phase,1);
    }
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
  const c=extractNeuralQR(renderNeuralQR(frame(16,2),512),{profile:profile(16),previousGeometry:b.geometry,now:10+geometryHintMaxAgeMs()});
  assert.equal(c.info.phase,2);assert.equal(c.stats.geometrySource,'detected');
  const d=extractNeuralQR(transform(renderNeuralQR(frame(16,3),512),1),{profile:profile(16),previousGeometry:c.geometry,now:220});
  assert.equal(d.ok,true);assert.equal(d.info.phase,3);assert.equal(d.stats.geometrySource,'detected');
  const bad=copy(renderNeuralQR(frame(16,0),512));
  for(const [x,y] of NQR_LAYOUT.phasePairs[0])paint(bad,x-10,y-10,20,20,[128,128,128]);
  const e=extractNeuralQR(bad,{profile:profile(16),previousGeometry:d.geometry,now:230});
  assert.equal(e.ok,false);assert.equal(e.input,undefined);
});

test('probe skips tensor work; arbitrary old matrices are not accepted through a fallback reader',()=>{
  const result=extractNeuralQR(renderNeuralQR(frame(16,1),512),{profile:profile(16),dense:false,now:0});
  assert.equal(result.ok,true);assert.equal(result.input,undefined);assert.equal(result.samples,undefined);assert.equal(result.info.phase,1);
  const old=renderFrame(encodeSequence(new Uint8Array(10),{size:16,colors:16})[0],512);
  assert.equal(extractNeuralQR(old,{profile:profile(16),now:0}).ok,false);
  assert.equal(extractNeuralQR(old,{now:0}).ok,false);
});



test('local reflection retains acquisition without darkening or modifying the RGB source',()=>{
  const image=renderNeuralQR(frame(64,0),512),side=image.width;
  for(let y=Math.floor(200*side/256);y<side;y++)for(let x=Math.floor(200*side/256);x<side;x++)
    for(let k=0;k<3;k++){const i=4*(y*side+x)+k;image.data[i]=.4*image.data[i]+140;}
  const original=image.data.slice(),result=extractNeuralQR(image,{now:0,dense:false});
  assert.equal(result.ok,true,result.error);assert.equal(result.info.size,64);assert.equal(result.info.phase,0);
  assert.equal(result.stats.detectionMode,'global');assert.ok(result.stats.finderContrasts[2]>80);
  assert.ok(result.stats.finderContrasts[2]<110);assert.ok(result.stats.orientationMargin>.08);
  assert.deepEqual(image.data,original);
  const dense=extractNeuralQR(image,{now:100,previousGeometry:result.geometry});
  assert.equal(dense.ok,true,dense.error);assert.deepEqual(image.data,original);
});

test('orientation tolerates one uncertain satellite but rejects ambiguous handedness',()=>{
  const original=renderNeuralQR(frame(32,0),512),uncertain=copy(original);
  paint(uncertain,211,231,6,6,[132,132,132]);
  const useful=extractNeuralQR(uncertain,{now:0,dense:false});
  assert.equal(useful.ok,true,useful.error);assert.ok(useful.stats.orientationError>0);
  const ambiguous=copy(uncertain);paint(ambiguous,231,211,6,6,[132,132,132]);
  const rejected=extractNeuralQR(ambiguous,{now:0,dense:false});
  assert.equal(rejected.ok,false);assert.equal(rejected.stage,'orientation');assert.match(rejected.error,/ambiguous/);
  assert.equal(rejected.geometry,undefined);assert.equal(rejected.input,undefined);
});

test('geometry hints expire at exactly twice the selected reading interval',()=>{
  const image=renderNeuralQR(frame(32,0),512);
  for(const captureIntervalMs of [100,500]){
    const ttl=2*captureIntervalMs;
    assert.equal(geometryHintMaxAgeMs(captureIntervalMs),ttl);
    const a=extractNeuralQR(image,{now:0,dense:false,captureIntervalMs});
    const b=extractNeuralQR(image,{now:ttl-1,dense:false,captureIntervalMs,previousGeometry:a.geometry});
    assert.equal(b.ok,true);assert.equal(b.stats.geometrySource,'previous');assert.equal(b.geometry.acquiredAt,0);
    assert.equal(b.stats.geometryMaxAgeMs,ttl);
    const c=extractNeuralQR(image,{now:ttl,dense:false,captureIntervalMs,previousGeometry:b.geometry});
    assert.equal(c.ok,true);assert.equal(c.stats.previousGeometryTried,false);assert.equal(c.stats.geometrySource,'detected');
    assert.equal(c.geometry.acquiredAt,ttl);
  }
  for(const value of [0,49,5001,Infinity,NaN,'100'])assert.throws(()=>geometryHintMaxAgeMs(value),/interval/);
});

test('phase failure preserves only verified geometry and never renews its lifetime',()=>{
  const image=renderNeuralQR(frame(32,0),512),a=extractNeuralQR(image,{now:0,dense:false}),bad=copy(image);
  for(const [x,y] of NQR_LAYOUT.phasePairs[0])paint(bad,x-10,y-10,20,20,[128,128,128]);
  const b=extractNeuralQR(bad,{now:100,dense:false,previousGeometry:a.geometry});
  assert.equal(b.ok,false);assert.equal(b.stage,'phase');assert.equal(b.stats.geometrySource,'previous');
  assert.equal(b.geometry.acquiredAt,0);assert.equal(b.tentativeCorners.length,4);
  for(const field of ['input','samples','info'])assert.equal(b[field],undefined);
  const c=extractNeuralQR(image,{now:200,dense:false,previousGeometry:b.geometry});
  assert.equal(c.ok,true);assert.equal(c.stats.previousGeometryTried,false);assert.equal(c.stats.geometrySource,'detected');
  const first=extractNeuralQR(bad,{now:300,dense:false});
  assert.equal(first.ok,false);assert.equal(first.stage,'phase');assert.equal(first.geometry.acquiredAt,300);
  assert.equal(first.info,undefined);assert.equal(first.input,undefined);
});

test('finder contrast failure is reported distinctly from satellite orientation',()=>{
  const image=renderNeuralQR(frame(32,0),512),a=extractNeuralQR(image,{now:0,dense:false}),bad=copy(image);
  paint(bad,220,220,28,28,[248,248,248]);
  const result=extractNeuralQR(bad,{now:100,dense:false,previousGeometry:a.geometry});
  assert.equal(result.ok,false);assert.equal(result.stage,'contrast');assert.match(result.error,/Finder contrast/);
  assert.equal(result.geometry,undefined);assert.equal(result.input,undefined);
});

test('small phone screens acquire through the relaxed area gate and high-resolution fallback',()=>{
  const original=renderNeuralQR(frame(16,0),512),width=1920,height=1080;
  for(const side of [128,160]){
    const data=new Uint8ClampedArray(4*width*height);
    for(let at=0;at<data.length;at+=4)data.set([90,90,90,255],at);
    const x0=Math.floor((width-side)/2),y0=Math.floor((height-side)/2);
    for(let y=0;y<side;y++)for(let x=0;x<side;x++){
      const from=4*(Math.floor(y*512/side)*512+Math.floor(x*512/side)),to=4*((y+y0)*width+x+x0);
      data.set(original.data.subarray(from,from+4),to);
    }
    const result=extractNeuralQR({width,height,data},{now:0,dense:false});
    assert.equal(result.ok,true,result.error);assert.equal(result.info.size,16);assert.equal(result.info.phase,0);
    assert.equal(result.stats.detectionMode,side===128?'adaptive-hires':'global');
  }
});
