import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
import {Worker} from 'node:worker_threads';
import {PALETTE,capacity,crc32,encodeSequence,parseHeader,symbolsToChunk,Assembler,renderFrame} from '../dist/js/format.js';
import {decodeImage,simulateCapture,homography,project} from '../dist/js/vision.js';

function fixture(length,seed=37){let s=seed;return Uint8Array.from({length},()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s>>>24;});}
const id=Uint8Array.of(1,2,3,4,5,6,7,8);
function frame(size=32,colors=4){const payload=fixture(capacity(size,colors));return {payload,frame:encodeSequence(payload,{size,colors,id})[0]};}
function recover(image,payload){const r=decodeImage(image,{preview:false});assert.equal(r.ok,true,r.error);assert.deepEqual(r.chunk,payload);return r;}

test('CRC32 matches the independent standard test vector',()=>{
  assert.equal(crc32(new TextEncoder().encode('123456789')),0xcbf43926);
  assert.equal(crc32(new Uint8Array()),0);
});
test('palette matches the existing Python optical utility exactly',()=>{
  assert.deepEqual(PALETTE,[[24,24,24],[240,240,240],[24,96,240],[96,240,24],[240,24,96],[24,240,168],[168,24,240],[240,168,24],[96,96,96],[168,168,168],[24,24,168],[24,168,24],[96,168,240],[168,24,24],[168,240,96],[240,96,168]]);
});
test('all palettes and odd sizes preserve bytes and out-of-order multi-frame assembly',()=>{
  for(const colors of [2,4,8,16])for(const size of [8,17,32,128,512]){
    const bytes=fixture(capacity(size,colors)*2+1),frames=encodeSequence(bytes,{size,colors,maxFrames:8,id}),assembler=new Assembler();
    for(const f of [...frames].reverse()){
      const info=parseHeader(f.header),chunk=symbolsToChunk(f.symbols,info);assembler.accept(info,chunk);assembler.accept(info,chunk);
    }
    assert.deepEqual(assembler.complete,bytes);assert.equal(assembler.parts.size,3);
  }
});
test('empty payload and maximum eight-frame sequence are unambiguous',()=>{
  const empty=encodeSequence(new Uint8Array(),{size:8,colors:2,id})[0],a=new Assembler();a.accept(parseHeader(empty.header),symbolsToChunk(empty.symbols,parseHeader(empty.header)));assert.equal(a.complete.length,0);
  const p=fixture(capacity(8,2)*8),frames=encodeSequence(p,{size:8,colors:2,maxFrames:8,id});assert.equal(frames.length,8);
  assert.throws(()=>encodeSequence(fixture(p.length+1),{size:8,colors:2,maxFrames:8,id}),/needs 9 frames/);
});
test('invalid dimensions, palettes, frame limits, and sequence IDs are rejected',()=>{
  for(const size of [7,513,1.5,NaN])assert.throws(()=>capacity(size,4));
  for(const colors of [1,3,32])assert.throws(()=>capacity(32,colors));
  for(const maxFrames of [0,9,1.5])assert.throws(()=>encodeSequence(new Uint8Array(),{maxFrames,id}));
  assert.throws(()=>encodeSequence(new Uint8Array(),{id:new Uint8Array(7)}));
});
test('header corruption, payload corruption, and truncated grids are rejected',()=>{
  const f=frame().frame,h=f.header.slice();h[24]^=1;assert.throws(()=>parseHeader(h),/checksum/);
  const symbols=f.symbols.slice();symbols[0]^=1;assert.throws(()=>symbolsToChunk(symbols,parseHeader(f.header)),/checksum/);
  assert.throws(()=>symbolsToChunk(f.symbols.subarray(1),parseHeader(f.header)),/grid/);
});
test('nonzero trailing bits in a non-byte-aligned grid are rejected',()=>{
  const f=frame(17,8).frame,s=f.symbols.slice();s[s.length-1]|=1;
  assert.throws(()=>symbolsToChunk(s,parseHeader(f.header)),/checksum/);
});
test('receiver locks onto a sequence and rejects mixed messages',()=>{
  const f=frame().frame,other=encodeSequence(f.chunk,{size:32,colors:4,id:Uint8Array.of(8,7,6,5,4,3,2,1)})[0],a=new Assembler();
  a.accept(parseHeader(f.header),f.chunk);assert.throws(()=>a.accept(parseHeader(other.header),other.chunk),/Different sequence/);
  a.reset();assert.doesNotThrow(()=>a.accept(parseHeader(other.header),other.chunk));
});
test('complete-message checksum catches a valid chunk with the wrong sequence checksum',()=>{
  const f=frame().frame,h=f.header.slice(),v=new DataView(h.buffer);v.setUint32(28,0);v.setUint32(h.length-4,crc32(h.subarray(0,h.length-4)));
  assert.throws(()=>new Assembler().accept(parseHeader(h),f.chunk),/Sequence checksum/);
});
test('homography matches four independent point correspondences',()=>{
  const src=[[0,0],[2,0],[2,2],[0,2]],dst=[[5,10],[80,12],[70,95],[-8,87]],h=homography(src,dst);
  for(let i=0;i<4;i++)assert.ok(Math.hypot(...project(h,...src[i]).map((v,k)=>v-dst[i][k]))<1e-9);
});
for(const colors of [2,4,8,16])test(`perspective and affine RGB distortion recover ${colors}-color payload exactly`,()=>{
  const f=frame(32,colors),r=recover(simulateCapture(renderFrame(f.frame),{preset:'perspective'}),f.payload);
  assert.ok(r.stats.colorRMSE<3);assert.ok(r.stats.pixelsPerCell>8);
});
for(const preset of ['rotated','mirrored'])test(`${preset} capture is oriented through a checksum-valid header`,()=>{
  const f=frame(32,16);recover(simulateCapture(renderFrame(f.frame),{preset}),f.payload);
});
test('64 and 128 cell color matrices survive the synthetic camera path',()=>{
  for(const size of [64,128]){const f=frame(size,16);recover(simulateCapture(renderFrame(f.frame)),f.payload);}
});
test('512 × 512, 16-color matrix recovers all 131072 bytes from a clean image',()=>{
  const f=frame(512,16);recover(renderFrame(f.frame,3328),f.payload);
});
test('rotation by an arbitrary angle can be decoded',()=>{
  const f=frame(32,4),source=renderFrame(f.frame,768),width=1100,height=1100,data=new Uint8ClampedArray(width*height*4);data.fill(245);
  const angle=.48,co=Math.cos(angle),si=Math.sin(angle);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const xx=Math.round((x-550)*co+(y-550)*si+384),yy=Math.round(-(x-550)*si+(y-550)*co+384),i=4*(y*width+x);data[i+3]=255;
    if(xx>=0&&xx<768&&yy>=0&&yy<768){const j=4*(yy*768+xx);data[i]=source.data[j];data[i+1]=source.data[j+1];data[i+2]=source.data[j+2];}
  }
  recover({width,height,data},f.payload);
});
test('a damaged color cell is rejected even when geometry and header remain valid',()=>{
  const f=frame(32,4),image=renderFrame(f.frame,1024),old=f.frame.symbols[0],replacement=PALETTE[(old+1)%4];
  for(let y=288;y<308;y++)for(let x=192;x<212;x++){const i=4*(y*1024+x);image.data.set([...replacement,255],i);}
  const r=decodeImage(image);assert.equal(r.ok,false);assert.equal(r.stage,'payload');assert.match(r.error,/checksum/);assert.ok(r.rectified);
});
test('occlusion, insufficient resolution, and a blank image never report success',()=>{
  const f=frame(128,16),source=renderFrame(f.frame),occluded={...source,data:source.data.slice()};
  for(let y=0;y<180;y++)for(let x=0;x<180;x++)occluded.data.set([255,255,255,255],4*(y*1024+x));
  assert.equal(decodeImage(occluded,{preview:false}).ok,false);
  assert.equal(decodeImage(simulateCapture(source,{width:320,height:253}),{preview:false}).ok,false);
  assert.equal(decodeImage({width:64,height:64,data:new Uint8ClampedArray(64*64*4).fill(255)}).ok,false);
});
test('browser decoder worker transports transferable bytes and preserves job metadata',async()=>{
  const worker=new Worker(new URL('./worker-harness.mjs',import.meta.url));
  try{
    const f=frame(),image=renderFrame(f.frame,768),response=await new Promise((resolve,reject)=>{
      worker.once('message',resolve);worker.once('error',reject);worker.postMessage({id:42,epoch:3,image},[image.data.buffer]);
    });
    assert.equal(image.data.byteLength,0);assert.equal(response.id,42);assert.equal(response.epoch,3);assert.equal(response.result.ok,true);assert.deepEqual(response.result.chunk,f.payload);assert.equal(response.result.rectified.data.length,320*320*4);
  }finally{await worker.terminate();}
});
test('static document IDs and local assets match the application bindings',()=>{
  const root=new URL('../dist/',import.meta.url),html=readFileSync(new URL('index.html',root),'utf8'),app=readFileSync(new URL('js/app.js',root),'utf8');
  const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);assert.equal(ids.length,new Set(ids).size);
  for(const match of app.matchAll(/\$\('([^']+)'\)/g))assert.ok(ids.includes(match[1]),`missing element ${match[1]}`);
  for(const match of html.matchAll(/(?:href|src)="(\.\/[^"#]+)"/g))assert.ok(existsSync(new URL(match[1],root)),`missing asset ${match[1]}`);
  assert.ok(!/https?:\/\//.test(html),'app should not load external assets');
});
