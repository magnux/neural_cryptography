import test from 'node:test';
import assert from 'node:assert/strict';
import {capacity,transportProfile,encodeSequence,chunkToSymbols,decodeSymbols} from '../dist/js/format.js';
import {decodeScores} from '../dist/js/vision.js';

const id=Uint8Array.of(10,20,30,40,50,60,70,80);
const fixture=n=>Uint8Array.from({length:n},(_,i)=>(i*71+37)&255);
function make(options={},shortLength=null){
  const o={size:64,colors:16,gridBlock:8,parity:32,...options};
  return encodeSequence(fixture(shortLength??capacity(o.size,o.colors,o)),{...o,id})[0];
}
function flipByte(symbols,info,profile,wireByte,value=255){
  for(let bit=0;bit<8;bit++)if(value&(1<<(7-bit))){const pos=wireByte*8+bit;symbols[profile.layout.cells[Math.floor(pos/info.bits)]]^=1<<(info.bits-1-pos%info.bits);}
}
function markByte(reliability,info,wireByte,value=0){
  for(let bit=0;bit<8;bit++)reliability[Math.floor((wireByte*8+bit)/info.bits)]=value;
}
function damaged24(){
  const frame=make(),profile=transportProfile(frame.size,frame.colors,frame),symbols=frame.symbols.slice(),reliability=new Uint8Array(profile.dataCells).fill(255);
  for(let col=0;col<24;col++){const wireByte=col*profile.blocks;flipByte(symbols,frame,profile,wireByte);if(col<16)markByte(reliability,frame,wireByte);}
  return {frame,profile,symbols,reliability};
}

test('ranked erasures recover a hard-failed block and preserve accepted-byte statistics',()=>{
  const {frame,profile,symbols,reliability}=damaged24();
  // Another block succeeds with ordinary RS and is reused without erasures.
  flipByte(symbols,frame,profile,1,31);
  assert.throws(()=>decodeSymbols(symbols,frame),/checksum/);
  const beforeSymbols=symbols.slice(),beforeReliability=reliability.slice();
  const r=decodeSymbols(symbols,frame,{reliability,recovery:'erasures'});
  assert.deepEqual(r.chunk,frame.chunk);assert.equal(r.rawCRC,false);assert.equal(r.hardCRC,false);
  assert.equal(r.recovery,'erasures');assert.equal(r.correctedBytes,25);assert.equal(r.erasedBytes,16);assert.equal(r.erasureTrials,3);
  assert.deepEqual(symbols,beforeSymbols);assert.deepEqual(reliability,beforeReliability);
});

test('score margins supply reliability and report erasure-assisted recovery',()=>{
  const {frame,profile,symbols,reliability}=damaged24(),scores=new Uint8Array(profile.dataCells*frame.colors);
  for(let i=0;i<profile.dataCells;i++){
    const cell=profile.layout.cells[i],winner=symbols[cell];
    if(reliability[i]===0){scores[i*frame.colors+winner]=128;scores[i*frame.colors+frame.symbols[cell]]=127;}
    else scores[i*frame.colors+winner]=255;
  }
  assert.equal(decodeScores(frame,scores).ok,false);
  const r=decodeScores(frame,scores,{}, {recovery:'erasures'});
  assert.equal(r.ok,true,r.error);assert.deepEqual(r.chunk,frame.chunk);assert.equal(r.stats.erasedBytes,16);assert.equal(r.stats.erasureTrials,3);assert.equal(r.stats.correctedBytes,24);assert.equal(r.stats.hardCRC,false);
});

test('three-bit cell reliability reaches both byte halves before deinterleaving',()=>{
  const frame=make({colors:8}),profile=transportProfile(frame.size,frame.colors,frame),symbols=frame.symbols.slice(),reliability=new Uint8Array(profile.dataCells).fill(255);
  // All low-confidence cells straddle a byte boundary. Only the right byte is
  // in the damaged codeword; dropping that contribution makes recovery fail.
  const lowColumns=[];for(let col=1;lowColumns.length<8;col++)if((col*profile.blocks*8)%3)lowColumns.push(col);
  const columns=lowColumns.slice();for(let col=40;columns.length<20;col++)columns.push(col);
  for(const col of columns)flipByte(symbols,frame,profile,col*profile.blocks);
  for(const col of lowColumns)reliability[Math.floor(col*profile.blocks*8/3)]=0;
  assert.throws(()=>decodeSymbols(symbols,frame),/checksum/);
  const r=decodeSymbols(symbols,frame,{reliability,recovery:'erasures'});
  assert.deepEqual(r.chunk,frame.chunk);assert.equal(r.correctedBytes,20);assert.equal(r.erasedBytes,8);assert.equal(r.erasureTrials,2);
});

test('incorrect reliability cannot bypass CRC and retries stay bounded',()=>{
  const {frame,profile,symbols}=damaged24(),reliability=new Uint8Array(profile.dataCells).fill(255);
  for(let col=80;col<112;col++)markByte(reliability,frame,col*profile.blocks);
  assert.throws(()=>decodeSymbols(symbols,frame,{reliability,recovery:'erasures'}),error=>{
    assert.match(error.message,/checksum/);assert.equal(error.erasureTrials,4);assert.equal(error.erasedBytes,0);assert.equal(error.hardCRC,false);return true;
  });
});

test('valid wrong RS codewords still retry and fail the payload CRC',()=>{
  const frame=make(),profile=transportProfile(frame.size,frame.colors,frame),other=frame.chunk.slice();other[0]^=19;
  const symbols=chunkToSymbols(other,frame),reliability=new Uint8Array(profile.dataCells);
  assert.throws(()=>decodeSymbols(symbols,frame,{reliability,recovery:'erasures'}),error=>error.erasureTrials===4&&error.correctedBytes===0&&error.hardCRC===false);
});

test('successful raw and hard candidates avoid erasure retries; padding remains mandatory',()=>{
  const frame=make(),profile=transportProfile(frame.size,frame.colors,frame),reliability=new Uint8Array(profile.dataCells);
  const raw=decodeSymbols(frame.symbols,frame,{reliability,recovery:'erasures'});
  assert.equal(raw.recovery,'raw');assert.equal(raw.hardCRC,true);assert.equal(raw.erasureTrials,0);
  const damaged=frame.symbols.slice();flipByte(damaged,frame,profile,0);
  const hard=decodeSymbols(damaged,frame,{reliability,recovery:'erasures'});
  assert.equal(hard.recovery,'hard');assert.equal(hard.hardCRC,true);assert.equal(hard.erasureTrials,0);assert.equal(hard.correctedBytes,1);
  const short=make({size:17,colors:8,gridBlock:0},3),shortProfile=transportProfile(short.size,short.colors,short),extra=Uint8Array.of(...short.chunk,1);
  assert.throws(()=>decodeSymbols(chunkToSymbols(extra,short),short,{reliability:new Uint8Array(shortProfile.dataCells),recovery:'erasures'}),/checksum/);
  const tail=short.symbols.slice();tail[shortProfile.layout.cells.at(-1)]|=1;
  assert.throws(()=>decodeSymbols(tail,short,{reliability:new Uint8Array(shortProfile.dataCells),recovery:'erasures'}),error=>error.erasureTrials===0);
  assert.throws(()=>decodeSymbols(frame.symbols,frame,{reliability:new Uint8Array(1)}),/evidence/);
  assert.throws(()=>decodeSymbols(frame.symbols,frame,{reliability:new Float32Array(profile.dataCells)}),/evidence/);
  assert.throws(()=>decodeSymbols(frame.symbols,frame,{recovery:'unknown'}),/evidence/);
});
