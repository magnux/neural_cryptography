import test from 'node:test';
import assert from 'node:assert/strict';
import {capacity,transportProfile,encodeSequence,phaseFrames,parseHeader,crc32,symbolsToChunk,decodeSymbols,renderFrame,PALETTE,Assembler,sequenceKey} from '../dist/js/format.js';
import {gridLayout,paletteMap} from '../dist/js/grid.js';
import {extractCapture,decodeSamples,decodeImage,simulateCapture} from '../dist/js/vision.js';
import {probePayload,measureErrors,ExperimentReport} from '../dist/js/diagnostics.js';
const id=Uint8Array.of(1,2,3,4,5,6,7,8);
const fixture=n=>Uint8Array.from({length:n},(_,i)=>(i*71+37)&255);
function make(options={}){const o={size:64,colors:16,gridBlock:8,parity:32,phases:2,permute:true,...options};return encodeSequence(fixture(capacity(o.size,o.colors,o)),{...o,id})[0];}
const logical=f=>{const out=f.symbols.slice();for(const cell of gridLayout(f.size,f.colors,f.gridBlock).cells)out[cell]=paletteMap(out[cell],f);return out;};

test('reference cells consume grid capacity without changing its dimensions',()=>{
  for(const size of [8,17,64,128,512])for(const gridBlock of [0,4,8,16])for(const colors of [2,4,8,16]){
    const l=gridLayout(size,colors,gridBlock);assert.equal(l.cells.length+l.pilots.length,size*size);
    assert.ok(l.cells.every(c=>l.reference[c]===-1));assert.ok(l.pilots.every(c=>l.reference[c]>=0&&l.reference[c]<colors));
    if(gridBlock)assert.ok(l.pilots.length>0);
  }
});
test('all palette phases and RS profiles preserve arbitrary bytes including shortened final frames',()=>{
  for(const colors of [2,4,8,16])for(const parity of [0,16,32])for(const gridBlock of [0,4,8,16]){
    const o={size:33,colors,parity,gridBlock,phases:4,permute:true,id,maxFrames:3},cap=capacity(o.size,colors,o),bytes=fixture(cap+7),frames=encodeSequence(bytes,o),a=new Assembler();
    for(const f of phaseFrames(frames).reverse()){assert.deepEqual(parseHeader(f.header).content,{type:'raw',length:bytes.length});a.accept(f,symbolsToChunk(f.symbols,f));}
    assert.deepEqual(a.complete,bytes);
  }
});
test('only the current format and consistent transport metadata are accepted',()=>{
  const f=make();assert.equal(f.header[4],5);assert.equal(new TextDecoder().decode(f.header.subarray(0,4)),'OBM5');assert.equal(f.holdMs,4000);
  for(const field of [44,45,46,47,48,49,51,52]){
    const h=f.header.slice();h[field]=255;new DataView(h.buffer).setUint32(60,crc32(h.subarray(0,60)));assert.throws(()=>parseHeader(h));
  }
  for(const version of [1,2,3,4,6]){const h=f.header.slice();h[4]=version;assert.throws(()=>parseHeader(h),/not recognized/);}
  assert.throws(()=>encodeSequence(new Uint8Array(),{phases:3}));assert.throws(()=>capacity(8,2,{gridBlock:8,parity:32}),/too small/);
  for(const holdMs of [249,10001,1000.5,NaN])assert.throws(()=>make({holdMs}),/hold time/);
  for(const holdMs of [250,10000]){const frame=make({holdMs});assert.equal(parseHeader(frame.header).holdMs,holdMs);assert.notEqual(sequenceKey(frame),sequenceKey(f));}
});
test('permutations are bijections; reference cells remain known in every phase',()=>{
  for(const colors of [2,4,8,16])for(const phases of [2,4,8,16]){
    const fs=phaseFrames([make({colors,phases})]),l=gridLayout(64,colors,8);
    for(const f of fs){const labels=Array.from({length:colors},(_,s)=>paletteMap(s,f));assert.equal(new Set(labels).size,colors);for(const s of labels)assert.equal(paletteMap(paletteMap(s,f),f),s);for(const cell of l.pilots)assert.equal(f.symbols[cell],l.reference[cell]);}
  }
});
test('extended phases use nested palette schedules and cycle only after every color',()=>{
  const schedules={2:[0,1],4:[0,3,1,2],8:[0,7,3,4,1,6,2,5],16:[0,15,7,8,3,12,4,11,1,14,6,9,2,13,5,10]};
  for(const colors of [2,4,8,16]){
    const full=phaseFrames([make({colors,phases:16})]);
    assert.deepEqual(full.map(f=>paletteMap(0,f)),Array.from({length:16},(_,i)=>schedules[colors][i%colors]));
    for(const symbol of Array.from({length:colors},(_,i)=>i))assert.equal(new Set(full.slice(0,colors).map(f=>paletteMap(symbol,f))).size,colors);
    for(const f of full)assert.deepEqual(symbolsToChunk(f.symbols,f),f.chunk);
    const shorter=phaseFrames([make({colors,phases:8})]);assert.deepEqual(shorter.map(f=>paletteMap(0,f)),full.slice(0,8).map(f=>paletteMap(0,f)));
  }
});
test('interleaving spreads a consecutive wire-byte burst across independently correctable blocks',()=>{
  const f=make({size:64,colors:4,phases:1,permute:false}),p=transportProfile(f.size,f.colors,f),s=logical(f),count=p.blocks*f.parity/2;
  for(let byte=0;byte<count;byte++){const pos=byte*8,cell=p.layout.cells[Math.floor(pos/f.bits)];s[cell]^=1<<(f.bits-1-pos%f.bits);}
  const r=decodeSymbols(s,f);assert.equal(r.rawCRC,false);assert.equal(r.correctedBytes,count);assert.deepEqual(r.chunk,f.chunk);
});
test('CRC remains mandatory after FEC; diagnostic knowledge cannot repair a payload',()=>{
  const f=make({phases:1,permute:false}),s=logical(f),p=transportProfile(f.size,f.colors,f);
  for(const cell of p.layout.cells)s[cell]=0;
  assert.throws(()=>decodeSymbols(s,f),/checksum|RS/);
  const probe={...f,probePattern:1,probeSeed:1234};assert.throws(()=>decodeSymbols(s,probe),/checksum|RS/);
});
test('diagnostics exclude references and measure pre-FEC coded symbols',()=>{
  const options={size:64,colors:16,gridBlock:8,parity:32,phases:2,permute:true,maxFrames:1,id,probePattern:1,probeSeed:1234};
  const f=encodeSequence(probePayload(options),options)[0],s=logical(f),p=transportProfile(f.size,f.colors,f);
  assert.equal(measureErrors(f,s).cellErrors,0);s[p.layout.cells[0]]^=1;s[p.layout.pilots[0]]^=1;
  const errors=measureErrors(f,s);assert.equal(errors.cellErrors,1);assert.equal(errors.bitErrors,1);assert.equal(errors.cells,p.dataCells);
  const decoded=decodeSymbols(s,f);assert.equal(decoded.rawCRC,false);assert.deepEqual(decoded.chunk,f.chunk);
});
test('same-exposure grid follows spatial color distortion without any previous sweep',()=>{
  const f=make({size:64,phases:1,permute:false,parity:0}),capture=extractCapture(renderFrame(f),{preview:false});
  // A smooth local tint in matrix cells; outside strip colors remain unchanged.
  for(let cell=0;cell<f.size**2;cell++)for(let k=0;k<3;k++)capture.samples[cell*3+k]=.75*capture.samples[cell*3+k]+30+(k===0?40*Math.sin(Math.PI*(cell%f.size)/f.size):0);
  const border=decodeSamples(f,capture.samples,{local:false}),local=decodeSamples(f,capture.samples,{local:true});
  const errors=r=>gridLayout(f.size,f.colors,f.gridBlock).cells.reduce((n,c)=>n+(r.symbols[c]!==f.symbols[c]),0);
  assert.ok(errors(border)>0);assert.ok(errors(local)<errors(border));assert.equal(local.ok,true,local.error);
});
test('permuted phases decode through perspective and color cast, using each capture alone',()=>{
  const f=make({size:32});for(const phase of phaseFrames([f])){
    const r=decodeImage(simulateCapture(renderFrame(phase)),{preview:false});assert.equal(r.ok,true,r.error);assert.deepEqual(r.chunk,f.chunk);
  }
});
test('reports distinguish raw acceptance from corrected acceptance with bounded history',()=>{
  const report=new ExperimentReport(),info=make({phases:1,permute:false});
  for(let i=0;i<140;i++)report.accept({info,stats:{temporal:{counts:[1]}},colorComparison:{fusedLocal:{ready:true,ok:true,stats:{rawCRC:false,correctedBytes:2}}}},i*100);
  const r=report.snapshot(14000);assert.equal(r.methods.fusedLocal.rawPasses,0);assert.equal(r.methods.fusedLocal.passes,140);assert.equal(r.methods.fusedLocal.frames.length,1);assert.equal(r.history.length,120);
});
test('512-cell grid with references, RS, and a nonzero palette phase survives clean rendering',()=>{
  const f=phaseFrames([make({size:512,phases:4})])[3];
  const r=decodeImage(renderFrame(f,3328),{preview:false});assert.equal(r.ok,true,r.error);assert.deepEqual(r.chunk,f.chunk);assert.equal(f.chunk.length,90098);
});
