import test from 'node:test';
import assert from 'node:assert/strict';
import {encodePacket as messagePacket,DATA_OVERHEAD} from '../dist/js/link/protocol.js';
import {gridLayout,paletteMap} from '../dist/js/grid.js';
import {encodeNeuralQR,decodeNeuralQR,decodeNeuralQRSymbols,neuralQRPacketProfile,packetCapacity,chunkBytesForSize} from '../dist/js/neural-qr/packet.js';
import {encodePacket as encodeFEC,decodePacket as decodeFEC} from '../dist/js/neural-qr/fec-packet.js';

const ids={sessionId:'12'.repeat(16),messageId:'34'.repeat(8),messageType:'raw'};
const bytes=n=>Uint8Array.from({length:n},(_,i)=>(i*17+(i>>8))&255);
const data=(n,index=0)=>messagePacket({...ids,kind:'data',index,data:bytes(n)});
const ack=()=>messagePacket({...ids,kind:'ack',base:17,bitmap:1,status:'progress',digest:'56'.repeat(32)});

function canonical(frame){
  const symbols=frame.symbols.slice();
  for(const cell of gridLayout(frame.size,16,8).cells)symbols[cell]=paletteMap(symbols[cell],frame);
  return symbols;
}
function logits(symbols){
  const output=new Float32Array(16*symbols.length).fill(-5);
  for(let cell=0;cell<symbols.length;cell++)output[symbols[cell]*symbols.length+cell]=5;
  return output;
}
function wireOf(symbols,size){
  const profile=neuralQRPacketProfile(size),wire=new Uint8Array(profile.wireBytes);
  for(let i=0;i<wire.length;i++)wire[i]=(symbols[profile.grid.cells[2*i]]<<4)|symbols[profile.grid.cells[2*i+1]];
  return wire;
}
function symbolsOf(wire,size){
  const profile=neuralQRPacketProfile(size),symbols=new Uint8Array(size*size);
  for(let i=0;i<wire.length;i++){symbols[profile.grid.cells[2*i]]=wire[i]>>>4;symbols[profile.grid.cells[2*i+1]]=wire[i]&15;}
  return symbols;
}

test('packet capacities account for pilots, outer CRC, RS, envelope and message framing',()=>{
  assert.equal(packetCapacity(16),40);
  for(const [size,capacity,chunk] of [[32,288,244],[64,1332,1288],[128,5460,5416]]){
    assert.equal(packetCapacity(size),capacity);assert.equal(chunkBytesForSize(size),chunk);
    assert.equal(capacity-DATA_OVERHEAD,chunk);
  }
  assert.throws(()=>encodeNeuralQR(ack(),{size:16}),/does not fit/);
  assert.throws(()=>chunkBytesForSize(16),/G32/);
});

for(const size of [32,64,128])test(`G${size}: every phase round-trips full payload and pilot references stay fixed`,()=>{
  const packet=data(chunkBytesForSize(size)),frames=encodeNeuralQR(packet,{size}),layout=gridLayout(size,16,8);
  assert.equal(frames.length,4);
  for(const frame of frames){
    assert.equal(frame.header,undefined);
    for(const cell of layout.pilots)assert.equal(frame.symbols[cell],layout.reference[cell]);
    const logical=canonical(frame),decoded=decodeNeuralQR(logits(logical),size);
    assert.equal(decoded.ok,true,decoded.error);assert.equal(decoded.kind,'data');assert.deepEqual(decoded.chunk,packet);
    assert.equal(decoded.stats.crcAccepted,1);assert.equal(decoded.stats.correctedBytes,0);
  }
  assert.throws(()=>encodeNeuralQR(data(chunkBytesForSize(size)+1),{size}),/does not fit/);
});

test('short control packets retain dense random padding shared across phases',()=>{
  const packet=ack(),first=encodeNeuralQR(packet,{size:32}),second=encodeNeuralQR(packet,{size:32});
  assert.notDeepEqual(first[0].symbols,second[0].symbols);
  for(const frame of first)assert.deepEqual(canonical(frame),first[0].symbols);
  assert.deepEqual(decodeNeuralQRSymbols(first[0].symbols,32).chunk,packet);
});

for(const size of [32,64,128])test(`G${size}: RS32 recovers 16 unknown byte errors in every interleaved block`,()=>{
  const packet=data(Math.min(150,chunkBytesForSize(size))),original=encodeNeuralQR(packet,{size})[0].symbols;
  const wire=wireOf(original,size),profile=neuralQRPacketProfile(size);
  for(let column=0;column<16;column++)for(let block=0;block<profile.blockCount;block++)wire[column*profile.blockCount+block]^=0x55;
  const decoded=decodeNeuralQRSymbols(symbolsOf(wire,size),size);
  assert.equal(decoded.ok,true,decoded.error);assert.deepEqual(decoded.chunk,packet);
  assert.equal(decoded.stats.correctedBytes,16*profile.blockCount);assert.equal(decoded.stats.rawCRCValid,0);
  for(let column=16;column<40;column++)wire[column*profile.blockCount]^=0x55;
  assert.equal(decodeNeuralQRSymbols(symbolsOf(wire,size),size).ok,false);
});

test('CRC-valid envelope still rejects wrong geometry/type/length and a broken embedded packet',()=>{
  const size=32,frame=encodeNeuralQR(ack(),{size})[0],wire=wireOf(frame.symbols,size),payload=decodeFEC(wire).payload;
  for(const mutate of [p=>{p[4]=2;},p=>{p[5]=2;},p=>{new DataView(p.buffer).setUint16(6,65535);},p=>{p[8+10]^=1;}]){
    const bad=payload.slice();mutate(bad);
    const decoded=decodeNeuralQRSymbols(symbolsOf(encodeFEC(bad,wire.length),size),size);
    assert.equal(decoded.ok,false);assert.equal(decoded.stage,'packet');assert.equal(decoded.stats.crcAccepted,1);
  }
});

test('mixed packet symbols are rejected; unused tail nibble and pilots do not affect payload',()=>{
  const size=32,packet=data(150),first=encodeNeuralQR(packet,{size})[0].symbols,other=encodeNeuralQR(data(150,1),{size})[0].symbols;
  const mixed=first.slice();mixed.set(other.subarray(mixed.length/2),mixed.length/2);
  assert.equal(decodeNeuralQRSymbols(mixed,size).ok,false);
  const layout=gridLayout(size,16,8),changed=first.slice();
  for(const cell of layout.pilots)changed[cell]^=15;
  changed[layout.cells.at(-1)]^=15; // G32 has an odd number of data cells.
  assert.deepEqual(decodeNeuralQRSymbols(changed,size).chunk,packet);
});

test('shape, palette, finite values and unsupported message kinds are rejected',()=>{
  assert.throws(()=>decodeNeuralQR(new Float32Array(1),32),/logits/);
  const bad=new Float32Array(16*32*32);bad[100]=NaN;assert.throws(()=>decodeNeuralQR(bad,32),/logits/);
  const symbols=new Uint8Array(32*32);symbols[0]=16;assert.throws(()=>decodeNeuralQRSymbols(symbols,32),/logical/);
  assert.throws(()=>encodeNeuralQR(messagePacket({...ids,kind:'poll',base:0}),{size:32}),/Unsupported/);
});
