import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {encodePacket as encodeOLM} from '../dist/js/link/protocol.js';
import {neuralQRPacketProfile,encodeNeuralQR,decodeNeuralQRSymbols} from '../dist/js/neural-qr/packet.js';
import {encodePacket as encodeFEC} from '../dist/js/neural-qr/fec-packet.js';
import {rsEncode} from '../dist/js/fec.js';
import {renderNeuralQR} from '../dist/js/neural-qr/layout.js';
import {simulateCapture} from '../dist/js/vision.js';
import {receipt16Digest,receipt16Frames,decodeReceipt16Capture,decodeReceipt16Symbols} from '../dist/js/neural-qr/receipt16.js';

const common={sessionId:'38'.repeat(16),messageId:'2a'.repeat(8),messageType:'raw'};
const ack=overrides=>encodeOLM({...common,kind:'ack',status:'progress',base:4,bitmap:1,digest:'66'.repeat(32),...overrides});
const profile=neuralQRPacketProfile(16), cells=profile.grid.cells;
const withWire=(source,wire)=>{
  const symbols=source.slice();
  for(let i=0;i<wire.length;i++){symbols[cells[2*i]]=wire[i]>>>4;symbols[cells[2*i+1]]=wire[i]&15;}
  return symbols;
};

test('receipt binds the whole canonical ACK; stale/foreign ACKs cannot match the outstanding digest',async()=>{
  const packet=ack(),expected=Uint8Array.from(createHash('sha256')
    .update('Neural-QR/G16/ACK-receipt/v1','ascii').update(Uint8Array.of(0)).update(packet).digest());
  assert.deepEqual(await receipt16Digest(packet),expected);
  assert.notDeepEqual(expected,Uint8Array.from(createHash('sha256').update(packet).digest()));
  for(const difference of [{base:5},{bitmap:3},{status:'complete'},{digest:'33'.repeat(32)},
      {sessionId:'08'.repeat(16)},{messageId:'77'.repeat(8)}]){
    const changed=await receipt16Digest(ack(difference)),frame=receipt16Frames(changed)[0];
    assert.notDeepEqual(decodeReceipt16Symbols(frame.symbols),expected);
    assert.deepEqual(decodeReceipt16Symbols(frame.symbols),changed);
  }
  const broken=packet.slice();broken[42]^=4;
  await assert.rejects(receipt16Digest(broken),/CRC/);
  await assert.rejects(receipt16Digest(encodeOLM({...common,kind:'poll',base:0})),/ACK/);
});

test('each independently rendered G16 phase decodes from a single measured exposure',async()=>{
  const expected=await receipt16Digest(ack()), phases=receipt16Frames(expected);
  for(const frame of phases){
    const image=simulateCapture(renderNeuralQR(frame,512),{preset:'perspective',blur:.2,noise:2});
    const read=decodeReceipt16Capture(image);
    assert.equal(read.ok,true,read.error);
    assert.equal(read.info.phase,frame.phase);
    assert.deepEqual(read.digest,expected);
    assert.equal(read.stats.crcAccepted,1);
  }
  const wrong=encodeNeuralQR(ack(),{size:32})[0];
  const read=decodeReceipt16Capture(renderNeuralQR(wrong,512));
  assert.equal(read.ok,false);
  assert.equal(read.digest,null);
});

test('16 erroneous wire bytes recover; bad CRC, type or reserved bits fail even with valid parity',async()=>{
  const expected=await receipt16Digest(ack()), {symbols}=receipt16Frames(expected)[0];
  const wire=Uint8Array.from({length:profile.wireBytes},(_,i)=>(symbols[cells[2*i]]<<4)|symbols[cells[2*i+1]]);
  const damaged=wire.slice();
  for(let i=0;i<16;i++)damaged[5*i]^=0xd3;
  assert.deepEqual(decodeReceipt16Symbols(withWire(symbols,damaged)),expected);
  const message=wire.slice(0,-32);
  message[message.length-1]^=1; // wrong CRC, but compute valid parity for it
  assert.equal(decodeReceipt16Symbols(withWire(symbols,rsEncode(message,32))),null);
  for(const index of [0,3,4,profile.payloadBytes-1]){
    const foreign=wire.slice(0,profile.payloadBytes);foreign[index]^=1;
    assert.equal(decodeReceipt16Symbols(withWire(symbols,encodeFEC(foreign,profile.wireBytes,32))),null);
  }
  assert.equal(decodeNeuralQRSymbols(symbols,16).ok,false,'receipt must not be a normal NQP packet');
  assert.equal(decodeReceipt16Symbols(symbols,32),null);
});
