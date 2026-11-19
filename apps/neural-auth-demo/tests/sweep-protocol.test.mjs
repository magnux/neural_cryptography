import test from 'node:test';
import assert from 'node:assert/strict';
import {createSweepProbe,acknowledgeSweepProbe,matchesSweepAck,newSweepRunId} from '../dist/sweep-protocol.js';
import {gridLayout,paletteMap} from '../dist/js/grid.js';
import {decodeNeuralQRSymbols,chunkBytesForSize} from '../dist/js/neural-qr/packet.js';
import {decodePacket,encodePacket} from '../dist/js/link/protocol.js';

// Optical symbol-layer roundtrip only: emulate correct phase unpermutation
// here, never in the camera or neural-inference path.
function restored(frame){
  const symbols=frame.symbols.slice();for(const i of gridLayout(frame.size,16,8).cells)symbols[i]=paletteMap(symbols[i],frame);
  const r=decodeNeuralQRSymbols(symbols,frame.size);assert.ok(r.ok,r.error);return r.chunk;
}
const settings=(size=64)=>({runId:newSweepRunId(),step:1,size,phaseMs:500,captureMs:450});

for(const size of [32,64,128])test('corrected sweep sends one full G'+size+' DATA and a G32 ACK bound to exactly those bytes',async()=>{
  const s=settings(size),probe=await createSweepProbe(s),data=restored(probe.frames[0]),p=decodePacket(data);
  assert.equal(p.data.length,chunkBytesForSize(size));
  const ack=await acknowledgeSweepProbe(data,{...s,runId:null});
  assert.equal(ack.runId,s.runId);assert.deepEqual(probe.frames.map(f=>f.phase),[0,1,2,3]);
  for(const f of probe.frames){assert.equal(f.size,size);assert.equal(f.holdMs,s.phaseMs);assert.deepEqual(restored(f),data);}
  for(const f of ack.frames){assert.equal(f.size,32);assert.equal(f.holdMs,s.phaseMs);assert.ok(matchesSweepAck(restored(f),probe));}
  // Valid OLM CRC but a different data byte. The receiver's ACK digest is
  // computed from the bytes actually read; it cannot finish the sender's step.
  p.data[p.data.length-1]^=1;
  const altered=await acknowledgeSweepProbe(encodePacket(p),s);
  assert.ok(altered);assert.equal(matchesSweepAck(restored(altered.frames[0]),probe),false);
});

test('protects against self, stale, unrelated and improperly configured sweep steps',async()=>{
  const s=settings(),probe=await createSweepProbe(s),d=restored(probe.frames[2]),ack=await acknowledgeSweepProbe(d,s);
  const bytes=restored(ack.frames[1]);assert.equal(await acknowledgeSweepProbe(bytes,s),null);assert.equal(matchesSweepAck(d,probe),false);
  for(const change of [{runId:newSweepRunId()},{size:128},{step:2},{phaseMs:450},{captureMs:400}]){
    assert.equal(await acknowledgeSweepProbe(d,{...s,...change}),null);
  }
  const retried=await createSweepProbe(s);
  assert.equal(matchesSweepAck(bytes,retried),false);
  const p=decodePacket(bytes);
  for(const change of [{base:1},{status:'progress'},{bitmap:2},{digest:'00'.repeat(32)},{sessionId:newSweepRunId()}]){
    assert.equal(matchesSweepAck(encodePacket({...p,...change}),probe),false);
  }
  d[40]^=1;bytes[20]^=1;
  assert.equal(await acknowledgeSweepProbe(d,s),null);assert.equal(matchesSweepAck(bytes,probe),false);
});
