import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {rsEncode} from '../dist/js/fec.js';
import {crc32} from '../dist/js/format.js';
import {packetLayout,encodePacket,decodePacket,replayCase} from '../../../scripts/lib/corrector-rs-replay.mjs';

const root=fileURLToPath(new URL('../../..',import.meta.url));
const fixture=n=>Uint8Array.from({length:n},(_,i)=>(i*71+(i>>>2)*13+9)&255);
const request=(id,expected,predicted)=>JSON.stringify({id,expected:Buffer.from(expected).toString('base64'),predicted:Buffer.from(predicted).toString('base64')});

test('replay accounts for RS and CRC overhead inside the original cell capacity',()=>{
  for(const [wire,blocks,payload] of [[84,1,48],[1568,7,1340],[6272,25,5468]]){
    const layout=packetLayout(wire);
    assert.equal(layout.blockCount,blocks);assert.equal(layout.payloadBytes,payload);
    assert.equal(layout.blockLengths.reduce((a,b)=>a+b),wire);
    assert.ok(layout.blockLengths.every(n=>n>32&&n<=255));
    assert.ok(Math.max(...layout.blockLengths)-Math.min(...layout.blockLengths)<=1);
    assert.equal(layout.payloadBytes+layout.parityBytes+layout.crcBytes,wire);
  }
  assert.equal(packetLayout(12).eligible,0);
  const result=replayCase(fixture(12),fixture(12));
  assert.equal(result.eligible,0);assert.equal(result.payloadBytes,0);assert.equal(result.crcAccepted,0);
});

test('clean packets round trip for shortened and full blocks through the canonical codec',()=>{
  for(const parity of [2,16,32,64])for(const wire of [84,255,256,1568,6272]){
    const layout=packetLayout(wire,parity);if(!layout.eligible)continue;
    const payload=fixture(layout.payloadBytes),wireBytes=encodePacket(payload,wire,parity);
    assert.equal(wireBytes.length,wire);
    const result=decodePacket(wireBytes,parity);
    assert.equal(result.rawCRCValid,1);assert.equal(result.rsDecoded,1);assert.equal(result.crcAccepted,1);
    assert.equal(result.correctedBytes,0);assert.deepEqual(result.payload,payload);
    const replay=replayCase(fixture(wire),fixture(wire),{parity});
    assert.equal(replay.exact,1);assert.equal(replay.wrongAccepted,0);assert.equal(replay.metric,'recorded-error-replay');
  }
});

test('every position including packet CRC and parity suffers replay errors and can be corrected',()=>{
  const expected=fixture(84),snapshot=expected.slice();
  for(let position=0;position<expected.length;position++){
    const predicted=expected.slice();predicted[position]^=0x91;const before=predicted.slice();
    const result=replayCase(expected,predicted);
    assert.equal(result.byteErrors,1);assert.equal(result.correctedBytes,1);assert.equal(result.exact,1);
    assert.equal(result.rsDecoded,1);assert.equal(result.crcAccepted,1);assert.equal(result.wrongAccepted,0);
    assert.deepEqual(predicted,before);assert.deepEqual(expected,snapshot);
  }
  // Last byte of one shortened block is parity. Raw payload CRC stays valid,
  // while RS nevertheless sees and corrects the error.
  const predicted=expected.slice();predicted[83]^=7;const result=replayCase(expected,predicted);
  assert.equal(result.rawCRCValid,1);assert.equal(result.correctedBytes,1);
});

test('unknown byte errors are corrected at the per-block bound without ground-truth erasures',()=>{
  for(const parity of [2,16,32,64]){
    const expected=fixture(255),predicted=expected.slice();
    for(let i=0;i<parity/2;i++)predicted[(i*29)%255]^=(i*3+1)&255;
    const result=replayCase(expected,predicted,{parity});
    assert.equal(result.correctedBytes,parity/2);assert.equal(result.exact,1);
  }
});

test('round-robin interleaving spreads a contiguous error burst across three blocks',()=>{
  const expected=fixture(600),predicted=expected.slice();
  // Three 200-byte blocks; 48 consecutive wire errors become 16 per block.
  for(let i=0;i<48;i++)predicted[i]^=17+i;
  const result=replayCase(expected,predicted);
  assert.equal(result.blockCount,3);assert.equal(result.byteErrors,48);
  assert.equal(result.correctedBytes,48);assert.equal(result.exact,1);
});

test('dense over-budget corruption is rejected rather than reported as recovery',()=>{
  const expected=fixture(255),predicted=expected.slice();
  for(let i=0;i<80;i++)predicted[i]^=91+i;
  const first=replayCase(expected,predicted),again=replayCase(expected,predicted);
  assert.deepEqual(first,again);assert.equal(first.crcAccepted,0);assert.equal(first.exact,0);
  assert.equal(first.wrongAccepted,0);assert.equal(first.rsDecoded,0);
});

test('a valid RS codeword with an invalid CRC fails the independent integrity check',()=>{
  const layout=packetLayout(84),data=new Uint8Array(layout.payloadBytes+4);
  data.set(fixture(layout.payloadBytes));
  new DataView(data.buffer).setUint32(layout.payloadBytes,crc32(data.subarray(0,layout.payloadBytes))^1,false);
  const received=rsEncode(data,32),result=decodePacket(received);
  assert.equal(result.rsDecoded,1);assert.equal(result.correctedBytes,0);
  assert.equal(result.rawCRCValid,0);assert.equal(result.crcAccepted,0);assert.equal(result.payload,null);
});

test('replay validates dimensions, types and parity instead of inventing capacity',()=>{
  for(const parity of [0,1,3,65,Infinity,NaN])assert.throws(()=>packetLayout(84,parity),/parity/);
  for(const capacity of [0,-1,32769,1.1,Infinity])assert.throws(()=>packetLayout(capacity),/capacity/);
  assert.throws(()=>replayCase(fixture(84),fixture(85)),/lengths differ/);
  assert.throws(()=>replayCase([1,2],[1,2]),/Uint8Array/);
  assert.throws(()=>encodePacket(fixture(1),12),/too small/);
  assert.throws(()=>encodePacket(fixture(49),84),/48 bytes/);
});

test('JSONL CLI produces independent ordered results and accepts a final line without newline',()=>{
  const expected=fixture(84),predicted=expected.slice();predicted[4]^=0xff;
  const input=request('clean',expected,expected)+'\n'+request('one',expected,predicted);
  const child=spawnSync(process.execPath,['scripts/corrector-rs-replay.mjs','--parity','32'],{cwd:root,input,encoding:'utf8'});
  assert.equal(child.status,0,child.stderr);assert.equal(child.stderr,'');
  const rows=child.stdout.trim().split('\n').map(JSON.parse);
  assert.deepEqual(rows.map(r=>r.id),['clean','one']);
  assert.deepEqual(rows.map(r=>r.correctedBytes),[0,1]);assert.ok(rows.every(r=>r.exact===1));
});

test('JSONL CLI rejects malformed or overlong input with an actionable line error',()=>{
  const valid=Buffer.from(fixture(84)).toString('base64');
  for(const input of [JSON.stringify({expected:'AA=',predicted:valid}),JSON.stringify({expected:valid,predicted:valid,parity:16}),'x'.repeat(100000)]){
    const child=spawnSync(process.execPath,['scripts/corrector-rs-replay.mjs'],{cwd:root,input,encoding:'utf8'});
    assert.equal(child.status,1);assert.match(child.stderr,/RS replay failed:/);assert.equal(child.stdout,'');
  }
});
