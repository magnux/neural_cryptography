import test from 'node:test';
import assert from 'node:assert/strict';
import {runFullHandshake} from '../dist/simulation-core.js';
import {TurnChannel} from '../dist/js/neural-auth/channel.js';
import {decodePacket} from '../dist/js/link/protocol.js';
import {chunkBytesForSize} from '../dist/js/neural-qr/packet.js';

test('a real short G64 exchange confirms 18 payload matrices and grants four turn changes', async t=>{
  // Observe the packets the real channel actually accepts. A held screen can
  // be read repeatedly; only a receipt that advances its sender counts as a trip.
  const receive=TurnChannel.prototype.receive, turns=new Map(), grants=new Set();
  t.mock.method(TurnChannel.prototype,'receive',async function(bytes){
    const packet=decodePacket(bytes), result=await receive.call(this,bytes);
    if(!result.accepted)return result;
    const id=packet.messageId;
    if(packet.kind==='start'&&!turns.has(id)){
      assert.equal(packet.chunkBytes,chunkBytesForSize(64));
      assert.ok(packet.data?.byteLength>0,'no standalone setup matrix');
      turns.set(id,{count:packet.chunkCount,bytes:packet.totalBytes,parts:new Map([[0,packet.data.length]]),confirmed:[]});
    }else if(packet.kind==='data'){
      assert.ok(packet.index>0,'chunk zero already traveled in START');
      turns.get(id).parts.set(packet.index,packet.data.length);
    }else if(packet.kind==='ack'){
      assert.notEqual(packet.status,'ready','no standalone readiness ACK');
      turns.get(id).confirmed.push(packet.base);
    }else if(packet.kind==='poll'&&result.reason==='requested'){
      assert.ok(!grants.has(id),'one handoff per reply');
      grants.add(id);
    }
    return result;
  });
  const result=await runFullHandshake({repetitions:32,optical:false,statistics:false});
  assert.equal(result.verifier.state,'verified');
  assert.equal(result.prover.state,'peer-accepted');
  assert.equal(result.finalResultAckAccepted,true);
  assert.equal(result.proofBytes,17474);
  assert.deepEqual(result.messages.map(m=>m.type),['HELLO','IDENTITY','CHALLENGE','PROOF','RESULT']);
  const messages=[...turns.values()];
  assert.deepEqual(messages.map(m=>m.confirmed.length),[1,1,1,14,1]);
  assert.deepEqual(messages.map(m=>m.bytes),result.messages.map(m=>m.bytes));
  for(const {count,bytes,parts,confirmed} of messages){
    assert.deepEqual(confirmed,Array.from({length:count},(_,i)=>i));
    assert.deepEqual([...parts.keys()],confirmed);
    assert.equal([...parts.values()].reduce((a,b)=>a+b,0),bytes);
  }
  assert.deepEqual([...grants],[...turns.keys()].slice(0,-1));
});
