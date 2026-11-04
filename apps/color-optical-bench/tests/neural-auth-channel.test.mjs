import test from 'node:test';
import assert from 'node:assert/strict';
import {TurnChannel} from '../dist/js/neural-auth/channel.js';
import {packetFrames,unpackOptical,DATA_PROFILE,ACK_PROFILE} from '../dist/js/neural-auth/optical.js';
import {decodePacket,encodePacket} from '../dist/js/link/protocol.js';
import {createTurnSender,TURN_LIMITS} from '../dist/js/neural-auth/transport.js';
import {capacity,parseHeader} from '../dist/js/format.js';
import {isTransportData,isTransportACK} from '../dist/js/corrector-lab/live-burst.js';

const flush=()=>new Promise(resolve=>setImmediate(resolve));
async function prepared(channel) {
  for(let i=0;i<100&&channel.snapshot().preparing;i++)await flush();
  assert.equal(channel.snapshot().preparing,false);
  assert.equal(channel.snapshot().mode,'send');
}
function bytes(n,seed=0) {return Uint8Array.from({length:n},(_,i)=>(i*19+seed)&255);}
async function transfer(sender,receiver) {
  const packets=[];
  while(sender.snapshot().outgoing) {
    const packet=sender.current();assert.ok(packet);packets.push(packet);
    const result=await receiver.receive(packet);assert.equal(result.accepted,true,result.reason);
    assert.equal(receiver.snapshot().mode,'ack');assert.ok(receiver.current());
    const ack=receiver.current();assert.equal(decodePacket(ack).kind,'ack');
    assert.equal((await sender.receive(ack)).accepted,true);
  }
  await flush();return packets;
}

test('manual reply retains the final ACK until explicit turn handoff, then completes both directions',async()=>{
  let now=0,replyDone;const receivedA=[],receivedB=[];
  const a=new TurnChannel({initiator:true,now:()=>++now,onMessage:value=>receivedA.push(value)});
  const b=new TurnChannel({now:()=>++now,onMessage:value=>{receivedB.push(value);replyDone=b.enqueue(bytes(1700,9));}});
  const firstDone=a.enqueue(bytes(2600));await prepared(a);
  const packets=await transfer(a,b);await firstDone;
  assert.equal(a.snapshot().mode,'receive');assert.equal(b.snapshot().mode,'ack');
  assert.equal(b.snapshot().canSendNext,true);assert.equal(b.snapshot().queued,true);
  const finalAck=b.current();assert.equal(decodePacket(finalAck).status,'complete');
  for(let i=0;i<5;i++){b.tick();assert.deepEqual(b.current(),finalAck);}
  const duplicate=await b.receive(packets.at(-1));assert.equal(duplicate.reason,'already-delivered');
  await flush();assert.equal(receivedB.length,1);assert.equal(b.snapshot().receivedMessages,1);
  assert.equal(await b.nextTurn(),true);assert.equal(b.snapshot().mode,'send');
  await transfer(b,a);await replyDone;
  assert.deepEqual(receivedA,[bytes(1700,9)]);assert.deepEqual(receivedB,[bytes(2600)]);
  assert.equal(a.snapshot().sessionId,b.snapshot().sessionId);
  a.close();b.close();
});

test('lost final ACK repeats safely; stale ACKs and foreign traffic cannot advance the current turn',async()=>{
  let now=0,deliveries=0;
  const a=new TurnChannel({initiator:true,now:()=>++now});
  const b=new TurnChannel({now:()=>++now,onMessage:()=>{deliveries++;}});
  const done=a.enqueue(bytes(12));await prepared(a);
  const start=a.current();await b.receive(start);const ready=b.current();await a.receive(ready);
  const data=a.current();await b.receive(data);const finalAck=b.current();
  assert.equal((await a.receive(ready)).reason,'stale-ack');
  assert.equal((await b.receive(data)).reason,'already-delivered');
  assert.deepEqual(b.current(),finalAck);
  const altered=encodePacket({...decodePacket(finalAck),sessionId:'aa'.repeat(16)});
  assert.equal((await a.receive(altered)).accepted,false);
  assert.equal(a.snapshot().outgoing.ackedChunks,0);
  await a.receive(finalAck);await done;await flush();assert.equal(deliveries,1);
  a.close();b.close();
});

test('cancel during asynchronous sender preparation rejects work and cannot resurrect a matrix',async()=>{
  const channel=new TurnChannel({initiator:true,now:()=>0});
  const done=channel.enqueue(bytes(1024)),rejected=assert.rejects(done,/stopped/);
  assert.equal(channel.snapshot().preparing,true);
  channel.close();await rejected;await flush();
  assert.equal(channel.current(),null);assert.equal(channel.snapshot().closed,true);
  assert.equal(channel.snapshot().queued,false);assert.equal(channel.snapshot().preparing,false);
  assert.equal(await channel.nextTurn(),false);
  await assert.rejects(channel.enqueue(bytes(1)),/closed/);
});

test('cancel while verifying a complete incoming message prevents application delivery and ACK resurrection',async()=>{
  let delivered=0;const channel=new TurnChannel({onMessage:()=>{delivered++;},now:()=>1});
  const sender=await createTurnSender(bytes(50),{now:0});
  await channel.receive(sender.next(1));sender.receive(channel.current(),2);
  const completing=channel.receive(sender.next(3));channel.close();
  assert.equal((await completing).reason,'closed');await flush();
  assert.equal(delivered,0);assert.equal(channel.current(),null);
});

test('timeouts release pending sender/receiver state and reject the sender completion promise',async()=>{
  let now=0;const errors=[],a=new TurnChannel({initiator:true,now:()=>now,onError:error=>errors.push(error.message)});
  const b=new TurnChannel({now:()=>now,onError:error=>errors.push(error.message)});
  const done=a.enqueue(bytes(2048)),rejected=assert.rejects(done,/deadline/);await prepared(a);
  await b.receive(a.current());await a.receive(b.current());
  now=TURN_LIMITS.timeoutMs+1;a.tick();b.tick();await rejected;
  for(const channel of [a,b]) {
    assert.equal(channel.snapshot().closed,true);assert.equal(channel.current(),null);
    assert.equal(channel.snapshot().incoming.bufferedBytes,0);
  }
  assert.equal(errors.length,2);
});

test('enqueue is bounded, and an application handler error closes cleanly after delivery',async()=>{
  let error=null;const a=new TurnChannel({initiator:true,now:()=>1});
  const b=new TurnChannel({now:()=>1,onMessage:()=>{throw Error('application failure');},onError:e=>{error=e;}});
  const done=a.enqueue(bytes(8));done.catch(()=>{});await prepared(a);
  await assert.rejects(a.enqueue(bytes(1)),/pending/);
  const start=a.current();await b.receive(start);await a.receive(b.current());
  await b.receive(a.current());await flush();
  assert.equal(b.snapshot().closed,true);assert.match(error.message,/application failure/);
  assert.equal(b.current(),null);a.close();
  const fresh=new TurnChannel();await assert.rejects(fresh.enqueue(bytes(524289)),/size/);
  await assert.rejects(fresh.enqueue(new Uint8Array()),/size/);fresh.close();
});

test('padded data uses the exact learned profile; ACKs remain small and match live classical acceptance',async()=>{
  const sender=await createTurnSender(bytes(1),{now:0}),start=sender.next(1);
  const dataFrames=packetFrames(start),otherFrames=packetFrames(start);
  assert.equal(dataFrames.length,4);
  for(const frame of dataFrames) {
    assert.equal(frame.holdMs,95);assert.equal(isTransportData(parseHeader(frame.header)),true);
    assert.equal(frame.chunk.length,capacity(DATA_PROFILE.size,DATA_PROFILE.colors,DATA_PROFILE));
    assert.deepEqual(unpackOptical(frame.chunk,'data'),start);
  }
  assert.notDeepEqual(dataFrames[0].chunk,otherFrames[0].chunk);
  const receiver=new TurnChannel({now:()=>1});await receiver.receive(start);
  const ack=receiver.current(),ackFrames=packetFrames(ack);
  assert.equal(ackFrames.length,1);assert.equal(isTransportACK(parseHeader(ackFrames[0].header)),true);
  assert.equal(ackFrames[0].chunk.length,ack.length);assert.deepEqual(unpackOptical(ackFrames[0].chunk,'ack'),ack);
  assert.equal(ACK_PROFILE.gridBlock,0);receiver.close();
});

test('optical container rejects wrong lengths, packet kinds and damaged embedded packet checksums',async()=>{
  const sender=await createTurnSender(bytes(20),{now:0}),start=sender.next(1),frame=packetFrames(start)[0];
  assert.throws(()=>unpackOptical(start,'data'),/container/);
  assert.throws(()=>unpackOptical(frame.chunk,'ack'));
  assert.throws(()=>unpackOptical(frame.chunk.subarray(1),'data'),/container/);
  const badLength=frame.chunk.slice();new DataView(badLength.buffer).setUint16(4,65535);
  assert.throws(()=>unpackOptical(badLength,'data'),/length/);
  const corrupt=frame.chunk.slice();corrupt[15]^=1;assert.throws(()=>unpackOptical(corrupt,'data'),/CRC|checksum/i);
  assert.throws(()=>packetFrames(start,{phaseMs:49}),/Phase/);
  assert.throws(()=>packetFrames(start,{phaseMs:95.5}),/Phase/);
});
