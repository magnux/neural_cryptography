import test from 'node:test';
import assert from 'node:assert/strict';
import {TurnChannel} from '../dist/js/neural-auth/channel.js';
import {packetFrames,unpackOptical} from '../dist/js/neural-auth/optical.js';
import {decodeNeuralQRSymbols,chunkBytesForSize} from '../dist/js/neural-qr/packet.js';
import {decodePacket,encodePacket} from '../dist/js/link/protocol.js';
import {createTurnSender,TURN_LIMITS} from '../dist/js/neural-auth/transport.js';

const flush=()=>new Promise(resolve=>setImmediate(resolve));
async function prepared(channel) {
  const deadline=performance.now()+5000;
  while(channel.snapshot().preparing&&performance.now()<deadline)await new Promise(resolve=>setTimeout(resolve,1));
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

test('an early handoff can replay the final ACK without changing the pending message',async t=>{
  const deliveredA=[],deliveredB=[];
  const a=new TurnChannel({initiator:true,now:()=>1,onMessage:value=>deliveredA.push(value)});
  const b=new TurnChannel({now:()=>1,onMessage:value=>deliveredB.push(value)});
  t.after(()=>{a.close();b.close();});
  assert.equal(a.repeatPreviousAck(),false);assert.equal(b.resumeMessage(),false);
  const first=a.enqueue(bytes(12));first.catch(()=>{});await prepared(a);
  assert.equal(a.snapshot().canRepeatAck,false);
  await b.receive(a.current());await a.receive(b.current());
  const lastData=a.current();await b.receive(lastData);const finalAck=b.current();await flush();
  // Simulate pressing Send next turn before the sender reads this final ACK.
  const reply=b.enqueue(bytes(40,9));reply.catch(()=>{});await b.nextTurn();
  const originalStart=b.current(),before=b.snapshot().outgoing;
  assert.equal((await a.receive(originalStart)).reason,'sending');
  assert.equal(b.snapshot().canRepeatAck,true);
  assert.equal(b.repeatPreviousAck(),true);assert.equal(b.snapshot().mode,'ack');
  assert.equal(b.snapshot().repeatingAck,true);assert.deepEqual(b.current(),finalAck);
  assert.equal(b.snapshot().outgoing.messageId,before.messageId);
  assert.equal(b.snapshot().outgoing.deadline,before.deadline);
  assert.equal((await a.receive(b.current())).delivered,true);await first;
  assert.equal(b.resumeMessage(),true);assert.deepEqual(b.current(),originalStart);
  assert.equal((await a.receive(b.current())).accepted,true);const ready=a.current();
  // A delayed, matching START ACK also ends replay automatically.
  b.repeatPreviousAck();assert.equal(b.snapshot().repeatingAck,true);
  assert.equal((await b.receive(ready)).accepted,true);
  assert.equal(b.snapshot().repeatingAck,false);assert.equal(b.snapshot().canRepeatAck,false);
  assert.equal(b.repeatPreviousAck(),false);assert.equal(b.resumeMessage(),false);
  assert.equal(decodePacket(b.current()).kind,'data');
  await transfer(b,a);await reply;
  assert.deepEqual(deliveredA,[bytes(40,9)]);assert.deepEqual(deliveredB,[bytes(12)]);
});

test('stopping an ACK replay clears the retained display and pending sender',async()=>{
  const a=new TurnChannel({initiator:true,now:()=>1}),b=new TurnChannel({now:()=>1});
  const first=a.enqueue(bytes(1));await prepared(a);await transfer(a,b);await first;
  const reply=b.enqueue(bytes(1));const rejected=assert.rejects(reply,/stopped/);await b.nextTurn();
  assert.equal(b.repeatPreviousAck(),true);b.close();await rejected;
  assert.equal(b.current(),null);assert.equal(b.snapshot().canRepeatAck,false);
  assert.equal(b.snapshot().repeatingAck,false);assert.equal(b.snapshot().outgoing,null);
  assert.equal(b.repeatPreviousAck(),false);assert.equal(b.resumeMessage(),false);a.close();
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

test('dense protected data and ACKs use four-phase Neural-QR without QR headers',async()=>{
  const sender=await createTurnSender(bytes(1),{now:0}),start=sender.next(1);
  const dataFrames=packetFrames(start),otherFrames=packetFrames(start);
  assert.equal(dataFrames.length,4);
  for(const frame of dataFrames){assert.equal(frame.holdMs,150);assert.equal(frame.header,undefined);}
  const decoded=decodeNeuralQRSymbols(dataFrames[0].symbols,64);
  assert.equal(decoded.ok,true,decoded.error);assert.deepEqual(unpackOptical(decoded.chunk,'data'),start);
  assert.notDeepEqual(dataFrames[0].symbols,otherFrames[0].symbols);
  const receiver=new TurnChannel({now:()=>1});await receiver.receive(start);
  const ack=receiver.current(),ackFrames=packetFrames(ack),ackDecoded=decodeNeuralQRSymbols(ackFrames[0].symbols,32);
  assert.equal(ackFrames.length,4);assert.equal(ackFrames[0].size,32);
  assert.equal(ackDecoded.ok,true,ackDecoded.error);assert.deepEqual(unpackOptical(ackDecoded.chunk,'ack'),ack);
  assert.equal(sender.receive(ackDecoded.chunk,2).accepted,true);receiver.close();
});

test('optical packet validation rejects wrong kinds, damaged CRCs and invalid phase timing',async()=>{
  const sender=await createTurnSender(bytes(20),{now:0}),start=sender.next(1);
  assert.deepEqual(unpackOptical(start,'data'),start);
  assert.throws(()=>unpackOptical(start,'ack'),/kind/);
  assert.throws(()=>unpackOptical(start.subarray(1),'data'));
  const corrupt=start.slice();corrupt[15]^=1;assert.throws(()=>unpackOptical(corrupt,'data'),/CRC|checksum/i);
  assert.throws(()=>packetFrames(start,{phaseMs:49}),/Phase/);
  assert.throws(()=>packetFrames(start,{phaseMs:95.5}),/Phase/);
});

test('outgoing geometry is independent per device and changes only between messages',async()=>{
  const pc=new TurnChannel({initiator:true,outgoingSize:128,now:()=>1});
  const phone=new TurnChannel({outgoingSize:64,now:()=>1});
  assert.equal(pc.snapshot().outgoingSize,128);assert.equal(phone.snapshot().outgoingSize,64);
  const done=pc.enqueue(bytes(6000));await prepared(pc);
  assert.equal(pc.snapshot().outgoing.chunkBytes,chunkBytesForSize(128));
  assert.throws(()=>pc.setOutgoingSize(64),/between messages/);
  await transfer(pc,phone);await done;
  pc.setOutgoingSize(64);assert.equal(pc.snapshot().outgoingSize,64);
  phone.setOutgoingSize(32);assert.equal(phone.snapshot().outgoingSize,32);
  const reply=phone.enqueue(bytes(500));await phone.nextTurn();
  assert.equal(phone.snapshot().outgoing.chunkBytes,chunkBytesForSize(32));
  await transfer(phone,pc);await reply;
  assert.throws(()=>phone.setOutgoingSize(16),/G32/);
  pc.close();phone.close();assert.throws(()=>phone.setOutgoingSize(64),/closed/);
});
