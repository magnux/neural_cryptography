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
    assert.deepEqual(sender.expectedAck(),ack);
    assert.equal(receiver.expectedAck(),null,'displaying an ACK does not mean waiting for one');
    assert.equal((await sender.receive(ack)).accepted,true);
  }
  await flush();return packets;
}

test('manual reply requires the predecessor REQUEST before explicit turn promotion',async t=>{
  let now=0,replyDone;const receivedA=[],receivedB=[];
  const a=new TurnChannel({initiator:true,now:()=>++now,onMessage:value=>receivedA.push(value)});
  const b=new TurnChannel({now:()=>++now,onMessage:value=>{receivedB.push(value);replyDone=b.enqueue(bytes(1700,9),{expectReply:false});}});
  t.after(()=>{a.close();b.close();});
  const firstDone=a.enqueue(bytes(2600));await prepared(a);
  const packets=await transfer(a,b);await firstDone;
  assert.equal(a.snapshot().mode,'request');assert.equal(a.snapshot().expected,'data');
  assert.equal(a.expectedAck(),null,'a held REQUEST cannot accept a stale digest receipt');
  assert.equal(a.snapshot().requestedReply,true);assert.equal(b.snapshot().mode,'ack');
  assert.equal(b.snapshot().expected,'request');assert.equal(b.snapshot().waitingForRequest,true);
  assert.equal(b.snapshot().canSendNext,false);assert.equal(b.snapshot().queued,true);
  assert.equal(await b.nextTurn(),false,'manual action cannot bypass the request');
  const finalAck=b.current();assert.equal(decodePacket(finalAck).status,'complete');
  for(let i=0;i<5;i++){b.tick();assert.deepEqual(b.current(),finalAck);}
  assert.equal((await b.receive(packets.at(-1))).reason,'already-delivered');
  const request=a.current(),p=decodePacket(request);
  assert.equal(request.length,44);assert.equal(p.kind,'poll');
  assert.equal(p.messageId,decodePacket(finalAck).messageId);assert.equal(p.base,decodePacket(finalAck).base);
  assert.equal((await b.receive(request)).reason,'requested');
  assert.equal(b.snapshot().canSendNext,true);assert.equal(b.snapshot().waitingForRequest,false);
  assert.equal(await b.nextTurn(),true);assert.equal(b.snapshot().mode,'send');
  await transfer(b,a);await replyDone;
  assert.deepEqual(receivedA,[bytes(1700,9)]);assert.deepEqual(receivedB,[bytes(2600)]);
  assert.equal(b.snapshot().mode,'receive');assert.equal(b.current(),null,'final RESULT requests no next turn');
  assert.equal(a.snapshot().sessionId,b.snapshot().sessionId);
});

test('automatic handoff holds ACK, REQUEST and START through losses without elapsed-time alternation',async t=>{
  let now=0,reply;
  const deliveredA=[],deliveredB=[];
  const a=new TurnChannel({initiator:true,automaticTurns:true,outgoingSize:32,now:()=>now,onMessage:value=>deliveredA.push(value)});
  const b=new TurnChannel({automaticTurns:true,outgoingSize:32,now:()=>now,onMessage:value=>{
    deliveredB.push(value);reply=b.enqueue(bytes(600,9),{expectReply:false});reply.catch(()=>{});
  }});
  t.after(()=>{a.close();b.close();});
  const first=a.enqueue(bytes(12));first.catch(()=>{});await prepared(a);
  const finalData=a.current();assert.equal(decodePacket(finalData).chunkCount,1);await b.receive(finalData);const finalAck=b.current();await flush();
  assert.equal(b.snapshot().queued,true);assert.equal(b.snapshot().outgoing,null);
  for(const time of [3000,9000,30000]){
    now=time;a.tick();b.tick();
    assert.deepEqual(a.current(),finalData,'lost singleton ACK keeps the full inline message displayed');
    assert.deepEqual(b.current(),finalAck,'queued reply cannot displace its predecessor final ACK');
  }
  const premature=encodePacket({...decodePacket(finalAck),status:'progress'});
  assert.equal((await a.receive(premature)).reason,'stale-ack');
  assert.equal((await b.receive(finalData)).reason,'already-delivered');
  assert.equal((await a.receive(finalAck)).delivered,true);await first;
  const request=a.current();assert.equal(decodePacket(request).kind,'poll');
  for(const time of [33000,39000,60000]){
    now=time;a.tick();b.tick();
    assert.deepEqual(a.current(),request,'lost REQUEST is redisplayed unchanged');
    assert.deepEqual(b.current(),finalAck);
  }
  assert.equal((await b.receive(request)).reason,'requested');await prepared(b);
  const before=b.snapshot().outgoing;
  const earlyAck=encodePacket({kind:'ack',sessionId:before.sessionId,messageId:before.messageId,
    messageType:before.messageType,base:0,bitmap:1,status:'progress',digest:before.digest});
  assert.equal((await b.receive(earlyAck)).reason,'unsent-ack');
  const start=b.current();assert.equal(decodePacket(start).kind,'start');
  for(const time of [63000,69000,90000]){
    now=time;a.tick();b.tick();
    assert.deepEqual(a.current(),request,'the requester waits for a fresh START');
    assert.deepEqual(b.current(),start,'lost START stays displayed without an ACK slot');
    assert.equal((await b.receive(request)).reason,'duplicate-request');
    assert.equal(b.snapshot().outgoing.deadline,before.deadline);
  }
  await transfer(b,a);await reply;
  assert.deepEqual(deliveredA,[bytes(600,9)]);assert.deepEqual(deliveredB,[bytes(12)]);
  assert.equal(a.snapshot().requestedReply,false);
  const lastAck=a.current();assert.equal(decodePacket(lastAck).status,'complete');
  now+=3000;a.tick();assert.deepEqual(a.current(),lastAck);
});

test('a matching REQUEST is remembered until slow proof computation queues its reply',async t=>{
  let now=0;
  const a=new TurnChannel({initiator:true,automaticTurns:true,now:()=>now});
  const b=new TurnChannel({automaticTurns:true,now:()=>now});
  t.after(()=>{a.close();b.close();});
  const first=a.enqueue(bytes(1));await prepared(a);await transfer(a,b);await first;
  const request=a.current();assert.equal((await b.receive(request)).reason,'requested');
  assert.equal(b.snapshot().waitingForRequest,false);assert.equal(b.snapshot().expected,'request');
  assert.equal(b.current(),null);
  now=45000;b.tick();assert.equal((await b.receive(request)).reason,'duplicate-request');
  const reply=b.enqueue(bytes(20),{expectReply:false});await prepared(b);
  await transfer(b,a);await reply;assert.equal(b.current(),null);
});

test('foreign and stale REQUESTs cannot release a final ACK or authorize another turn',async t=>{
  const a=new TurnChannel({initiator:true,automaticTurns:true,now:()=>1});
  const b=new TurnChannel({automaticTurns:true,now:()=>1});
  t.after(()=>{a.close();b.close();});
  const first=a.enqueue(bytes(20));await prepared(a);await transfer(a,b);await first;
  const request=a.current(),p=decodePacket(request),ack=b.current();
  const reply=b.enqueue(bytes(30));
  for(const patch of [{sessionId:'ab'.repeat(16)},{messageId:'cd'.repeat(8)},{base:p.base+1},{messageType:'hello'}]){
    assert.equal((await b.receive(encodePacket({...p,...patch}))).accepted,false);
    assert.deepEqual(b.current(),ack);assert.equal(b.snapshot().outgoing,null);
  }
  await b.receive(request);await prepared(b);await transfer(b,a);await reply;
  const second=a.enqueue(bytes(10),{expectReply:false});
  await a.receive(b.current());await prepared(a);await transfer(a,b);await second;
  const latestAck=b.current();assert.notEqual(decodePacket(latestAck).messageId,p.messageId);
  assert.equal((await b.receive(request)).reason,'stale-request');assert.deepEqual(b.current(),latestAck);
  assert.equal(b.snapshot().waitingForRequest,true);
});

test('completed START and DATA duplicates cannot replace a pending REQUEST or newer final ACK',async t=>{
  const a=new TurnChannel({initiator:true,automaticTurns:true,now:()=>1});
  const b=new TurnChannel({automaticTurns:true,now:()=>1});
  t.after(()=>{a.close();b.close();});
  const first=a.enqueue(bytes(1500));await prepared(a);const oldA=await transfer(a,b);await first;
  const reply=b.enqueue(bytes(1500));await b.receive(a.current());await prepared(b);
  const oldB=await transfer(b,a);await reply;
  const second=a.enqueue(bytes(20));await a.receive(b.current());await prepared(a);await transfer(a,b);await second;
  const request=a.current(),ack=b.current();
  for(const wire of oldB){assert.equal((await a.receive(wire)).reason,'already-delivered');assert.deepEqual(a.current(),request);}
  for(const wire of oldA){
    await b.receive(wire);assert.deepEqual(b.current(),ack);
  }
  assert.equal(a.snapshot().requestedReply,true);assert.equal(b.snapshot().waitingForRequest,true);
});

test('delayed duplicate DATA cannot regress the currently displayed progress ACK',async t=>{
  const a=new TurnChannel({initiator:true,outgoingSize:32,now:()=>1}),b=new TurnChannel({now:()=>1});
  t.after(()=>{a.close();b.close();});
  const done=a.enqueue(bytes(900));done.catch(()=>{});await prepared(a);
  await b.receive(a.current());await a.receive(b.current());
  const old=a.current();await b.receive(old);await a.receive(b.current());
  await b.receive(a.current());const latest=b.current();assert.equal(decodePacket(latest).base,2);
  assert.equal((await b.receive(old)).reason,'duplicate');assert.deepEqual(b.current(),latest);
});

test('handoff deadlines bound lost requests and slow proof queues; duplicates cannot extend them',async()=>{
  for(const requested of [false,true]){
    let now=0;const errors=[];
    const a=new TurnChannel({initiator:true,automaticTurns:true,now:()=>now});
    const b=new TurnChannel({automaticTurns:true,now:()=>now,onError:error=>errors.push(error.message)});
    const first=a.enqueue(bytes(1));await prepared(a);await transfer(a,b);await first;
    const request=a.current();if(requested)await b.receive(request);
    const reply=requested?null:b.enqueue(bytes(1));
    const rejected=reply?assert.rejects(reply,/deadline/):Promise.resolve();
    now=TURN_LIMITS.timeoutMs-1;if(requested)await b.receive(request);b.tick();assert.equal(b.snapshot().closed,false);
    now++;b.tick();a.tick();await rejected;
    assert.equal(b.snapshot().closed,true);assert.equal(b.current(),null);assert.equal(a.snapshot().closed,true);
    assert.equal(errors.length,1);now+=30000;b.tick();assert.equal(errors.length,1);
  }
});

test('the total session deadline is bounded even before the initial message',()=>{
  let now=0,error;
  const channel=new TurnChannel({now:()=>now,onError:value=>{error=value;}});
  now=TURN_LIMITS.maxTimeoutMs;channel.tick();
  assert.equal(channel.snapshot().closed,true);assert.match(error.message,/Session deadline/);
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
  const completing=channel.receive(sender.next(1));channel.close();
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
  const start=a.current();await b.receive(start);await flush();
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
  const reply=phone.enqueue(bytes(500));await phone.receive(pc.current());await phone.nextTurn();
  assert.equal(phone.snapshot().outgoing.chunkBytes,chunkBytesForSize(32));
  await transfer(phone,pc);await reply;
  assert.throws(()=>phone.setOutgoingSize(16),/G32/);
  pc.close();phone.close();assert.throws(()=>phone.setOutgoingSize(64),/closed/);
});
