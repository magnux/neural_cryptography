import test from 'node:test';
import assert from 'node:assert/strict';
import {TurnChannel} from '../dist/js/neural-auth/channel.js';
import {TurnNotices,NOTICE_DURATION_MS} from '../dist/js/neural-auth/notices.js';
import {decodePacket,encodePacket} from '../dist/js/link/protocol.js';
import {chunkBytesForSize} from '../dist/js/neural-qr/packet.js';

function endpoint(initiator=false){
  let next=0;
  const timers=new Map(),changes=[];
  const notices=new TurnNotices({onChange:value=>changes.push(value),
    setTimer:(callback,ms)=>{assert.equal(ms,NOTICE_DURATION_MS);timers.set(++next,callback);return next;},
    clearTimer:id=>timers.delete(id)});
  const channel=new TurnChannel({initiator,outgoingSize:32,
    onChange:snapshot=>notices.update({},snapshot)});
  return {channel,notices,timers,changes,last:()=>changes.at(-1),
    timer:()=>[...timers.values()][0],close:()=>{channel.close();notices.reset();}};
}
async function prepared(channel){
  const until=performance.now()+5000;
  while(channel.snapshot().preparing&&performance.now()<until)await new Promise(resolve=>setTimeout(resolve,1));
  assert.equal(channel.snapshot().mode,'send');
}
async function quiet(endpoint,operation){
  const changes=endpoint.changes.length,timers=[...endpoint.timers],notice=endpoint.last();
  const result=await operation();
  assert.equal(endpoint.changes.length,changes,'duplicate or rejected traffic cannot replace/dismiss a notice');
  assert.deepEqual([...endpoint.timers],timers,'duplicate or rejected traffic cannot restart its timer');
  assert.equal(endpoint.last(),notice);
  return result;
}

test('multipart packet progress replaces notices while repeats stay quiet and final ACK persists',async t=>{
  const a=endpoint(true),b=endpoint();t.after(()=>{a.close();b.close();});
  const payload=Uint8Array.from({length:2*chunkBytesForSize(32)+13},(_,i)=>i&255);
  const done=a.channel.enqueue(payload);done.catch(()=>{});await prepared(a.channel);
  assert.equal(a.last().title,'Sending this round');
  const oldSendingTimer=a.timer(),start=a.channel.current();
  assert.equal((await b.channel.receive(start)).accepted,true);
  assert.equal(b.last().title,'Message start received · ACK ready');
  const oldStartTimer=b.timer();
  assert.equal((await quiet(b,()=>b.channel.receive(start))).reason,'duplicate-start');

  const ready=b.channel.current();
  assert.equal((await a.channel.receive(ready)).accepted,true);
  assert.equal(a.last().title,'ACK received · sending chunk 1/3');
  oldSendingTimer();assert.equal(a.last().title,'ACK received · sending chunk 1/3');
  const foreign=encodePacket({...decodePacket(ready),sessionId:'ff'.repeat(16)});
  assert.equal((await quiet(a,()=>a.channel.receive(foreign))).accepted,false);

  for(let index=0;index<3;index++){
    const data=a.channel.current();assert.equal(decodePacket(data).index,index);
    assert.equal((await b.channel.receive(data)).accepted,true);
    if(index<2)assert.equal(b.last().title,`Chunk ${index+1}/3 received · ACK ready`);
    else assert.match(b.last().title,/Round received/);
    if(index===0){oldStartTimer();assert.equal(b.last().title,'Chunk 1/3 received · ACK ready');}
    const ack=b.channel.current();
    assert.equal(decodePacket(ack).kind,'ack');
    const duplicate=await quiet(b,()=>b.channel.receive(data));
    assert.equal(duplicate.reason,index===2?'already-delivered':'duplicate');
    assert.equal((await quiet(a,()=>a.channel.receive(ready))).accepted,false);
    assert.equal((await a.channel.receive(ack)).accepted,true);
    if(index<2)assert.equal(a.last().title,`ACK received · sending chunk ${index+2}/3`);
    else assert.equal(a.last().title,'ACK received · round delivered');
  }
  await done;
  assert.equal(a.channel.current(),null,'the sender has no stale outgoing packet after final ACK');
  const finalAck=b.channel.current();assert.equal(decodePacket(finalAck).status,'complete');
  const repeated=await quiet(a,()=>a.channel.receive(finalAck));assert.equal(repeated.accepted,false);
  assert.deepEqual(b.channel.current(),finalAck);

  const reply=b.channel.enqueue(Uint8Array.of(7));reply.catch(()=>{});
  assert.equal(b.last().title,'Your next turn is ready');
  assert.deepEqual(b.channel.current(),finalAck,'queueing a reply must retain the final ACK for the other camera');
  assert.equal(await b.channel.nextTurn(),true);
  assert.equal(b.last().title,'Sending this round');
  assert.equal(decodePacket(b.channel.current()).kind,'start');
});
