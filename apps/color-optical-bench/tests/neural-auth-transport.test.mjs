import test from 'node:test';
import assert from 'node:assert/strict';
import {createTurnSender,TurnReceiver,TURN_LIMITS} from '../dist/js/neural-auth/transport.js';
import {encodePacket,decodePacket,sha256} from '../dist/js/link/protocol.js';
import {paletteMap,gridLayout} from '../dist/js/grid.js';
import {decodeNeuralQRSymbols,chunkBytesForSize} from '../dist/js/neural-qr/packet.js';
import {packetFrames,unpackOptical} from '../dist/js/neural-auth/optical.js';

const sessionId='ab'.repeat(16),messageId='cd'.repeat(8);
const ids={sessionId,messageId,messageType:'raw'};
const payload=n=>Uint8Array.from({length:n},(_,i)=>(i*17+(i>>8))&255);
async function sender(n=2600,extra={}) {return createTurnSender(payload(n),{sessionId,messageId,now:0,...extra});}
async function start(s,r,now=1) {const value=await r.receive(s.next(now),now);assert.equal(value.accepted,true);assert.equal(s.receive(value.ack,now).accepted,true);}

for(const n of [0,1,1155,1156,1157,1200,1201,2356,2357,17474,119249])test(`stop-and-wait transfers ${n} bytes exactly with sticky ACKs`,async()=>{
  const s=await sender(n),r=new TurnReceiver(),deliveries=[];
  let now=1,trips=0;
  while(s.snapshot().state!=='delivered') {
    const bytes=s.next(now),previous=s.next(now+1);assert.deepEqual(bytes,previous);
    const current=decodePacket(bytes);
    assert.equal(current.kind,trips===0?'start':'data');
    assert.equal(current.index??0,trips++);
    const got=await r.receive(bytes,now+2);assert.equal(got.accepted,true);assert.ok(got.ack);
    assert.deepEqual(r.ack(),got.ack);
    if(got.delivery)deliveries.push(got.delivery);
    assert.equal(s.receive(got.ack,now+3).accepted,true);now+=4;
  }
  assert.equal(deliveries.length,1);assert.deepEqual(deliveries[0].bytes,payload(n));
  assert.equal(deliveries[0].digest,await sha256(payload(n)));
  const count=1+Math.ceil(Math.max(0,n-1156)/1200);
  assert.equal(trips,count);assert.equal(s.snapshot().ackedChunks,count);
  assert.equal(s.snapshot().ackedBytes,n);
  assert.equal(s.snapshot().bufferedBytes,0);assert.equal(r.snapshot().bufferedBytes,0);
  assert.equal(decodePacket(r.ack()).status,'complete');assert.equal(s.next(now),null);
});

test('lost first, intermediate and final ACKs do not duplicate delivery',async()=>{
  const s=await sender(),r=new TurnReceiver(),startBytes=s.next(1);
  const progress=await r.receive(startBytes,1);
  const retry=await r.receive(s.next(1),1);assert.equal(retry.reason,'duplicate-start');
  assert.deepEqual(retry.ack,progress.ack);assert.equal(s.receive(retry.ack,1).accepted,true);
  const first=s.next(2),got=await r.receive(first,3);
  assert.equal(got.delivery,undefined);
  const dup=await r.receive(first,4);assert.equal(dup.reason,'duplicate');assert.deepEqual(dup.ack,got.ack);
  assert.equal(s.receive(dup.ack,5).accepted,true);
  const final=s.next(9),done=await r.receive(final,10);assert.ok(done.delivery);
  const again=await r.receive(final,11);assert.equal(again.reason,'already-delivered');assert.equal(again.delivery,undefined);
  assert.deepEqual(again.ack,done.ack);assert.equal(s.receive(again.ack,12).delivered,true);
  assert.equal(r.snapshot().completedMessages,1);
});

test('sender rejects unsent, stale, foreign, incorrect-digest and premature completion ACKs',async()=>{
  const s=await sender(),r=new TurnReceiver(),initial=await r.receive(s.current.slice(),0);
  assert.equal(s.receive(initial.ack,1).reason,'unsent-ack');
  s.next(2);
  assert.equal(s.receive(encodePacket({...decodePacket(initial.ack),status:'ready',bitmap:0}),2).reason,'stale-ack');
  assert.equal(s.receive(initial.ack,3).accepted,true);
  assert.equal(s.receive(initial.ack,4).reason,'unsent-ack');
  s.next(5);assert.equal(s.receive(initial.ack,6).reason,'stale-ack');
  const expected={...ids,kind:'ack',base:1,bitmap:1,status:'progress',digest:s.meta.digest};
  for(const patch of [{sessionId:'ef'.repeat(16)},{messageId:'ab'.repeat(8)},{digest:'00'.repeat(32)},{base:0},{status:'complete'},{bitmap:0}]) {
    assert.equal(s.receive(encodePacket({...expected,...patch}),7).accepted,false);
  }
  assert.equal(s.snapshot().ackedChunks,1);
});

test('expected receipt is scoped to the displayed chunk and matches the independent receiver',async()=>{
  const s=await sender(2600),r=new TurnReceiver();
  assert.equal(s.expectedAck(),null);
  let old=null;
  for(let i=0;i<3;i++){
    const actual=(await r.receive(s.next(i),i)).ack,expected=s.expectedAck();
    assert.deepEqual(expected,actual);
    expected.fill(0);assert.deepEqual(s.expectedAck(),actual,'cannot mutate the cached bytes');
    if(old){assert.notDeepEqual(s.expectedAck(),old);assert.equal(s.receive(old,i).reason,'stale-ack');}
    assert.equal(s.receive(actual,i).accepted,true);
    assert.equal(s.expectedAck(),null,'a new chunk is not acknowledged before presentation');
    old=actual;
  }
  const timed=await sender(100,{timeoutMs:5});
  timed.next(1);assert.ok(timed.expectedAck(4));assert.equal(timed.expectedAck(5),null);
  timed.cancel();assert.equal(timed.expectedAck(),null);
});

test('receiver rejects reordered/conflicting chunks and conflicting START',async()=>{
  const s=await sender(4000),r=new TurnReceiver(),startBytes=s.next(0),initial=decodePacket(startBytes);
  await start(s,r);
  const first=s.next(2),p=decodePacket(first);
  assert.equal((await r.receive(encodePacket({...p,index:2}),3)).reason,'out-of-order');
  assert.equal((await r.receive(encodePacket({...p,index:0}),3)).reason,'invalid-chunk');
  const accepted=await r.receive(first,4);assert.equal(accepted.accepted,true);
  const bad=p.data.slice();bad[0]^=1;
  assert.equal((await r.receive(encodePacket({...p,data:bad}),5)).reason,'conflicting-chunk');
  assert.equal((await r.receive(encodePacket({...initial,digest:'00'.repeat(32)}),6)).reason,'conflicting-start');
  const badFirst=initial.data.slice();badFirst[0]^=1;
  assert.equal((await r.receive(encodePacket({...initial,data:badFirst}),7)).reason,'conflicting-start');
  const duplicate=await r.receive(startBytes,8);
  assert.equal(duplicate.reason,'duplicate-start');assert.deepEqual(duplicate.ack,accepted.ack);
  assert.equal(r.snapshot().receivedChunks,2);
});

test('singleton is verified and delivered directly from START; replay never delivers twice',async()=>{
  const s=await sender(100),r=new TurnReceiver(),bytes=s.next(1),p=decodePacket(bytes);
  assert.deepEqual(p.data,payload(100));assert.equal(p.chunkCount,1);
  const done=await r.receive(bytes,2);
  assert.deepEqual(done.delivery.bytes,payload(100));
  const ack=decodePacket(done.ack);assert.deepEqual([ack.base,ack.bitmap,ack.status],[0,1,'complete']);
  const again=await r.receive(bytes,3);assert.equal(again.reason,'already-delivered');
  assert.deepEqual(again.ack,done.ack);assert.equal(again.delivery,undefined);
  p.data[0]^=1;
  assert.equal((await r.receive(encodePacket(p),4)).reason,'conflicting-start');
  assert.equal(s.receive(done.ack,5).delivered,true);
});

for(const n of [100,2600])test(`full-message digest failure yields no final ACK for ${n} bytes`,async()=>{
  const s=await sender(n),r=new TurnReceiver();
  let now=0,got;
  do {
    const p=decodePacket(s.next(++now));p.data[0]^=1;
    got=await r.receive(encodePacket(p),++now);
    if(got.ack)s.receive(got.ack,++now);
  }while(got.ack);
  assert.equal(got.reason,'digest-mismatch');assert.equal(got.ack,null);assert.equal(got.delivery,undefined);
  assert.equal(r.ack(),null);assert.equal(r.snapshot().state,'failed');assert.equal(r.snapshot().bufferedBytes,0);
  assert.equal((await r.receive(s.next(++now),now)).reason,'terminal');
});

test('malformed and oversized input fails before state allocation',async()=>{
  const s=await sender(100),r=new TurnReceiver({maxMessageBytes:99});
  assert.equal((await r.receive(s.next(1),1)).reason,'unsupported-message');assert.equal(r.snapshot().bufferedBytes,0);
  assert.equal((await r.receive(encodePacket({...s.meta,kind:'start'}),1)).reason,'unsupported-message');
  const broken=s.next(2);broken[10]^=1;assert.equal((await r.receive(broken,2)).reason,'malformed');
  assert.equal((await r.receive(new Uint8Array(2000),3)).reason,'malformed');
  await assert.rejects(()=>sender(0,{chunkBytes:TURN_LIMITS.chunkBytes+1}),/chunk size/);
  await assert.rejects(()=>createTurnSender(new Uint8Array(TURN_LIMITS.maxMessageBytes+1)),/2 MiB/);
});

test('receiver binds first session, retains final receipts, and requires reset for a fresh session',async()=>{
  const r=new TurnReceiver();
  for(let message=0;message<2;message++) {
    const s=await sender(1,{messageId:message.toString(16).padStart(16,'0')});
    const got=await r.receive(s.next(message*10+2),message*10+3);assert.ok(got.delivery);s.receive(got.ack,message*10+4);
  }
  assert.equal(r.snapshot().completedMessages,2);
  const foreign=await sender(1,{sessionId:'ef'.repeat(16)});
  assert.equal((await r.receive(foreign.next(50),50)).reason,'foreign-session');
  r.reset();assert.equal((await r.receive(foreign.next(51),51)).accepted,true);
});

test('timeouts free buffers and reject late packets; duplicate traffic does not extend deadline',async()=>{
  const s=await sender(2600,{timeoutMs:100}),r=new TurnReceiver({timeoutMs:100});await start(s,r,1);
  const bytes=s.next(2);assert.ok(bytes);assert.equal(s.next(100),null);assert.equal(s.snapshot().state,'timed-out');
  r.tick(101);assert.equal(r.snapshot().state,'timed-out');assert.equal(r.snapshot().bufferedBytes,0);
  assert.equal((await r.receive(bytes,102)).reason,'terminal');
});

test('sender and receiver expose copies of packets, and sender snapshots caller payload',async()=>{
  const data=payload(4),s=await createTurnSender(data,{sessionId,messageId,now:0}),r=new TurnReceiver();data.fill(99);
  const one=s.next(1);one.fill(0);assert.notDeepEqual(s.next(2),one);
  const got=await r.receive(s.next(4),5);assert.deepEqual(got.delivery.bytes,payload(4));
  got.ack.fill(0);assert.notDeepEqual(r.ack(),got.ack);
});

test('data and full ACK survive the same header-free Neural-QR framing',async()=>{
  const s=await sender(),r=new TurnReceiver(),startBytes=s.next(1),ready=await r.receive(startBytes,1);s.receive(ready.ack,2);
  for(const [bytes,expected] of [[s.next(3),'data'],[ready.ack,'ack']]){
    const frames=packetFrames(bytes);assert.equal(frames.length,4);
    for(const frame of frames){
      const logical=frame.symbols.slice(),layout=gridLayout(frame.size,16,8);
      for(const cell of layout.cells)logical[cell]=paletteMap(logical[cell],frame);
      const decoded=decodeNeuralQRSymbols(logical,frame.size);
      assert.equal(decoded.ok,true,decoded.error);assert.deepEqual(unpackOptical(decoded.chunk,expected),bytes);
      assert.equal(frame.size,expected==='ack'?32:64);
    }
  }
});

for(const size of [32,64,128])test(`G${size} negotiated chunks transfer exactly at optical capacity`,async()=>{
  const chunkBytes=chunkBytesForSize(size),s=await sender(chunkBytes*2+1,{chunkBytes}),r=new TurnReceiver();
  let delivered=null,now=0;
  while(s.snapshot().state!=='delivered'){
    const bytes=s.next(++now),frame=packetFrames(bytes,{size})[0];
    const decoded=decodeNeuralQRSymbols(frame.symbols,size);assert.equal(decoded.ok,true,decoded.error);
    const received=await r.receive(decoded.chunk,++now);assert.equal(received.accepted,true,received.reason);
    if(received.delivery)delivered=received.delivery.bytes;
    const ack=decodeNeuralQRSymbols(packetFrames(received.ack)[0].symbols,32);
    assert.equal(ack.ok,true,ack.error);assert.equal(s.receive(ack.chunk,++now).accepted,true);
  }
  assert.deepEqual(delivered,payload(chunkBytes*2+1));assert.equal(s.snapshot().ackedChunks,3);
});

test('completed receipt replay returns its ACK without replacing the latest receipt',async()=>{
  const r=new TurnReceiver(),messages=[];
  for(let index=0;index<2;index++){
    const s=await sender(1200,{messageId:index.toString(16).padStart(16,'0')});
    const startBytes=s.next(1),ready=await r.receive(startBytes,1);s.receive(ready.ack,1);
    const data=s.next(2),done=await r.receive(data,2);s.receive(done.ack,2);
    messages.push({start:startBytes,data,ack:done.ack});
  }
  const latest=r.ack();assert.deepEqual(latest,messages[1].ack);
  for(const wire of [messages[0].start,messages[0].data]){
    const repeated=await r.receive(wire,3);
    assert.equal(repeated.reason,'already-delivered');assert.deepEqual(repeated.ack,messages[0].ack);
    assert.deepEqual(r.ack(),latest,'an old completed message cannot regress the displayed receipt');
  }
});
