import test from 'node:test';
import assert from 'node:assert/strict';
import {PacketTelemetry} from '../dist/js/neural-auth/telemetry.js';
import {encodePacket} from '../dist/js/link/protocol.js';

const meta={sessionId:'11'.repeat(16),messageId:'22'.repeat(8),messageType:'raw',digest:'33'.repeat(32),
  totalBytes:9,chunkBytes:47,chunkCount:2,windowSize:1};
const start=(overrides={})=>encodePacket({...meta,kind:'start',data:Uint8Array.of(1,2,3),...overrides});
const data=(index,overrides={})=>encodePacket({...meta,kind:'data',index,data:Uint8Array.of(4,5,6,7,8,9),...overrides});
const ack=(index,status,overrides={})=>encodePacket({...meta,kind:'ack',base:index,bitmap:status==='ready'?0:1,status,...overrides});
const request=(overrides={})=>encodePacket({...meta,kind:'poll',base:0,...overrides});

test('inline START and DATA have one RTT each; redisplays preserve the first timestamp',()=>{
  const t=new PacketTelemetry({originMs:1000});
  t.displayed(start(),{grid:128,atMs:1100});t.displayed(start(),{grid:128,atMs:1400});
  assert.equal(t.accepted(ack(0,'progress'),{accepted:true},{grid:32,atMs:1600}),true);
  t.displayed(data(1),{grid:128,atMs:1700});t.displayed(data(1),{grid:128,atMs:1800});
  assert.equal(t.accepted(ack(1,'complete'),{accepted:true,delivered:true},{grid:32,atMs:2700}),true);
  const report=t.snapshot();
  assert.deepEqual(report.packets.map(row=>[row.kind,row.index,row.payloadBytes,row.displayedMs,row.roundTripMs,row.displayCount,row.ack.grid]),
    [['start',0,3,100,500,2,32],['data',1,6,700,1000,2,32]]);
  assert.deepEqual(report.roundTrips.overall,{count:2,minMs:500,medianMs:750,p95Ms:1000,meanMs:750,maxMs:1000});
  assert.deepEqual(report.roundTrips.byGridAndKind.map(row=>[row.grid,row.kind,row.count,row.minMs]),[[128,'start',1,500],[128,'data',1,1000]]);
  assert.equal(t.minimumRoundTripMs,500);
});

test('stale, foreign, unsent, invalid and duplicated ACKs cannot invent or reset RTT',()=>{
  const t=new PacketTelemetry({originMs:0});
  assert.equal(t.accepted(ack(0,'progress'),{accepted:true},{grid:32,atMs:1}),false);
  t.displayed(start(),{grid:64,atMs:10});
  for(const packet of [ack(0,'ready'),ack(0,'complete'),ack(0,'progress',{messageId:'aa'.repeat(8)}),
    ack(0,'progress',{digest:'ff'.repeat(32)}),Uint8Array.of(0)]) {
    assert.equal(t.accepted(packet,{accepted:true},{grid:32,atMs:20}),false);
  }
  assert.equal(t.accepted(ack(0,'progress'),{accepted:false,reason:'stale-ack'},{grid:32,atMs:20}),false);
  assert.equal(t.accepted(ack(0,'progress'),{accepted:true},{grid:32,atMs:5}),false);
  assert.equal(t.accepted(ack(0,'progress'),{accepted:true},{grid:32,atMs:40}),true);
  assert.equal(t.accepted(ack(0,'progress'),{accepted:true},{grid:32,atMs:60}),false);
  t.displayed(data(1),{grid:64,atMs:65});
  assert.equal(t.accepted(ack(1,'progress'),{accepted:true},{grid:32,atMs:80}),false,'final DATA needs a complete ACK');
  const report=t.snapshot();assert.equal(report.roundTrips.overall.count,1);assert.equal(report.roundTrips.overall.minMs,30);
  assert.equal(report.packets[1].roundTripMs,null,'pending work has no zero-valued RTT');
});

test('incoming multipart and singleton ACKs correlate even when painted before the receive callback returns',()=>{
  const t=new PacketTelemetry({originMs:1000});
  t.displayed(ack(0,'progress'),{grid:32,atMs:1120});
  t.accepted(start(),{accepted:true},{grid:128,atMs:1100});
  t.displayed(ack(1,'complete'),{grid:32,atMs:1325});
  t.accepted(data(1),{accepted:true},{grid:128,atMs:1300});
  t.displayed(ack(1,'complete'),{grid:32,atMs:1400});
  assert.equal(t.accepted(data(1),{accepted:true,reason:'duplicate'},{grid:128,atMs:1400}),false);
  assert.equal(t.accepted(start(),{accepted:true,reason:'duplicate-start'},{grid:128,atMs:1500}),false);
  const id={messageId:'aa'.repeat(8)};
  t.displayed(ack(0,'complete',id),{grid:32,atMs:1520});
  t.accepted(start({...id,totalBytes:3,chunkCount:1}),{accepted:true},{grid:64,atMs:1500});
  const report=t.snapshot();
  assert.deepEqual(report.packets.map(row=>[row.kind,row.index,row.receivedMs,row.ack.displayedMs,row.ackDisplayDelayMs,row.ack.displayCount]),
    [['start',0,100,120,20,1],['data',1,300,325,25,2],['start',0,500,520,20,1]]);
  assert.equal(report.roundTrips.overall.count,0,'incoming read-to-ACK is not a round trip');
  assert.equal(report.receiveToAckDisplay.medianMs,20);
});

test('bounded telemetry reports omitted observations and never retains packet payloads',()=>{
  const t=new PacketTelemetry({originMs:0,maxRows:2});
  t.displayed(start(),{grid:64,atMs:10});t.displayed(data(1),{grid:64,atMs:20});
  assert.equal(t.displayed(start({messageId:'aa'.repeat(8)}),{grid:64,atMs:30}),false);
  assert.equal(t.accepted(ack(1,'complete'),{accepted:true},{grid:32,atMs:40}),true);
  const report=t.snapshot();assert.equal(report.packets.length,2);assert.equal(report.droppedEvents,1);assert.equal(report.truncated,true);
  assert.equal(report.roundTrips.overall.minMs,20);
  assert.ok(report.packets.every(row=>!('data'in row)&&!('digest'in row)));
  report.packets[0].ack.receivedMs=99;assert.equal(t.snapshot().packets[0].ack.receivedMs,null,'report mutation cannot alter tracking');
  assert.throws(()=>new PacketTelemetry({maxRows:0}));
});

test('REQUEST controls retain predecessor and local display/acceptance times separately from packet RTT',()=>{
  const t=new PacketTelemetry({originMs:1000});
  t.displayed(start({...meta,totalBytes:3,chunkCount:1}),{grid:64,atMs:1100});
  t.accepted(ack(0,'complete'),{accepted:true},{grid:32,atMs:1200});
  assert.equal(t.displayed(request(),{grid:32,atMs:1300}),true);
  assert.equal(t.displayed(request(),{grid:32,atMs:1400}),true);
  assert.equal(t.accepted(request(),{accepted:false,reason:'foreign-request'},{grid:32,atMs:1450}),false);
  assert.equal(t.accepted(request(),{accepted:true},{grid:32,atMs:1500}),true);
  assert.equal(t.accepted(request(),{accepted:true},{grid:32,atMs:1600}),false);
  assert.equal(t.accepted(request({messageId:'aa'.repeat(8)}),{accepted:true,reason:'duplicate-request'},{grid:32,atMs:1700}),false);
  assert.equal(t.displayed(request({messageId:'bb'.repeat(8)}),{grid:64,atMs:1800}),false,'REQUEST uses the G32 control profile');
  const report=t.snapshot();
  assert.deepEqual(report.requestCounts,{sent:1,received:1});
  assert.deepEqual(report.requests.map(row=>[row.direction,row.previousMessageId,row.previousFinalIndex,row.displayedMs,row.lastDisplayedMs,row.displayCount,row.receivedMs]),
    [['sent',meta.messageId,0,300,400,2,null],['received',meta.messageId,0,null,null,0,500]]);
  assert.equal(report.requests[0].packetBytes,44);
  assert.equal(report.packets.length,1);
  assert.equal(report.roundTrips.overall.count,1);assert.equal(t.minimumRoundTripMs,100);
  assert.equal(report.receiveToAckDisplay.count,0,'REQUEST has no ACK and is not a DATA round trip');
  report.requests[0].displayCount=99;assert.equal(t.snapshot().requests[0].displayCount,2);
});

test('REQUEST control records share the bounded telemetry row budget',()=>{
  const t=new PacketTelemetry({originMs:0,maxRows:2});
  t.displayed(request(),{grid:32,atMs:10});
  t.accepted(request(),{accepted:true},{grid:32,atMs:20});
  assert.equal(t.displayed(start(),{grid:64,atMs:30}),false);
  assert.equal(t.displayed(request(),{grid:32,atMs:40}),true,'retained controls still update at capacity');
  const report=t.snapshot();
  assert.equal(report.requests.length+report.packets.length,2);
  assert.equal(report.droppedEvents,1);assert.equal(report.truncated,true);
});

test('G16 timings correlate the independently matched logical ACK but count only the transmitted digest',()=>{
  const t=new PacketTelemetry({originMs:0}),receipt=ack(0,'progress');
  assert.equal(t.displayed(start(),{grid:16,atMs:0}),false);
  assert.equal(t.accepted(request(),{accepted:true},{grid:16,atMs:1}),false);
  t.displayed(start(),{grid:64,atMs:10});
  t.accepted(start(),{accepted:true},{grid:64,atMs:20});
  t.displayed(receipt,{grid:16,atMs:25});
  assert.equal(t.accepted(receipt,{accepted:false},{grid:16,atMs:29}),false);
  t.accepted(receipt,{accepted:true},{grid:16,atMs:30});
  const snap=t.snapshot();
  assert.equal(snap.roundTrips.overall.meanMs,20);
  assert.equal(snap.receiveToAckDisplay.meanMs,5);
  for(const row of snap.packets){assert.equal(row.ack.packetBytes,32);assert.equal(row.ack.grid,16);}
});
