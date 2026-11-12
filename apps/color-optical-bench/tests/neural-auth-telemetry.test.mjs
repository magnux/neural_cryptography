import test from 'node:test';
import assert from 'node:assert/strict';
import {PacketTelemetry} from '../dist/js/neural-auth/telemetry.js';
import {encodePacket} from '../dist/js/link/protocol.js';

const meta={sessionId:'11'.repeat(16),messageId:'22'.repeat(8),messageType:'raw',digest:'33'.repeat(32),
  totalBytes:6,chunkBytes:3,chunkCount:2,windowSize:1};
const start=(overrides={})=>encodePacket({...meta,kind:'start',...overrides});
const data=(index,overrides={})=>encodePacket({...meta,kind:'data',index,data:Uint8Array.of(4,5,6),...overrides});
const ack=(index,status,overrides={})=>encodePacket({...meta,kind:'ack',base:index,bitmap:status==='ready'?0:1,status,...overrides});

test('actual redisplays preserve START/DATA RTT origins and summaries separate grid and kind',()=>{
  const t=new PacketTelemetry({originMs:1000});
  t.displayed(start(),{grid:128,atMs:1100});
  t.displayed(start(),{grid:128,atMs:1400});
  t.accepted(ack(0,'ready'),{accepted:true},{grid:32,atMs:1600});
  t.displayed(data(0),{grid:64,atMs:1700});
  t.displayed(data(0),{grid:64,atMs:1800});
  t.accepted(ack(0,'progress'),{accepted:true},{grid:32,atMs:2700});
  t.displayed(data(1),{grid:64,atMs:2800});
  t.accepted(ack(1,'complete'),{accepted:true,delivered:true},{grid:32,atMs:3100});
  const report=t.snapshot();
  assert.deepEqual(report.packets.map(row=>[row.kind,row.index,row.displayedMs,row.roundTripMs,row.displayCount]),
    [['start',null,100,500,2],['data',0,700,1000,2],['data',1,1800,300,1]]);
  assert.deepEqual(report.roundTrips.overall,{count:3,minMs:300,medianMs:500,p95Ms:1000,meanMs:600,maxMs:1000});
  assert.deepEqual(report.roundTrips.byGridAndKind.map(row=>[row.grid,row.kind,row.count,row.minMs]),[[64,'data',2,300],[128,'start',1,500]]);
  assert.equal(t.minimumRoundTripMs,300);
  assert.equal(report.packets[1].payloadBytes,3);
  assert.equal(report.packets[1].ack.grid,32);
});

test('stale, foreign, unsent, invalid and duplicated ACKs cannot invent or reset RTT',()=>{
  const t=new PacketTelemetry({originMs:0});
  assert.equal(t.accepted(ack(0,'ready'),{accepted:true},{grid:32,atMs:1}),false);
  t.displayed(start(),{grid:64,atMs:10});
  assert.equal(t.accepted(ack(0,'ready'),{accepted:false,reason:'stale-ack'},{grid:32,atMs:20}),false);
  assert.equal(t.accepted(ack(0,'ready',{messageId:'aa'.repeat(8)}),{accepted:true},{grid:32,atMs:20}),false);
  assert.equal(t.accepted(ack(0,'ready',{digest:'ff'.repeat(32)}),{accepted:true},{grid:32,atMs:20}),false);
  assert.equal(t.accepted(Uint8Array.of(0),{accepted:true},{grid:32,atMs:20}),false);
  assert.equal(t.accepted(ack(0,'ready'),{accepted:true},{grid:32,atMs:5}),false);
  assert.equal(t.accepted(ack(0,'ready'),{accepted:true},{grid:32,atMs:40}),true);
  assert.equal(t.accepted(ack(0,'ready'),{accepted:true},{grid:32,atMs:60}),false);
  t.displayed(data(0),{grid:64,atMs:65});
  assert.equal(t.accepted(ack(0,'complete'),{accepted:true},{grid:32,atMs:80}),false,'non-final DATA needs a progress ACK');
  const report=t.snapshot();assert.equal(report.roundTrips.overall.count,1);assert.equal(report.roundTrips.overall.minMs,30);
  assert.equal(report.packets[1].roundTripMs,null,'pending work has no zero-valued RTT');
});

test('incoming packet timestamps correlate with ACKs displayed before acceptance callback returns',()=>{
  const t=new PacketTelemetry({originMs:1000});
  t.displayed(ack(0,'ready'),{grid:32,atMs:1120});
  t.accepted(start(),{accepted:true},{grid:128,atMs:1100});
  t.displayed(ack(0,'progress'),{grid:32,atMs:1325});
  t.accepted(data(0),{accepted:true},{grid:128,atMs:1300});
  t.displayed(ack(0,'progress'),{grid:32,atMs:1400});
  assert.equal(t.accepted(data(0),{accepted:true,reason:'duplicate'},{grid:128,atMs:1400}),false);
  assert.equal(t.accepted(start(),{accepted:true,reason:'duplicate-start'},{grid:128,atMs:1500}),false);
  const report=t.snapshot();
  assert.deepEqual(report.packets.map(row=>[row.receivedMs,row.ack.displayedMs,row.ackDisplayDelayMs,row.ack.displayCount]),[[100,120,20,1],[300,325,25,2]]);
  assert.equal(report.roundTrips.overall.count,0,'incoming read-to-ACK is not a round trip');
  assert.equal(report.receiveToAckDisplay.medianMs,22.5);
});

test('bounded telemetry reports omitted observations and never retains packet payloads',()=>{
  const t=new PacketTelemetry({originMs:0,maxRows:2});
  t.displayed(start(),{grid:64,atMs:10});t.displayed(data(0),{grid:64,atMs:20});
  assert.equal(t.displayed(data(1),{grid:64,atMs:30}),false);
  assert.equal(t.accepted(ack(0,'progress'),{accepted:true},{grid:32,atMs:40}),true);
  const report=t.snapshot();assert.equal(report.packets.length,2);assert.equal(report.droppedEvents,1);assert.equal(report.truncated,true);
  assert.equal(report.roundTrips.overall.minMs,20);
  assert.ok(report.packets.every(row=>!('data'in row)&&!('digest'in row)));
  report.packets[0].ack.receivedMs=99;assert.equal(t.snapshot().packets[0].ack.receivedMs,null,'report mutation cannot alter tracking');
  assert.throws(()=>new PacketTelemetry({maxRows:0}));
});
