import test from 'node:test';
import assert from 'node:assert/strict';
import {SweepRun,CapturePacer} from '../dist/sweep-run.js';

test('manual cadence descends together, never auto-arms, and stops before either interval falls below 50 ms',()=>{
  const r=new SweepRun({role:'receiver'});assert.equal(r.count,9);
  for(let n=1;n<=9;n++){
    const t=1000*n,a=r.begin(t);assert.equal(r.next,null);assert.throws(()=>r.begin(t));
    assert.equal(a.step,n);assert.equal(a.phaseMs,550-50*n);assert.equal(a.captureMs,500-50*n);
    if(n===1)r.finish('timeout',t+900);
    else{r.mark('ackDraw',t+800);const row=r.finish('ack-shown',t+900);
      assert.equal(row.armToAckMs,800);assert.equal(row.rttMs,null);}
    assert.equal(r.active,null);
  }
  assert.equal(r.next,null);assert.equal(r.report().steps.length,9);assert.equal(r.report().steps[0].rttMs,null);
});

test('sender reports time between paint and actual matched ACK; independent step stages and boundaries are not summed',()=>{
  const r=new SweepRun({role:'sender'});r.begin(0);assert.throws(()=>r.finish('ack-read',100));
  r.mark('probeDraw',300);r.mark('firstLock',700);r.mark('firstLock',900);r.mark('allPhases',1300);
  r.mark('packetDecoded',1400);r.mark('ackDecoded',1450);r.sample('readback',4);r.sample('readback',8);
  const first=r.finish('ack-read',1600,{inferences:1});
  assert.equal(first.rttMs,1150);assert.equal(first.lockToPacketMs,700);assert.equal(first.elapsedMs,1600);
  assert.deepEqual(first.stages.readback,{count:2,totalMs:12,minMs:4,maxMs:8,avgMs:6});
  r.begin(25000);assert.throws(()=>r.finish('ack-shown',25001));
  const second=r.finish('timeout',55000);assert.deepEqual(second.stages,{});assert.equal(second.rttMs,null);
  assert.equal(r.report({runId:'same-on-both-phones'}).steps[0].rttMs,1150);
});

test('held ACK redraws measure physical phase cadence on the original step without extending its duration',()=>{
  const r=new SweepRun({role:'receiver'});r.begin(0);r.mark('ackDraw',100);r.finish('ack-shown',100);
  r.painted(1,'ack',0,100);r.painted(1,'ack',0,101);
  r.painted(1,'ack',1,600);r.painted(1,'ack',3,1600);
  r.begin(1800);r.painted(1,'ack',4,2100); // A stale callback after Next is ignored.
  const row=r.report().steps[0];assert.equal(row.elapsedMs,100);assert.equal(row.display.skippedSlots,1);
  assert.deepEqual(row.display.byPhase,[1,1,0,1]);assert.equal(row.display.spacingMs.avgMs,750);
  assert.equal(r.display(2),null);
});

test('optional capture variation samples one deadline per accepted frame and never catches up after a busy scan',()=>{
  let calls=0;const p=new CapturePacer(250,true,()=>[0,1,.5][calls++]);
  assert.deepEqual(p.take(10),{interval:null});
  for(const stamp of [20,100,209])assert.equal(p.take(stamp),null);
  assert.equal(calls,1);assert.deepEqual(p.take(211).interval,{requested:200,actual:201});
  assert.deepEqual(p.take(810).interval,{requested:250*(.8+.4),actual:599});
  assert.equal(p.take(811),null);assert.equal(calls,3);
  const fixed=new CapturePacer(250,false,()=>{throw Error('unexpected timing variation');});
  fixed.take(20);assert.equal(fixed.take(269),null);assert.equal(fixed.take(270).interval.requested,250);
});

test('zero decrement repeats a fixed pace for five independently armed packets, including misses',()=>{
  const r=new SweepRun({role:'receiver',startPhaseMs:350,startCaptureMs:250,decrementMs:0});
  for(let i=0;i<5;i++){
    assert.deepEqual(r.next,{step:i+1,phaseMs:350,captureMs:250});
    const t=i*500;r.begin(t);if(i>0)r.mark('ackDraw',t+100);
    r.finish(i===0?'timeout':'ack-shown',t+101);
  }
  assert.equal(r.next,null);assert.equal(r.report().successfulSteps,4);
});
