import test from 'node:test';
import assert from 'node:assert/strict';
import {encodeSequence} from '../dist/js/format.js';
import {ExperimentReport} from '../dist/js/diagnostics.js';

function info(id=1){
  const frame=encodeSequence(Uint8Array.of(1,2,3),{id:Uint8Array.of(id,2,3,4,5,6,7,8),size:32,colors:16,phases:4,permute:true})[0];
  return {...frame,phases:16,holdMs:4000};
}
function candidate(ok,stats={}){return {ready:true,ok,stage:ok?'complete':'payload',stats:{rawCRC:false,hardCRC:false,correctedBytes:0,erasedBytes:0,erasureTrials:0,...stats}};}
function capture(frame,methods,counts=Array(16).fill(0)){
  return {info:frame,stats:{temporal:{filter:'robust',samples:5,target:5,requested:5,poolCapacity:7,requestedPool:7,counts,phases:16,ready:true,eligiblePhases:[0,1,2,3],minPhases:4,retainedPhaseLimit:16,memoryLimited:false,memoryBytes:1234,evicted:0,ageLimitMs:128000}},colorComparison:methods};
}

test('each method records its first CRC-verified time, capture, and phase coverage once',()=>{
  const report=new ExperimentReport(),frame=info(),counts=[5,5,5,5,...Array(12).fill(0)];
  report.accept({ok:false,stats:{}},100);
  report.accept(capture(frame,{robustGlobal:candidate(false)}),200);
  report.accept(capture(frame,{robustGlobal:candidate(true,{rawCRC:true,hardCRC:true,recovery:'raw'}),accepted:candidate(true,{rawCRC:true,hardCRC:true,recovery:'raw'})},counts),350);
  counts[0]=99; // Worker/result buffers may be reused after report ingestion.
  report.accept(capture({...frame,index:1},{robustGlobal:candidate(true,{hardCRC:true,recovery:'hard'}),robustLocal:candidate(true,{recovery:'erasures',correctedBytes:2,erasedBytes:4,erasureTrials:1})},Array(16).fill(7)),500);
  const snapshot=report.snapshot(600),global=snapshot.methods.robustGlobal,local=snapshot.methods.robustLocal;
  assert.equal(global.firstVerifiedMs,250);assert.equal(global.firstVerifiedCapture,3);
  assert.deepEqual(global.firstVerifiedCounts,[5,5,5,5,...Array(12).fill(0)]);
  assert.equal(local.firstVerifiedMs,400);assert.equal(local.firstVerifiedCapture,4);
  assert.deepEqual(local.firstVerifiedCounts,Array(16).fill(7));
  assert.equal(global.attempts,3);assert.equal(global.passes,2);assert.equal(global.rawPasses,1);assert.equal(global.hardPasses,2);assert.equal(global.erasurePasses,0);
  assert.equal(local.erasurePasses,1);assert.equal(local.hardPasses,0);assert.equal(local.erasedBytes,4);assert.equal(local.erasureTrials,1);
  assert.equal(snapshot.profile.holdMs,4000);assert.equal(snapshot.history[1].accepted.rawCRC,true);
  assert.equal(snapshot.temporal.ready,true);assert.deepEqual(snapshot.temporal.eligiblePhases,[0,1,2,3]);
});

test('failed candidates never increment acceptance counters even if lower-stage CRC flags remain set',()=>{
  const report=new ExperimentReport(),frame=info();
  report.accept(capture(frame,{fusedLocal:candidate(false,{rawCRC:true,hardCRC:true,recovery:'erasures',erasureTrials:3,erasedBytes:16})}),0);
  const row=report.snapshot(100).methods.fusedLocal;
  assert.equal(row.attempts,1);assert.equal(row.passes,0);assert.equal(row.rawPasses,0);assert.equal(row.hardPasses,0);assert.equal(row.erasurePasses,0);
  assert.equal(row.firstVerifiedMs,null);assert.equal(row.firstVerifiedCapture,null);assert.equal(row.firstVerifiedCounts,null);
  assert.equal(row.erasureTrials,3);assert.equal(row.erasedBytes,16);assert.equal(row.bytes,0);assert.equal(row.frames.length,0);
});

test('foreign sequences cannot alter the profile, methods, temporal status, or history',()=>{
  const report=new ExperimentReport(),first=info(1),foreign=info(2);
  report.accept(capture(first,{fusedGlobal:candidate(true,{hardCRC:true,recovery:'hard'})},[5,5,5,5]),0);
  const before=structuredClone(report.snapshot(100));
  report.accept(capture({...foreign,holdMs:1000},{robustLocal:candidate(true,{recovery:'erasures'})},Array(16).fill(7)),100);
  const after=report.snapshot(100);
  assert.equal(after.rejectedSequences,1);assert.equal(after.captures,2);assert.equal(after.headers,2);
  assert.deepEqual(after.profile,before.profile);assert.deepEqual(after.methods,before.methods);assert.deepEqual(after.temporal,before.temporal);assert.deepEqual(after.history,before.history);
});

test('history stays bounded and stores compact robust statistics without image, payload, or evidence arrays',()=>{
  const report=new ExperimentReport(),frame=info(),counts=Array(20).fill(7);
  const stats={phase:0,cells:100,candidates:7,kept:5,trimmedCells:20,ambiguousCells:2,meanSelectionDistance:.1,meanDiscardedDistance:.8,scores:new Uint8Array(1000)};
  const result=capture(frame,{robustGlobal:{...candidate(true,{hardCRC:true,recovery:'hard',scores:new Uint8Array(1000)}),chunk:new Uint8Array(1000),scores:new Uint8Array(1000)},robustLocal:candidate(false),accepted:candidate(true)},counts);
  result.stats.temporal.robust={global:Array(20).fill(stats),local:[stats]};result.stats.temporal.values=new Float32Array(1000);
  for(let i=0;i<140;i++)report.accept(result,i*100);
  const snapshot=report.snapshot(14000);
  assert.equal(snapshot.history.length,120);assert.equal(snapshot.history[0].elapsedMs,2000);
  assert.equal(snapshot.methods.robustGlobal.frames.length,1);assert.equal(snapshot.methods.robustGlobal.bytes,3);
  assert.equal(snapshot.temporal.counts.length,16);assert.equal(snapshot.temporal.robust.global.length,16);
  assert.equal(snapshot.history[0].temporal.robust.local[0].kept,5);
  assert.equal(snapshot.history[0].robustGlobal.ok,true);assert.equal(snapshot.history[0].robustLocal.ok,false);assert.equal(snapshot.history[0].accepted.ok,true);
  for(const forbidden of ['"scores"','"chunk"','"values"','"symbols"','"image"'])assert.equal(JSON.stringify(snapshot).includes(forbidden),false,forbidden);
  counts[0]=99;stats.kept=99;
  assert.equal(snapshot.temporal.counts[0],7);assert.equal(snapshot.temporal.robust.global[0].kept,5);
  report.reset();assert.equal(report.snapshot(15000).temporal,null);assert.equal(report.snapshot(15000).captures,0);
});
