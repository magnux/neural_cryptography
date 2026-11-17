import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {runFullHandshake} from '../dist/simulation-core.js';

test('the runnable protocol benchmark performs the complete fresh-key handshake and verifies a real 32-repetition proof',async()=>{
  const result=await runFullHandshake({optical:false,statistics:true});
  assert.equal(result.success,true);assert.equal(result.proofVerified,true);assert.equal(result.resultDelivered,true);assert.equal(result.peerReported,true);
  assert.equal(result.proofBytes,17474);assert.equal(result.finalResultAckAccepted,true);
  assert.equal(result.verifier.peerDisplayName,'Simulated phone');assert.equal(result.prover.peerDisplayName,'Simulated computer');
  assert.deepEqual(result.messages.map(item=>item.type),['HELLO','IDENTITY','CHALLENGE','PROOF','RESULT']);
  assert.deepEqual(result.cryptoCounts,{createKey:2,newChallenge:1,prove:1,verify:1});
  assert.equal(result.inferenceCalls,0);assert.ok(result.cryptoIdleWaits>=1);
  for(const stage of ['crypto.createKey','crypto.prove','crypto.verify','transport.receive']){
    const row=result.stages.find(item=>item.stage===stage);assert.ok(row?.count>0);assert.ok(row.minMs>=0);assert.ok(row.maxMs>=row.avgMs);assert.ok(row.avgMs>=row.minMs);
  }
});

test('statistics can be disabled without bypassing the real verification',async()=>{
  const result=await runFullHandshake({optical:false,statistics:false});
  assert.equal(result.success,true);assert.equal(result.statisticsEnabled,false);assert.deepEqual(result.stages,[]);assert.equal(result.cryptoCounts.verify,1);
});

test('tampering the actual proof does not produce simulated authentication success',async()=>{
  const result=await runFullHandshake({optical:false,tamperProof:true});
  assert.equal(result.success,false);assert.equal(result.proofVerified,false);assert.equal(result.verifier.state,'rejected');assert.equal(result.resultDelivered,true);
});

test('optical simulation has no silent proxy or fake inference fallback',async()=>{
  await assert.rejects(runFullHandshake(),/Actual simulation inference must be initialized/);
  const runtime=await readFile(new URL('../dist/simulation-runtime.js',import.meta.url),'utf8');
  assert.match(runtime,/manifest.status!=='untrained'/);
  assert.match(runtime,/session.run/);
  assert.match(runtime,/compareLogits/);
});

test('simulation checks cancellation before performing a handshake',async()=>{
  const controller=new AbortController();controller.abort();
  await assert.rejects(runFullHandshake({optical:false,signal:controller.signal}),{name:'AbortError'});
});


test('request handoffs survive lost final ACKs and requests without manual coordination',async()=>{
  const result=await runFullHandshake({optical:false,dropFinalAckWindows:true,dropRequestWindows:true});
  assert.equal(result.success,true);assert.equal(result.finalResultAckAccepted,true);
  assert.equal(result.automaticTurns,true);assert.equal(result.manualHandoffs,0);
  assert.equal(result.droppedFinalAckWindows,5);assert.equal(result.requestWindows,4);assert.equal(result.droppedRequestWindows,4);assert.equal(result.handoff,'request');
  assert.equal(result.messages.length,5);assert.equal(result.cryptoCounts.prove,1);assert.equal(result.cryptoCounts.verify,1);
});


test('synthetic images follow the actual receiving turn through automatic optical handoffs',async()=>{
  // Explicit test-only inference stub: the production simulator requires the
  // actual untrained ONNX loader. This case isolates acquisition/turn routing.
  const manifest={modelProfile:'VW',pixelsPerCell:4,channels:32,framesPerInput:8,
    minFramesPerInput:4,frameCounts:[4,5,6,7,8],inputLayout:'phase-paired-rgb-presence-planes',
    outputLabelSpace:'logical',phaseMasks:[0,15,7,8]};
  const corrector={loadContext:async size=>({manifest,entry:{size},effectiveBackend:'test-only-stub'}),
    infer:async context=>({values:new Float32Array(16*context.entry.size**2),ms:0})};
  const result=await runFullHandshake({corrector,repetitions:1,statistics:false});
  assert.equal(result.success,true);assert.equal(result.finalResultAckAccepted,true);assert.equal(result.manualHandoffs,0);
  assert.ok(result.inferenceCalls>0);assert.ok(result.scans>result.inferenceCalls);assert.ok(result.wrongTurnScans>0);
  assert.deepEqual(result.stages,[]);
});
