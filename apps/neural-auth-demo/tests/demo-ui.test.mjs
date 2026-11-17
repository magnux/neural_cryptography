import test from 'node:test';
import assert from 'node:assert/strict';
import {detectDevice,sessionMessage,addTiming,mergeTimings,timingSummary} from '../dist/app.js';

test('desktop detection is independent of a narrow viewport and recognizes iPad desktop mode',()=>{
  assert.equal(detectDevice({userAgent:'Mozilla/5.0 (X11; Linux x86_64)',maxTouchPoints:0}),'desktop');
  assert.equal(detectDevice({userAgent:'Mozilla/5.0 (iPhone; CPU iPhone OS 18)',maxTouchPoints:5}),'mobile');
  assert.equal(detectDevice({userAgent:'Mozilla/5.0 Macintosh',platform:'MacIntel',maxTouchPoints:5}),'mobile');
  assert.equal(detectDevice({userAgent:'Mozilla/5.0 Macintosh',platform:'MacIntel',maxTouchPoints:0}),'desktop');
  assert.equal(detectDevice({userAgent:'custom',userAgentData:{mobile:true}}),'mobile');
});

test('a locally verified proof is not shown as a completed exchange before result ACK',()=>{
  const state={state:'verified',result:true,peerDisplayName:'Pat’s phone'};
  const pending=sessionMessage(state,{mode:'send'});
  assert.equal(pending.complete,false);assert.match(pending.title,/locally/);assert.match(pending.result,/delivery pending/);
  const complete=sessionMessage({...state,resultDelivered:true},{mode:'receive'});
  assert.equal(complete.complete,true);assert.match(complete.result,/Verified “Pat’s phone” locally/);
});

test('prover copy never treats an unauthenticated peer result as local verification',()=>{
  for(const accepted of [true,false]){
    const message=sessionMessage({state:accepted?'peer-accepted':'peer-rejected',peerReported:true,result:accepted,peerDisplayName:'Verifier <b>name</b>'},{mode:'ack'});
    assert.equal(message.complete,true);assert.match(message.title,/Peer reports/);assert.match(message.result,/Peer reports/);
    assert.doesNotMatch(message.result,/Verified.*locally/);assert.match(message.instruction,/Keep this code visible/);
    assert.match(message.result,/Verifier <b>name<\/b>/); // UI inserts this using textContent.
  }
});

test('aggregate timings merge counts and extrema without retaining frame histories',()=>{
  const operations=Object.create(null);addTiming(operations,'readback',8);addTiming(operations,'readback',4);
  mergeTimings(operations,{'readback':{count:3,totalMs:9,minMs:2,maxMs:4},'inference':{count:2,totalMs:100,minMs:45,maxMs:55}});
  assert.deepEqual(timingSummary(operations).readback,{count:5,totalMs:21,minMs:2,maxMs:8,meanMs:4.2});
  assert.equal(timingSummary(operations).inference.meanMs,50);
  for(let i=0;i<100;i++)addTiming(operations,`stage${i}`,1);
  assert.equal(Object.keys(operations).length,48);
  assert.ok(Object.values(operations).every(row=>!Array.isArray(row)));
});

test('invalid and absent timings do not create misleading measurements',()=>{
  const operations=Object.create(null);
  addTiming(operations,'bad',NaN);addTiming(operations,'negative',-1);
  mergeTimings(operations,{bad:{count:1,totalMs:Infinity,minMs:0,maxMs:1},empty:{count:0,totalMs:0,minMs:0,maxMs:0}});
  mergeTimings(operations,undefined);
  assert.deepEqual(timingSummary(operations),{});
});
