import test from 'node:test';
import assert from 'node:assert/strict';
import {Worker} from 'node:worker_threads';
import {runSimulation} from '../dist/js/link/simulation.js';
import {diagnosticPayload} from '../dist/js/link/payload.js';
import {sha256} from '../dist/js/link/protocol.js';

test('165 KB survives dropped, duplicated and corrupt optical packets exactly once',async()=>{
  const r=await runSimulation();
  assert.equal(r.status,'delivered');assert.equal(r.exact,true);
  assert.equal(r.payloadBytes,165000);assert.equal(r.sender.chunkCount,31);
  assert.equal(r.expectedDigest,r.receivedDigest);assert.equal(r.receiver.bufferedBytes,0);
  assert.equal(r.stats.deliveries,1);assert.ok(r.sender.retries>0);
  for(const counter of ['dropped','duplicated','corrupted','headerRejected','payloadRejected'])assert.ok(r.stats[counter]>0,counter);
  assert.ok(r.stats.physicalPhases>r.stats.transmissions);
  assert.ok(r.events.length<=100);
});

test('lossy runs cover small and large matrices, multiple windows and independent channel seeds',async()=>{
  for(const [size,windowSize,seed] of [[64,1,27],[128,8,91],[256,4,12]]){
    const r=await runSimulation({bytes:165000,windowSize,seed,profile:{size,phases:4,holdMs:250},dropRate:.2,corruptRate:.12,duplicateRate:.3});
    assert.equal(r.status,'delivered',`${size}/${windowSize}: ${r.sender.error}`);
    assert.equal(r.exact,true);assert.equal(r.stats.deliveries,1);
  }
});

test('a blackholed channel terminates at its configured deadline without delivering bytes',async()=>{
  const r=await runSimulation({bytes:1024,dropRate:1,duplicateRate:0,corruptRate:0,timeoutMs:3000,retryMs:1000,maxRetries:100});
  assert.equal(r.status,'timed-out');assert.equal(r.exact,false);
  assert.equal(r.stats.deliveries,0);assert.equal(r.receiver.bufferedBytes,0);
  assert.ok(r.simulatedMs<=5000);
});

test('an entirely corrupted channel stops after its retry budget',async()=>{
  const r=await runSimulation({bytes:1024,dropRate:0,duplicateRate:0,corruptRate:1,retryMs:1000,maxRetries:2});
  assert.equal(r.status,'failed');assert.equal(r.stats.deliveries,0);
  assert.ok(r.stats.headerRejected+r.stats.payloadRejected>0);
});

test('empty messages and supplied binary files preserve their exact SHA-256',async()=>{
  for(const payload of [new Uint8Array(),Uint8Array.from([0,255,0,128,65,10,13])]){
    const r=await runSimulation({payload,dropRate:0,duplicateRate:1,corruptRate:0});
    assert.equal(r.status,'delivered');assert.equal(r.exact,true);
    assert.equal(r.expectedDigest,await sha256(payload));assert.equal(r.stats.deliveries,1);
    assert.equal(r.options.source,'file');
  }
  assert.deepEqual(diagnosticPayload(100,17),diagnosticPayload(100,17));
  assert.notDeepEqual(diagnosticPayload(100,17),diagnosticPayload(100,18));
});

test('simulation rejects invalid and excessive work before starting',async()=>{
  for(const options of [{bytes:1048577},{bytes:-1},{bytes:NaN},{windowSize:9},{dropRate:-.01},{corruptRate:Infinity},{payload:[]},{seed:-1}]){
    await assert.rejects(runSimulation(options));
  }
});

test('the actual browser simulation worker streams progress and completes the 165 KB run',async()=>{
  const worker=new Worker(new URL('./link-worker-harness.mjs',import.meta.url));
  try{
    let updates=0;
    const report=await new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(Error('Worker simulation deadline')),30000);
      worker.on('error',e=>{clearTimeout(timer);reject(e);});
      worker.on('message',message=>{
        assert.equal(message.id,17);
        if(message.type==='progress')updates++;
        if(message.type==='error'){clearTimeout(timer);reject(Error(message.error));}
        if(message.type==='result'){clearTimeout(timer);resolve(message.report);}
      });
      worker.postMessage({type:'run',id:17,options:{bytes:165000}});
    });
    assert.equal(report.exact,true);assert.equal(report.status,'delivered');assert.ok(updates>0);
  }finally{await worker.terminate();}
});
