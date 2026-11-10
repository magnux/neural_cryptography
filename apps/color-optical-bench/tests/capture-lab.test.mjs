import test from 'node:test';
import assert from 'node:assert/strict';
import {createSender,captureProfile,base64ToBytes} from '../dist/js/dataset/contract.js';
import {renderNeuralQR} from '../dist/js/neural-qr/layout.js';
import {CaptureClient} from '../dist/js/dataset/client.js';

class FakeWorker{
  constructor(){this.messages=[];this.terminated=false;}
  postMessage(message,transfer){this.messages.push({message,transfer});}
  terminate(){this.terminated=true;}
  receive(message){this.onmessage?.({data:message});}
}
test('capture client limits in-flight work and transfers only current image bytes',async()=>{
  const worker=new FakeWorker(),client=new CaptureClient({workerFactory:()=>worker});
  const image={width:32,height:32,data:new Uint8ClampedArray(32*32*4)};
  const pending=client.call('capture',{image,timestamp:10},[image.data.buffer]);
  await assert.rejects(client.call('capture',{image}),/already running/);
  assert.equal(worker.messages.length,1);
  assert.equal(worker.messages[0].transfer[0],image.data.buffer);
  const id=worker.messages[0].message.id;
  worker.receive({id,type:'result',result:{ok:false,stage:'detect',error:'No markers'}});
  assert.deepEqual(await pending,{ok:false,stage:'detect',error:'No markers'});
  client.cancel();
});
test('cancel terminates extraction and a stale worker cannot replace a later result',async()=>{
  const workers=[],client=new CaptureClient({workerFactory:()=>{const w=new FakeWorker();workers.push(w);return w;}});
  const first=client.call('capture'),rejected=assert.rejects(first,{name:'AbortError'});
  client.cancel();await rejected;assert.equal(workers[0].terminated,true);
  const second=client.call('capture'),id=workers[1].messages[0].message.id;
  workers[0].receive({id,type:'result',result:'stale'});
  workers[1].receive({id,type:'result',result:'current'});
  assert.equal(await second,'current');client.cancel();
});
test('worker errors and timeouts release the client for a new capture',async()=>{
  const workers=[],client=new CaptureClient({timeoutMs:10,workerFactory:()=>{const w=new FakeWorker();workers.push(w);return w;}});
  await assert.rejects(client.call('capture'),/timed out/);
  assert.equal(workers[0].terminated,true);
  const pending=client.call('capture'),rejected=assert.rejects(pending,/camera crash/);
  workers[1].onerror({message:'camera crash'});await rejected;
  assert.equal(workers[1].terminated,true);assert.equal(client.pending,null);
});
test('capture worker rejects oversized frames before extraction',async()=>{
  const previous=globalThis.self,messages=[];
  globalThis.self={postMessage:message=>messages.push(message)};
  try{
    await import('../dist/js/dataset/worker.js');
    await self.onmessage({data:{id:1,operation:'capture',options:{image:{width:6000,height:4000,data:new Uint8ClampedArray(1)}}}});
    assert.equal(messages[0].type,'error');assert.match(messages[0].error,/20-megapixel processing limit/);
    await self.onmessage({data:{id:2,operation:'unrecognized'}});
    assert.equal(messages[1].type,'error');assert.match(messages[1].error,/Unknown dataset/);
    const sender=await createSender({seed:4,sessionId:'worker'}),image=renderNeuralQR(sender.frames[0],1024),assignment={collectionCode:sender.manifest.collectionCode,profile:sender.manifest.profile};
    await self.onmessage({data:{id:3,operation:'probe',options:{image,timestamp:1000,...assignment}}});
    assert.equal(messages[2].result.ok,true);assert.equal(messages[2].result.probe,true);
    assert.equal(messages[2].result.info.phase,0);assert.equal(messages[2].result.info.size,64);assert.equal(messages[2].result.info.pixelsPerCell,4);assert.ok(messages[2].result.transform);
    assert.equal(messages[2].result.inputBase64,undefined,'alignment does not serialize a training tensor');
    await self.onmessage({data:{id:4,operation:'capture',options:{image,timestamp:1600,...assignment}}});
    assert.equal(messages[3].result.ok,true);assert.equal(messages[3].result.signature.length,3072);
    assert.equal(base64ToBytes(messages[3].result.inputBase64).byteLength,3*256*256*4,'capture stores the native4ppc crop');
    await self.onmessage({data:{id:5,operation:'compare',options:{signatures:[messages[3].result.signature,messages[3].result.signature]}}});
    assert.equal(messages[4].result.mean,0);
    await self.onmessage({data:{id:6,operation:'select',options:{key:messages[3].result.key,signatures:[messages[3].result.signature,messages[3].result.signature],incomingSignature:messages[3].result.signature}}});
    assert.equal(messages[5].result.replaceIndex,null);assert.equal(messages[5].result.reason,'near-duplicate');
    await self.onmessage({data:{id:7,operation:'capture',options:{image,timestamp:1700,collectionCode:assignment.collectionCode,profile:captureProfile(128)}}});
    assert.equal(messages[6].type,'result');assert.equal(messages[6].result.ok,false);
    assert.equal(messages[6].result.stage,'profile');assert.equal(messages[6].result.stats.detectedSize,64);
    assert.equal(messages[6].result.inputBase64,undefined,'a mismatched optical grid never becomes a saved tensor');
    const highResolution={width:3840,height:2160,data:new Uint8ClampedArray(3840*2160*4)};
    await self.onmessage({data:{id:8,operation:'probe',options:{image:highResolution,timestamp:1800,...assignment}}});
    assert.equal(messages[7].type,'result','a complete4K image reaches acquisition instead of the old1080p rejection');
    assert.equal(messages[7].result.ok,false);assert.equal(messages[7].result.stage,'detect');
  }finally{if(previous===undefined)delete globalThis.self;else globalThis.self=previous;}
});


