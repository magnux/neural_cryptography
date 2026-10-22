import test from 'node:test';
import assert from 'node:assert/strict';
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
    await self.onmessage({data:{id:1,operation:'capture',options:{image:{width:2048,height:2048,data:new Uint8ClampedArray(1)}}}});
    assert.equal(messages[0].type,'error');assert.match(messages[0].error,/1080p pixel limit/);
    await self.onmessage({data:{id:2,operation:'unrecognized'}});
    assert.equal(messages[1].type,'error');assert.match(messages[1].error,/Unknown dataset/);
  }finally{if(previous===undefined)delete globalThis.self;else globalThis.self=previous;}
});
