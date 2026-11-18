import test from 'node:test';
import assert from 'node:assert/strict';
import {timingWorkerHandler} from '../dist/roundtrip-worker.js';
import {inputPresence} from '../dist/js/corrector-lab/runtime.js';

test('classical timer runs without model files and bounds camera captures',async()=>{
  const replies=[],handle=timingWorkerHandler(m=>replies.push(m),{createCorrector:()=>assert.fail('Classical mode opened a model')});
  const send=async(op,args)=>{await handle({data:{op,id:1,...args}});return replies.at(-1);};
  assert.equal((await send('init',{size:128,inference:false})).value.inference,false);
  assert.match((await send('scan',{image:{width:8192,height:8192,data:new Uint8ClampedArray(0)},captureIntervalMs:100})).error,/bounds/);
  assert.match((await send('infer')).error,/disabled/);
});

test('dummy inference uses a masked synthetic tensor and never publishes its logits',async()=>{
  const manifest={framesPerInput:8,modelProfile:'VW',inputLayout:'phase-paired-rgb-presence-planes',channels:32,frameCounts:[4,5,6,7,8]};
  const replies=[],read=[],size=32,context={entry:{size}},model={manifest,loadContext:async()=>context,
    async infer(ctx,pixels){assert.equal(ctx,context);assert.deepEqual(inputPresence(pixels,manifest,size),[0,2,4,6]);read.push(pixels);return {ms:17,values:new Float32Array([42])};}};
  let release;
  const handler=timingWorkerHandler(m=>replies.push(m),{createCorrector:()=>new Promise(resolve=>{release=()=>resolve(model);})});
  const run=(id,op,options={})=>handler({data:{id,op,...options}});
  await run(0,'init',{size:128,inference:true});
  assert.match(replies.pop().error,/G32 or G64/);
  const init=run(1,'init',{size,inference:true});
  await run(2,'infer');assert.match(replies.pop().error,/busy/);
  release();await init;await run(3,'infer');await run(4,'infer');
  assert.deepEqual(replies.at(-1).value,{inferenceMs:17,backend:'wasm',size});
  assert.equal(read[0],read[1]);
});
