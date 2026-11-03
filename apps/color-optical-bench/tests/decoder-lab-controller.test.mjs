import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {validateOptions,summarizeTimings,validFrameCount,MAX_INPUT_FRAMES} from '../dist/js/decoder-lab/runtime.js';

async function harness({advance=true,alwaysWaiting=false,duplicateFrame=false,frames=4,frameInterval=100,capacity=frames,captureFrames=0}={}) {
  const nodes=new Map(),calls=[],clients=[],pendingFrames=new Map();let stopped=0,requestedFrames=0,canceledFrames=0,presented=0,clock=0;
  class Element {
    constructor(id='') {this.id=id;this.value='';this.hidden=false;this.disabled=false;this.dataset={};this.children=[];this.listeners={};}
    addEventListener(name,fn) {this.listeners[name]=fn;}
    replaceChildren(...items) {this.children=items;}
    append(...items) {this.children.push(...items);}
    add(item) {this.children.push(item);}
    getContext() {return {drawImage(){},getImageData:(_x,_y,width,height)=>({width,height,data:new Uint8ClampedArray(width*height*4)})};}
  }
  function element(id) {if(!nodes.has(id))nodes.set(id,new Element(id));return nodes.get(id);}
  for(const [id,value] of Object.entries({modelSize:'64',backend:'wasm',iterations:'1',resolution:'720',captureFrames:String(captureFrames)}))element(id).value=value;
  const video=element('video');Object.assign(video,{videoWidth:32,videoHeight:32,readyState:2,currentTime:1,play:async()=>{}});
  video.requestVideoFrameCallback=callback=>{
    const id=++requestedFrames;pendingFrames.set(id,callback);
    if(advance||id===1)queueMicrotask(()=>{
      if(!pendingFrames.has(id))return;
      pendingFrames.delete(id);video.currentTime+=.1;clock+=frameInterval;
      if(!(duplicateFrame&&id===2))presented++;
      callback(clock,{mediaTime:presented/30,presentedFrames:presented});
    });
    return id;
  };
  video.cancelVideoFrameCallback=id=>{pendingFrames.delete(id);canceledFrames++;};
  const track={getSettings:()=>({width:32,height:32,frameRate:30}),stop:()=>{stopped++;}};
  const stream={getVideoTracks:()=>[track],getTracks:()=>[track]};
  class FakeClient {
    constructor() {this.canceled=0;clients.push(this);}
    async call(operation,options,transfer) {
      calls.push({operation,options,transfer,captureStartedMs:clock});
      // The playback clock may advance independently of presented frames.
      video.currentTime+=.01;
      if(alwaysWaiting||calls.length%frames!==0)return {waiting:true,collected:calls.length%frames||1,framesPerInput:capacity,requiredCaptures:frames,operation:'capture'};
      return {operation:'capture',modelStatus:'trained',size:64,paletteColors:16,framesPerInput:capacity,requiredCaptures:frames,inputCaptures:frames,
        inputStats:{preprocessMs:8},inferenceMs:1,effectiveBackend:'wasm',logitsFinite:true,note:'fixture'};
    }
    cancel() {this.canceled++;}
  }
  const document={getElementById:element,createElement:tag=>new Element(tag),addEventListener(){},hidden:false};
  const context=vm.createContext({DecoderClient:FakeClient,validateOptions,summarizeTimings,validFrameCount,MAX_INPUT_FRAMES,document,
    window:{addEventListener(){}},navigator:{mediaDevices:{getUserMedia:async()=>stream,enumerateDevices:async()=>[]}},
    isSecureContext:true,Worker:function(){},crypto:globalThis.crypto,performance:{now:()=>clock},DOMException,Date,
    Uint8ClampedArray,Option:class{},requestAnimationFrame:()=>1,cancelAnimationFrame(){},setTimeout:(fn,ms)=>setTimeout(fn,ms===3000?30:ms),clearTimeout});
  const source=(await readFile(new URL('../dist/js/decoder-lab/lab.js',import.meta.url),'utf8')).replace(/^import .*;\n/gm,'');
  vm.runInContext(source+'\nglobalThis.testLab={run,startCamera,stopCamera,cancel,get records(){return records},get busy(){return busy}};',context);
  await context.testLab.startCamera();
  return {lab:context.testLab,video,nodes,calls,clients,get stopped(){return stopped;},pendingFrames,get canceledFrames(){return canceledFrames},get requestedFrames(){return requestedFrames}};
}

test('Decoder Lab camera action gathers four distinct frames with one run identity',async()=>{
  const h=await harness();await h.lab.run('capture');
  assert.equal(h.calls.length,4);assert.equal(new Set(h.calls.map(c=>c.options.mediaTime)).size,4);
  assert.equal(new Set(h.calls.map(c=>c.options.presentedFrames)).size,4);
  assert.equal(new Set(h.calls.map(c=>c.options.runId)).size,1);
  for(const call of h.calls)assert.equal(call.transfer[0],call.options.image.data.buffer);
  assert.equal(h.lab.records.length,1);assert.equal(h.lab.records[0].inputCaptures,4);
  assert.equal(h.lab.busy,false);assert.equal(h.nodes.get('runCapture').disabled,false);
});

test('camera stop cancels a burst waiting for a new exposure; restart uses a new run',async()=>{
  const h=await harness({advance:false}),pending=h.lab.run('capture');
  while(!h.calls.length)await new Promise(resolve=>setTimeout(resolve,1));
  const oldRun=h.calls[0].options.runId;
  h.lab.stopCamera();await pending;
  assert.equal(h.calls.length,1);assert.equal(h.clients[0].canceled,1);assert.equal(h.stopped,1);
  assert.equal(h.pendingFrames.size,0);assert.equal(h.canceledFrames,1);
  assert.equal(h.lab.busy,false);assert.equal(h.lab.records[0].canceled,true);
  await h.lab.startCamera();
  const emit=h.video.requestVideoFrameCallback;let restartFrame=true;
  h.video.requestVideoFrameCallback=callback=>{const id=emit(callback);if(restartFrame){restartFrame=false;queueMicrotask(()=>{if(h.pendingFrames.has(id)){h.pendingFrames.delete(id);callback(performance.now(),{mediaTime:2,presentedFrames:2});}});}return id;};
  const next=h.lab.run('capture');
  while(h.calls.length<2)await new Promise(resolve=>setTimeout(resolve,1));
  assert.notEqual(h.calls[1].options.runId,oldRun);
  h.lab.cancel();await next;
  assert.equal(h.lab.records[1].canceled,true);
});

test('camera collection stops after a bounded number of phase resets',async()=>{
  for(const frames of [4,5,64]) {
    const h=await harness({alwaysWaiting:true,frames});await h.lab.run('capture');
    assert.equal(h.calls.length,4*frames);assert.equal(h.clients[0].canceled,1);
    assert.equal(h.lab.records[0].ok,false);
    assert.match(h.lab.records[0].error,/phase kept changing/);
    assert.equal(h.lab.busy,false);
  }
});

test('five-frame camera model samples distinct exposures at 100 ms start-to-start',async()=>{
  const h=await harness({frames:5,frameInterval:25});await h.lab.run('capture');
  assert.equal(h.calls.length,5);
  assert.deepEqual(h.calls.map(c=>c.captureStartedMs),[25,125,225,325,425]);
  assert.deepEqual(h.calls.map(c=>c.options.presentedFrames),[1,5,9,13,17]);
  assert.equal(h.lab.records[0].inputCaptures,5);
  assert.equal(h.lab.records[0].camera.requestedSampleIntervalMs,100);
});


test('playback-clock changes cannot make the same presented frame count twice',async()=>{
  const h=await harness({duplicateFrame:true});await h.lab.run('capture');
  assert.equal(h.calls.length,4);assert.equal(h.requestedFrames,5);
  assert.deepEqual(h.calls.map(c=>c.options.presentedFrames),[1,2,3,4]);
  assert.deepEqual(h.calls.map(c=>c.options.mediaTime),[1/30,2/30,3/30,4/30]);
});
test('a stalled camera times out and cancels its pending frame callback',async()=>{
  const h=await harness({advance:false});await h.lab.run('capture');
  assert.equal(h.calls.length,1);assert.equal(h.pendingFrames.size,0);assert.equal(h.canceledFrames,1);
  assert.match(h.lab.records[0].error,/No new camera exposure/);assert.equal(h.lab.busy,false);
});
test('unsupported browsers fail clearly instead of substituting playback time',async()=>{
  const h=await harness();delete h.video.requestVideoFrameCallback;
  await h.lab.run('capture');assert.equal(h.calls.length,0);
  assert.match(h.lab.records[0].error,/requestVideoFrameCallback/);
});


test('variable-frame camera selection sends actual count while preserving eight-slot model capacity',async()=>{
  for(const frames of [5,6,7,8]) {
    const h=await harness({frames,capacity:8,captureFrames:frames});await h.lab.run('capture');
    assert.equal(h.calls.length,frames);
    assert.ok(h.calls.every(call=>call.options.captureFrames===frames));
    assert.equal(h.lab.records[0].inputCaptures,frames);
    assert.equal(h.lab.records[0].framesPerInput,8);
    assert.equal(h.lab.records[0].requiredCaptures,frames);
  }
});
