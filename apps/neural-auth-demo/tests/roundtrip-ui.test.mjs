import test from 'node:test';
import assert from 'node:assert/strict';
import {setTimeout as delay} from 'node:timers/promises';
import {initRoundtrip,timingSummary} from '../dist/roundtrip-ui.js';

const settings={role:'sender',size:'16',facing:'user',device:'',resolution:'720',sample:'100',count:'2',deadline:'500',exposure:'50'};

async function until(predicate){
  for(let i=0;i<200&&!predicate();i++)await delay(10);
  assert.ok(predicate(),'expected controller transition');
}

// Hold the camera between exposures, as happens while two users are aligning
// their phones. Use the actual matrix generator, renderer and deadline loop.
function harness(t,{camera}={}){
  const nodes=new Map(),workers=[],raf=new Set(),tracks=[];
  function element(key){
    if(!nodes.has(key)){
      const listeners=new Map(),name=key.match(/^\[name="(\w+)"\]$/)?.[1],value=settings[name]??'';
      nodes.set(key,{value,hidden:false,disabled:false,checked:false,textContent:'',dataset:{},draws:0,
        get valueAsNumber(){return this.value===''?NaN:Number(this.value);},
        addEventListener:(event,fn)=>listeners.set(event,fn),fire:(event)=>listeners.get(event)?.(),
        getContext(){return {putImageData:()=>{this.draws++;},clearRect(){}};}});
    }
    return nodes.get(key);
  }
  const video=element('video');Object.assign(video,{requestVideoFrameCallback(){return 37;},cancelVideoFrameCallback(){},async play(){}});
  const stream=()=>{const track={stopped:false,stop(){this.stopped=true;},addEventListener(){},getSettings:()=>({width:1280,height:720})};tracks.push(track);return {getVideoTracks:()=>[track],getTracks:()=>[track]};};
  const replacements={window:{addEventListener(){}},document:{addEventListener(){},createElement:element},
    isSecureContext:true,navigator:{mediaDevices:{getUserMedia:()=>camera?camera(stream):stream()}},
    Worker:class{constructor(){workers.push(this);}postMessage(m){if(m.op==='init')queueMicrotask(()=>this.onmessage?.({data:{id:m.id,value:{}}}));}
      terminate(){this.terminated=true;}},
    ImageData:class{constructor(data,width,height){Object.assign(this,{data,width,height});}},
    requestAnimationFrame(fn){const timer=setTimeout(()=>{raf.delete(timer);fn(performance.now());},0);raf.add(timer);return timer;},
    cancelAnimationFrame(timer){clearTimeout(timer);raf.delete(timer);}
  };
  const old=Object.keys(replacements).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]);
  for(const [key,value] of Object.entries(replacements))Object.defineProperty(globalThis,key,{configurable:true,value});
  const activity=[],panel=initRoundtrip({querySelector:element},{onActive:flag=>activity.push(flag)});
  t.after(()=>{panel.stop();for(const id of raf)clearTimeout(id);for(const [key,descriptor] of old){
    if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}});
  return {element,panel,activity,workers,tracks};
}

test('sender blanks an exposed probe, times out, advances exactly once and releases all resources on cancel',async t=>{
  const {element:el,panel,activity,workers,tracks}=harness(t),matrix=el('.rt-matrix');
  el('.rt-go').fire('click');
  await until(()=>matrix.draws===1);
  assert.equal(panel.active,true);assert.match(el('.rt-msg').textContent,/probe 1 \/ 2 · phase 1/);
  await until(()=>matrix.hidden);assert.equal(el('.rt-empty').textContent,'Waiting for echo');
  await until(()=>matrix.draws===2);
  assert.match(el('.rt-msg').textContent,/probe 2 \/ 2 · phase 2/);
  assert.equal(el('[data-stat="timeouts"]').textContent,1);assert.equal(el('[data-stat="count"]').textContent,'0 / 2');
  panel.stop();assert.equal(tracks[0].stopped,true);assert.equal(workers[0].terminated,true);
  assert.equal(el('video').srcObject,null);assert.deepEqual(activity,[true,false]);assert.equal(matrix.hidden,true);
  await delay(550);assert.equal(matrix.draws,2);assert.equal(el('[data-stat="timeouts"]').textContent,1);
});

test('a late camera permission response after tab cancellation releases the stream and never starts a trial',async t=>{
  let grant;
  const {element:el,panel,activity,workers,tracks}=harness(t,{camera:create=>new Promise(resolve=>{grant=()=>resolve(create());})});
  el('.rt-go').fire('click');await until(()=>grant);panel.stop();grant();await until(()=>tracks[0]?.stopped);
  assert.equal(el('.rt-matrix').draws,0);assert.equal(el('[data-stat="count"]').textContent,'0 / 0');
  assert.equal(workers[0].terminated,true);assert.equal(panel.active,false);assert.deepEqual(activity,[true,false]);
});

test('roundtrip reports count misses separately from RTT and pool only payload cells actually read',()=>{
  assert.deepEqual(timingSummary([{status:'timeout'}]),{attempts:1,matched:0,timeouts:1,minMs:null,meanMs:null,maxMs:null,cer:null});
  const result=timingSummary([{status:'matched',rttMs:50,errors:1,cells:8},{status:'timeout'},
    {status:'matched',rttMs:100,errors:0,cells:12},{status:'cancelled'}]);
  assert.deepEqual(result,{attempts:4,matched:2,timeouts:1,minMs:50,meanMs:75,maxMs:100,cer:.05});
});
