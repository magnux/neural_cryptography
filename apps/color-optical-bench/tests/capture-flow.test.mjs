/** Exercise the real page controller with a deterministic camera/worker/DOM harness. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createSender,captureKey} from '../dist/js/dataset/contract.js';
import {captureFrame} from '../dist/js/dataset/capture.js';
import {meanCaptureDistance} from '../dist/js/dataset/diversity.js';
import {renderFrame} from '../dist/js/format.js';

test('page scans automatically, fills/thins/resumes pools, switches phases, and stops safely',async()=>{
  const html=await readFile(new URL('../dist/capture.html',import.meta.url),'utf8');
  const sender=await createSender({seed:90,sessionId:'flow'});
  const record=captureFrame(renderFrame(sender.frames[0],1024),{signature:true});
  let phase=0,clock=1000,media=1,stopped=0,filled=0;
  const timers=new Map();let timerId=0;const operations=[];
  const context={clearRect(){},drawImage(){},getImageData:()=>({width:32,height:32,data:new Uint8ClampedArray(4096)}),
    beginPath(){},moveTo(){},lineTo(){},closePath(){},stroke(){},fill(){filled++;}};
  class Element{
    constructor(id){this.id=id;this.value='';this.disabled=false;this.hidden=false;this.handlers={};this.options=[];this.width=32;this.height=32;}
    addEventListener(type,handler){this.handlers[type]=handler;}
    setAttribute(){}removeAttribute(){}getContext(){return context;}
    replaceChildren(...children){this.options=children;}add(child){this.options.push(child);}
    async fire(type='click'){if(type==='click')assert.equal(this.disabled,false,`${this.id} is enabled`);await this.handlers[type]?.();}
  }
  const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element(m[1])]));
  const get=id=>{assert.ok(elements.has(id),`real HTML contains #${id}`);return elements.get(id);};
  for(const [id,value] of Object.entries({phaseMaximum:'32',keepCount:'16',scanInterval:'500',resolution:'720'}))get(id).value=value;
  get('keepCount').options=[4,8,16,32].map(value=>({value:String(value)}));
  Object.assign(get('video'),{readyState:2,videoWidth:32,videoHeight:32,play:async()=>{}});
  Object.defineProperty(get('video'),'currentTime',{get:()=>media++});
  const track={stop:()=>stopped++,getSettings:()=>({width:32,height:32,frameRate:30})};
  const documentEvents={},windowEvents={};
  class Worker{
    postMessage({id,operation,options}){
      operations.push(operation);const info={...record.info,phase},key=captureKey(info);
      const result=operation==='compare'?meanCaptureDistance(options.signatures):operation==='probe'
        ?{ok:true,probe:true,info,key,transform:record.transform,stats:record.stats}
        :{...record,info,key,timestamp:options.timestamp};
      queueMicrotask(()=>this.onmessage?.({data:{id,type:'result',result}}));
    }
    terminate(){}
  }
  const replacements={Worker,isSecureContext:true,Option:class{constructor(label,value){this.label=label;this.value=value;}},
    navigator:{mediaDevices:{enumerateDevices:async()=>[],getUserMedia:async()=>({getVideoTracks:()=>[track],getTracks:()=>[track]})}},
    document:{getElementById:get,createElement:id=>new Element(id),addEventListener:(k,f)=>documentEvents[k]=f,
      body:{classList:{remove(){},contains:()=>false}}},window:{addEventListener:(k,f)=>windowEvents[k]=f},
    setTimeout:(fn,ms)=>{const id=++timerId;timers.set(id,{fn,ms});return id;},clearTimeout:id=>timers.delete(id)};
  const saved=new Map(Object.keys(replacements).map(k=>[k,Object.getOwnPropertyDescriptor(globalThis,k)]));
  const now=Date.now;Date.now=()=>clock;
  const flush=()=>new Promise(resolve=>setImmediate(resolve));
  async function tick(){await flush();const next=[...timers].find(([,x])=>x.ms<1000);assert.ok(next,'one scanner tick scheduled');timers.delete(next[0]);clock+=600;next[1].fn();await flush();}
  try{
    for(const [k,value] of Object.entries(replacements))Object.defineProperty(globalThis,k,{value,configurable:true,writable:true});
    await import('../dist/js/dataset/lab.js');
    await get('receiverRole').fire();await get('startCamera').fire();await flush();
    assert.ok(operations.includes('probe'));assert.ok(filled>0,'matrix overlay fills its area');
    assert.equal(get('sampleCount').textContent,'0 / 128');
    await get('collectBurst').fire();await flush();
    for(let i=0;i<32;i++)await tick();
    assert.equal(get('phaseCount').textContent,'32 / 32');assert.equal(get('meanDistance').textContent,'0.00%');
    await tick();assert.equal(operations.at(-1),'probe','full pool keeps scanning without storing tensors');
    await get('thinCaptures').fire();await flush();assert.equal(get('phaseCount').textContent,'16 / 32');
    for(let i=0;i<16;i++)await tick();assert.equal(get('phaseCount').textContent,'32 / 32');
    phase=1;await tick();await tick();await tick();
    assert.equal(get('phaseCount').textContent,'1 / 32');assert.equal(get('sampleCount').textContent,'33 / 128');
    await get('cancelBurst').fire();await flush();const count=get('sampleCount').textContent;
    await tick();assert.equal(get('sampleCount').textContent,count);assert.equal(get('downloadCaptures').disabled,false);
    await get('stopCamera').fire();await flush();assert.equal(timers.size,0);assert.equal(stopped,1);
    await get('startCamera').fire();await flush();document.hidden=true;documentEvents.visibilitychange();await flush();
    assert.equal(timers.size,0);assert.equal(stopped,2);assert.equal(get('sampleCount').textContent,count);
  }finally{
    windowEvents.pagehide?.();Date.now=now;
    for(const [k,descriptor] of saved)if(descriptor)Object.defineProperty(globalThis,k,descriptor);else delete globalThis[k];
  }
});
