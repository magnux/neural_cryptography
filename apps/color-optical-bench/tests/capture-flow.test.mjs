/** Exercise the real page controller with a deterministic camera/worker/DOM harness. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createSender,captureKey} from '../dist/js/dataset/contract.js';
import {captureFrame} from '../dist/js/dataset/capture.js';
import {recordingFilename} from '../dist/js/dataset/filenames.js';
import {meanCaptureDistance} from '../dist/js/dataset/diversity.js';
import {renderFrame} from '../dist/js/format.js';

test('page keeps sampling full pools, applies selection, isolates phases, and pauses without losing records',async()=>{
  const html=await readFile(new URL('../dist/capture.html',import.meta.url),'utf8');
  const css=await readFile(new URL('../dist/js/dataset/style.css',import.meta.url),'utf8');
  const sender=await createSender({seed:90,sessionId:'flow'});
  const record=captureFrame(renderFrame(sender.frames[0],1024),{signature:true});
  let phase=0,grid=64,clock=1000,media=1,stopped=0,filled=0,replacementChoice=null,holdSelection=false,lateSelection=null,download=null,downloadName=null;
  const uploaded=[];let failUpload=true;
  const timers=new Map();let timerId=0;const operations=[],cameraInputs=[],draws=[];
  const rawPixels=Uint8ClampedArray.from({length:4096},(_,i)=>(i*17+Math.floor(i/128))%256);
  const context={clearRect(){},drawImage(...args){draws.push(args);},getImageData:()=>({width:32,height:32,data:rawPixels.slice()}),
    beginPath(){},moveTo(){},lineTo(){},closePath(){},stroke(){},fill(){filled++;}};
  class Element{
    constructor(id){this.id=id;this.value='';this.disabled=false;this.hidden=false;this.handlers={};this.attributes={};this.options=[];this.width=32;this.height=32;}
    addEventListener(type,handler){this.handlers[type]=handler;}
    setAttribute(name,value){this.attributes[name]=value;}getAttribute(name){return this.attributes[name];}removeAttribute(name){delete this.attributes[name];}getContext(){return context;}click(){downloadName=this.download;}
    replaceChildren(...children){this.options=children;}add(child){this.options.push(child);}
    async fire(type='click'){if(type==='click')assert.equal(this.disabled,false,`${this.id} is enabled`);await this.handlers[type]?.();}
  }
  const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element(m[1])]));
  const get=id=>{assert.ok(elements.has(id),`real HTML contains #${id}`);return elements.get(id);};
  const intervalOptions=html.match(/<select id="scanInterval">(.*?)<\/select>/s)[1];
  assert.deepEqual([...intervalOptions.matchAll(/value="(\d+)"/g)].map(match=>Number(match[1])),[100,250,500,1000,2000]);
  assert.match(intervalOptions,/<option value="100" selected>/);
  for(const [id,value] of Object.entries({phaseMaximum:'32',scanInterval:'100',resolution:'720',datasetSplit:'train'}))get(id).value=value;
  assert.equal(elements.has('keepCount'),false);assert.equal(elements.has('thinCaptures'),false);
  assert.ok(html.indexOf('id="collectBurst"')<html.indexOf('id="video"'));
  assert.ok(html.indexOf('id="phaseCount"')<html.indexOf('id="video"'));
  assert.match(css,/\.capture-camera\[data-mirrored=true\]>video,\.capture-camera\[data-mirrored=true\]>canvas\{transform:scaleX\(-1\)\}/,
    'the same CSS transform covers the video and overlay, not the button or capture canvas');
  Object.assign(get('video'),{readyState:2,videoWidth:32,videoHeight:32,play:async()=>{}});
  Object.defineProperty(get('video'),'currentTime',{get:()=>media++});
  const track={label:'Flow camera',stop:()=>stopped++,getSettings:()=>({width:32,height:32,frameRate:30,deviceId:'flow-camera-id',facingMode:'user'})};
  const documentEvents={},windowEvents={};
  class Worker{
    postMessage({id,operation,options}){
      operations.push(operation);const info={...record.info,size:grid,phase},key=captureKey(info);
      if(options.image)cameraInputs.push(options.image.data.slice());
      const result=operation==='select'?{replaceIndex:replacementChoice,reason:'test selection'}:operation==='compare'?meanCaptureDistance(options.signatures):operation==='probe'
        ?{ok:true,probe:true,info,key,transform:record.transform,stats:record.stats}
        :{...record,info,key,timestamp:options.timestamp};
      const reply=()=>this.onmessage?.({data:{id,type:'result',result}});
      if(operation==='select'&&holdSelection){holdSelection=false;lateSelection=reply;}else queueMicrotask(reply);
    }
    terminate(){}
  }
  const NativeURL=URL;
  class UploadRequest{
    constructor(){this.upload={};this.status=201;}
    open(){}setRequestHeader(){}abort(){this.onabort?.();}
    send(body){uploaded.push(body);queueMicrotask(()=>{if(failUpload){failUpload=false;this.onerror?.();}else{this.responseText=JSON.stringify({ok:true,storedPath:'train/g64/receiver/test.json'});this.onload?.();}});}
  }
  const replacements={URL:class extends NativeURL{static createObjectURL(blob){download=blob;return 'blob:download';}static revokeObjectURL(){}},Worker,isSecureContext:true,Option:class{constructor(label,value){this.label=label;this.value=value;}},
    fetch:async()=>({ok:true,json:async()=>({enabled:true,maxUploadBytes:160*1048576,uploadToken:'test-token'})}),XMLHttpRequest:UploadRequest,
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
    assert.equal(get('mirrorPreview').getAttribute('aria-pressed'),'false');
    // The display preference can be set before starting the camera.
    await get('mirrorPreview').fire();
    assert.equal(get('cameraPreview').getAttribute('data-mirrored'),'true');
    assert.equal(get('mirrorPreview').getAttribute('aria-pressed'),'true');
    await get('receiverRole').fire();await get('startCamera').fire();await flush();
    assert.ok(operations.includes('probe'));assert.ok(filled>0,'matrix overlay fills its area');
    assert.equal(get('sampleCount').textContent,'0 / 128');
    await get('mirrorPreview').fire();await tick();
    assert.equal(get('cameraPreview').getAttribute('data-mirrored'),'false');
    assert.equal(get('mirrorPreview').getAttribute('aria-pressed'),'false');
    await get('mirrorPreview').fire();await tick();
    for(const pixels of cameraInputs)assert.deepEqual(pixels,rawPixels,'preview mirroring never flips worker pixels');
    for(const args of draws)assert.deepEqual(args,[get('video'),0,0,32,32],'capture uses the untransformed video source');
    await get('collectBurst').fire();await flush();
    for(let i=0;i<32;i++)await tick();
    assert.equal(get('phaseCount').textContent,'32 / 32');assert.equal(get('meanDistance').textContent,'0.00%');
    await tick();assert.equal(operations.at(-1),'select','full pool continues capturing and selecting');
    assert.equal(get('phaseCount').textContent,'32 / 32');assert.equal(get('replacementCount').textContent,'0');
    replacementChoice=0;await tick();
    assert.equal(get('phaseCount').textContent,'32 / 32');assert.equal(get('replacementCount').textContent,'1');
    const seen=Number(get('seenCount').textContent);assert.ok(seen>32);
    // A selection already in flight must not replace a record after pause.
    holdSelection=true;await tick();assert.ok(lateSelection);
    await get('cancelBurst').fire();await flush();lateSelection();await flush();
    assert.equal(get('replacementCount').textContent,'1');
    await get('collectBurst').fire();await flush();
    phase=1;await tick();await tick();
    assert.equal(get('phaseCount').textContent,'1 / 32');assert.equal(get('sampleCount').textContent,'33 / 128');
    grid=128;await tick();assert.equal(get('sampleCount').textContent,'33 / 128');
    assert.match(get('receiverStatus').textContent,/different profile/);assert.equal(get('cancelBurst').disabled,true);
    assert.equal(get('datasetSplit').disabled,true,'split remains fixed for the physical session');
    grid=64;await get('collectBurst').fire();await flush();
    await get('cancelBurst').fire();await flush();const count=get('sampleCount').textContent;
    await tick();assert.equal(get('sampleCount').textContent,count);assert.equal(get('downloadCaptures').disabled,false);
    await get('stopCamera').fire();await flush();assert.equal(timers.size,0);assert.equal(stopped,1);
    assert.equal(get('cameraPreview').getAttribute('data-mirrored'),'true','camera stop preserves the preview preference');
    await get('startCamera').fire();await flush();document.hidden=true;documentEvents.visibilitychange();await flush();
    assert.equal(get('mirrorPreview').getAttribute('aria-pressed'),'true');
    assert.equal(timers.size,0);assert.equal(stopped,2);assert.equal(get('sampleCount').textContent,count);
    await get('downloadCaptures').fire();
    const exported=JSON.parse(await download.text());
    assert.equal(downloadName,recordingFilename(exported));
    assert.ok(downloadName.startsWith(`${record.info.chunkCRC.toString(16).padStart(8,'0')}_`));
    assert.equal(exported.format,'optical-decoder-captures');assert.equal(exported.version,1);
    assert.equal(exported.datasetSplit,'train');assert.ok(exported.environment.deviceId);
    assert.equal(exported.camera.label,'Flow camera');assert.equal(exported.camera.deviceId,'flow-camera-id');
    assert.equal(exported.camera.active,false,'camera metadata survives stopping capture');
    assert.equal(exported.records.length,33);assert.equal(exported.collection.keepRandom,0);
    assert.equal(exported.collection.scanIntervalMs,100);
    assert.deepEqual(Object.keys(exported.collection).sort(),['distances','keepRandom','maxPerPhase','scanIntervalMs']);
    assert.ok(exported.records.every(row=>row.inputBase64&&row.samplesBase64&&row.stability&&!('signature' in row)));
    assert.ok(exported.records.every(row=>row.inputBase64===record.inputBase64&&row.samplesBase64===record.samplesBase64),
      'mirroring does not change recorded tensors or decoder samples');
    assert.ok(Array.isArray(exported.selection.recentThinning));
    await get('uploadCaptures').fire();
    assert.equal(get('sampleCount').textContent,count,'failed upload preserves every capture');
    assert.match(get('receiverUploadStatus').textContent,/failed/);
    await get('uploadCaptures').fire();
    assert.equal(await uploaded[0].text(),await uploaded[1].text(),'retry reuses identical timestamp and snapshot bytes');
    assert.match(get('receiverUploadStatus').textContent,/Saved: train\/g64\/receiver/);
    await get('clearCaptures').fire();assert.equal(get('sampleCount').textContent,'0 / 128');
    assert.equal(get('datasetSplit').disabled,true,'saving and clearing a batch does not reset its physical split');
    await get('newReceiverSession').fire();assert.equal(get('datasetSplit').disabled,false);
    timers.clear();
  }finally{
    windowEvents.pagehide?.();Date.now=now;
    for(const [k,descriptor] of saved)if(descriptor)Object.defineProperty(globalThis,k,descriptor);else delete globalThis[k];
  }
});

