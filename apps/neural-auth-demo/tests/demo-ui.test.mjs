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

test('display defaults and saved preferences cannot change captured or transmitted pixels',async()=>{
  const {displayPreferences,matrixEdge}=await import('../dist/app.js');
  assert.deepEqual(displayPreferences(),{mirrored:true,qrPosition:'bottom'});
  assert.deepEqual(displayPreferences('{bad json'),{mirrored:true,qrPosition:'bottom'});
  assert.deepEqual(displayPreferences('{"mirrored":false,"qrPosition":"top"}'),{mirrored:false,qrPosition:'top'});
  assert.deepEqual(displayPreferences('{"mirrored":0,"qrPosition":"left","outgoingSize":128}'),{mirrored:true,qrPosition:'bottom'});
  for(const [width,height,edge] of [[360,520,360],[360,233.9,233],[540,297,297],[0,250,0],[300,-10,0],[NaN,300,0]])assert.equal(matrixEdge(width,height),edge);
});

test('preview and placement toggles persist independently, size the remaining slot, and leave pixel canvases untouched',async t=>{
  const previous=new Map(['document','window','navigator','localStorage','ResizeObserver'].map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
  t.after(()=>{for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}});
  const elements=new Map(),saved=new Map(),windowListeners=new Map();let matrixObserver;
  function element(id){
    if(!elements.has(id)){
      const attrs=new Map(),styles=new Map(),listeners=new Map();
      elements.set(id,{id,querySelector:selector=>element(id+'/'+selector),dataset:{},textContent:'',hidden:false,disabled:false,open:false,scrollTop:0,
        clientWidth:id==='matrixSlot'?340:0,clientHeight:id==='matrixSlot'?270:0,
        style:{setProperty:(name,value)=>styles.set(name,value),getPropertyValue:name=>styles.get(name)||''},
        setAttribute:(name,value)=>attrs.set(name,value),getAttribute:name=>attrs.get(name),
        addEventListener:(name,callback)=>listeners.set(name,callback),dispatch:name=>listeners.get(name)?.(),
        // No view preference may call a 2D context or touch camera pixels.
        getContext:()=>{throw Error(`Unexpected pixel access: ${id}`);}});
    }
    return elements.get(id);
  }
  const root=element('root');
  Object.defineProperty(globalThis,'document',{configurable:true,value:{body:element('body'),documentElement:root,getElementById:element,createElement:tag=>element('detached/'+tag),addEventListener(){}}});
  Object.defineProperty(globalThis,'window',{configurable:true,value:{innerHeight:844,visualViewport:{height:844,addEventListener(){}},addEventListener:(name,callback)=>windowListeners.set(name,callback)}});
  Object.defineProperty(globalThis,'navigator',{configurable:true,value:{userAgent:'iPhone',mediaDevices:{}}});
  Object.defineProperty(globalThis,'localStorage',{configurable:true,value:{getItem:key=>saved.get(key)??null,setItem:(key,value)=>saved.set(key,value)}});
  Object.defineProperty(globalThis,'ResizeObserver',{configurable:true,value:class{constructor(callback){matrixObserver=callback;}observe(){}}});
  await import(`../dist/app.js?display-ui-test=${Date.now()}`);
  assert.equal(element('app').dataset.previewMirrored,'true');assert.equal(element('app').dataset.qrPosition,'bottom');
  assert.equal(element('matrixSlot').style.getPropertyValue('--matrix-edge'),'270px');
  element('mirrorPreview').dispatch('click');
  assert.equal(element('app').dataset.previewMirrored,'false');assert.equal(element('mirrorPreview').textContent,'Normal');
  assert.equal(element('mirrorPreview').getAttribute('aria-pressed'),'false');
  element('qrPosition').dispatch('click');
  assert.equal(element('app').dataset.qrPosition,'top');assert.equal(element('qrPosition').getAttribute('aria-pressed'),'true');
  assert.deepEqual(JSON.parse(saved.get('neural-key-display')),{mirrored:false,qrPosition:'top'});
  element('matrixSlot').clientWidth=510;element('matrixSlot').clientHeight=290;matrixObserver();
  assert.equal(element('matrixSlot').style.getPropertyValue('--matrix-edge'),'290px');
  window.visualViewport.height=390;windowListeners.get('resize')();
  assert.equal(root.style.getPropertyValue('--viewport-height'),'390px');
  element('app').scrollTop=450;element('settings').dispatch('toggle');assert.equal(element('app').scrollTop,0);
  // Visiting timing must not open a camera or start cryptography. Returning to
  // the other tabs restores their independent panels and hides timing controls.
  element('roundtripTab').dispatch('click');
  assert.equal(element('app').dataset.mode,'roundtrip');
  assert.equal(element('roundtripTab').getAttribute('aria-pressed'),'true');
  assert.equal(element('roundtripPanel').hidden,false);assert.equal(element('cameraPanel').hidden,true);
  assert.equal(element('simulationPanel').hidden,true);
  element('simulationTab').dispatch('click');
  assert.equal(element('roundtripPanel').hidden,true);assert.equal(element('simulationPanel').hidden,false);
  element('cameraTab').dispatch('click');
  assert.equal(element('roundtripTab').getAttribute('aria-pressed'),'false');assert.equal(element('cameraPanel').hidden,false);
  assert.equal(elements.has('captureCanvas'),false);assert.equal(elements.has('gateCanvas'),false);assert.equal(elements.has('matrix'),false);
});
