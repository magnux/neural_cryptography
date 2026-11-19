import test from 'node:test';
import assert from 'node:assert/strict';
import {setTimeout as wait} from 'node:timers/promises';
import {initSweep} from '../dist/sweep-ui.js';
import {SweepReader} from '../dist/sweep-reader.js';
import {PALETTE} from '../dist/js/format.js';
import {gridLayout,paletteMap} from '../dist/js/grid.js';
import {NQR_LAYOUT} from '../dist/js/neural-qr/layout.js';
import {decodeNeuralQRSymbols} from '../dist/js/neural-qr/packet.js';

// Only the camera/worker boundary is substituted. Read the actual opposite
// canvas at known, perfectly aligned cell centers. This is not a model,
// a four-frame accuracy test, or a measurement of physical latency.
function readCanvas(image,size){
  if(!image)return {ok:false,stage:'detect'};
  const pixel=(x,y)=>{
    const i=4*(Math.floor(y*image.height/256)*image.width+Math.floor(x*image.width/256));
    return PALETTE.findIndex(c=>c.every((v,j)=>v===image.data[i+j]));
  };
  const phase=pixel(...NQR_LAYOUT.phasePairs[0][0]);assert.ok(phase>=0&&phase<4);
  const symbols=new Uint8Array(size*size),{x,y,side}=NQR_LAYOUT.payload,cell=side/size;
  const info={size,phase,phases:4,colors:16,permute:true};
  for(const i of gridLayout(size,16,8).cells)symbols[i]=paletteMap(pixel(x+((i%size)+.5)*cell,y+((i/size|0)+.5)*cell),info);
  const decoded=decodeNeuralQRSymbols(symbols,size);assert.ok(decoded.ok,decoded.error);
  return {ok:true,chunk:decoded.chunk,info,inputCaptures:4,phaseCounts:[1,1,1,1],corners:[[0,0],[1,0],[1,1],[0,1]]};
}
async function until(pred,description){
  for(let i=0;i<500&&!pred();i++)await wait(4);
  assert.ok(pred(),description);
}

function devices(t){
  const urls=[],tracks=[],closed=[],raf=new Set(),inFlight=new Map();let seq=0,blocked=null;
  class Element{
    constructor(sel='',value=''){this.value=value;this.id=sel;this.dataset={};this.listeners={};this.hidden=false;this.draws=0;}
    get valueAsNumber(){return Number(this.value);}
    addEventListener(k,fn){this.listeners[k]=fn;}click(){if(!this.disabled)this.listeners.click?.();}
    getContext(){const self=this;return {clearRect(){self.image=null;},beginPath(){},moveTo(){},lineTo(){},closePath(){},stroke(){},
      putImageData(img){self.image=img;self.draws++;},drawImage(v){self.video=v;},
      getImageData(){const {peer}=self.video;return (self.video.override??(!peer.hidden&&peer.image))||null;}};}
    play(){} requestVideoFrameCallback(fn){assert.ok(!this.cb,'at most one camera callback is pending');this.cb=fn;return ++seq;}
    cancelVideoFrameCallback(){this.cb=null;}
    emit(){if(!this.cb)return false;const fn=this.cb;this.cb=null;const n=++seq;fn(500*n,{mediaTime:n/2,presentedFrames:n});return true;}
  }
  const media={async getUserMedia(){const tk={stop(){this.stopped=true;},addEventListener(){},getSettings:()=>({width:512,height:512})};
    tracks.push(tk);return {getTracks:()=>[tk],getVideoTracks:()=>[tk]};}};
  const globals={window:{addEventListener(){}},document:{addEventListener(){},createElement:tag=>new Element(tag)},
    navigator:{mediaDevices:media},isSecureContext:true,ImageData:class{constructor(data,width,height){Object.assign(this,{data,width,height});}},
    requestAnimationFrame(fn){const id=setTimeout(()=>{raf.delete(id);fn(performance.now());},1);raf.add(id);return id;},
    cancelAnimationFrame(id){clearTimeout(id);raf.delete(id);}};
  const old=Object.keys(globals).map(k=>[k,Object.getOwnPropertyDescriptor(globalThis,k)]);
  for(const [k,value] of Object.entries(globals))Object.defineProperty(globalThis,k,{configurable:true,value});
  t.mock.method(URL,'createObjectURL',b=>{urls.push(b);return 'blob:sweep-test';});
  t.mock.method(SweepReader.prototype,'init',async()=>({modelStatus:'test-boundary'}));
  t.mock.method(SweepReader.prototype,'beginStep',function(){this.calls=0;});
  t.mock.method(SweepReader.prototype,'snapshot',function(){return {scans:this.calls};});
  t.mock.method(SweepReader.prototype,'close',function(){closed.push(this);});
  t.mock.method(SweepReader.prototype,'scan',async function(im){
    assert.ok(!inFlight.has(this),'no overlapping scan');inFlight.set(this,true);this.calls++;
    try{if(blocked&&this.expected==='data'){const gate=blocked;blocked=null;await gate;}
      return readCanvas(im,this.size);}finally{inFlight.delete(this);}
  });
  function phone(role){
    const vals={role,size:'64',phase:'200',capture:'150',decrement:'50',timeout:'3',backend:'wasm',facing:'user',resolution:'720',device:''};
    const nodes=new Map();const get=s=>{if(!nodes.has(s))nodes.set(s,new Element(s,vals[s.match(/^\[name="(\w+)"\]$/)?.[1]]));return nodes.get(s);};
    const video=get('video');Object.assign(video,{videoWidth:512,videoHeight:512,readyState:4});
    const activity=[],panel=initSweep({classList:{add(){}},querySelector:get},{onActive:v=>activity.push(v)});
    return {panel,activity,video,matrix:get('.sw-slot canvas'),next:get('.sw-next'),start:get('.sw-start'),status:get('.sw-status'),history:get('.sw-history'),
      report:async()=>{get('.sw-save').click();return JSON.parse(await urls.at(-1).text());}};
  }
  const S=phone('sender'),R=phone('receiver');S.video.peer=R.matrix;R.video.peer=S.matrix;
  t.after(()=>{S.panel.stop();R.panel.stop();for(const id of raf)clearTimeout(id);
    for(const [k,d] of old){if(d)Object.defineProperty(globalThis,k,d);else delete globalThis[k];}});
  return {S,R,tracks,closed,block(){let done;blocked=new Promise(r=>{done=r;});return done;}};
}

test('Next clears all previous ACK phases before reading; old DATA and delayed ACK exposures do not complete the next step',async t=>{
  const {S,R,tracks,closed,block}=devices(t);
  R.start.click();S.start.click();await until(()=>!R.next.disabled&&!S.next.disabled,'both cameras ready');
  assert.equal(S.matrix.draws+R.matrix.draws,0,'opening a camera must not start a step');
  R.next.click();S.next.click();await until(()=>S.matrix.draws>0&&S.video.cb,'first probe painted');
  const heldData=S.matrix.image,unblock=block();
  assert.ok(R.video.emit());assert.equal(R.video.emit(),false,'no fresh work while reader waits');
  unblock();await until(()=>!R.next.disabled&&R.matrix.draws>0,'step 1 ACK shown');
  const oldAck=R.matrix.image;assert.equal(R.matrix.hidden,false,'the ACK stays visible between steps');
  assert.ok(S.video.emit());await until(()=>!S.next.disabled,'sender accepts ACK 1');
  R.next.click();const oldDrawCount=R.matrix.draws;
  assert.equal(R.matrix.hidden,true);assert.equal(R.matrix.image,null,'old pixels erased immediately');
  R.video.override=heldData;assert.ok(R.video.emit());
  await until(()=>R.video.cb,'late DATA from prior step ignored');R.video.override=null;
  await wait(225); // cross the old ACK phase timer
  assert.equal(R.matrix.draws,oldDrawCount,'an old phase cannot repaint');assert.equal(R.matrix.hidden,true);
  assert.match(R.history.textContent,/Step 1.*ack-shown/);
  const c=S.matrix.draws;S.next.click();await until(()=>S.matrix.draws>c&&S.video.cb,'second probe painted');
  S.video.override=oldAck;assert.ok(S.video.emit());await until(()=>S.video.cb,'old ACK ignored for step 2');S.video.override=null;
  assert.ok(S.next.disabled);assert.ok(R.video.emit());await until(()=>!R.next.disabled,'new ACK shown');
  assert.equal(R.matrix.hidden,false);assert.ok(R.matrix.draws>oldDrawCount);
  assert.ok(S.video.emit());await until(()=>!S.next.disabled,'sender step 2 complete');
  const a=await S.report(),b=await R.report();assert.ok(a.runId);assert.equal(a.runId,b.runId);
  assert.deepEqual(a.steps.map(x=>[x.step,x.phaseMs,x.captureMs,x.status]),[[1,200,150,'ack-read'],[2,150,100,'ack-read']]);
  assert.deepEqual(b.steps.map(x=>x.status),['ack-shown','ack-shown']);assert.ok(a.steps.every(x=>x.rttMs>=0&&x.armToAckMs===null));
  assert.ok(b.steps.every(x=>x.rttMs===null&&x.armToAckMs>=0));assert.equal(a.steps[1].reader.scans,2);assert.equal(b.steps[1].reader.scans,2);
  // Sender-first: previous ACK is still visible until the receiver advances.
  const d=S.matrix.draws;S.next.click();await until(()=>S.matrix.draws>d&&S.video.cb,'step 3 painted');
  assert.equal(R.matrix.hidden,false);assert.ok(S.video.emit());await until(()=>S.video.cb,'old ACK 2 cannot confirm step 3');
  assert.equal(R.video.emit(),false);R.next.click();assert.equal(R.matrix.hidden,true);
  const release=block();R.video.override=heldData;assert.ok(R.video.emit());
  await wait(3200);assert.match(S.history.textContent,/Step 3.*timeout/);assert.match(R.history.textContent,/Step 3.*timeout/);
  release();await wait(10);assert.equal(R.video.emit(),false);assert.equal((await S.report()).steps[2].rttMs,null);
  assert.equal((await R.report()).steps[2].status,'timeout');R.panel.stop();S.panel.stop();
  assert.equal(S.matrix.hidden,true);assert.equal(R.matrix.hidden,true);assert.equal(closed.length,2);
  assert.ok(tracks.every(x=>x.stopped));assert.deepEqual(R.activity,[true,false]);assert.deepEqual(S.activity,[true,false]);
});

test('early receiver advance withdraws an unread ACK rather than pretending it was delivered',async t=>{
  const {S,R}=devices(t);R.start.click();S.start.click();await until(()=>!R.next.disabled&&!S.next.disabled,'both cameras ready');
  R.next.click();S.next.click();await until(()=>S.matrix.draws>0&&S.video.cb,'first DATA visible');
  assert.ok(R.video.emit());await until(()=>!R.next.disabled&&R.matrix.draws>0,'ACK 1 is available');
  R.next.click();assert.equal(R.matrix.hidden,true);assert.equal(R.matrix.image,null);
  assert.ok(S.video.emit());await until(()=>S.video.cb,'sender continues looking for withdrawn ACK 1');
  assert.ok(S.next.disabled);assert.doesNotMatch(S.history.textContent,/ack-read/);
  assert.ok(R.video.emit());await until(()=>R.video.cb,'old DATA still held by sender is ignored');
  assert.equal(R.matrix.hidden,true);assert.equal(R.matrix.image,null);
});
