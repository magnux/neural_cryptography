import test from 'node:test';
import assert from 'node:assert/strict';
import {setImmediate as turn} from 'node:timers/promises';
import {browser} from './helpers/live-browser.mjs';
import {CorrectorClient} from '../dist/js/corrector-lab/client.js';
import {CryptoClient} from '../dist/js/crypto-lab/client.js';
import {CryptoOperations} from '../dist/js/crypto-lab/operations.js';
import {TurnChannel} from '../dist/js/neural-auth/channel.js';
import {TurnReceiver,createTurnSender} from '../dist/js/neural-auth/transport.js';
import {decodePacket} from '../dist/js/link/protocol.js';
import {PALETTE} from '../dist/js/format.js';
import {NQR_LAYOUT} from '../dist/js/neural-qr/layout.js';

async function until(check,why){
  const end=Date.now()+3000;
  while(!check()&&Date.now()<end)await turn();
  assert.ok(check(),why);
}

// The camera, screen refresh and worker output are doubles. Rendered marker
// colors, packet bytes, turn transitions and the app's schedulers are real.
async function device(t,role,{holdKey=false,firstCycle=true}={}){
  const {get,track}=browser(t,role,true),video=get('video'),full=get('captureCanvas');
  for(const id of ['phaseWatch','optimized'])get(id).checked=true;get('firstCycle').checked=firstCycle;
  let now=100,id=0,frame=0,camera=null,pixels=0,reply=null,keyReady;
  const scans=[],paints=[],rafs=new Map(),timers=new Map(),ops=new CryptoOperations();
  const key=holdKey?new Promise(r=>{keyReady=r;}):Promise.resolve();
  function rect(cb){rafs.set(++id,cb);return id;}
  t.mock.method(globalThis,'requestAnimationFrame',rect);t.mock.method(window,'requestAnimationFrame',rect);
  for(const where of [window,globalThis])t.mock.method(where,'cancelAnimationFrame',k=>rafs.delete(k));
  t.mock.method(performance,'now',()=>now);
  for(const [set,clear,repeat] of [['setTimeout','clearTimeout',false],['setInterval','clearInterval',true]]){
    t.mock.method(globalThis,set,(cb,delay)=>{timers.set(++id,{cb,at:now+delay,interval:repeat?delay:0});return id;});
    t.mock.method(globalThis,clear,k=>timers.delete(k));
  }
  function refresh(dt){
    now+=dt;
    for(const [k,v] of [...timers])if(v.at<=now&&timers.has(k)){
      if(v.interval)v.at=now+v.interval;else timers.delete(k);v.cb();
    }
    for(const [k,cb] of [...rafs])if(rafs.delete(k))cb(now);
  }
  function pulse(){frame++;if(camera){const cb=camera;camera=null;cb(now,{presentedFrames:frame,mediaTime:now/1000});}}
  Object.assign(video,{videoWidth:256,videoHeight:256,readyState:4,
    requestVideoFrameCallback(cb){assert.equal(camera,null,'at most one camera callback');camera=cb;return 1;},
    cancelVideoFrameCallback(){camera=null;}});
  const context={clearRect(){},drawImage(){pixels++;},
    getImageData(x,y,w,h){return {width:w,height:h,data:new Uint8ClampedArray(w*h*4)};}};
  full.getContext=get('gateCanvas').getContext=()=>context;
  const create=document.createElement;
  document.createElement=k=>{const n=create(k);if(k==='canvas')n.getContext=()=>context;return n;};
  get('overlay').getContext=()=>({clearRect(){}});
  get('matrix').getContext=()=>({putImageData({data,width}){
    const [x,y]=NQR_LAYOUT.phasePairs[0][0],scale=width/NQR_LAYOUT.units,i=(y*scale*width+x*scale)*4;
    paints.push(PALETTE.findIndex(rgb=>rgb.every((v,j)=>v===data[i+j])));
  }});
  t.mock.method(CryptoClient.prototype,'request',async(type,args,options)=>{
    if(type==='createKey')await key;
    return ops.run(type,args,options?.onProgress);
  });
  t.mock.method(CorrectorClient.prototype,'call',async(op,opts)=>{
    if(op==='transport-init')return {ready:true,offlineReady:true,parity:{passed:true},modelStatus:'trained',modelProfile:'VW'};
    assert.equal(op,'transport-scan');scans.push(opts.expected);
    const output=reply;reply=null;
    if(output)return {ok:true,chunk:await output};
    return {ok:false,stage:'detect'};
  });
  const read=t.mock.method(TurnChannel.prototype,'current');
  await import('../dist/app.js?presentation='+role+'-'+firstCycle);
  get('start').click();await until(()=>read.mock.callCount(),'channel did not start');
  const channel=read.mock.calls[0].this;read.mock.restore();
  return {get,track,refresh,pulse,channel,paints,scans,get pixels(){return pixels;},keyReady,
    get listening(){return !!camera;},setReply(value){reply=value;}};
}

async function presentFour(d){
  // A late OS timer skipping multiple nominal slots must not make the reader
  // start early. Repeated notifications of the same packet must not re-arm it.
  const count=d.scans.length,reads=d.pixels,offset=d.paints.length;
  for(const [n,dt] of [16,850,350,350].entries()){
    d.refresh(dt);d.channel.notify();d.pulse();await turn();
    assert.deepEqual(d.paints.slice(offset),[0,1,2,3].slice(0,n+1));
    assert.equal(d.scans.length,count,'no worker job before four phases have been presented');
    assert.equal(d.pixels,reads,'no full-frame or tiny ROI reads during presentation');
  }
  d.refresh(17); // one render opportunity after the fourth canvas submission
}

test('Singleton HELLO, its matching ACK, then REQUEST: no reads during either first pass or after cancelling',async t=>{
  const d=await device(t,'verifier',{holdKey:true});
  try{
    // Start a read while there is no outbound HELLO. The old worker result
    // cannot stand in for an ACK once the screen switches to START.
    let unblock;const delayed=new Promise(r=>{unblock=r;});d.setReply(delayed);d.pulse();
    assert.deepEqual(d.scans,['data']);d.keyReady();
    await until(()=>d.channel.current(),'verifier never produced HELLO');
    const stale=(await createTurnSender(Uint8Array.of(5))).next(100);
    const received=t.mock.method(d.channel,'receive');unblock(stale);await turn();
    assert.equal(received.mock.callCount(),0,'prior exposure must not reach the new transport turn');
    const peer=new TurnReceiver();
    await presentFour(d);
    const hello=await peer.receive(d.channel.current(),0);assert.ok(hello.delivery);
    assert.equal(decodePacket(hello.ack).status,'complete');
    d.setReply(hello.ack);d.pulse();
    await until(()=>decodePacket(d.channel.current()).kind==='poll','matching final ACK did not advance HELLO to REQUEST');
    assert.deepEqual(d.scans,['data','ack']);
    await presentFour(d);
    d.pulse();await turn();assert.deepEqual(d.scans,['data','ack','data']);
    const n=d.scans.length,p=d.paints.length;
    d.get('stop').click();d.refresh(1400);d.pulse();
    assert.equal(d.scans.length,n);assert.equal(d.paints.length,p);assert.equal(d.track.stopped,true);
  }finally{if(!d.get('stop').hidden)d.get('stop').click();}
});

test('blank prover reads immediately; its ACK is guarded once even if the peer repeats START',async t=>{
  const d=await device(t,'prover');
  try{
    assert.equal(d.channel.current(),null);assert.equal(d.listening,true);
    // Use a multipart fixture: chunk zero produces progress, not an auth envelope.
    const start=(await createTurnSender(new Uint8Array(2000))).next(100);
    d.setReply(start);d.pulse();
    await until(()=>d.channel.current(),'prover did not present the chunk-zero ACK');
    const ack=d.channel.current();assert.equal(decodePacket(ack).kind,'ack');
    assert.deepEqual(d.scans,['data']);await presentFour(d);
    d.setReply(start);d.pulse();await turn();
    assert.deepEqual(d.scans,['data','data']);assert.deepEqual(d.channel.current(),ack);
    d.channel.notify();d.refresh(260);d.pulse();await turn();
    assert.deepEqual(d.scans,['data','data','data'],'held identical ACK must not restart the pause');
  }finally{d.get('stop').click();}
});

test('turning off the display-first guard permits a scan during the first outgoing phase',async t=>{
  const d=await device(t,'verifier',{firstCycle:false});
  try{
    await until(()=>d.channel.current(),'HELLO not displayed');d.refresh(16);d.pulse();await turn();
    assert.deepEqual(d.paints,[0]);assert.deepEqual(d.scans,['ack']);assert.ok(d.pixels>0);
  }finally{d.get('stop').click();}
});
