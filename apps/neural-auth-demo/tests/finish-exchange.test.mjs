import test from 'node:test';
import assert from 'node:assert/strict';
import {setTimeout as wait} from 'node:timers/promises';
import {AuthSession} from '../dist/js/auth-core/session.js';
import {decodeEnvelope} from '../dist/js/auth-core/envelope.js';
import {TurnChannel} from '../dist/js/neural-auth/channel.js';
import {POCAutoPairing} from '../dist/js/neural-auth/poc-pairing.js';
import {decodePacket} from '../dist/js/link/protocol.js';
import {CryptoOperations} from '../dist/js/crypto-lab/operations.js';
import {CryptoClient} from '../dist/js/crypto-lab/client.js';
import {CorrectorClient} from '../dist/js/corrector-lab/client.js';

import {browser} from './helpers/live-browser.mjs';

for(const [role,statistics] of [['prover',true],['verifier',false]])test(
  'Finish exchange on '+role+' retains optical delivery, then exports a short report with statistics '+(statistics?'on':'off'),async t=>{
    const {get,track}=browser(t,role,statistics);
    const localOps=new CryptoOperations(),otherOps=new CryptoOperations();
    let online=true;const warmed=new Set(),cryptoJobs=[];
    t.mock.method(CryptoClient.prototype,'request',function(type,payload,opts){
      if(type==='selfTest'){if(!online)throw Error('Cannot load a new worker in flight mode');warmed.add(this);}
      else assert.ok(warmed.has(this),'offline auth reuses the already loaded crypto worker');
      cryptoJobs.push(type);return localOps.run(type,payload,opts?.onProgress);
    });
    const corrector=t.mock.method(CorrectorClient.prototype,'call',async operation=>{
      assert.ok(online,'model initialization needs the server');assert.equal(operation,'transport-init');
      return {ready:true,offlineReady:true,parity:{passed:true},modelStatus:'trained',modelProfile:'VW',effectiveBackend:'wasm'};
    });
    // Observe the live controller's channel; the remote device will deliver to
    // its public receive() just as a successfully decoded camera packet would.
    const read=t.mock.method(TurnChannel.prototype,'current');
    await import('../dist/app.js?finish-flow='+role);
    const expires=Date.now()+20000;
    while(get('preloadNotice').dataset.status!=='ready'&&Date.now()<expires)await wait(1);
    assert.equal(get('preloadNotice').dataset.status,'ready');
    assert.deepEqual(cryptoJobs,['selfTest'],'opening the page does not create a session key');
    assert.ok(!get('video').srcObject);assert.equal(corrector.mock.callCount(),1);
    online=false;get('start').click();
    while(!read.mock.callCount()&&Date.now()<expires)await wait(1);
    const local=read.mock.calls[0]?.this;assert.ok(local,get('error').textContent);read.mock.restore();
    const remoteRole=role==='prover'?'verifier':'prover',pair=new POCAutoPairing();
    const remote=new TurnChannel({initiator:remoteRole==='verifier',automaticTurns:true,onMessage:b=>peer.receive(b)});
    const peer=new AuthSession({role:remoteRole,deviceId:crypto.randomUUID(),displayName:'Other phone',repetitions:32,
      send:b=>remote.enqueue(b,{expectReply:decodeEnvelope(b).header.type!=='RESULT'}),
      cryptoClient:{request:(type,payload,opts)=>otherOps.run(type,payload,opts?.onProgress)},
      onChange:state=>pair.update(peer,state)});
    t.after(()=>{if(!get('stop').hidden)get('stop').click();remote.close();pair.close();peer.close();});
    await peer.start();
    let sawFinalAck=false;
    while(!sawFinalAck&&Date.now()<expires){
      for(const [sender,receiver] of [[local,remote],[remote,local]]){
        const wire=sender.current();if(!wire)continue;
        const final=(role==='prover'&&sender===local&&get('stop').textContent==='Finish exchange')||
          (role==='verifier'&&sender===remote&&peer.snapshot().peerReported);
        if(final&&decodePacket(wire).kind==='ack'){
          assert.equal(get('exchangeSummary').hidden,true);assert.equal(track.stopped,undefined);
          if(role==='verifier')assert.equal(get('stop').textContent,'Cancel exchange','verifier must wait for delivery');
          else assert.equal(get('matrixZone').dataset.visible,'true','prover still displays its final ACK');
          sawFinalAck=true;
        }
        await receiver.receive(wire);
      }
      await wait(1);
    }
    assert.ok(sawFinalAck,'the final optical ACK reached the verifier');assert.equal(get('stop').textContent,'Finish exchange');
    assert.equal(get('exchangeSummary').hidden,true);get('stop').click();
    assert.equal(track.stopped,true);assert.equal(get('exchangeSummary').hidden,false);
    const link=get('downloadSummary');assert.equal(link.hidden,false);assert.match(link.download,/\.json$/);
    const url=link.href,summary=await (await fetch(url)).json();assert.equal(summary.role,role);
    assert.equal(summary.peerName,'Other phone');assert.equal(summary.statisticsEnabled,statistics);
    assert.equal(summary.steps.length,statistics?5:0);assert.ok(statistics?summary.elapsedMs>0:summary.elapsedMs===null);
    assert.equal('packets' in summary,false);
    assert.equal(corrector.mock.callCount(),1,'no model reload during this exchange');
    online=true;
    if(statistics)get('start').click();else get('simulationTab').click();
    assert.equal(get('exchangeSummary').hidden,true);await assert.rejects(fetch(url)); // backing Blob URL was revoked
    if(statistics){get('stop').click();assert.equal(get('exchangeSummary').hidden,true,'cancel never presents a completion report');}
});

test('live capture watches the submitted matrix, waits for a fresh settled frame, and searches again after lock loss',async t=>{
  const {get}=browser(t,'prover',true),video=get('video'),full=get('captureCanvas');
  get('phaseWatch').checked=true;get('optimized').checked=true;
  Object.assign(video,{videoWidth:256,videoHeight:256,readyState:4,level:24});
  let next=null,frame=0,release,fullReads=0;
  video.requestVideoFrameCallback=cb=>{assert.equal(next,null,'only one outstanding callback');next=cb;return 1;};
  video.cancelVideoFrameCallback=()=>{next=null;};
  const fire=(stamp,mediaAt=stamp)=>{assert.ok(next);const run=next;next=null;run(stamp,{presentedFrames:++frame,mediaTime:mediaAt/1000});};
  function pixContext(canvas,fullSize=false){
    return {drawImage(src){canvas.level=src.level;},getImageData(x,y,w,h){
      if(fullSize)fullReads++;
      return {data:new Uint8ClampedArray(w*h*4).fill(canvas.level),width:w,height:h};
    }};
  }
  full.getContext=()=>pixContext(full,true);
  const gateInputs=[],tiny=get('gateCanvas'),gateContext=pixContext(tiny);
  tiny.getContext=()=>({...gateContext,drawImage(source){gateInputs.push(source);gateContext.drawImage(source);}});
  const create=document.createElement;
  document.createElement=tag=>{const el=create(tag);if(tag==='canvas')el.getContext=()=>pixContext(el);return el;};
  get('overlay').getContext=()=>({clearRect(){},beginPath(){},closePath(){},moveTo(){},lineTo(){},stroke(){},fill(){}});
  const localOps=new CryptoOperations();
  t.mock.method(CryptoClient.prototype,'request',(type,payload,opts)=>localOps.run(type,payload,opts?.onProgress));
  const scans=[];
  t.mock.method(CorrectorClient.prototype,'call',async(op,opts)=>{
    if(op==='transport-init')return {ready:true,offlineReady:true,parity:{passed:true},modelStatus:'trained',modelProfile:'VW',effectiveBackend:'wasm'};
    assert.equal(op,'transport-scan');scans.push(opts.presentedFrames);
    if(scans.length===1)await new Promise(resolve=>{release=resolve;});
    return scans.length===1?{stage:'collect',info:{size:64,phase:0},corners:[[20,20],[235,20],[235,235],[20,235]]}
      :{stage:'detect',ok:false};
  });
  await import('../dist/app.js?live-change-camera');
  get('start').click();
  try{
    for(let n=0;n<200&&!next;n++)await wait(5);
    assert.ok(next,get('error').textContent);
    fire(100);assert.deepEqual(scans,[1]);assert.equal(next,null,'never queues full frames while the worker is busy');
    video.level=195;release();await wait(0);
    fire(120);fire(125);assert.deepEqual(scans,[1]);assert.equal(fullReads,1,'edge and settle callbacks read only the small ROI');
    fire(136,125);assert.equal(fullReads,1,'a duplicate media timestamp is not a fresh exposure');
    fire(137);assert.deepEqual(scans,[1,5],'settled exposure arrives well before the 250 ms search interval');
    await wait(0);fire(200);assert.equal(fullReads,2,'a real miss returns to timed search');
    fire(400);await wait(0);assert.deepEqual(scans,[1,5,7]);
    assert.deepEqual(gateInputs,[full,full,full],'before acceptance, search and change reads derive thumbnails from the submitted canvas');
  }finally{get('stop').click();}
});
