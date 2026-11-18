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

// The screen/camera are doubles. Real browser controllers, auth messages, proof,
// chunking, turns, result and final ACK run in-process at the packet boundary.
function browser(t,role,statistics) {
  const values={role,backend:'wasm',repetitions:'32',resolution:'720',outgoingSize:'64',incomingSize:'64',
    phaseMs:'350',captureMs:'250',camera:''};
  const elements=new Map(),listeners=new Map(),track={stop(){this.stopped=true;},addEventListener(){},getSettings(){return {};}};
  class Node {
    constructor(id){this.id=id;this.hidden=['exchangeSummary','downloadSummary'].includes(id);this.value=values[id]??'';
      this.checked=id==='statistics'&&statistics;this.style={getPropertyValue(){return '';},setProperty(){}};this.dataset={};this.children=[];}
    textContent='';setAttribute(k,v){this[k]=v;} removeAttribute(k){delete this[k];}
    addEventListener(k,fn){listeners.set(this.id+':'+k,fn);}click(){listeners.get(this.id+':click')?.();}
    replaceChildren(...children){this.children=children;}append(...children){this.children.push(...children);}appendChild(child){this.append(child);return child;}
    getContext(){return {clearRect(){},putImageData(){}};}focus(){} requestVideoFrameCallback(){return 1;} cancelVideoFrameCallback(){} play(){}
  }
  const get=id=>{if(!elements.has(id))elements.set(id,new Node(id));return elements.get(id);};
  const document={body:get('body'),documentElement:get('html'),getElementById:get,createElement:k=>new Node(k),addEventListener(){}};
  const globals={document,window:{innerHeight:800,addEventListener(){}},navigator:{mediaDevices:{getUserMedia:async()=>({getTracks:()=>[track],getVideoTracks:()=>[track]})}},
    isSecureContext:true,ImageData:class{constructor(data,width,height){Object.assign(this,{data,width,height});}},Worker:class{}};
  for(const [key,value] of Object.entries(globals)){
    const prior=Object.getOwnPropertyDescriptor(globalThis,key);
    Object.defineProperty(globalThis,key,{configurable:true,value});
    t.after(()=>prior?Object.defineProperty(globalThis,key,prior):delete globalThis[key]);
  }
  return {get,track};
}

for(const [role,statistics] of [['prover',true],['verifier',false]])test(
  'Finish exchange on '+role+' retains optical delivery, then exports a short report with statistics '+(statistics?'on':'off'),async t=>{
    const {get,track}=browser(t,role,statistics);
    const localOps=new CryptoOperations(),otherOps=new CryptoOperations();
    t.mock.method(CryptoClient.prototype,'request',(type,payload,opts)=>localOps.run(type,payload,opts?.onProgress));
    t.mock.method(CorrectorClient.prototype,'call',async operation=>{assert.equal(operation,'transport-init');return {modelProfile:'VW',effectiveBackend:'wasm'};});
    // Observe the live controller's channel; the remote device will deliver to
    // its public receive() just as a successfully decoded camera packet would.
    const read=t.mock.method(TurnChannel.prototype,'current');
    await import('../dist/app.js?finish-flow='+role);
    get('start').click();
    const expires=Date.now()+20000;
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
    if(statistics)get('start').click();else get('simulationTab').click();
    assert.equal(get('exchangeSummary').hidden,true);await assert.rejects(fetch(url)); // backing Blob URL was revoked
    if(statistics){get('stop').click();assert.equal(get('exchangeSummary').hidden,true,'cancel never presents a completion report');}
});
