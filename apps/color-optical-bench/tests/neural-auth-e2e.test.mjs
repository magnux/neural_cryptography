/** Actual cryptography and optical framing. Clean-image classification is the
 * channel fixture here; trained camera/ONNX quality is tested on the devices. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { setImmediate as settle } from 'node:timers/promises';
import { AuthSession } from '../dist/js/auth-core/session.js';
import { decodeEnvelope, encodeEnvelope } from '../dist/js/auth-core/envelope.js';
import { CryptoOperations } from '../dist/js/crypto-lab/operations.js';
import { createTurnSender, TurnReceiver } from '../dist/js/neural-auth/transport.js';
import { packetFrames, unpackOptical } from '../dist/js/neural-auth/optical.js';
import { decodePacket } from '../dist/js/link/protocol.js';
import { renderFrame } from '../dist/js/format.js';
import { decodeImage } from '../dist/js/vision.js';

const roles=['verifier','prover'];
const opposite=role=>role==='verifier'?'prover':'verifier';
const ids=['11111111-1111-4111-8111-111111111111','22222222-2222-4222-8222-222222222222'];

class Pair {
  constructor({images=false,tamper=false,loseFinalAcks=false}={}) {
    this.images=images;this.tamper=tamper;this.loseFinalAcks=loseFinalAcks;
    this.clock=0;this.pending=[];this.sent=[];this.deliveries=[];this.calls=[];this.sessions={};this.receivers={};this.phases=new Set();this.opticalPackets=0;this.finalRetries=0;
    for(const [index,role] of roles.entries()) {
      const operations=new CryptoOperations();
      this.sessions[role]=new AuthSession({role,deviceId:ids[index],repetitions:32,now:()=>this.clock,
        cryptoClient:{request:async(type,payload,options={})=>{this.calls.push({role,type});return operations.run(type,payload,options.onProgress);},cancel:()=>operations.clear()},
        send:bytes=>new Promise((resolve,reject)=>{this.pending.push({role,bytes:bytes.slice(),resolve,reject});}),
      });
      this.receivers[role]=new TurnReceiver({maxMessageBytes:524288});
    }
  }
  async start(){await this.sessions.prover.start();await this.sessions.verifier.start();await settle();}
  async channel(bytes) {
    if(!this.images)return bytes.slice();
    const frames=await packetFrames(bytes,{phaseMs:95}),kind=decodePacket(bytes).kind;
    assert.equal(frames.length,kind==='ack'?1:4);
    const frame=frames[kind==='ack'?0:this.opticalPackets%4];
    this.opticalPackets++;
    if(kind!=='ack') {assert.equal(frame.size,64);assert.equal(frame.colors,16);assert.equal(frame.holdMs,95);this.phases.add(frame.phase);}
    else {assert.equal(frame.size,32);assert.equal(frame.colors,4);}
    const decoded=decodeImage(renderFrame(frame,768),{preview:false,recovery:'erasures'});
    assert.equal(decoded.ok,true,`${kind}: ${decoded.error}`);
    const packet=unpackOptical(decoded.chunk,kind==='ack'?'ack':'data');
    assert.deepEqual(packet,bytes);
    return packet;
  }
  async transfer(message) {
    const to=opposite(message.role),envelope=decodeEnvelope(message.bytes);
    this.sent.push({role:message.role,...envelope,bytes:message.bytes.slice()});
    let bytes=message.bytes;
    if(this.tamper&&envelope.header.type==='PROOF') {
      const body=envelope.body.slice();body[body.length-1]^=1;bytes=encodeEnvelope(envelope.header,body);
    }
    const sender=await createTurnSender(bytes,{sessionId:envelope.header.sessionId,now:this.clock});
    let drops=0;
    for(let limit=0;limit<1000;limit++) {
      const wire=sender.next(this.clock);
      assert.ok(wire,'A packet remains visible until its exact ACK arrives.');
      const incoming=await this.receivers[to].receive(await this.channel(wire),this.clock);
      assert.equal(incoming.accepted,true,incoming.reason);
      if(incoming.delivery) {
        const receivedEnvelope=decodeEnvelope(incoming.delivery.bytes);
        this.deliveries.push({role:to,...receivedEnvelope});
        const received=await this.sessions[to].receive(incoming.delivery.bytes);
        if(this.tamper&&receivedEnvelope.header.type==='RESULT')assert.equal(received.accepted,false);
        else assert.equal(received.accepted,true,received.reason);
      }
      assert.ok(incoming.ack);
      const ack=decodePacket(incoming.ack);
      if(this.loseFinalAcks&&ack.status==='complete'&&drops<2){drops++;this.finalRetries++;this.clock+=5000;continue;}
      const accepted=sender.receive(await this.channel(incoming.ack),this.clock);
      assert.equal(accepted.accepted,true,accepted.reason);
      this.clock+=500;
      if(accepted.delivered){message.resolve();await settle();return;}
    }
    throw Error('Turn transfer did not finish');
  }
  async run() {
    await this.start();
    for(let limit=0;limit<12;limit++) {
      await settle();
      const snapshots=roles.map(role=>this.sessions[role].snapshot());
      if(snapshots.every(s=>s.canConfirm)) {
        assert.equal(snapshots[0].pairingCode,snapshots[1].pairingCode);
        await Promise.all(roles.map(role=>this.sessions[role].confirmPairing()));await settle();
      }
      if(this.pending.length){await this.transfer(this.pending.shift());continue;}
      const v=this.sessions.verifier.snapshot(),p=this.sessions.prover.snapshot();
      if(v.resultDelivered&&(this.tamper||p.state==='peer-accepted'))return {verifier:v,prover:p};
    }
    assert.fail(`Authentication did not finish: ${JSON.stringify(roles.map(role=>this.sessions[role].snapshot()))}`);
  }
  close(){for(const role of roles)this.sessions[role].close('Test cleanup');}
}

test('both optical applications use identical prepared Neural-ZKBoo session and envelope sources',async()=>{
  for(const name of ['session.js','envelope.js']){
    const canonical=await readFile(new URL(`../../../shared/browser-crypto/auth/${name}`,import.meta.url));
    const color=await readFile(new URL(`../dist/js/auth-core/${name}`,import.meta.url));
    const qr=await readFile(new URL(`../../qr-neural-key/dist/js/auth-core/${name}`,import.meta.url));
    assert.deepEqual(color,canonical);assert.deepEqual(qr,canonical);
  }
});

test('32-repetition Neural-ZKBoo completes in turns through rendered color DATA and ACK matrices',async t=>{
  const pair=new Pair({images:true,loseFinalAcks:true});t.after(()=>pair.close());
  const {verifier,prover}=await pair.run();
  assert.equal(verifier.state,'verified');assert.equal(verifier.result,true);assert.equal(verifier.resultDelivered,true);
  assert.equal(prover.state,'peer-accepted');assert.equal(prover.peerReported,true);
  assert.equal(verifier.proofBytes,17474);assert.equal(verifier.proofDigest,prover.proofDigest);
  assert.deepEqual(pair.sent.map(m=>m.header.type),['HELLO','IDENTITY','CHALLENGE','PROOF','RESULT']);
  assert.equal(pair.deliveries.length,5);assert.equal(pair.finalRetries,10);
  assert.equal(pair.calls.filter(c=>c.type==='prove').length,1);assert.equal(pair.calls.filter(c=>c.type==='verify').length,1);
  assert.deepEqual([...pair.phases].sort(),[0,1,2,3]);
  const proof=pair.sent.find(m=>m.header.type==='PROOF');
  assert.equal((await pair.sessions.verifier.receive(proof.bytes)).accepted,false);
  assert.equal(pair.calls.filter(c=>c.type==='verify').length,1);
});

test('a tampered proof passes color transport integrity but fails Neural-ZKBoo verification',async t=>{
  const pair=new Pair({tamper:true});t.after(()=>pair.close());
  // A negative result names the altered proof digest. The original prover
  // rejects that peer report because it refers to different proof bytes.
  const {verifier,prover}=await pair.run();
  assert.equal(verifier.state,'rejected');assert.equal(verifier.result,false);assert.equal(verifier.resultDelivered,true);
  assert.equal(prover.result,null);assert.equal(prover.state,'sending-proof');
  assert.equal(pair.calls.filter(c=>c.type==='verify').length,1);
});

test('the actual turn scheduler keeps final ACKs visible until the next manual handoff',async t=>{
  const {TurnChannel}=await import('../dist/js/neural-auth/channel.js');
  let clock=0,busy=0,manualHandoffs=0,heldReply=false;
  const sessions={},channels={},delivered=[],failures=[],dropped=new Set();
  for(const [index,role] of roles.entries()){
    const operations=new CryptoOperations();
    channels[role]=new TurnChannel({initiator:role==='verifier',now:()=>clock,
      onError:error=>failures.push(error.message),onMessage:async bytes=>{
        delivered.push(decodeEnvelope(bytes).header.type);
        const result=await sessions[role].receive(bytes);assert.equal(result.accepted,true,result.reason);
      }});
    sessions[role]=new AuthSession({role,deviceId:ids[index],repetitions:32,now:()=>clock,send:bytes=>channels[role].enqueue(bytes),
      cryptoClient:{cancel:()=>operations.clear(),request:async(type,payload,options={})=>{
        busy++;try{return await operations.run(type,payload,options.onProgress);}finally{busy--;}
      }}});
  }
  t.after(()=>{for(const role of roles){sessions[role].close('Test cleanup');channels[role].close();}});
  await sessions.prover.start();await sessions.verifier.start();
  for(let step=0;step<20000;step++){
    await settle();
    if(roles.every(role=>sessions[role].snapshot().canConfirm)){
      assert.equal(sessions.verifier.snapshot().pairingCode,sessions.prover.snapshot().pairingCode);
      await Promise.all(roles.map(role=>sessions[role].confirmPairing()));
    }
    for(const role of roles){
      const local=channels[role].snapshot(),peer=channels[opposite(role)].snapshot();
      if(local.canSendNext&&peer.outgoing){assert.equal(local.mode,'ack');heldReply=true;}
      if(local.canSendNext&&!peer.outgoing){assert.equal(await channels[role].nextTurn(),true);manualHandoffs++;}
    }
    for(const role of roles){
      const wire=channels[role].current();if(!wire)continue;
      const packet=decodePacket(wire);
      if(packet.kind==='ack'&&packet.status==='complete'){
        if(!dropped.has(packet.messageId)){dropped.add(packet.messageId);continue;}
        // Hold the acknowledgment while the peer prepares its response, as a
        // person turning the phone can do. The final RESULT needs no response.
        if(!channels[role].snapshot().queued&&delivered.at(-1)!=='RESULT')continue;
      }
      await channels[opposite(role)].receive(wire);
    }
    if(!busy)clock+=20;
    for(const role of roles){channels[role].tick();sessions[role].tick();}
    assert.deepEqual(failures,[]);
    if(sessions.verifier.snapshot().resultDelivered&&sessions.prover.snapshot().state==='peer-accepted')break;
    if(step===19999)assert.fail('Turn-channel authentication did not complete');
  }
  assert.equal(sessions.verifier.snapshot().state,'verified');
  assert.equal(sessions.prover.snapshot().proofBytes,17474);
  assert.equal(heldReply,true,'A queued response waits while the peer still needs the final ACK.');
  assert.equal(manualHandoffs,4);assert.equal(dropped.size,5);
  assert.deepEqual(delivered,['HELLO','IDENTITY','CHALLENGE','PROOF','RESULT']);
});
