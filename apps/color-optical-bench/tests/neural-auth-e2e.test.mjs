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
import { chunkBytesForSize, decodeNeuralQR } from '../dist/js/neural-qr/packet.js';
import { renderNeuralQR } from '../dist/js/neural-qr/layout.js';
import { extractNeuralQR } from '../dist/js/neural-qr/vision.js';
import { LivePhaseBurst } from '../dist/js/corrector-lab/live-burst.js';
import { packFrameStack, inputPresence } from '../dist/js/corrector-lab/runtime.js';
import { gridLayout } from '../dist/js/grid.js';
import { decodePacket } from '../dist/js/link/protocol.js';
import { PALETTE } from '../dist/js/format.js';

// Deterministic clean-image classifier for this transport/crypto fixture only.
// It reads the packed camera RGB and public phase/mask contract, never sender
// bytes or expected symbols. Trained-model/camera quality needs device testing.
const modelContract={modelProfile:'VW',pixelsPerCell:4,channels:32,framesPerInput:8,
  minFramesPerInput:4,frameCounts:[4,5,6,7,8],inputLayout:'phase-paired-rgb-presence-planes',
  outputLabelSpace:'logical',phaseMasks:[0,15,7,8]};
function classifyCleanPacked(packed,size){
  const active=inputPresence(packed,modelContract,size),side=4*size,plane=side*side,cells=size*size;
  const logits=new Float32Array(16*cells),layout=gridLayout(size,16,8);
  for(const cell of layout.cells){
    const x=cell%size,y=Math.floor(cell/size);
    const points=[(4*y+1)*side+4*x+1,(4*y+1)*side+4*x+2,
      (4*y+2)*side+4*x+1,(4*y+2)*side+4*x+2];
    for(const slot of active){
      const observed=[0,1,2].map(channel=>points.reduce((sum,point)=>
        sum+(packed[(slot*3+channel)*plane+point]+1)*127.5,0)/points.length);
      for(let color=0;color<16;color++){
        const physical=color^modelContract.phaseMasks[Math.floor(slot/2)];
        for(let channel=0;channel<3;channel++)logits[color*cells+cell]-=
          (observed[channel]-PALETTE[physical][channel])**2/active.length;
      }
    }
  }
  return logits;
}

const roles=['verifier','prover'];
const opposite=role=>role==='verifier'?'prover':'verifier';
const ids=['11111111-1111-4111-8111-111111111111','22222222-2222-4222-8222-222222222222'];

class Pair {
  constructor({images=false,tamper=false,loseFinalAcks=false}={}) {
    this.images=images;this.tamper=tamper;this.loseFinalAcks=loseFinalAcks;
    this.clock=0;this.pending=[];this.sent=[];this.deliveries=[];this.calls=[];this.sessions={};this.receivers={};this.phases=new Set();this.opticalPackets=0;this.neuralACKs=0;this.finalRetries=0;this.geometries=new Set();this.chunksByRole={};
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
  async channel(bytes,size) {
    if(!this.images)return bytes.slice();
    const kind=decodePacket(bytes).kind,expectedSize=kind==='ack'?32:size;
    const frames=packetFrames(bytes,{size}),burst=new LivePhaseBurst();
    assert.equal(frames.length,4);
    let geometry=null;
    for(const [index,frame] of frames.entries()){
      assert.equal(frame.size,expectedSize);assert.equal(frame.colors,16);
      assert.equal(frame.holdMs,150);this.phases.add(frame.phase);this.geometries.add(frame.size);
      const crop=extractNeuralQR(renderNeuralQR(frame,frame.size===128?1024:frame.size===64?768:512),
        {previousGeometry:geometry,now:150*index});
      assert.equal(crop.ok,true,`${kind} G${frame.size}: ${crop.error}`);
      assert.equal(crop.info.phase,index);geometry=crop.geometry;
      const state=burst.add(crop,{context:'clean-transport-fixture',mediaTime:.15*(index+1),presentedFrames:index+1});
      assert.equal(state.ready,index===3);
    }
    const captures=burst.snapshot();
    assert.equal(captures.inputCaptures,4);assert.deepEqual(captures.phaseCounts,[1,1,1,1]);
    const packed=packFrameStack(captures.input,modelContract,expectedSize,
      {phases:captures.phases,slots:captures.slots});
    const decoded=decodeNeuralQR(classifyCleanPacked(packed,expectedSize),expectedSize);
    this.opticalPackets++;if(kind==='ack')this.neuralACKs++;
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
    const size=message.role==='verifier'?128:64,chunkBytes=chunkBytesForSize(size);
    this.chunksByRole[message.role]=chunkBytes;
    const sender=await createTurnSender(bytes,{sessionId:envelope.header.sessionId,now:this.clock,chunkBytes});
    let drops=0;
    for(let limit=0;limit<1000;limit++) {
      const wire=sender.next(this.clock);
      assert.ok(wire,'A packet remains visible until its exact ACK arrives.');
      const incoming=await this.receivers[to].receive(await this.channel(wire,size),this.clock);
      assert.equal(incoming.accepted,true,incoming.reason);
      if(incoming.delivery) {
        const receivedEnvelope=decodeEnvelope(incoming.delivery.bytes);
        this.deliveries.push({role:to,...receivedEnvelope});
        if(receivedEnvelope.header.type==='PROOF'){
          const before=this.calls.filter(call=>call.type==='verify').length;
          for(const field of ['sessionId','binding','nonce']){
            const header={...receivedEnvelope.header},value=header[field];
            header[field]=(value[0]==='0'?'1':'0')+value.slice(1);
            assert.equal((await this.sessions[to].receive(encodeEnvelope(header,receivedEnvelope.body))).accepted,false,
              `A proof with a different ${field} must be rejected before verification.`);
          }
          assert.equal(this.calls.filter(call=>call.type==='verify').length,before);
        }
        const received=await this.sessions[to].receive(incoming.delivery.bytes);
        if(this.tamper&&receivedEnvelope.header.type==='RESULT')assert.equal(received.accepted,false);
        else assert.equal(received.accepted,true,received.reason);
      }
      assert.ok(incoming.ack);
      const ack=decodePacket(incoming.ack);
      if(this.loseFinalAcks&&ack.status==='complete'&&drops<2){drops++;this.finalRetries++;this.clock+=5000;continue;}
      const accepted=sender.receive(await this.channel(incoming.ack,32),this.clock);
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

test('32-repetition Neural-ZKBoo completes through asymmetric Neural-QR DATA and ACKs despite lost final ACKs',async t=>{
  const pair=new Pair({images:true,loseFinalAcks:true});t.after(()=>pair.close());
  const {verifier,prover}=await pair.run();
  assert.equal(verifier.state,'verified');assert.equal(verifier.result,true);assert.equal(verifier.resultDelivered,true);
  assert.equal(prover.state,'peer-accepted');assert.equal(prover.peerReported,true);
  assert.equal(verifier.proofBytes,17474);assert.equal(verifier.proofDigest,prover.proofDigest);
  assert.deepEqual(pair.sent.map(m=>m.header.type),['HELLO','IDENTITY','CHALLENGE','PROOF','RESULT']);
  assert.equal(pair.deliveries.length,5);assert.equal(pair.finalRetries,10);
  assert.ok(pair.neuralACKs>5,'ACKs for the multipart proof pass through G32 Neural-QR acquisition and RS/CRC.');
  assert.deepEqual([...pair.geometries].sort((a,b)=>a-b),[32,64,128]);
  assert.deepEqual(pair.chunksByRole,{verifier:chunkBytesForSize(128),prover:chunkBytesForSize(64)});
  assert.equal(pair.calls.filter(c=>c.type==='prove').length,1);assert.equal(pair.calls.filter(c=>c.type==='verify').length,1);
  assert.deepEqual([...pair.phases].sort(),[0,1,2,3]);
  const proof=pair.sent.find(m=>m.header.type==='PROOF');
  assert.equal((await pair.sessions.verifier.receive(proof.bytes)).accepted,false);
  assert.equal(pair.calls.filter(c=>c.type==='verify').length,1);
  const fresh=new Pair();t.after(()=>fresh.close());await fresh.start();
  assert.deepEqual(await fresh.sessions.verifier.receive(proof.bytes),{accepted:false,reason:'Session ID mismatch'});
  assert.equal(fresh.calls.some(call=>call.type==='verify'),false);
});

test('a tampered proof passes multipart transport integrity but fails Neural-ZKBoo verification',async t=>{
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
    channels[role]=new TurnChannel({initiator:role==='verifier',outgoingSize:role==='verifier'?128:64,now:()=>clock,
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

