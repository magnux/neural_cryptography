/** The automatic UI policy runs the real crypto/session/channel stack here.
 * Camera recovery is replaced by byte delivery; physical reading needs devices.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {setImmediate as settle} from 'node:timers/promises';
import {AuthSession} from '../dist/js/auth-core/session.js';
import {decodeEnvelope} from '../dist/js/auth-core/envelope.js';
import {CryptoOperations} from '../dist/js/crypto-lab/operations.js';
import {decodePacket} from '../dist/js/link/protocol.js';
import {TurnChannel} from '../dist/js/neural-auth/channel.js';
import {POCAutoPairing} from '../dist/js/neural-auth/poc-pairing.js';

const roles=['verifier','prover'];
const peer=role=>role==='verifier'?'prover':'verifier';
const ids=['11111111-1111-4111-8111-111111111111','22222222-2222-4222-8222-222222222222'];

test('automatic POC completes all five crypto messages despite lost ACKs, REQUESTs and STARTs',async t=>{
  let clock=0,busy=0,active=true,duplicates=0,repeatedRequests=0;
  const sessions={},channels={},pairing={},calls=[],deliveries=[],failures=[],dropUntil=new Map(),delayed=[];
  const sentPackets=new Map();
  for(const [index,role] of roles.entries()){
    const operations=new CryptoOperations();
    pairing[role]=new POCAutoPairing({isActive:()=>active,onError:error=>failures.push(error.message)});
    channels[role]=new TurnChannel({initiator:role==='verifier',automaticTurns:true,
      outgoingSize:role==='verifier'?128:64,now:()=>clock,
      onError:error=>failures.push(error.message),onMessage:async bytes=>{
        deliveries.push({role,type:decodeEnvelope(bytes).header.type});
        const result=await sessions[role].receive(bytes);
        assert.equal(result.accepted,true,result.reason);
      }});
    sessions[role]=new AuthSession({role,deviceId:ids[index],repetitions:32,now:()=>clock,
      send:bytes=>channels[role].enqueue(bytes,{expectReply:decodeEnvelope(bytes).header.type!=='RESULT'}),
      onChange:snapshot=>pairing[role].update(sessions[role],snapshot),
      cryptoClient:{cancel:()=>operations.clear(),request:async(type,payload,options={})=>{
        busy++;calls.push({role,type});
        try{return await operations.run(type,payload,options.onProgress);}finally{busy--;}
      }}});
  }
  t.after(()=>{
    active=false;
    for(const role of roles){pairing[role].close();channels[role].close();sessions[role].close('Test cleanup');}
  });
  await sessions.prover.start();await sessions.verifier.start();
  for(let step=0;step<20000;step++){
    await settle();
    for(const role of roles){
      const wire=channels[role].current();if(!wire)continue;
      const packet=decodePacket(wire),other=peer(role);
      if(packet.kind==='start'||packet.kind==='data'){
        if(!sentPackets.has(packet.messageId))sentPackets.set(packet.messageId,new Set());
        sentPackets.get(packet.messageId).add(packet.index??0);
      }
      const lossMs=packet.kind==='ack'&&packet.status==='complete'?7000:packet.kind==='poll'?4000:packet.kind==='start'?2000:0;
      if(lossMs){
        const key=`${packet.kind}:${packet.messageId}`;
        if(!dropUntil.has(key))dropUntil.set(key,clock+lossMs);
        if(clock<dropUntil.get(key)){
          if(packet.kind==='poll')repeatedRequests++;
          continue;
        }
      }
      // Delivery and a later stale copy model repeated captures and delayed work.
      const result=await channels[other].receive(wire);
      if(result.accepted){
        const repeated=await channels[other].receive(wire);
        if(repeated.reason==='duplicate'||repeated.reason==='duplicate-start'||repeated.reason==='already-delivered'||repeated.reason==='duplicate-request')duplicates++;
        if(step%7===0)delayed.push({to:other,wire:wire.slice(),due:clock+500});
      }
    }
    for(let i=delayed.length-1;i>=0;i--){
      if(delayed[i].due>clock)continue;
      const row=delayed.splice(i,1)[0];await channels[row.to].receive(row.wire);
    }
    if(!busy&&!roles.some(role=>channels[role].snapshot().preparing))clock+=100;
    for(const role of roles){channels[role].tick();sessions[role].tick();}
    assert.deepEqual(failures,[]);
    if(sessions.verifier.snapshot().resultDelivered&&sessions.prover.snapshot().peerReported)break;
    if(step===19999)assert.fail(`Automatic exchange did not finish: ${JSON.stringify(roles.map(role=>({auth:sessions[role].snapshot(),link:channels[role].snapshot()})))}`);
  }
  const verifier=sessions.verifier.snapshot(),prover=sessions.prover.snapshot();
  assert.equal(verifier.result,true);assert.equal(verifier.resultDelivered,true);
  assert.equal(prover.state,'peer-accepted');assert.equal(prover.peerReported,true);
  assert.equal(verifier.proofBytes,17474);assert.equal(prover.proofBytes,17474);
  assert.equal(verifier.binding,prover.binding);assert.equal(verifier.pairingCode,prover.pairingCode);
  assert.equal(verifier.proofDigest,prover.proofDigest);
  assert.deepEqual(deliveries.map(row=>row.type),['HELLO','IDENTITY','CHALLENGE','PROOF','RESULT']);
  assert.deepEqual([...sentPackets.values()].map(chunks=>chunks.size),[1,1,1,14,1],
    '32 repetitions with a G64 proof take 18 acknowledged matrices, including the first chunk of each envelope');
  assert.equal(calls.filter(row=>row.type==='prove').length,1);
  assert.equal(calls.filter(row=>row.type==='verify').length,1);
  assert.equal([...dropUntil.keys()].filter(key=>key.startsWith('ack:')).length,5,'every final ACK was initially unavailable');
  assert.equal([...dropUntil.keys()].filter(key=>key.startsWith('poll:')).length,4,'each non-final message requests its reply');
  assert.equal([...dropUntil.keys()].filter(key=>key.startsWith('start:')).length,5,'every START was initially unavailable');
  assert.ok(repeatedRequests>0,'REQUEST stays current throughout loss');
  assert.equal(channels.verifier.snapshot().requestedReply,false,'final RESULT requests no next turn');
  assert.ok(duplicates>0,'duplicate camera deliveries do not duplicate auth messages');
  assert.equal(channels.prover.snapshot().mode,'ack','the final result receipt remains displayed');
});

test('automatic pairing confirms once despite repeated notifications',async()=>{
  let calls=0;const errors=[];
  const policy=new POCAutoPairing({onError:error=>errors.push(error)});
  const session={confirmPairing:async()=>{calls++;policy.update(session,{canConfirm:true});return true;}};
  policy.update(session,{canConfirm:false});
  for(let i=0;i<10;i++)policy.update(session,{canConfirm:true});
  await settle();assert.equal(calls,1);assert.deepEqual(errors,[]);policy.close();
});

test('closing or invalidating the run prevents queued automatic confirmation',async()=>{
  for(const cancel of ['close','inactive']){
    let active=true,calls=0;
    const policy=new POCAutoPairing({isActive:()=>active});
    policy.update({confirmPairing:async()=>{calls++;return true;}},{canConfirm:true});
    if(cancel==='close')policy.close();else active=false;
    await settle();assert.equal(calls,0,cancel);policy.close();
  }
});
