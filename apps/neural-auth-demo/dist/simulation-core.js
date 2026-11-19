/** End-to-end diagnostic only. No live camera module imports this file.
 * An actual untrained VW inference runs first; explicit target logits are then
 * substituted so that transport and cryptographic timings can be measured.
 * This assumption never establishes corrector accuracy or physical throughput.
 */
import {AuthSession} from './js/auth-core/session.js';
import {decodeEnvelope,encodeEnvelope} from './js/auth-core/envelope.js';
import {CryptoOperations} from './js/crypto-lab/operations.js';
import {TurnChannel} from './js/neural-auth/channel.js';
import {POCAutoPairing} from './js/neural-auth/poc-pairing.js';
import {packetFrames,unpackOptical} from './js/neural-auth/optical.js';
import {ReceivePipeline} from './js/neural-auth/receive-pipeline.js';
import {StageProfiler} from './js/neural-auth/timing.js';
import {decodePacket} from './js/link/protocol.js';
import {renderNeuralQR} from './js/neural-qr/layout.js';
import {decodeReceipt16Capture,receipt16Digest,receipt16Frame} from './js/neural-qr/receipt16.js';
import {paletteMap} from './js/grid.js';
import {sameBytes} from './js/format.js';

const ROLES=['verifier','prover'];
const DEVICE_NAMES={verifier:'Simulated computer',prover:'Simulated phone'};
const opposite=role=>role==='verifier'?'prover':'verifier';
const settle=()=>new Promise(resolve=>setTimeout(resolve,0));
const cancelled=signal=>{if(signal?.aborted)throw Object.assign(Error('Simulation cancelled'),{name:'AbortError'});};

function idealSimulationLogits(frame) {
  const cells=frame.size**2,values=new Float32Array(16*cells).fill(-12);
  // The sender target is available only in this isolated diagnostic. It is
  // deliberately used after real inference, never by live capture/recovery.
  for(let cell=0;cell<cells;cell++)values[paletteMap(frame.symbols[cell],frame)*cells+cell]=12;
  return values;
}
function combineStages(...snapshots) {
  const rows=new Map();
  for(const snapshot of snapshots)for(const [stage,item] of Object.entries(snapshot??{})){
    const row=rows.get(stage)??{stage,count:0,totalMs:0,minMs:Infinity,maxMs:0};
    row.count+=item.count;row.totalMs+=item.totalMs;row.minMs=Math.min(row.minMs,item.minMs);row.maxMs=Math.max(row.maxMs,item.maxMs);rows.set(stage,row);
  }
  return [...rows.values()].map(row=>({...row,avgMs:row.totalMs/row.count}));
}

export async function runFullHandshake({corrector,optimized=true,statistics=true,optical=true,
  repetitions=32,ackSize=16,heldCycles=2,onProgress=()=>{},signal,tamperProof=false,dropFinalAckWindows=false,dropRequestWindows=false}={}) {
  cancelled(signal);
  if(optical&&(!corrector?.infer||!corrector?.loadContext))throw Error('Actual simulation inference must be initialized before the optical handshake.');
  if(![16,32].includes(ackSize)||!Number.isInteger(repetitions)||repetitions<1||repetitions>219||!Number.isInteger(heldCycles)||heldCycles<1||heldCycles>4)
    throw Error('Invalid bounded simulation settings.');
  const started=performance.now(),profiler=new StageProfiler(statistics),sessions={},channels={},pipelines={},operations={},pairing={};
  const cryptoJobs=new Set();
  const messages=[],errors=[],cryptoCounts={},modelSizes=new Set(),scanFrames={verifier:0,prover:0};
  const displayedFrames={verifier:[],prover:[]};
  const requestWindows=new Set(),droppedAckWindows=new Map(),droppedRequests=new Map();let active=true;
  let clock=0,busy=0,cryptoIdleWaits=0,scans=0,inferenceCalls=0,skippedScans=0,packets=0,acceptedPackets=0,duplicatePackets=0,wrongTurnScans=0,waitingPacketWindows=0,containerRejectedScans=0;
  let targetFrame=null,finalResultAckAccepted=false,matchedReceipts=0;
  const reportProgress=message=>onProgress(`${optimized?'Optimized':'Baseline'}: ${message}`);
  for(const role of ROLES){
    operations[role]=new CryptoOperations();
    pairing[role]=new POCAutoPairing({isActive:()=>active,onError:error=>errors.push(error.message)});
    channels[role]=new TurnChannel({initiator:role==='verifier',outgoingSize:64,automaticTurns:true,now:()=>clock,
      onError:error=>errors.push(error.message),onMessage:async bytes=>{
        const envelope=decodeEnvelope(bytes);messages.push({type:envelope.header.type,to:role,bytes:bytes.length,atVirtualMs:clock});
        reportProgress(`${envelope.header.type} delivered (${messages.length}/5)…`);
        const received=await profiler.measure('authentication.receive',()=>sessions[role].receive(bytes));
        if(!received.accepted&&!(tamperProof&&envelope.header.type==='RESULT'))throw Error(`${envelope.header.type}: ${received.reason}`);
      }});
    sessions[role]=new AuthSession({role,deviceId:globalThis.crypto.randomUUID(),displayName:DEVICE_NAMES[role],repetitions,now:()=>clock,
      onChange:snapshot=>pairing[role].update(sessions[role],snapshot),
      send:bytes=>{
        if(tamperProof){const envelope=decodeEnvelope(bytes);if(envelope.header.type==='PROOF'){const body=envelope.body.slice();body[body.length-1]^=1;bytes=encodeEnvelope(envelope.header,body);}}
        return channels[role].enqueue(bytes,{expectReply:decodeEnvelope(bytes).header.type!=='RESULT'});
      },
      cryptoClient:{cancel:()=>operations[role].clear(),request:async(type,payload,options={})=>{
        busy++;cryptoCounts[type]=(cryptoCounts[type]??0)+1;
        const job=Promise.resolve().then(()=>profiler.measure(`crypto.${type}`,()=>operations[role].run(type,payload,options.onProgress)));
        cryptoJobs.add(job);
        try{return await job;}finally{busy--;cryptoJobs.delete(job);}
      }}});
    if(optical)pipelines[role]=new ReceivePipeline({loadContext:corrector.loadContext,infer:async(context,packed)=>{
      cancelled(signal);const current=targetFrame;
      if(!current||context.entry.size!==current.size)throw Error('Simulation inference target/profile mismatch.');
      const actual=await corrector.infer(context,packed,{statistics});inferenceCalls++;modelSizes.add(current.size);
      profiler.record('inference.actual-onnx',actual.ms);
      // Do not skip, cache, estimate, or shorten model execution. Its measured
      // output is intentionally discarded only at this simulation boundary.
      const values=profiler.measure('simulation.assumed-output',()=>idealSimulationLogits(current));
      return {values,ms:actual.ms};
    }});
  }
  async function deliver(role,wire) {
    const to=opposite(role),packet=decodePacket(wire);
    packets++;
    if(optical&&packets%8===0)reportProgress(`${packets} optical packets; ${inferenceCalls} actual model calls…`);
    if(!optical){
      const result=await profiler.measure('transport.receive',()=>channels[to].receive(wire));
      if(result.accepted)acceptedPackets++;
      if(to==='verifier'&&packet.kind==='ack'&&packet.status==='complete'&&result.delivered&&messages.at(-1)?.type==='RESULT')finalResultAckAccepted=true;
      return;
    }
    // Hold a single static image for G16 ACKs, or the same four phase images
    // (and padding) for DATA, REQUEST and the optional neural G32 ACK.
    let cached=displayedFrames[role].find(item=>sameBytes(item.wire,wire));
    if(!cached){
      const frames=await profiler.measure('render.packet-framing',async()=>packet.kind==='ack'&&ackSize===16
        ?[receipt16Frame(await receipt16Digest(wire),150)]:packetFrames(wire,{size:64,phaseMs:150}));
      cached={wire:wire.slice(),frames};
      displayedFrames[role].push(cached);if(displayedFrames[role].length>4)displayedFrames[role].shift();
    }
    const frames=cached.frames;
    let delivered=false;
    // Repeated cycles deliberately model the sender leaving a packet visible
    // while the receiver is still looking at it. Both modes get equal scans.
    for(let cycle=0;cycle<heldCycles;cycle++)for(const frame of frames){
      cancelled(signal);targetFrame=frame;
      const image=profiler.measure('render.raster',()=>renderNeuralQR(frame,frame.size<=32?512:768));
      scans++;scanFrames[to]++;clock+=150;
      // Use the receiving device's current turn, exactly as the camera UI does.
      // A response may begin while the previous display is still being held.
      const expected=channels[to].snapshot().expected;
      if(expected!==(packet.kind==='ack'?'ack':packet.kind==='poll'?'request':'data'))wrongTurnScans++;
      let result,chunk;
      // Select the reader from the local turn alone. Even a stale/unrelated
      // image must go through the same independent optical checks as live.
      if(expected==='ack'&&ackSize===16){
        result=profiler.measure('receipt.single-exposure',()=>decodeReceipt16Capture(image,{now:clock,captureIntervalMs:150,collectTimings:statistics}));
        if(result.ok){
          const outstanding=channels[to].expectedAck();
          if(outstanding&&sameBytes(result.digest,await receipt16Digest(outstanding)))chunk=outstanding;
          else containerRejectedScans++;
        }
      }else{
        result=await pipelines[to].scan({image,runId:1,mediaTime:clock/1000,presentedFrames:scanFrames[to],
          expected,captureIntervalMs:150,maxSize:64,optimized,statistics});
        if(result.skipped)skippedScans++;
        chunk=result.chunk??result.decoded?.chunk;
      }
      if(chunk){
        let recovered;
        try{recovered=profiler.measure('transport.unpack',()=>unpackOptical(chunk,expected));}
        catch{containerRejectedScans++;continue;}
        if(recovered.length!==wire.length||recovered.some((value,i)=>value!==wire[i]))throw Error('Simulation recovery differed from transmitted wire bytes.');
        const received=await profiler.measure('transport.receive',()=>channels[to].receive(recovered));
        if(to==='verifier'&&packet.kind==='ack'&&packet.status==='complete'&&received.delivered&&messages.at(-1)?.type==='RESULT')finalResultAckAccepted=true;
        if(received.accepted){
          acceptedPackets++;if(delivered)duplicatePackets++;delivered=true;
          if(expected==='ack'&&ackSize===16)matchedReceipts++;
          if(result.acceptToken){
            const acceptance=pipelines[to].accept({token:result.acceptToken,runId:1,expected});
            if(!acceptance.accepted)throw Error('Simulation receiver rejected its own accepted-packet token.');
          }
        }
      }
    }
    // Wrong-turn images, an accepted held matrix, or transient acquisition
    // misses cannot advance transfer. Leave the real sender/ACK scheduler to
    // retry; its deadlines and the outer step limit bound a failed simulation.
    if(!delivered)waitingPacketWindows++;
  }
  try{
    reportProgress('generating fresh keys and starting the full five-message handshake…');
    await sessions.prover.start();await sessions.verifier.start();
    for(let step=0;step<20000;step++){
      cancelled(signal);await settle();
      if(errors.length)throw Error(errors.join('; '));
      const states=ROLES.map(role=>sessions[role].snapshot());
      const failure=states.find(state=>['failed','timed-out'].includes(state.state));
      if(failure)throw Error(`${failure.role}: ${failure.error}`);
      // Live cryptography and inference run in separate workers. Here they share
      // one thread: repeated WASM calls would starve the sequential WebCrypto
      // commitments while no new screen is possible. Once the REQUEST has been
      // read, finish the pending real computation before resuming camera time.
      // Do not use this pause to skip an unread ACK/REQUEST or an outgoing packet.
      const idle=ROLES.every(role=>!channels[role].snapshot().outgoing&&!channels[role].snapshot().preparing);
      const requested=ROLES.some(role=>channels[role].snapshot().requestedReply);
      const ungranted=ROLES.some(role=>channels[role].snapshot().waitingForRequest);
      if(busy&&idle&&requested&&!ungranted){
        cryptoIdleWaits++;reportProgress('request accepted; completing pending cryptography before the next screen…');
        await Promise.all([...cryptoJobs]);cancelled(signal);continue;
      }
      let scanned=false;
      for(const role of ROLES){
        const wire=channels[role].current();if(!wire)continue;
        const packet=decodePacket(wire);
        if(packet.kind==='poll')requestWindows.add(packet.messageId);
        if(dropFinalAckWindows&&packet.kind==='ack'&&packet.status==='complete'){
          if(!droppedAckWindows.has(packet.messageId))droppedAckWindows.set(packet.messageId,clock+7000);
          if(clock<droppedAckWindows.get(packet.messageId))continue;
        }
        if(dropRequestWindows&&packet.kind==='poll'){
          if(!droppedRequests.has(packet.messageId))droppedRequests.set(packet.messageId,clock+2000);
          if(clock<droppedRequests.get(packet.messageId))continue;
        }
        // A held final ACK with an idle peer has no camera work to perform.
        if(decodePacket(wire).kind==='ack'&&!channels[opposite(role)].snapshot().outgoing)continue;
        scanned=true;await deliver(role,wire);
      }
      // Advance bounded transfer deadlines without sleeping for virtual display
      // seconds. Busy cryptographic work is not expired.
      if(!busy&&!ROLES.some(role=>channels[role].snapshot().preparing))clock+=scanned?20:100;
      for(const role of ROLES){channels[role].tick();sessions[role].tick();}
      const verifier=sessions.verifier.snapshot(),prover=sessions.prover.snapshot();
      if(verifier.resultDelivered&&(prover.peerReported||tamperProof)){
        const expectedMessages=['HELLO','IDENTITY','CHALLENGE','PROOF','RESULT'];
        if(messages.map(row=>row.type).join(',')!==expectedMessages.join(','))throw Error('The full authentication message sequence did not complete.');
        if(cryptoCounts.prove!==1||cryptoCounts.verify!==1)throw Error('The handshake did not perform exactly one real proof and verification.');
        if(verifier.binding!==prover.binding||verifier.pairingCode!==prover.pairingCode)throw Error('The automatically paired identities have different bindings.');
        if(!finalResultAckAccepted)throw Error('Final RESULT acknowledgment was not accepted by the actual turn channel.');
        if(verifier.peerDisplayName!==DEVICE_NAMES.prover||prover.peerDisplayName!==DEVICE_NAMES.verifier)throw Error('The human-readable peer names did not traverse the authentication handshake.');
        const success=verifier.result===true&&verifier.state==='verified'&&prover.state==='peer-accepted';
        if(!success&&!tamperProof)throw Error('Real Neural-ZKBoo proof verification or result delivery failed.');
        return {mode:optimized?'optimized':'baseline',success,proofVerified:verifier.result===true,resultDelivered:verifier.resultDelivered,
          peerReported:prover.peerReported,finalResultAckAccepted,proofBytes:verifier.proofBytes,messages,wallMs:performance.now()-started,virtualOpticalMs:clock,
          scans,inferenceCalls,ackSize:optical?ackSize:null,matchedReceipts,skippedScans,wrongTurnScans,waitingPacketWindows,containerRejectedScans,packets,acceptedPackets,duplicatePackets,automaticTurns:true,handoff:'request',manualHandoffs:0,cryptoIdleWaits,
          requestWindows:requestWindows.size,droppedRequestWindows:droppedRequests.size,droppedFinalAckWindows:droppedAckWindows.size,cryptoCounts,
          modelSizes:[...modelSizes].sort((a,b)=>a-b),statisticsEnabled:statistics,
          stages:statistics?combineStages(profiler.snapshot(),...ROLES.map(role=>pipelines[role]?.snapshot?.().timings??pipelines[role]?.profiler?.snapshot?.())):[],
          verifier:{localDisplayName:verifier.localDisplayName,peerDisplayName:verifier.peerDisplayName,state:verifier.state,result:verifier.result,resultDelivered:verifier.resultDelivered,timings:verifier.timings},
          prover:{localDisplayName:prover.localDisplayName,peerDisplayName:prover.peerDisplayName,state:prover.state,peerReported:prover.peerReported,timings:prover.timings}};
      }
    }
    throw Error('Simulation exceeded its bounded handshake steps.');
  }finally{active=false;targetFrame=null;for(const role of ROLES){pairing[role].close();sessions[role].close('Simulation cleanup');channels[role].close();operations[role].clear();}}
}

export async function runSimulationComparison({corrector,ackSize=16,statistics=true,optimized=true,compare=true,onProgress=()=>{},signal}={}) {
  const report={format:'neural-auth-demo-simulation',version:1,kind:'target-assisted-simulation',timestamp:new Date().toISOString(),
    assumedOutput:true,modelAccuracyValidated:false,physicalCameraTest:false,
    inference:corrector?.description,configuration:{repetitions:32,outgoingGrid:64,ackGrid:ackSize,requestGrid:32,heldCycles:2,automaticTurns:true,handoff:'request',statisticsEnabled:statistics},
    recoveryOutput:ackSize===16?'DATA and REQUEST use target logits after actual untrained inference; static G16 receipts use independently decoded rendered pixels':'target logits after actual untrained inference for every packet kind',
    warnings:[
      `Simulation only: real untrained VW inference is ${statistics?'timed':'executed without detailed timing'}, then target logits replace its output to assume successful correction.`,
      'Authentication, framing, acquisition, RS+CRC recovery, and verification execute real code. Corrector accuracy is not measured.',
      'Clean synthetic images and virtual display time do not measure camera reliability or phone-to-computer throughput.',
      'The simulator waits for pending cryptography after an accepted request when neither side has outgoing data; live cryptography and inference use separate workers.',
      'Baseline and optimized use the same scan schedule with independent fresh keys, challenges and proofs. Wall times include rendering; stage timings may overlap.',
    ]};
  if(compare||!optimized)report.baseline=await runFullHandshake({corrector,ackSize,statistics,optimized:false,onProgress,signal});
  if(compare||optimized)report.optimized=await runFullHandshake({corrector,ackSize,statistics,optimized:true,onProgress,signal});
  report.success=Boolean((report.baseline??report.optimized)?.success&&(!compare||report.optimized.success));
  if(compare){
    const baseline=report.baseline,current=report.optimized;
    report.comparison={inferenceCallsSaved:baseline.inferenceCalls-current.inferenceCalls,
      inferenceReductionPercent:baseline.inferenceCalls?100*(baseline.inferenceCalls-current.inferenceCalls)/baseline.inferenceCalls:0,
      wallMsSaved:baseline.wallMs-current.wallMs,baselineWallMs:baseline.wallMs,optimizedWallMs:current.wallMs,
      equalScans:baseline.scans===current.scans,stageCountsComparable:baseline.packets===current.packets,
      caution:'One sequential synthetic comparison, not a device benchmark or statistically stable speedup estimate.'};
  }
  return report;
}
