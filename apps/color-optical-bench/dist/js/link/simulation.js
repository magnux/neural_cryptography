import {createSender,Receiver} from './session.js';
import {decodePacket,sha256} from './protocol.js';
import {DATA_PROFILE,CONTROL_PROFILE,normalizeProfile,getChunkBytes,packetToFrames} from './optical.js';
import {parseHeader,symbolsToChunk,sameBytes} from '../format.js';
import {diagnosticPayload,seededRandom} from './payload.js';

const TERMINAL=new Set(['delivered','failed','timed-out','cancelled']);
const MAX_SIM_BYTES=1024*1024;
function integer(value,min,max,name){if(!Number.isSafeInteger(value)||value<min||value>max)throw Error(`${name} must be ${min}..${max}.`);return value;}
function probability(value,name){if(!Number.isFinite(value)||value<0||value>1)throw Error(`${name} must be 0..1.`);return value;}

/** Exercise the real packet state machines and OBM5 header/FEC/symbol codec.
 * The synthetic channel erases whole displayed packet cycles, duplicates decoded
 * packets, and corrupts a monochrome header or payload beyond FEC recovery.
 * It does not simulate camera images, motion, or color classification. Virtual
 * optical time counts advertised phase holds; execution time is measured apart.
 */
export async function runSimulation(options={},onProgress=()=>{}){
  if(options.payload!==undefined&&!(options.payload instanceof Uint8Array))throw Error('Simulation payload must be Uint8Array.');
  const size=integer(options.payload?.length??options.bytes??165000,0,MAX_SIM_BYTES,'Simulation payload bytes');
  const seed=integer(options.seed??1234,0,0xffffffff,'Seed');
  const dropRate=probability(options.dropRate??.15,'Drop probability');
  const duplicateRate=probability(options.duplicateRate??.15,'Duplicate probability');
  const corruptRate=probability(options.corruptRate??.05,'Corruption probability');
  const windowSize=integer(options.windowSize??4,1,8,'Window');
  const profile=normalizeProfile(options.profile??DATA_PROFILE),control={...CONTROL_PROFILE};
  const chunkBytes=getChunkBytes(profile),chunks=Math.max(1,Math.ceil(size/chunkBytes));
  // Waiting starts after a POLL, not when the first DATA frame is displayed.
  const retryMs=integer(options.retryMs??10000,100,3600000,'Retry interval');
  const expectedMs=chunks*profile.phases*profile.holdMs+Math.ceil(chunks/windowSize)*4000;
  const timeoutMs=integer(options.timeoutMs??Math.max(120000,Math.ceil(expectedMs*5+retryMs*32)),1000,86400000,'Timeout');
  const maxRetries=integer(options.maxRetries??24,0,100,'Retry limit');
  const source=options.source??(options.payload?'file':'generated');
  if(!['generated','file'].includes(source))throw Error('Unknown simulation source.');
  const payload=options.payload?options.payload.slice():diagnosticPayload(size,seed),random=seededRandom(seed),started=performance.now();
  let now=0,delivery=null,deliveredCount=0,iterations=0;
  const sender=await createSender(payload,{chunkBytes,windowSize,retryMs,maxRetries,timeoutMs,now});
  const receiver=new Receiver({timeoutMs,completedTtlMs:timeoutMs,maxMessageBytes:MAX_SIM_BYTES});
  const stats={transmissions:0,dataTransmissions:0,controlTransmissions:0,physicalPhases:0,
    dropped:0,duplicated:0,corrupted:0,headerRejected:0,payloadRejected:0,decoded:0,deliveries:0};
  const events=[];
  const note=(kind,detail)=>{events.push({atMs:now,kind,detail});if(events.length>100)events.shift();};
  const summary=()=>({payloadBytes:size,simulatedMs:now,elapsedMs:performance.now()-started,
    stats:{...stats},sender:sender.snapshot(),receiver:receiver.snapshot()});

  async function channel(packet){
    const kind=decodePacket(packet).kind,isData=kind==='data',selected=isData?profile:control;
    const frames=await packetToFrames(packet,selected);
    stats.transmissions++;stats[isData?'dataTransmissions':'controlTransmissions']++;
    stats.physicalPhases+=frames.length;now+=frames.length*selected.holdMs;
    if(random()<dropRate){stats.dropped++;note('drop',kind);return [];}
    // Alternate phase observations, exercising the real palette inverse.
    const frame=frames[Math.floor(random()*frames.length)],header=frame.header.slice(),symbols=frame.symbols.slice();
    if(random()<corruptRate){
      stats.corrupted++;
      if(stats.corrupted%2){header[20]^=1;}
      else {for(let i=0;i<symbols.length;i++)symbols[i]=0;}
    }
    let info,decoded;
    try{info=parseHeader(header);}catch{stats.headerRejected++;note('reject',`${kind} header CRC`);return [];}
    try{decoded=symbolsToChunk(symbols,info);}catch{stats.payloadRejected++;note('reject',`${kind} payload CRC/FEC`);return [];}
    // This is an assertion after decoding, never input to repair or acceptance.
    if(!sameBytes(decoded,packet))throw Error('Optical codec returned unexpected bytes.');
    stats.decoded++;
    if(random()<duplicateRate){stats.duplicated++;note('duplicate',kind);return [decoded,decoded.slice()];}
    return [decoded];
  }

  async function returnPackets(packets){
    for(const packet of packets??[])for(const decoded of await channel(packet))sender.receive(decoded,now);
  }

  while(!TERMINAL.has(sender.snapshot().state)){
    if(++iterations>200000)throw Error('Simulation iteration bound reached.');
    const packet=sender.next(now);
    if(packet){
      for(const decoded of await channel(packet)){
        const result=await receiver.receive(decoded,now);
        if(result.delivery){delivery=result.delivery;deliveredCount++;stats.deliveries=deliveredCount;note('received',`${delivery.bytes.length} bytes`);}
        await returnPackets(result.outbound);
      }
    }else now+=Math.min(1000,retryMs);
    await returnPackets(receiver.tick(now)?.outbound);
    if(iterations%8===0){onProgress(summary());await new Promise(resolve=>setTimeout(resolve,0));}
  }
  const sentDigest=await sha256(payload),receivedDigest=delivery?await sha256(delivery.bytes):null;
  const exact=Boolean(delivery)&&sameBytes(payload,delivery.bytes);
  const state=sender.snapshot().state,report={schema:'neural-key-link-report-v1',mode:'simulation',
    status:state,exact,confirmationReceived:state==='delivered',...summary(),
    options:{bytes:size,source,seed,dropRate,duplicateRate,corruptRate,windowSize,profile,control,retryMs,maxRetries,timeoutMs},
    expectedDigest:sentDigest,receivedDigest,events,
    limits:{maxSimulationBytes:MAX_SIM_BYTES,eventHistory:100},
    scope:'Packet-cycle loss, duplication, and header/payload corruption through the OBM5 codec. No camera/color or cryptographic-authentication test.'};
  if(deliveredCount>1)throw Error('Receiver delivered the same message more than once.');
  if(state==='delivered'&&!exact)throw Error('Sender completed without byte-exact receiver delivery.');
  onProgress(summary());return report;
}
