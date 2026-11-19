// A timing step is one full-size OLM DATA and its G32 ACK, not a multipart
// transfer. Both are ordinary four-phase, RS32+CRC-protected Neural-QRs.
// Only a successfully corrected byte packet enters these helpers.
import {randomBytes} from './js/format.js';
import {decodePacket,encodePacket,randomId,sha256} from './js/link/protocol.js';
import {chunkBytesForSize,CONTROL_SIZE,packetFrames,unpackOptical} from './js/neural-auth/optical.js';

const magic=new TextEncoder().encode('NQSWEEP1');
const int=(value,min,max)=>Number.isSafeInteger(value)&&value>=min&&value<=max;
const validRun=id=>typeof id==='string'&&/^[0-9a-f]{32}$/.test(id);
function validate({runId,step,size,phaseMs,captureMs},receive=false){
  if(!(receive&&runId===null)&&!validRun(runId))throw Error('Invalid timing sweep run ID.');
  if(!int(step,1,65535)||![32,64,128].includes(size)||!int(phaseMs,50,10000)||!int(captureMs,30,2000))
    throw Error('Invalid timing sweep step, grid or intervals.');
}
// Checks both OLM CRC and expected optical turn. Reject unrelated observations
// quietly and keep collecting; a bad scan is not a failed sweep step.
function read(chunk,kind){
  try{const p=decodePacket(unpackOptical(chunk,kind));return p.kind===kind?p:null;}
  catch{return null;}
}

export const newSweepRunId=()=>randomId(16);

export async function createSweepProbe({runId,step,size,phaseMs,captureMs}){
  validate({runId,step,size,phaseMs,captureMs});
  const data=randomBytes(chunkBytesForSize(size)),v=new DataView(data.buffer),messageId=randomId(8);
  data.set(magic);v.setUint8(8,size);v.setUint16(9,phaseMs);v.setUint16(11,captureMs);
  const bytes=encodePacket({kind:'data',messageType:'raw',sessionId:runId,messageId,index:step-1,data});
  return {runId,step,size,phaseMs,captureMs,messageId,digest:await sha256(data),
    frames:packetFrames(bytes,{size,phaseMs})};
}

// The caller pins the first runId and arms the next expected step only on an
// explicit user action. A stale, foreign or mismatched-cadence probe cannot
// silently advance the sweep. Hash the observed data, never a seeded target.
export async function acknowledgeSweepProbe(chunk,{runId,step,size,phaseMs,captureMs}){
  validate({runId,step,size,phaseMs,captureMs},true);
  const p=read(chunk,'data');
  if(!p||p.index!==step-1||(runId!==null&&runId!==p.sessionId)||p.data.length!==chunkBytesForSize(size)||
    !magic.every((v,i)=>p.data[i]===v))return null;
  const v=new DataView(p.data.buffer,p.data.byteOffset,p.data.byteLength);
  if(v.getUint8(8)!==size||v.getUint16(9)!==phaseMs||v.getUint16(11)!==captureMs)return null;
  const digest=await sha256(p.data);
  const bytes=encodePacket({kind:'ack',messageType:'raw',sessionId:p.sessionId,messageId:p.messageId,
    base:p.index,bitmap:1,status:'complete',digest});
  return {runId:p.sessionId,step,size:CONTROL_SIZE,phaseMs,captureMs,messageId:p.messageId,
    frames:packetFrames(bytes,{size:CONTROL_SIZE,phaseMs})};
}

export function matchesSweepAck(chunk,probe){
  const p=read(chunk,'ack');
  return !!p&&!!probe&&p.sessionId===probe.runId&&p.messageId===probe.messageId&&p.base===probe.step-1&&
    p.status==='complete'&&p.bitmap===1&&p.digest===probe.digest;
}
