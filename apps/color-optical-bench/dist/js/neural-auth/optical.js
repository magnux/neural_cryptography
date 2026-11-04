/** Optical framing for the turn demo. Random fill keeps short data matrices dense.
 * This padding has no cryptographic security role; OLM1 and Neural-ZKBoo own theirs.
 */
import {capacity,encodeSequence,phaseFrames,randomBytes} from '../format.js';
import {decodePacket,DATA_OVERHEAD} from '../link/protocol.js';
import {TURN_LIMITS} from './transport.js';

export const DATA_PROFILE=Object.freeze({size:64,colors:16,gridBlock:8,parity:32,phases:4,permute:true});
export const ACK_PROFILE=Object.freeze({size:32,colors:4,gridBlock:0,parity:16,phases:1,permute:false});
const MAGIC=Uint8Array.of(78,90,77,49); // NZM1: current padded optical packet container.
const HEADER_BYTES=6;
const DATA_BYTES=capacity(DATA_PROFILE.size,DATA_PROFILE.colors,DATA_PROFILE);
export function packetFrames(bytes,{phaseMs=95}={}) {
  if(!Number.isInteger(phaseMs)||phaseMs<50||phaseMs>10000)throw Error('Phase time must be 50–10000 ms.');
  const packet=decodePacket(bytes),isAck=packet.kind==='ack';
  if(!isAck&&!['start','data'].includes(packet.kind))throw Error('Unsupported optical turn packet.');
  if(bytes.length>TURN_LIMITS.chunkBytes+DATA_OVERHEAD)throw Error('Optical packet exceeds the turn limit.');
  let payload=bytes;
  if(!isAck){
    if(bytes.length+HEADER_BYTES>DATA_BYTES)throw Error('Turn packet does not fit a data matrix.');
    payload=randomBytes(DATA_BYTES);payload.set(MAGIC);new DataView(payload.buffer).setUint16(4,bytes.length);payload.set(bytes,HEADER_BYTES);
  }
  return phaseFrames(encodeSequence(payload,{...(isAck?ACK_PROFILE:DATA_PROFILE),holdMs:phaseMs,maxFrames:1}));
}
export function unpackOptical(bytes,expected) {
  if(!(bytes instanceof Uint8Array)||!['data','ack'].includes(expected))throw Error('Invalid optical packet input.');
  let packetBytes=bytes;
  if(expected==='data'){
    if(bytes.length!==DATA_BYTES||MAGIC.some((v,i)=>bytes[i]!==v))throw Error('Unexpected data matrix container.');
    const n=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength).getUint16(4);
    if(n<40||n>TURN_LIMITS.chunkBytes+DATA_OVERHEAD||n+HEADER_BYTES>bytes.length)throw Error('Invalid optical packet length.');
    packetBytes=bytes.slice(HEADER_BYTES,HEADER_BYTES+n);
  }
  const packet=decodePacket(packetBytes);
  if(expected==='ack'?packet.kind!=='ack':!['start','data'].includes(packet.kind))throw Error('Unexpected optical packet kind.');
  return packetBytes;
}
