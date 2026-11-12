/** Header-free Neural-QR framing for both directions of the turn demo. */
import {decodePacket} from '../link/protocol.js';
import {encodeNeuralQR,CONTROL_SIZE,packetCapacity} from '../neural-qr/packet.js';
export {CONTROL_SIZE,chunkBytesForSize} from '../neural-qr/packet.js';

export const DATA_PROFILE=Object.freeze({size:64,colors:16,gridBlock:8,parity:32,phases:4,permute:true,pixelsPerCell:4});

export function packetFrames(bytes,{size=64,phaseMs=150}={}){
  const packet=decodePacket(bytes);
  return encodeNeuralQR(bytes,{size:packet.kind==='ack'?CONTROL_SIZE:size,phaseMs});
}

/** FEC decoding already returned the protected OLM bytes; enforce expected turn. */
export function unpackOptical(bytes,expected){
  if(!(bytes instanceof Uint8Array)||!['data','ack'].includes(expected)||bytes.length>packetCapacity(128))
    throw Error('Invalid optical packet input.');
  const packet=decodePacket(bytes);
  if(packet.messageType!=='raw'||(expected==='ack'?packet.kind!=='ack':!['start','data'].includes(packet.kind)))
    throw Error('Unexpected optical packet kind.');
  return bytes.slice();
}
