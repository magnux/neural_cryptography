import {capacity,encodeSequence,phaseFrames} from '../format.js';
import {DATA_OVERHEAD,CONTROL_MAX_BYTES} from './protocol.js';

/** A packet occupies one OBM5 logical frame; phases repeat the same bytes.
 * Multipart messages are assembled by the link, never by increasing OBM5's
 * existing eight-frame sequence limit. */
export const CONTROL_PROFILE=Object.freeze({size:32,colors:4,gridBlock:8,parity:32,phases:1,permute:false,holdMs:1000});
export const DATA_PROFILE=Object.freeze({size:128,colors:16,gridBlock:8,parity:32,phases:8,permute:true,holdMs:4000});

export function normalizeProfile(profile=DATA_PROFILE){
  if(!profile||typeof profile!=='object'||Array.isArray(profile))throw Error('An optical profile is required.');
  const {size=DATA_PROFILE.size,colors=DATA_PROFILE.colors,gridBlock=DATA_PROFILE.gridBlock,
    parity=DATA_PROFILE.parity,phases=DATA_PROFILE.phases,holdMs=DATA_PROFILE.holdMs}=profile;
  const permute=profile.permute??phases>1;
  if(![1,2,4,8,16].includes(phases)||typeof permute!=='boolean'||(phases===1&&permute)||
    !Number.isInteger(holdMs)||holdMs<250||holdMs>10000)throw Error('Invalid optical link phase profile.');
  const out={size,colors,gridBlock,parity,phases,permute,holdMs};
  if(capacity(size,colors,out)<=DATA_OVERHEAD)throw Error('The optical profile cannot fit the link DATA header and content.');
  return out;
}

export function getChunkBytes(profile=DATA_PROFILE){
  const selected=normalizeProfile(profile);
  return capacity(selected.size,selected.colors,selected)-DATA_OVERHEAD;
}

export function getControlCapacity(profile=CONTROL_PROFILE){
  const selected=normalizeProfile(profile),bytes=capacity(selected.size,selected.colors,selected);
  if(bytes<CONTROL_MAX_BYTES)throw Error('The optical control profile cannot fit the largest link control packet.');
  return bytes;
}

/** The ID follows packet bytes, so a repeated packet can reuse accumulated
 * observations, while a changed ACK or DATA packet starts independent evidence.
 * It is a routing discriminator, not an authentication tag. No unbounded cache
 * is retained; a caller can retain these frames while displaying one packet. */
export async function packetToFrames(packet,profile=DATA_PROFILE){
  if(!(packet instanceof Uint8Array)||packet.length<1)throw Error('An optical link packet must contain bytes.');
  const selected=normalizeProfile(profile),limit=capacity(selected.size,selected.colors,selected);
  if(packet.length>limit)throw Error(`Link packet has ${packet.length} bytes; this optical profile holds ${limit}.`);
  if(!globalThis.crypto?.subtle)throw Error('Web Crypto is required; open the app over HTTPS or localhost.');
  // Snapshot before awaiting digest; caller mutation cannot split ID and content.
  const bytes=new Uint8Array(packet),digest=new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256',bytes));
  return phaseFrames(encodeSequence(bytes,{...selected,id:digest.slice(0,8),contentType:'raw',maxFrames:1}));
}

getControlCapacity();
