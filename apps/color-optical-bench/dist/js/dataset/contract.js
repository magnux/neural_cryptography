/** Current offline dataset contract. Sender truth is never a receiver input. */
import {capacity, encodeSequence, phaseFrames, parseHeader, frameKey, randomBytes, hex} from '../format.js';

export const DATASET_FORMAT = 'optical-decoder-dataset';
export const DATASET_VERSION = 1;
export const MAX_CAPTURE_BYTES = 32 * 1024 * 1024;
export const MAX_RECORDS = 20;
export const captureKey = info => `${frameKey(info)}/${info.phase}`;
export function canonicalJSON(value) {
  if(Array.isArray(value))return `[${value.map(canonicalJSON).join(',')}]`;
  if(value&&typeof value==='object')return `{${Object.keys(value).sort().map(key=>`${JSON.stringify(key)}:${canonicalJSON(value[key])}`).join(',')}}`;
  return JSON.stringify(value);
}

export function bytesToBase64(bytes) {
  if (!(bytes instanceof Uint8Array)) throw Error('Expected bytes.');
  if(globalThis.Buffer) return globalThis.Buffer.from(bytes.buffer,bytes.byteOffset,bytes.byteLength).toString('base64');
  let text='';
  for(let i=0;i<bytes.length;i+=32768) text+=String.fromCharCode(...bytes.subarray(i,i+32768));
  return btoa(text);
}
export function base64ToBytes(text, limit=MAX_CAPTURE_BYTES) {
  if(typeof text!=='string'||text.length>Math.ceil(limit/3)*4||! /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(text)) throw Error('Invalid or oversized base64 bytes.');
  if(globalThis.Buffer) {
    const decoded=globalThis.Buffer.from(text,'base64');
    if(decoded.length>limit)throw Error('Decoded bytes exceed limit.');
    return new Uint8Array(decoded.buffer,decoded.byteOffset,decoded.byteLength);
  }
  const decoded=atob(text);
  if(decoded.length>limit) throw Error('Decoded bytes exceed limit.');
  return Uint8Array.from(decoded,c=>c.charCodeAt(0));
}
export function floatBytes(values) {
  const bytes=new Uint8Array(values.length*4),view=new DataView(bytes.buffer);
  for(let i=0;i<values.length;i++) {
    if(!Number.isFinite(values[i])) throw Error('Non-finite tensor value.');
    view.setFloat32(i*4,values[i],true);
  }
  return bytes;
}
export function floatsFromBytes(bytes, count) {
  if(!(bytes instanceof Uint8Array)||bytes.length!==count*4) throw Error('Wrong float32 tensor byte length.');
  const result=new Float32Array(count),view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  for(let i=0;i<count;i++) {result[i]=view.getFloat32(i*4,true);if(!Number.isFinite(result[i])) throw Error('Non-finite tensor value.');}
  return result;
}
export async function sha256(bytes) {return hex(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)));}
export function validateSessionId(value) {
  if(typeof value!=='string'||! /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}$/.test(value)) throw Error('Invalid dataset session ID.');
  return value;
}
export function validateProfile(profile) {
  if(!profile||![64,128].includes(profile.size)||![4,16].includes(profile.colors)||
    ![1,2,4,8,16].includes(profile.phases)||![0,4,8,16].includes(profile.gridBlock)||
    ![0,16,32].includes(profile.parity)||!Number.isInteger(profile.holdMs)||profile.holdMs<250||profile.holdMs>10000) throw Error('Unsupported capture profile.');
  return profile;
}
function seededBytes(length,seed) {
  let state=seed>>>0;const bytes=new Uint8Array(length);
  for(let i=0;i<length;i++) {state=(Math.imul(state,1664525)+1013904223)>>>0;bytes[i]=state>>>24;}
  return bytes;
}

/** One static logical frame; the UI advances phases manually, with a settling guard. */
export async function createSender({size=64,colors=16,phases=4,gridBlock=8,parity=32,holdMs=4000,sessionId=crypto.randomUUID(),seed}={}) {
  validateSessionId(sessionId);
  const profile=validateProfile({size,colors,phases,gridBlock,parity,holdMs});
  if(seed!==undefined&&(!Number.isInteger(seed)||seed<0||seed>0xffffffff)) throw Error('Invalid public synthetic seed.');
  const bytes=seed===undefined?randomBytes(capacity(size,colors,profile)+8):seededBytes(capacity(size,colors,profile)+8,seed);
  const id=bytes.slice(0,8),payload=bytes.slice(8);
  const frames=phaseFrames(encodeSequence(payload,{...profile,id,permute:phases>1,contentType:'raw'}));
  const manifest={format:'optical-decoder-sender',version:1,created:new Date().toISOString(),sessionId,
    payloadId:await sha256(payload),payloadBase64:bytesToBase64(payload),sequenceId:hex(id),profile,
    frames:frames.map(frame=>({key:captureKey(frame),info:parseHeader(frame.header),headerBase64:bytesToBase64(frame.header)}))};
  return {manifest,frames};
}

/** Recreate only offline or on the sender. Never call this on the receive path. */
export async function senderFrames(manifest) {
  if(!manifest||manifest.format!=='optical-decoder-sender'||manifest.version!==1) throw Error('Unsupported sender manifest.');
  validateSessionId(manifest.sessionId);const profile=validateProfile(manifest.profile);
  if(typeof manifest.sequenceId!=='string'||! /^[a-f0-9]{16}$/.test(manifest.sequenceId)||! /^[a-f0-9]{64}$/.test(manifest.payloadId)) throw Error('Invalid sender identity.');
  const payload=base64ToBytes(manifest.payloadBase64,capacity(profile.size,profile.colors,profile));
  if(payload.length!==capacity(profile.size,profile.colors,profile)||await sha256(payload)!==manifest.payloadId) throw Error('Sender payload hash/length mismatch.');
  const id=Uint8Array.from(manifest.sequenceId.match(/../g),s=>parseInt(s,16));
  const frames=phaseFrames(encodeSequence(payload,{...profile,id,permute:profile.phases>1,contentType:'raw'}));
  if(!Array.isArray(manifest.frames)||manifest.frames.length!==frames.length) throw Error('Sender phase count mismatch.');
  for(let i=0;i<frames.length;i++) {
    const row=manifest.frames[i],frame=frames[i];
    if(row.key!==captureKey(frame)||row.headerBase64!==bytesToBase64(frame.header)||canonicalJSON(row.info)!==canonicalJSON(parseHeader(frame.header))) throw Error('Sender header or phase metadata mismatch.');
  }
  return frames;
}
