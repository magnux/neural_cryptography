/** Current offline dataset contract. Sender truth is never a receiver input. */
import {randomBytes, hex, crc32} from '../format.js';
import {gridLayout,paletteMap} from '../grid.js';

export const DATASET_FORMAT = 'optical-decoder-dataset'; // Established offline training experiment.
export const NEURAL_QR_DATASET_FORMAT = 'neural-qr-dataset';
export const DATASET_VERSION = 1;
export const MAX_CAPTURE_BYTES = 800 * 1024 * 1024;
export const DEFAULT_CAPTURE_BYTES = 400 * 1024 * 1024;
export const CAPTURE_BUDGET_MIB = Object.freeze([200,400,800]);
export const MAX_RECORDS = 128;
export const CAPTURE_SIZES = Object.freeze([8,16,32,64,128,256]);
export const CAPTURE_PIXELS_PER_CELL = 2;
/** Header-free Neural-QR acquisition profile; training symbols have no packet framing. */
export function captureProfile(size=64,colors=16,phases=4){
  const profile={size,colors,phases,gridBlock:8,parity:0,holdMs:4000,permute:true,pixelsPerCell:CAPTURE_PIXELS_PER_CELL};
  validateProfile(profile);return profile;
}
/** Conservative JSON budget across all phases, including camera metadata and two PNGs. */
export function recommendedPhaseMaximum(size,phases=4,budgetBytes=DEFAULT_CAPTURE_BYTES){
  if(!CAPTURE_SIZES.includes(size)||![1,2,4,8,16].includes(phases))throw Error('Unsupported capture pool profile.');
  if(!Number.isSafeInteger(budgetBytes)||budgetBytes<1024||budgetBytes>MAX_CAPTURE_BYTES)throw Error('Invalid capture byte budget.');
  const inputBytes=3*(CAPTURE_PIXELS_PER_CELL*size)**2*4,sampleBytes=(3*size*size+96)*4;
  const perRecord=4*Math.ceil(inputBytes/3)+4*Math.ceil(sampleBytes/3)+8192;
  const allowance=Math.min(Math.floor(MAX_RECORDS/phases),Math.floor((budgetBytes-3*1024*1024)/(perRecord*phases)));
  return [32,16,8,6,4,2,1].find(n=>n<=allowance)??1;
}
export function validateCollectionCode(value){
  if(typeof value!=='string'||! /^[a-f0-9]{8}$/.test(value))throw Error('Collection code must be eight lowercase hexadecimal characters.');
  return value;
}
export const captureKey = info => `${validateCollectionCode(info.collectionCode??info.id)}/${info.size}/${info.colors}/${info.gridBlock}/${info.phase}`;
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
  if(!profile||!CAPTURE_SIZES.includes(profile.size)||profile.colors!==16||profile.phases!==4||
    profile.gridBlock!==8||profile.parity!==0||profile.permute!==true||profile.pixelsPerCell!==2||
    !Number.isInteger(profile.holdMs)||profile.holdMs<500||profile.holdMs>10000) throw Error('Unsupported Neural-QR capture profile.');
  return profile;
}
function seededBytes(length,seed) {
  let state=seed>>>0;const bytes=new Uint8Array(length);
  for(let i=0;i<length;i++) {state=(Math.imul(state,1664525)+1013904223)>>>0;bytes[i]=state>>>24;}
  return bytes;
}

/** One static logical matrix; identity is assigned publicly, never read from hidden truth. */
export async function createSender({size=64,colors=16,phases=4,gridBlock=8,parity=0,holdMs=4000,
  pixelsPerCell=2,permute=true,sessionId=crypto.randomUUID(),seed}={}) {
  validateSessionId(sessionId);
  const profile=validateProfile({size,colors,phases,gridBlock,parity,holdMs,pixelsPerCell,permute});
  if(seed!==undefined&&(!Number.isInteger(seed)||seed<0||seed>0xffffffff))throw Error('Invalid public synthetic seed.');
  const logical=seed===undefined?randomBytes(size*size):seededBytes(size*size,seed);
  for(let i=0;i<logical.length;i++)logical[i]&=15;
  const layout=gridLayout(size,colors,gridBlock);
  for(const cell of layout.pilots)logical[cell]=layout.reference[cell];
  const collectionCode=crc32(logical).toString(16).padStart(8,'0');
  const manifest={format:'neural-qr-sender',version:1,created:new Date().toISOString(),sessionId,
    collectionCode,payloadId:await sha256(logical),payloadBase64:bytesToBase64(logical),profile};
  const frames=makeFrames(logical,profile,collectionCode);
  manifest.frames=frames.map(frame=>({key:captureKey(frame),info:frame.info}));
  return {manifest,frames};
}
function makeFrames(logical,profile,collectionCode){
  const layout=gridLayout(profile.size,profile.colors,profile.gridBlock);
  return Array.from({length:4},(_,phase)=>{
    const info={...profile,id:collectionCode,collectionCode,phase,bits:4};
    const symbols=logical.slice();for(const cell of layout.cells)symbols[cell]=paletteMap(symbols[cell],info);
    return {...info,info,symbols,logicalSymbols:logical.slice(),chunk:logical.slice()};
  });
}
/** Offline/sender only. The receiver never imports a manifest or clean symbols. */
export async function senderFrames(manifest){
  if(!manifest||manifest.format!=='neural-qr-sender'||manifest.version!==1)throw Error('Unsupported Neural-QR sender manifest.');
  validateSessionId(manifest.sessionId);const profile=validateProfile(manifest.profile);
  const code=validateCollectionCode(manifest.collectionCode),logical=base64ToBytes(manifest.payloadBase64,profile.size**2);
  if(logical.length!==profile.size**2||logical.some(v=>v>=16)||await sha256(logical)!==manifest.payloadId||
    crc32(logical).toString(16).padStart(8,'0')!==code)throw Error('Sender logical matrix hash/length/code mismatch.');
  const layout=gridLayout(profile.size,16,8);
  if(Array.from(layout.pilots).some(cell=>logical[cell]!==layout.reference[cell]))throw Error('Sender grid references disagree.');
  const frames=makeFrames(logical,profile,code);
  if(!Array.isArray(manifest.frames)||manifest.frames.length!==4)throw Error('Sender phase count mismatch.');
  for(let i=0;i<4;i++)if(manifest.frames[i].key!==captureKey(frames[i])||canonicalJSON(manifest.frames[i].info)!==canonicalJSON(frames[i].info))throw Error('Sender phase metadata mismatch.');
  return frames;
}
