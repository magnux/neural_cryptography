/** Timing lab only. One exposure carries the fingerprint of an OLM ACK,
 * not an OLM packet or an authentication proof. Compare with the currently
 * outstanding ACK only AFTER blind optical classification, RS and CRC.
 */
import {decodePacket as decodeMessagePacket} from '../link/protocol.js';
import {paletteMap} from '../grid.js';
import {patch, classifySamples} from '../vision.js';
import {NQR_LAYOUT} from './layout.js';
import {neuralQRPacketProfile} from './packet.js';
import {encodePacket as encodeFEC, decodePacket as decodeFEC} from './fec-packet.js';
import {extractNeuralQR} from './vision.js';

const SIZE=16, DIGEST_BYTES=32;
const profile=neuralQRPacketProfile(SIZE), {grid}=profile;
// 'NQR', wire version 1, type 1 (ACK fingerprint). Distinct from 'NQP' packets.
// Followed by the 32-byte digest, then reserved zeroes to fill the FEC payload.
const TAG=Uint8Array.of(0x4e,0x51,0x52,1,1);
const END=TAG.length+DIGEST_BYTES;
const DOMAIN=new TextEncoder().encode('Neural-QR/G16/ACK-receipt/v1\0');

export async function receipt16Digest(ackBytes){
  const packet=decodeMessagePacket(ackBytes);
  if(packet.kind!=='ack'||packet.messageType!=='raw')throw Error('Expected a canonical raw OLM ACK.');
  const input=new Uint8Array(DOMAIN.length+ackBytes.length);
  input.set(DOMAIN);input.set(ackBytes,DOMAIN.length);
  return new Uint8Array(await crypto.subtle.digest('SHA-256',input));
}

/** Each phase is independently readable. Hold frames[0] for a static receipt. */
export function receipt16Frames(digest,phaseMs=350){
  if(!(digest instanceof Uint8Array)||digest.length!==DIGEST_BYTES)throw Error('Expected a 32-byte ACK fingerprint.');
  if(!Number.isInteger(phaseMs)||phaseMs<50||phaseMs>10000)throw Error('Phase time must be 50–10000 ms.');
  const payload=new Uint8Array(profile.payloadBytes);
  payload.set(TAG);payload.set(digest,TAG.length);
  const wire=encodeFEC(payload,profile.wireBytes,profile.parity);
  return Array.from({length:4},(_,phase)=>{
    const frame={size:SIZE,colors:16,bits:4,phases:4,gridBlock:8,permute:true,pixelsPerCell:4,phase,holdMs:phaseMs,parity:profile.parity};
    const symbols=new Uint8Array(SIZE*SIZE);
    for(const cell of grid.pilots)symbols[cell]=grid.reference[cell];
    for(let i=0;i<grid.cells.length;i++){
      const byte=wire[i>>1]??0;
      symbols[grid.cells[i]]=paletteMap(i%2?byte&15:byte>>>4,frame);
    }
    return {...frame,symbols};
  });
}

function readSymbols(symbols,size){
  if(size!==SIZE||!(symbols instanceof Uint8Array)||symbols.length!==SIZE*SIZE||symbols.some(v=>v>15))
    return {stage:'profile',error:'Expected a logical G16 matrix.',digest:null};
  const wire=new Uint8Array(profile.wireBytes);
  for(let i=0;i<wire.length;i++)wire[i]=(symbols[grid.cells[2*i]]<<4)|symbols[grid.cells[2*i+1]];
  const {payload,...stats}=decodeFEC(wire,profile.parity);
  if(!payload)return {stage:stats.rsDecoded?'crc':'fec',error:'G16 receipt did not pass RS+CRC.',stats,digest:null};
  if(TAG.some((v,i)=>v!==payload[i])||payload.subarray(END).some(v=>v!==0))
    return {stage:'receipt-type',error:'Not a G16 ACK receipt.',stats,digest:null};
  return {stage:'decoded',error:null,stats,digest:payload.slice(TAG.length,END)};
}

export function decodeReceipt16Symbols(symbols,size=SIZE){
  return readSymbols(symbols,size).digest;
}

/** Sample each G16 cell with the standard camera patch and calibrated pilot
 * classifier. It scores LOGICAL colors using the *observed* phase: no phase
 * bundle, dense neural input, expected ACK, or trained checkpoint is needed.
 */
export function decodeReceipt16Capture(image,{now,previousGeometry,captureIntervalMs,profiler}={}){
  const acq=extractNeuralQR(image,{profile,now,previousGeometry,captureIntervalMs,profiler,dense:false});
  if(!acq.ok)return {...acq,digest:null};
  const measure=(stage,fn)=>profiler?profiler.measure(stage,fn):fn();
  try{
    const {x,y,side}=NQR_LAYOUT.payload, unit=side/SIZE;
    const samples=measure('cell-sampling',()=>{
      const values=new Float32Array(3*SIZE*SIZE+96);
      for(let i=0;i<SIZE*SIZE;i++)values.set(patch(image,acq.transform,
        x+(i%SIZE+.5)*unit,y+(Math.floor(i/SIZE)+.5)*unit,.16*unit),3*i);
      acq.observedPalette.forEach((rgb,i)=>values.set(rgb,3*(SIZE*SIZE+i)));
      return values;
    });
    const logical=measure('cell-classification',()=>{
      const {scores}=classifySamples(acq.info,samples),symbols=new Uint8Array(SIZE*SIZE);
      for(let i=0;i<grid.cells.length;i++){
        let best=0;
        for(let color=1;color<16;color++)if(scores[16*i+color]>scores[16*i+best])best=color;
        symbols[grid.cells[i]]=best;
      }
      return symbols;
    });
    const decoded=measure('rs-crc',()=>readSymbols(logical,SIZE));
    return {...acq,...decoded,ok:Boolean(decoded.digest),stats:{...acq.stats,...decoded.stats}};
  }catch(error){return {...acq,ok:false,stage:'cell-classification',error:error.message,digest:null};}
}
