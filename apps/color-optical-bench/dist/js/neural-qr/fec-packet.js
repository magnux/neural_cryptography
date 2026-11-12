/** Shared Neural-QR RS+CRC envelope, also used by offline recorded-error replay.
 * CRC detects damaged channels; it does not authenticate the sender.
 */
import {rsEncode,rsDecode} from '../fec.js';
import {crc32} from '../format.js';

export const DEFAULT_PARITY=32;
export const MAX_WIRE_BYTES=32768;
export const CRC_BYTES=4;

export function packetLayout(wireBytes,parity=DEFAULT_PARITY){
  if(!Number.isInteger(wireBytes)||wireBytes<1||wireBytes>MAX_WIRE_BYTES)
    throw Error(`RS packet wire capacity must be 1..${MAX_WIRE_BYTES} bytes.`);
  if(!Number.isInteger(parity)||parity<2||parity>64||parity%2)
    throw Error('RS packet parity must be an even integer from 2 to 64.');
  const blockCount=Math.ceil(wireBytes/255),short=Math.floor(wireBytes/blockCount),extra=wireBytes%blockCount;
  const blockLengths=Array.from({length:blockCount},(_,i)=>short+Number(i<extra));
  const parityBytes=parity*blockCount,payloadBytes=wireBytes-parityBytes-CRC_BYTES;
  return {wireBytes,parity,parityBytes,crcBytes:CRC_BYTES,payloadBytes,blockCount,blockLengths,
    eligible:Number(payloadBytes>0&&blockLengths.every(n=>n>parity))};
}

function bytes(value,name){
  if(!(value instanceof Uint8Array))throw Error(`${name} must be a Uint8Array.`);
  return value;
}

function requireLayout(wireBytes,parity){
  const layout=packetLayout(wireBytes,parity);
  if(!layout.eligible)throw Error('RS packet capacity is too small for parity, CRC and a nonempty payload.');
  return layout;
}

/** Round-robin columns spread adjacent recorded errors across RS blocks. */
function interleave(blocks,wireBytes){
  const output=new Uint8Array(wireBytes);let offset=0;
  for(let column=0;column<Math.max(...blocks.map(b=>b.length));column++)
    for(const block of blocks)if(column<block.length)output[offset++]=block[column];
  return output;
}

function deinterleave(wire,layout){
  const blocks=layout.blockLengths.map(n=>new Uint8Array(n));let offset=0;
  for(let column=0;column<Math.max(...layout.blockLengths);column++)
    for(const block of blocks)if(column<block.length)block[column]=wire[offset++];
  return blocks;
}

function unpackData(blocks,layout){
  const data=new Uint8Array(layout.payloadBytes+CRC_BYTES);let offset=0;
  for(const block of blocks){const count=block.length-layout.parity;data.set(block.subarray(0,count),offset);offset+=count;}
  return data;
}

function validCRC(data){
  const count=data.length-CRC_BYTES;
  return crc32(data.subarray(0,count))===new DataView(data.buffer,data.byteOffset,data.byteLength).getUint32(count,false);
}

/** Packet envelope: payload + CRC32, then shortened systematic RS. */
export function encodePacket(payload,wireBytes,parity=DEFAULT_PARITY){
  bytes(payload,'Payload');const layout=requireLayout(wireBytes,parity);
  if(payload.length!==layout.payloadBytes)throw Error(`RS packet payload must contain ${layout.payloadBytes} bytes.`);
  const data=new Uint8Array(payload.length+CRC_BYTES);data.set(payload);
  new DataView(data.buffer).setUint32(payload.length,crc32(payload),false);
  const blocks=[];let offset=0;
  for(const length of layout.blockLengths){const count=length-parity;blocks.push(rsEncode(data.subarray(offset,offset+count),parity));offset+=count;}
  return interleave(blocks,wireBytes);
}

/** Blind receiver. It is intentionally unaware of expected bytes and errors. */
export function decodePacket(wire,parity=DEFAULT_PARITY){
  bytes(wire,'Wire');const layout=requireLayout(wire.length,parity),blocks=deinterleave(wire,layout);
  const rawCRCValid=Number(validCRC(unpackData(blocks,layout)));
  const repaired=[];let correctedBytes=0;
  try{
    for(const block of blocks){const result=rsDecode(block,parity);repaired.push(result.bytes);correctedBytes+=result.corrected;}
  }catch{
    return {rawCRCValid,rsDecoded:0,crcAccepted:0,correctedBytes:0,payload:null};
  }
  const data=unpackData(repaired,layout),crcAccepted=Number(validCRC(data));
  return {rawCRCValid,rsDecoded:1,crcAccepted,correctedBytes,
    payload:crcAccepted?data.slice(0,layout.payloadBytes):null};
}

