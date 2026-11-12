/** Neural-QR in-band packets: geometry/type/length, random padding, RS32 + CRC.
 * Acquisition markers reveal geometry and phase only. All message identities,
 * sequence numbers and digests remain inside the protected OLM packet bytes.
 */
import {gridLayout,paletteMap} from '../grid.js';
import {randomBytes} from '../format.js';
import {decodePacket as decodeMessagePacket,DATA_OVERHEAD} from '../link/protocol.js';
import {NQR_LAYOUT,validateNeuralQRProfile} from './layout.js';
import {packetLayout,encodePacket as encodeFEC,decodePacket as decodeFEC} from './fec-packet.js';

export const CONTROL_SIZE=32;
export const DATA_SIZES=Object.freeze([32,64,128]);
export const NQR_PARITY=32;
export const NQR_PACKET_HEADER_BYTES=8;
const MAGIC=Uint8Array.of(78,81,80,1); // NQP1: one current in-band envelope.
const KINDS=Object.freeze({start:1,data:2,ack:3});

export function neuralQRPacketProfile(size){
  const geometry=validateNeuralQRProfile({size,colors:16,phases:4,gridBlock:8});
  const grid=gridLayout(size,16,8),wireBytes=Math.floor(grid.cells.length/2);
  const fec=packetLayout(wireBytes,NQR_PARITY);
  return {...geometry,parity:NQR_PARITY,grid,...fec,packetBytes:fec.payloadBytes-NQR_PACKET_HEADER_BYTES};
}

export function packetCapacity(size){return neuralQRPacketProfile(size).packetBytes;}

export function chunkBytesForSize(size){
  if(!DATA_SIZES.includes(size))throw Error('Data geometry must be G32, G64 or G128.');
  return packetCapacity(size)-DATA_OVERHEAD;
}

function messagePacket(bytes){
  const packet=decodeMessagePacket(bytes);
  if(!Object.hasOwn(KINDS,packet.kind)||packet.messageType!=='raw')throw Error('Unsupported Neural-QR turn packet.');
  return packet;
}

/** Four physical phase matrices from one logical RS codeword and padding draw. */
export function encodeNeuralQR(bytes,{size=64,phaseMs=150}={}){
  if(!Number.isInteger(phaseMs)||phaseMs<50||phaseMs>10000)throw Error('Phase time must be 50–10000 ms.');
  const packet=messagePacket(bytes),profile=neuralQRPacketProfile(size);
  if(bytes.length>profile.packetBytes)throw Error(`Turn packet does not fit G${size}: ${bytes.length} > ${profile.packetBytes} bytes.`);
  const padded=randomBytes(profile.payloadBytes);
  padded.set(MAGIC);padded[4]=NQR_LAYOUT.sizes.indexOf(size);padded[5]=KINDS[packet.kind];
  new DataView(padded.buffer).setUint16(6,bytes.length,false);padded.set(bytes,NQR_PACKET_HEADER_BYTES);
  const wire=encodeFEC(padded,profile.wireBytes,NQR_PARITY),logical=new Uint8Array(size*size);
  for(const cell of profile.grid.pilots)logical[cell]=profile.grid.reference[cell];
  for(let i=0;i<profile.grid.cells.length;i++)logical[profile.grid.cells[i]]=i<wire.length*2
    ?(i%2?wire[i>>1]&15:wire[i>>1]>>>4):randomBytes(1)[0]&15;
  return Array.from({length:4},(_,phase)=>{
    const info={size,colors:16,bits:4,phases:4,gridBlock:8,permute:true,pixelsPerCell:4,phase,holdMs:phaseMs,parity:NQR_PARITY};
    const symbols=logical.slice();
    for(const cell of profile.grid.cells)symbols[cell]=paletteMap(symbols[cell],info);
    return {...info,symbols};
  });
}

/** Decode canonical symbols. Pilots and a possible unused final nibble are ignored. */
export function decodeNeuralQRSymbols(symbols,size){
  const profile=neuralQRPacketProfile(size);
  if(!(symbols instanceof Uint8Array)||symbols.length!==size*size||symbols.some(v=>v>15))
    throw Error('Expected a canonical 16-color logical matrix.');
  const wire=new Uint8Array(profile.wireBytes);
  for(let i=0;i<wire.length;i++)wire[i]=(symbols[profile.grid.cells[2*i]]<<4)|symbols[profile.grid.cells[2*i+1]];
  const decoded=decodeFEC(wire,NQR_PARITY);
  const stats={wireBytes:profile.wireBytes,rsBlocks:profile.blockCount,rsParity:NQR_PARITY,
    rawCRCValid:decoded.rawCRCValid,rsDecoded:decoded.rsDecoded,crcAccepted:decoded.crcAccepted,correctedBytes:decoded.correctedBytes};
  if(!decoded.payload)return {ok:false,stage:decoded.rsDecoded?'crc':'fec',error:'Gathering clearer phases for RS+CRC recovery.',stats};
  try{
    const padded=decoded.payload;
    if(MAGIC.some((value,i)=>padded[i]!==value)||padded[4]!==NQR_LAYOUT.sizes.indexOf(size))throw Error('Unexpected Neural-QR envelope or geometry.');
    const length=new DataView(padded.buffer,padded.byteOffset,padded.byteLength).getUint16(6,false);
    if(length<40||length>profile.packetBytes)throw Error('Invalid protected packet length.');
    const chunk=padded.slice(NQR_PACKET_HEADER_BYTES,NQR_PACKET_HEADER_BYTES+length),packet=messagePacket(chunk);
    if(padded[5]!==KINDS[packet.kind])throw Error('Protected packet kind disagrees with its message.');
    return {ok:true,stage:'decoded',error:null,chunk,kind:packet.kind,stats};
  }catch(error){return {ok:false,stage:'packet',error:error.message,stats};}
}

/** Network output is canonical NCHW logits, never phase-inverted a second time. */
export function decodeNeuralQR(logits,size){
  const cells=size*size;
  neuralQRPacketProfile(size);
  if(!(logits instanceof Float32Array)||logits.length!==16*cells||logits.some(v=>!Number.isFinite(v)))
    throw Error('Invalid Neural-QR corrector logits.');
  const symbols=new Uint8Array(cells);
  for(let cell=0;cell<cells;cell++){
    let best=0;for(let color=1;color<16;color++)if(logits[color*cells+cell]>logits[best*cells+cell])best=color;
    symbols[cell]=best;
  }
  return decodeNeuralQRSymbols(symbols,size);
}
