import {contentDescriptor,parseContentDescriptor} from './content.js';
import {gridLayout,paletteMap} from './grid.js';
import {rsEncode,rsDecode} from './fec.js';

/** One experimental wire format. Every device must run the same bench release. */
export const FORMAT = Object.freeze({version:5, units:256, headerBytes:64,
  header:{x:48,y:8,cols:32,rows:16,cell:5,cellHeight:3}, payload:{x:48,y:72,side:160},
  centers:[[22,22],[234,22],[234,234],[22,234]], markerSize:28});

const crcTable = Uint32Array.from({length:256},(_,i)=>{
  let c=i; for(let j=0;j<8;j++) c=(c&1)?0xedb88320^(c>>>1):c>>>1; return c>>>0;
});
export function crc32(bytes){let c=0xffffffff;for(const b of bytes)c=crcTable[(c^b)&255]^(c>>>8);return (c^0xffffffff)>>>0;}
export const hex = bytes => Array.from(bytes,b=>b.toString(16).padStart(2,'0')).join('');
export const sameBytes = (a,b) => a.length===b.length && a.every((v,i)=>v===b[i]);

// Same nested farthest-point palettes as neural_optical.py, including tie order.
export function palette(colors=16){
  if(![2,4,8,16].includes(colors))throw Error('Use 2, 4, 8, or 16 colors.');
  const candidates=[];for(const r of [24,96,168,240])for(const g of [24,96,168,240])for(const b of [24,96,168,240])candidates.push([r,g,b]);
  const chosen=[0,63];
  while(chosen.length<colors){let best=-1,index=-1;for(let i=0;i<64;i++){
    if(chosen.includes(i))continue;
    const d=Math.min(...chosen.map(j=>candidates[i].reduce((s,v,k)=>s+(v-candidates[j][k])**2,0)));
    if(d>best){best=d;index=i;}
  }chosen.push(index);}
  return chosen.map(i=>candidates[i]);
}
export const PALETTE = palette(16);
export function transportProfile(size,colors,{gridBlock=0,parity=0}={}){
  const layout=gridLayout(size,colors,gridBlock),wireBytes=Math.floor(layout.cells.length*Math.log2(colors)/8);
  if(![0,16,32].includes(parity))throw Error('Use 0, 16, or 32 RS parity bytes.');
  const blocks=parity?Math.ceil(wireBytes/255):1,capacity=wireBytes-blocks*parity;
  if(capacity<1)throw Error('Grid too small for this reference/FEC profile. Increase its size or disable FEC.');
  const lengths=Array.from({length:blocks},(_,i)=>Math.floor(capacity/blocks)+(i<capacity%blocks?1:0));
  return {layout,wireBytes,capacity,blocks,lengths,parity,pilotCells:layout.pilots.length,dataCells:layout.cells.length};
}
export const capacity=(size,colors,options={})=>transportProfile(size,colors,options).capacity;
export function randomBytes(length){
  const bytes=new Uint8Array(length);
  for(let start=0;start<length;start+=65536)globalThis.crypto.getRandomValues(bytes.subarray(start,start+65536));
  return bytes;
}
export function encodeSequence(payload,{size=32,colors=4,maxFrames=1,id=randomBytes(8),contentType='raw',gridBlock=0,parity=0,phases=1,permute=false,holdMs=4000,probePattern=0,probeSeed=0}={}){
  if(!(payload instanceof Uint8Array))throw Error('Payload must be bytes.');
  if(!(id instanceof Uint8Array)||id.length!==8)throw Error('Sequence ID must be eight bytes.');
  if(!Number.isInteger(maxFrames)||maxFrames<1||maxFrames>8)throw Error('Frame limit must be 1..8.');
  if(![1,2,4,8,16].includes(phases)||typeof permute!=='boolean'||(phases===1&&permute))throw Error('Invalid palette phase profile.');
  if(!Number.isInteger(holdMs)||holdMs<250||holdMs>10000)throw Error('Phase hold time must be 250..10000 ms.');
  if(![0,1,2,3].includes(probePattern)||!Number.isInteger(probeSeed)||probeSeed<0||probeSeed>0xffffffff||(!probePattern&&probeSeed))throw Error('Invalid diagnostic pattern.');
  const descriptor=contentDescriptor(payload,contentType),cap=capacity(size,colors,{gridBlock,parity}),count=Math.max(1,Math.ceil(payload.length/cap));
  if(count>maxFrames)throw Error(`Payload needs ${count} frames; the selected limit is ${maxFrames}.`);
  const fullCRC=crc32(payload),bits=Math.log2(colors),frames=[];
  for(let index=0;index<count;index++){
    const chunk=payload.slice(index*cap,(index+1)*cap),header=new Uint8Array(FORMAT.headerBytes),v=new DataView(header.buffer);
    header.set([79,66,77,53,5,bits]);v.setUint16(6,size);header.set(id,8);
    v.setUint16(16,index);v.setUint16(18,count);v.setUint32(20,chunk.length);v.setUint32(24,payload.length);
    v.setUint32(28,fullCRC);v.setUint32(32,crc32(chunk));header.set(descriptor,36);
    header[44]=gridBlock;header[46]=phases;header[47]=Number(permute);header[48]=parity;v.setUint16(49,holdMs);header[53]=probePattern;v.setUint32(56,probeSeed);
    v.setUint32(60,crc32(header.subarray(0,60)));
    const info=parseHeader(header);frames.push({...info,header,chunk,symbols:chunkToSymbols(chunk,info)});
  }
  return frames;
}
export function parseHeader(header){
  if(!(header instanceof Uint8Array)||header.length!==64||header[4]!==5||header[0]!==79||header[1]!==66||header[2]!==77||header[3]!==53)throw Error('Header not recognized. Reload both devices with this bench release.');
  const v=new DataView(header.buffer,header.byteOffset,header.byteLength);
  if(crc32(header.subarray(0,60))!==v.getUint32(60))throw Error('Header checksum failed. Hold the matrix steady.');
  const bits=header[5],size=v.getUint16(6),colors=2**bits,gridBlock=header[44],phase=header[45],phases=header[46],permute=Boolean(header[47]),parity=header[48],holdMs=v.getUint16(49);
  if(![1,2,4,8,16].includes(phases)||phase>=phases||header[47]>1||(phases===1&&permute)||holdMs<250||holdMs>10000||v.getUint16(51)||v.getUint16(54))throw Error('Invalid transport header.');
  const cap=capacity(size,colors,{gridBlock,parity}),index=v.getUint16(16),count=v.getUint16(18),length=v.getUint32(20),total=v.getUint32(24);
  if(count<1||count>8||index>=count||count!==Math.max(1,Math.ceil(total/cap))||length!==Math.min(cap,Math.max(0,total-index*cap)))throw Error('Invalid frame lengths.');
  const content=parseContentDescriptor(header.subarray(36,44));if(content.length!==total)throw Error('Content length disagrees with the optical header.');
  const probePattern=header[53],probeSeed=v.getUint32(56);
  if(![0,1,2,3].includes(probePattern)||(!probePattern&&probeSeed)||(probePattern&&(content.type!=='raw'||total!==cap*count)))throw Error('Invalid diagnostic payload.');
  return {version:5,content,size,colors,bits,index,count,length,total,id:hex(header.subarray(8,16)),fullCRC:v.getUint32(28),chunkCRC:v.getUint32(32),gridBlock,phase,phases,permute,parity,holdMs,probePattern,probeSeed};
}
export const sequenceKey=i=>`${i.id}/${i.size}/${i.colors}/${i.count}/${i.total}/${i.fullCRC}/${i.content.type}/${i.gridBlock}/${i.phases}/${Number(i.permute)}/${i.parity}/${i.holdMs}/${i.probePattern}/${i.probeSeed}`;
export const frameKey=i=>`${sequenceKey(i)}/${i.index}/${i.length}/${i.chunkCRC}`;
export function phaseFrames(frames){
  return frames.flatMap(first=>Array.from({length:first.phases},(_,phase)=>{
    const header=first.header.slice();header[45]=phase;new DataView(header.buffer).setUint32(60,crc32(header.subarray(0,60)));
    const info=parseHeader(header);return {...info,header,chunk:first.chunk,symbols:chunkToSymbols(first.chunk,info)};
  }));
}
// Round-robin codeword interleaving spreads nearby wire-byte errors over blocks.
function interleave(blocks){
  const out=new Uint8Array(blocks.reduce((n,b)=>n+b.length,0));let at=0;
  for(let col=0;col<Math.max(...blocks.map(b=>b.length));col++)for(const block of blocks)if(col<block.length)out[at++]=block[col];
  return out;
}
function deinterleave(bytes,profile){
  const blocks=profile.lengths.map(n=>new Uint8Array(n+profile.parity));let at=0;
  for(let col=0;col<Math.max(...blocks.map(b=>b.length));col++)for(const block of blocks)if(col<block.length)block[col]=bytes[at++];
  return blocks;
}
export function chunkToSymbols(chunk,info){
  const p=transportProfile(info.size,info.colors,info),padded=new Uint8Array(p.capacity);padded.set(chunk);
  let offset=0;const blocks=p.lengths.map(n=>{const data=padded.slice(offset,offset+n);offset+=n;return info.parity?rsEncode(data,info.parity):data;});
  const wire=interleave(blocks),symbols=new Uint8Array(info.size**2);
  for(const cell of p.layout.pilots)symbols[cell]=p.layout.reference[cell];
  for(let i=0;i<p.layout.cells.length;i++){
    let symbol=0;for(let bit=0;bit<info.bits;bit++){const pos=i*info.bits+bit;symbol=(symbol<<1)|((wire[pos>>>3]??0)>>(7-(pos&7))&1);}
    symbols[p.layout.cells[i]]=paletteMap(symbol,info);
  }
  return symbols;
}
/** Logical labels; reliability is one byte per data-cell ordinal, excluding pilots. */
export function decodeSymbols(symbols,info,{reliability=null,recovery='hard'}={}){
  if(symbols.length!==info.size**2||symbols.some(v=>!Number.isInteger(v)||v<0||v>=info.colors))throw Error('Invalid payload grid.');
  const p=transportProfile(info.size,info.colors,info),wire=new Uint8Array(p.wireBytes);let tailBad=false;
  if(!['hard','erasures'].includes(recovery)||reliability!==null&&(!(reliability instanceof Uint8Array)||reliability.length!==p.dataCells))throw Error('Invalid recovery evidence.');
  const wireReliability=reliability?new Uint8Array(p.wireBytes).fill(255):null;
  for(let i=0;i<p.layout.cells.length;i++)for(let bit=0;bit<info.bits;bit++){
    const pos=i*info.bits+bit,b=(symbols[p.layout.cells[i]]>>(info.bits-1-bit))&1;
    if(pos<wire.length*8){wire[pos>>>3]|=b<<(7-(pos&7));if(wireReliability)wireReliability[pos>>>3]=Math.min(wireReliability[pos>>>3],reliability[i]);}else if(b)tailBad=true;
  }
  const blocks=deinterleave(wire,p);
  const unpack=blocks=>{const data=new Uint8Array(p.capacity);let at=0;blocks.forEach((b,i)=>{data.set(b.subarray(0,p.lengths[i]),at);at+=p.lengths[i];});return data;};
  const valid=data=>!tailBad&&!data.subarray(info.length).some(Boolean)&&crc32(data.subarray(0,info.length))===info.chunkCRC;
  const rawCRC=valid(unpack(blocks));let erasureTrials=0;
  const accepted=(data,method,correctedBytes,erasedBytes=0)=>({chunk:data.slice(0,info.length),rawCRC,hardCRC:method!=='erasures',correctedBytes,erasedBytes,erasureTrials,recovery:method,blocks:p.blocks,parity:info.parity});
  if(rawCRC)return accepted(unpack(blocks),'raw',0);
  if(info.parity){
    // Attempt every block independently so successful hard decisions can be reused.
    const hard=blocks.map(b=>{try{return rsDecode(b,info.parity);}catch{return null;}}),allHard=hard.every(Boolean);
    if(allHard){const data=unpack(hard.map(r=>r.bytes));if(valid(data))return accepted(data,'hard',hard.reduce((n,r)=>n+r.corrected,0));}
    if(recovery==='erasures'&&wireReliability&&!tailBad){
      const reliabilities=deinterleave(wireReliability,p),ranked=reliabilities.map(r=>Array.from(r.keys()).sort((a,b)=>r[a]-r[b]||a-b));
      const budgets=[...new Set([4,8,16,32].map(n=>Math.min(n,info.parity)))];
      for(const budget of budgets){
        erasureTrials++;const candidate=[];let correctedBytes=0,erasedBytes=0,failed=false;
        for(let i=0;i<blocks.length;i++){
          // If all blocks passed their syndromes but CRC failed, none is trusted.
          if(!allHard&&hard[i]){candidate.push(hard[i].bytes);correctedBytes+=hard[i].corrected;continue;}
          try{const erased=ranked[i].slice(0,budget),decoded=rsDecode(blocks[i],info.parity,erased);candidate.push(decoded.bytes);correctedBytes+=decoded.corrected;erasedBytes+=erased.length;}
          catch{failed=true;break;}
        }
        if(!failed){const data=unpack(candidate);if(valid(data))return accepted(data,'erasures',correctedBytes,erasedBytes);}
      }
    }
  }
  throw Object.assign(Error('Payload checksum failed. Move closer or lower the density.'),{rawCRC,hardCRC:false,correctedBytes:0,erasedBytes:0,erasureTrials,recovery:erasureTrials?'erasures':info.parity?'hard':'raw'});
}
export function symbolsToChunk(symbols,info){
  const logical=symbols.slice(),layout=gridLayout(info.size,info.colors,info.gridBlock);
  for(const cell of layout.cells)logical[cell]=paletteMap(logical[cell],info);
  return decodeSymbols(logical,info).chunk;
}
export class Assembler{
  constructor(){this.reset();}
  reset(){this.key=null;this.parts=new Map();this.info=null;this.complete=null;}
  accept(info,chunk){
    if(chunk.length!==info.length||crc32(chunk)!==info.chunkCRC)throw Error('Invalid received chunk.');
    const key=sequenceKey(info);
    if(this.key&&this.key!==key)throw Error('Different sequence detected. Reset the receiver to accept it.');
    this.key=key;this.info=info;
    if(this.parts.has(info.index)&&!sameBytes(this.parts.get(info.index),chunk))throw Error('Conflicting repeated frame.');
    this.parts.set(info.index,chunk.slice());
    if(this.parts.size===info.count){
      const all=new Uint8Array(info.total);let offset=0;
      for(let i=0;i<info.count;i++){all.set(this.parts.get(i),offset);offset+=this.parts.get(i).length;}
      if(crc32(all)!==info.fullCRC)throw Error('Sequence checksum failed.');
      this.complete=all;
    }
    return {received:this.parts.size,count:info.count,complete:this.complete};
  }
}

/** Pure RGBA rendering, shared by browser, camera simulation, and Node tests. */
export function renderFrame(frame,pixels=1024){
  if(!Number.isInteger(pixels)||pixels<256||pixels>4096)throw Error('Render size must be 256..4096.');
  const data=new Uint8ClampedArray(pixels*pixels*4);data.fill(255);const scale=pixels/256;
  function rect(x,y,w,h,color){
    const left=Math.round(x*scale),top=Math.round(y*scale),right=Math.round((x+w)*scale),bottom=Math.round((y+h)*scale);
    for(let yy=top;yy<bottom;yy++)for(let xx=left;xx<right;xx++){const i=(yy*pixels+xx)*4;data[i]=color[0];data[i+1]=color[1];data[i+2]=color[2];}
  }
  for(const [cx,cy] of FORMAT.centers){rect(cx-14,cy-14,28,28,[16,16,16]);rect(cx-10,cy-10,20,20,[248,248,248]);rect(cx-6,cy-6,12,12,[16,16,16]);}
  // Known all-color pilots on both sides; these do not depend on the payload palette.
  for(let i=0;i<16;i++){rect(14,48+10*i,16,8,PALETTE[i]);rect(226,48+10*i,16,8,PALETTE[i]);}
  const h=FORMAT.header;
  for(let bit=0;bit<frame.header.length*8;bit++)rect(h.x+(bit%h.cols)*h.cell,h.y+Math.floor(bit/h.cols)*h.cellHeight,h.cell,h.cellHeight,(frame.header[bit>>>3]>>(7-(bit&7)))&1?[16,16,16]:[248,248,248]);
  const p=FORMAT.payload,cell=p.side/frame.size;
  for(let y=0;y<frame.size;y++)for(let x=0;x<frame.size;x++)rect(p.x+x*cell,p.y+y*cell,cell,cell,PALETTE[frame.symbols[y*frame.size+x]]);
  // Alternating timing tracks separated from payload by a small white gutter.
  for(let i=0;i<32;i++){const c=i%2?[248,248,248]:[16,16,16];rect(48+i*5,65,5,3,c);rect(41,72+i*5,3,5,c);}
  return {width:pixels,height:pixels,data};
}
