/** Single-exposure timing diagnostic. Test payload symbols are observations,
 * not protected messages. Only correlation metadata is checked by CRC.
 * Never use this codec, its colors, or its error tolerance for authentication. */
import {crc32,randomBytes} from './js/format.js';
import {gridLayout} from './js/grid.js';
import {NQR_LAYOUT} from './js/neural-qr/layout.js';
import {extractNeuralQR} from './js/neural-qr/vision.js';
import {patch} from './js/vision.js';

const sizes=NQR_LAYOUT.sizes;
const requestTag=0xe2,echoTag=0xe3;
const integral=(v,min,max)=>Number.isInteger(v)&&v>=min&&v<=max;
const clock=()=>performance.now();
const layoutCache=new Map();

// Each metadata bit is its own black/white macrocell. Two spatially separate
// copies survive localized reflection; require agreement if both CRCs pass.
// At larger G, tags grow in cells so they retain their apparent angular size.
function layout(size){
  if(!sizes.includes(size))throw Error('Choose G16, G32, G64 or G128.');
  if(layoutCache.has(size))return layoutCache.get(size);
  const scale=Math.max(1,size/32),tagCols=12,tagRows=8;
  const copies=[[0,0],[size-tagCols*scale,size-tagRows*scale]];
  const grid=gridLayout(size,16,8);
  const cells=grid.cells.filter(c=>{const x=c%size,y=Math.floor(c/size);
    return !copies.some(([ox,oy])=>x>=ox&&x<ox+tagCols*scale&&y>=oy&&y<oy+tagRows*scale);});
  const value={scale,copies,grid,cells};layoutCache.set(size,value);return value;
}
function tagBytes(kind,size,phase,run,sequence){
  const bytes=new Uint8Array(12),view=new DataView(bytes.buffer);
  bytes[0]=kind==='request'?requestTag:echoTag;
  bytes[1]=0xb0|(sizes.indexOf(size)<<2)|phase;
  view.setUint32(2,run);view.setUint16(6,sequence);view.setUint32(8,crc32(bytes.subarray(0,8)));return bytes;
}
function tag(frame){
  const {size,phase,symbols,run,sequence,kind}=frame,{copies,scale}=layout(size);
  const bytes=tagBytes(kind,size,phase,run,sequence);
  for(const [ox,oy] of copies)for(let bit=0;bit<96;bit++){
    const v=(bytes[bit>>3]>>(7-(bit&7)))&1;
    const x=ox+(bit%12)*scale,y=oy+Math.floor(bit/12)*scale;
    for(let dy=0;dy<scale;dy++)symbols.fill(v,(y+dy)*size+x,(y+dy)*size+x+scale);
  }
  return frame;
}
function readTag(bytes,size,phase){
  const view=new DataView(bytes.buffer);
  if(![requestTag,echoTag].includes(bytes[0])||bytes[1]!== (0xb0|(sizes.indexOf(size)<<2)|phase)||
    crc32(bytes.subarray(0,8))!==view.getUint32(8))return null;
  const sequence=view.getUint16(6);
  if(!sequence||((sequence-1)&3)!==phase)return null;
  return {kind:bytes[0]===requestTag?'request':'echo',run:view.getUint32(2),sequence};
}
function frame(size,phase,symbols,meta){
  return {size,phase,symbols,colors:16,phases:4,gridBlock:8,pixelsPerCell:4,...meta};
}
export function createTimingTrial({size,run,sequence}){
  if(!integral(run,0,0xffffffff)||!integral(sequence,1,0xffff))throw Error('Invalid matrix timing run or trial.');
  const {grid}=layout(size),symbols=randomBytes(size*size);
  for(let i=0;i<symbols.length;i++)symbols[i]=grid.reference[i]<0?symbols[i]&15:grid.reference[i];
  return tag(frame(size,(sequence-1)&3,symbols,{kind:'request',run,sequence}));
}

const distance=(rgb,ref)=>(rgb[0]-ref[0])**2+(rgb[1]-ref[1])**2+(rgb[2]-ref[2])**2;
function nearest(rgb,refs,n){
  let index=0,best=Infinity;
  for(let i=0;i<n;i++){const d=distance(rgb,refs[i]);if(d<best){index=i;best=d;}}
  return index;
}
// Interpolate the measured left/right references to the sampled column so
// modest illumination gradients do not require recalibration between trials.
function atColumn(refs,x){
  const [left,right]=NQR_LAYOUT.referenceX,t=Math.max(0,Math.min(1,(x-left)/(right-left)));
  return Array.from({length:16},(_,i)=>refs[i].map((v,j)=>v+t*(refs[16+i][j]-v)));
}
export class TimingScanner{
  constructor(){this.reset();}
  reset(){this.geometry=null;}
  scan(image,{now=clock(),captureIntervalMs=100}={}){
    const t=clock(),acq=extractNeuralQR(image,{now,captureIntervalMs,previousGeometry:this.geometry,dense:false,collectTimings:false});
    this.geometry=acq.geometry??null;
    const endDetection=clock(),timings={detectionMs:endDetection-t,decodeMs:0};
    const base={corners:acq.corners??[],tentativeCorners:acq.tentativeCorners??[],info:acq.info??null,timings};
    if(!acq.ok)return {...base,ok:false,stage:acq.stage,error:acq.error};
    try{
      const {size,phase}=acq.info,{copies,scale,cells,grid}=layout(size),box=NQR_LAYOUT.payload,unit=box.side/size,h=acq.transform;
      const refs=Array.from({length:size},(_,x)=>atColumn(acq.observedPalette,box.x+(x+.5)*unit));
      const sample=(x,y,r)=>patch(image,h,box.x+x*unit,box.y+y*unit,r*unit);
      const read=copies.map(([ox,oy])=>{
        const bytes=new Uint8Array(12);
        for(let bit=0;bit<96;bit++){
          const x=ox+(bit%12+.5)*scale,y=oy+(Math.floor(bit/12)+.5)*scale;
          bytes[bit>>3]|=nearest(sample(x,y,.16*scale),refs[Math.floor(x)],2)<<(7-(bit&7));
        }
        return readTag(bytes,size,phase);
      }).filter(Boolean);
      const meta=read[0];
      if(!meta||read.some(v=>v.kind!==meta.kind||v.run!==meta.run||v.sequence!==meta.sequence))
        throw Error('Waiting for a readable roundtrip trial ID.');
      const symbols=new Uint8Array(size*size);
      // The grid pilots are public render decoration, and are not scored.
      // Every scored cell, even one far from every palette color, comes from
      // the camera. An echo never regenerates random challenge data from its ID.
      for(const i of grid.pilots)symbols[i]=grid.reference[i];
      for(const i of cells){const x=i%size,y=Math.floor(i/size);symbols[i]=nearest(sample(x+.5,y+.5,.16),refs[x],16);}
      const physical=tag(frame(size,phase,symbols,meta));timings.decodeMs=clock()-endDetection;
      return {...base,...meta,ok:true,stage:'decoded',frame:physical};
    }catch(e){timings.decodeMs=clock()-endDetection;return {...base,ok:false,stage:'timing-id',error:e.message};}
  }
}

export function makeTimingEcho(observation){
  if(!observation?.ok||observation.kind!=='request')throw Error('Expected a decoded timing request.');
  return tag({...observation.frame,kind:'echo',symbols:observation.frame.symbols.slice()});
}

/** One receiver run, one sender. A held echo remains on the screen while
 * duplicate/self/foreign/old trials are ignored. Restart to pair a new sender. */
export class TimingPeer{
  constructor(){this.reset();}
  reset(){this.run=null;this.sequence=0;}
  accept(observation){
    if(!observation?.ok||observation.kind!=='request'||(this.run!==null&&this.run!==observation.run)||observation.sequence<=this.sequence)return null;
    const echo=makeTimingEcho(observation);this.run=observation.run;this.sequence=observation.sequence;return echo;
  }
}

export function compareTimingEcho(original,observation){
  if(original?.kind!=='request'||!observation?.ok||observation.kind!=='echo'||observation.run!==original.run||
    observation.sequence!==original.sequence||observation.frame.size!==original.size||observation.frame.phase!==original.phase)return null;
  const {cells}=layout(original.size);
  let errors=0;for(const cell of cells)if(original.symbols[cell]!==observation.frame.symbols[cell])errors++;
  return {errors,cells:cells.length,cer:errors/cells.length};
}
