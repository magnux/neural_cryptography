/** Neural-QR acquisition layout. Packet framing is a separate, in-band layer. */
import {PALETTE} from '../format.js';

export const NQR_LAYOUT = Object.freeze({
  name:'neural-qr',version:2,units:256,sizes:Object.freeze([16,32,64,128]),
  centers:Object.freeze([[22,22],[234,22],[234,234],[22,234]].map(Object.freeze)),
  payload:Object.freeze({x:44,y:44,side:168}),
  referenceX:Object.freeze([22,234]),referenceY:52,referenceStep:10,
  // The lower-right finder retains its localization core. These satellites
  // give it handedness: transposition changes the two unequal bit sequences.
  orientation:Object.freeze([[214,226,1],[214,234,0],[214,242,0],
    [226,214,1],[234,214,1],[242,214,0]].map(Object.freeze)),
  // Eight coarse black/white patches encode the grid independently of payload
  // color and phase. Balanced codewords have pairwise Hamming distance four.
  // Copies occupy opposite free border regions, outside every existing cue.
  sizeCodes:Object.freeze(['11110000','11001100','10101010','10011001']),
  sizePatchSide:8,
  sizeCopies:Object.freeze([
    [16,32].flatMap(y=>[48,60,72,84].map(x=>Object.freeze([x,y]))),
    [224,240].flatMap(y=>[160,172,184,196].map(x=>Object.freeze([x,y]))),
  ].map(Object.freeze)),
  phasePairs:Object.freeze([[[104,20],[132,20]],[[104,236],[132,236]]]
    .map(pair=>Object.freeze(pair.map(Object.freeze)))),
});

export function validateNeuralQRProfile(profile){
  if(!profile||!NQR_LAYOUT.sizes.includes(profile.size)||profile.colors!==16||
    profile.phases!==4||profile.gridBlock!==8||profile.pixelsPerCell!==undefined&&profile.pixelsPerCell!==4)
    throw Error('Neural-QR acquisition requires G16/32/64/128, 16 colors, four phases, grid 8 and 4×4 pixels/cell.');
  return {size:profile.size,colors:16,bits:4,phases:4,gridBlock:8,permute:true,pixelsPerCell:4};
}

/** Symbols are physical palette indices, already permuted by the sender. */
export function renderNeuralQR(frame,pixels=1024){
  const profile=validateNeuralQRProfile(frame);
  if(!Number.isInteger(frame.phase)||frame.phase<0||frame.phase>3||
    !(frame.symbols instanceof Uint8Array)||frame.symbols.length!==profile.size**2||frame.symbols.some(v=>v>15))
    throw Error('Invalid Neural-QR symbols or phase.');
  if(!Number.isInteger(pixels)||pixels<256||pixels>4096)throw Error('Render size must be 256..4096.');
  const data=new Uint8ClampedArray(pixels*pixels*4);data.fill(255);const scale=pixels/256;
  const rect=(x,y,w,h,color)=>{
    for(let yy=Math.round(y*scale);yy<Math.round((y+h)*scale);yy++)
      for(let xx=Math.round(x*scale);xx<Math.round((x+w)*scale);xx++){
        const i=4*(yy*pixels+xx);data[i]=color[0];data[i+1]=color[1];data[i+2]=color[2];
      }
  };
  const black=[16,16,16],white=[248,248,248];
  for(const [x,y] of NQR_LAYOUT.centers){rect(x-14,y-14,28,28,black);rect(x-10,y-10,20,20,white);rect(x-6,y-6,12,12,black);}
  for(const [x,y,bit] of NQR_LAYOUT.orientation)rect(x-3,y-3,6,6,bit?black:white);
  const sizeCode=NQR_LAYOUT.sizeCodes[NQR_LAYOUT.sizes.indexOf(profile.size)],patchSide=NQR_LAYOUT.sizePatchSide;
  for(const copy of NQR_LAYOUT.sizeCopies)for(const [i,[x,y]] of copy.entries())
    rect(x-patchSide/2,y-patchSide/2,patchSide,patchSide,sizeCode[i]==='1'?black:white);
  for(const x of NQR_LAYOUT.referenceX)for(let i=0;i<16;i++)rect(x-8,48+10*i,16,8,PALETTE[i]);
  for(const pair of NQR_LAYOUT.phasePairs)pair.forEach(([x,y],i)=>rect(x-10,y-10,20,20,PALETTE[(frame.phase+i)%4]));
  const p=NQR_LAYOUT.payload,cell=p.side/profile.size;
  for(let y=0;y<profile.size;y++)for(let x=0;x<profile.size;x++)
    rect(p.x+x*cell,p.y+y*cell,cell,cell,PALETTE[frame.symbols[y*profile.size+x]]);
  return {width:pixels,height:pixels,data};
}
