// Timing DATA only. Retain sparse canonical wire symbols after display or inference.
// No pixels, OLM contents, ground truth, or matching machinery enter the reader.
import {gridLayout} from './js/grid.js';

const cache=new Map();
function positions(size){
  if(![32,64,128].includes(size))throw Error('Sweep samples require G32, G64 or G128.');
  if(cache.has(size))return cache.get(size);
  const {cells}=gridLayout(size,16,8),wire=cells.length-(cells.length%2);
  const witness=[],score=[],n=352;
  // One position per equal-sized stratum in raster-ordered wire cells,
  // jittered with a fixed public sequence used ONLY to select metric positions.
  let state=0xa6bdc19^size;
  for(let i=0;i<n;i++){
    state^=state<<13;state^=state>>>17;state^=state<<5;
    const lo=Math.floor(i*wire/n),hi=Math.floor((i+1)*wire/n);
    const cell=cells[lo+Math.floor((state>>>0)/2**32*(hi-lo))];
    (Math.floor((i+1)*96/n)>Math.floor(i*96/n)?witness:score).push(cell);
  }
  const result={witness,score};cache.set(size,result);return result;
}

export function sweepCellSketch(symbols,size){
  if(!(symbols instanceof Uint8Array)||symbols.length!==size*size)throw Error('Expected canonical sweep symbols.');
  const {witness,score}=positions(size);
  const read=indices=>indices.map(i=>{
    if(symbols[i]>15)throw Error('Invalid sweep symbol.');
    return symbols[i].toString(16);
  }).join('');
  return 's1:'+read(witness)+':'+read(score);
}
