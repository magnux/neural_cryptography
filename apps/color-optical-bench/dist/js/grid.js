/** Shared geometry only: an N×N optical grid includes its reference cells. */
const cache=new Map();
export function gridLayout(size,colors,gridBlock=0){
  if(!Number.isInteger(size)||size<8||size>512||![2,4,8,16].includes(colors)||![0,4,8,16].includes(gridBlock))throw Error('Invalid grid profile.');
  const key=`${size}/${colors}/${gridBlock}`;if(cache.has(key))return cache.get(key);
  const reference=new Int8Array(size*size);reference.fill(-1);const cells=[],pilots=[],step=gridBlock+1;
  for(let y=0;y<size;y++)for(let x=0;x<size;x++){
    const cell=y*size+x,horizontal=gridBlock&&(y%step===0||y===size-1),vertical=gridBlock&&(x%step===0||x===size-1);
    if(horizontal||vertical){const color=(horizontal?x+5*Math.floor(y/step):y+7*Math.floor(x/step))&(colors-1);reference[cell]=color;pilots.push(cell);}
    else cells.push(cell);
  }
  const layout={size,colors,gridBlock,reference,cells:Uint32Array.from(cells),pilots:Uint32Array.from(pilots)};
  cache.set(key,layout);if(cache.size>8)cache.delete(cache.keys().next().value);return layout;
}
const phaseMasks=Object.freeze({2:[0,1],4:[0,3,1,2],8:[0,7,3,4,1,6,2,5],16:[0,15,7,8,3,12,4,11,1,14,6,9,2,13,5,10]});
/** Nested public XOR mappings; a complete palette cycle visits every color once. */
export function paletteMap(symbol,info){
  if(!info.permute||!info.phase)return symbol;
  const masks=phaseMasks[info.colors];return symbol^masks[info.phase%masks.length];
}
