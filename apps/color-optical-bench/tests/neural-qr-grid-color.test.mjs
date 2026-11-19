import test from 'node:test';
import assert from 'node:assert/strict';
import {PALETTE} from '../dist/js/format.js';
import {gridLayout,paletteMap} from '../dist/js/grid.js';
import {NQR_LAYOUT,renderNeuralQR} from '../dist/js/neural-qr/layout.js';
import {applyGridColorCorrection} from '../dist/js/neural-qr/grid-color.js';
import {extractNeuralQR} from '../dist/js/neural-qr/vision.js';

function frame(size=64,phase=2){
  const info={size,phase,colors:16,phases:4,gridBlock:8,permute:true,pixelsPerCell:4};
  const {reference}=gridLayout(size,16,8);
  let seed=91;
  return {...info,symbols:Uint8Array.from(reference,r=>{
    seed=(Math.imul(seed,1664525)+1013904223)>>>0;
    return r>=0?r:paletteMap(seed>>>28,info);
  })};
}
function dense(source){
  const size=source.size,s=4*size,rgb=new Float32Array(3*s*s);
  for(let y=0;y<s;y++)for(let x=0;x<s;x++)rgb.set(PALETTE[source.symbols[(y>>2)*size+(x>>2)]],3*(y*s+x));
  return rgb;
}
test('ideal phase-permuted payload is unchanged, and unusable collapsed grid measurements cannot recolor the data',()=>{
  for(const size of [16,32,64,128]){
    const input=dense(frame(size)),original=input.slice();
    const stats=applyGridColorCorrection(input,size);
    assert.deepEqual(input,original);assert.equal(stats.fallbackChannels,0);
  }
  const size=32,source=frame(size),bad=dense(source),{pilots}=gridLayout(size,16,8);
  for(const cell of pilots)for(let j=0;j<4;j++)for(let i=0;i<4;i++){
    const idx=3*((4*(cell>>5)+j)*(4*size)+4*(cell%size)+i);
    bad.fill(127,idx,idx+3);
  }
  const copy=bad.slice(),stats=applyGridColorCorrection(bad,size);
  assert.equal(stats.fallbackChannels,3*stats.nodes);assert.deepEqual(bad,copy);
});

test('a local tint on a rendered G128 improves non-pilot colors; disabled and non-dense paths are untouched',()=>{
  const size=128,source=frame(size),image=renderNeuralQR(source,2048),p=NQR_LAYOUT.payload;
  const unit=image.width/256,lo=p.x*unit,hi=(p.x+p.side)*unit;
  for(let y=lo;y<hi;y++)for(let x=lo;x<hi;x++){
    // Spatially varying glare/tint affects only part of the payload, not the borders.
    const bump=Math.exp(-8*(((x-lo)/(hi-lo)-.58)**2+((y-lo)/(hi-lo)-.46)**2));
    const i=4*(y*image.width+x);
    image.data[i]=image.data[i]*(1-.22*bump)+54*bump;
    image.data[i+1]=image.data[i+1]*(1-.23*bump)+3*bump;
    image.data[i+2]=image.data[i+2]*(1-.12*bump)+29*bump;
  }
  const base=extractNeuralQR(image,{includeSamples:false});
  assert.ok(base.ok,base.error);
  const coords={previousGeometry:base.geometry,now:base.geometry.acquiredAt,includeSamples:false};
  const disabled=extractNeuralQR(image,{...coords,localColorCorrection:false});
  const local=extractNeuralQR(image,{...coords,localColorCorrection:true});
  assert.deepEqual(base.input,disabled.input);assert.deepEqual(base.info,local.info);
  assert.ok(local.stats.gridColor.pilots>0);assert.equal(base.stats.gridColor,undefined);
  const sparse=extractNeuralQR(image,{...coords,localColorCorrection:true,dense:false});
  assert.ok(sparse.ok);assert.equal(sparse.input,undefined);assert.equal(sparse.stats.gridColor,undefined);
  let baseError=0,localError=0;
  for(const i of gridLayout(size,16,8).cells){
    const q=(4*Math.floor(i/size)+2)*4*size+4*(i%size)+2;
    for(let k=0;k<3;k++){
      const expected=PALETTE[source.symbols[i]][k]/127.5-1;
      const offset=k*(4*size)**2+q;
      baseError+=(base.input[offset]-expected)**2;localError+=(local.input[offset]-expected)**2;
    }
  }
  assert.ok(localError<.30*baseError,`Grid/global held-out cell MSE ratio = ${localError/baseError}`);
});

test('changing arbitrary data cells leaves the correction on the rest of the crop identical',()=>{
  const size=32,src=frame(size),layout=gridLayout(size,16,8),rgb=dense(src),s=4*size;
  for(let i=0;i<rgb.length;i++)rgb[i]=rgb[i]*.85+25;
  // Isolated faulty pilots do not dominate a fit. The algorithm never uses the identity of a data cell.
  for(const c of layout.pilots)if(c%73===0){
    for(let dy=0;dy<4;dy++)for(let dx=0;dx<4;dx++){
      const p=3*((4*Math.floor(c/size)+dy)*s+4*(c%size)+dx);rgb.fill(0,p,p+3);
    }
  }
  const other=rgb.slice();
  for(const c of layout.cells)if(c%2)for(let dy=0;dy<4;dy++)for(let dx=0;dx<4;dx++){
    const p=3*((4*Math.floor(c/size)+dy)*s+4*(c%size)+dx);other.fill(255,p,p+3);
  }
  applyGridColorCorrection(rgb,size);applyGridColorCorrection(other,size);
  let worst=0;
  for(const c of layout.cells)if(c%2===0){
    const p=3*((4*Math.floor(c/size)+2)*s+4*(c%size)+2);
    for(let k=0;k<3;k++){
      assert.equal(rgb[p+k],other[p+k]);
      worst=Math.max(worst,Math.abs(rgb[p+k]-PALETTE[src.symbols[c]][k]));
    }
  }
  assert.ok(worst<12,`worst channel error after sparse pilot corruption: ${worst}`);
});
