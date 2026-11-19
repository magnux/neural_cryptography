import test from 'node:test';
import assert from 'node:assert/strict';
import {PhaseWatch} from '../dist/phase-watch.js';
import {renderNeuralQR} from '../dist/js/neural-qr/layout.js';
import {extractNeuralQR} from '../dist/js/neural-qr/vision.js';
import {gridLayout,paletteMap} from '../dist/js/grid.js';

// Feed the real layout, pilot lines and phase masks to the scheduler. This
// samples a clean stationary image, not a claim about clocks or camera noise.
function phases(size) {
  const {cells,reference}=gridLayout(size,16,8);
  const logical=new Uint8Array(reference);
  let state=7;
  for(const i of cells){state=(Math.imul(1664525,state)+1013904223)>>>0;logical[i]=state>>>28;}
  return [0,1,2,3].map(phase=>{
    const profile={size,colors:16,phases:4,phase,gridBlock:8,permute:true};
    const symbols=logical.slice();for(const i of cells)symbols[i]=paletteMap(symbols[i],profile);
    return renderNeuralQR({...profile,symbols},512);
  });
}
function samplePatch(image,{x,y,w,h}) {
  const out=new Uint8ClampedArray(12*12*4);
  for(let j=0;j<12;j++)for(let i=0;i<12;i++){
    const src=(Math.floor((y+(j+.5)*h/12)*image.height)*image.width+Math.floor((x+(i+.5)*w/12)*image.width))*4;
    out.set(image.data.subarray(src,src+4),(j*12+i)*4);
  }
  return out;
}
for(const size of [32,64,128])test('phase watch wakes on each rendered transition, including 3 → 0, G'+size,()=>{
  const images=phases(size),watch=new PhaseWatch({captureMs:300,statistics:true});
  let stamp=0,frame=0;
  const present=p=>watch.decide(stamp+=17,++frame,roi=>samplePatch(images[p],roi),()=>true);
  const observe=p=>{
    const image=images[p],r=extractNeuralQR(image,{beforeRaster:()=>false});
    assert.equal(r.info?.phase,p,r.error);
    watch.observe(r,image.width,image.height,roi=>samplePatch(image,roi));
    assert.equal(watch.locked,true);
  };
  assert.equal(present(0),'search');observe(0);
  for(const p of [1,2,3,0]){
    assert.equal(present((p+3)%4),null,'unchanged matrix should not wake a full read');
    assert.equal(present(p),null,'transition frame is not the settled frame');
    assert.equal(present(p),'change');observe(p);
  }
  assert.equal(watch.snapshot().changeReads,4);
  assert.equal(watch.snapshot().watchdogReads,0);
});
