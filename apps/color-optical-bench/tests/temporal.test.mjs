import test from 'node:test';
import assert from 'node:assert/strict';
import {Worker} from 'node:worker_threads';
import {encodeSequence,phaseFrames,renderFrame,capacity,Assembler} from '../dist/js/format.js';
import {extractCapture,simulateCapture} from '../dist/js/vision.js';
import {TemporalDecoder,filterSamples} from '../dist/js/temporal.js';
const id=Uint8Array.of(1,2,3,4,5,6,7,8);
function make(options={}){const o={size:32,colors:16,gridBlock:8,parity:32,phases:2,permute:true,...options};return encodeSequence(Uint8Array.from({length:capacity(o.size,o.colors,o)},(_,i)=>(i*47+91)&255),{...o,id})[0];}
const values=f=>extractCapture(renderFrame(f,768),{preview:false}).samples;

test('mean and median accept byte evidence; a median rejects an outlier',()=>{
  const good=Uint8Array.of(220,12,11,12),bad=Uint8Array.of(0,0,0,255);
  assert.deepEqual(Array.from(filterSamples([good,bad,good],'median')),Array.from(good));
  assert.deepEqual(Array.from(filterSamples([Uint8Array.of(10,20),Uint8Array.of(30,40)],'mean')),[20,30]);
  assert.throws(()=>filterSamples([new Uint8Array(1),new Uint8Array(2)],'mean'));
});
test('all phases must arrive; repeated captures of one phase do not impersonate missing phases',()=>{
  const fs=phaseFrames([make({phases:4})]),decoder=new TemporalDecoder(),v=fs.map(values);let result;
  for(let i=0;i<8;i++)result=decoder.process(fs[0],v[0],{filter:'mean',window:3});
  assert.equal(result.ok,false);assert.deepEqual(result.stats.temporal.counts,[3,0,0,0]);
  for(const phase of [3,1,2])for(let i=0;i<3;i++)result=decoder.process(fs[phase],v[phase],{filter:'mean',window:3});
  assert.equal(result.ok,true,result.error);assert.deepEqual(result.chunk,fs[0].chunk);assert.deepEqual(result.stats.temporal.counts,[3,3,3,3]);
});
test('out-of-order data frames and phases remain separate and assemble exact bytes',()=>{
  const o={size:32,colors:4,gridBlock:8,parity:16,phases:2,permute:true,maxFrames:2,id},bytes=Uint8Array.from({length:capacity(32,4,o)*2},(_,i)=>i),frames=phaseFrames(encodeSequence(bytes,o)),decoder=new TemporalDecoder(),a=new Assembler();
  for(const f of [frames[3],frames[0],frames[2],frames[1]]){const r=decoder.process(f,values(f),{filter:'none'});if(r.ok)a.accept(f,r.chunk);}
  assert.deepEqual(a.complete,bytes);
});
test('memory caps reduce per-phase targets, expire stale phases, and evict old frame groups',()=>{
  const fs=phaseFrames([make()]),entry={global:new Uint8Array(128),local:new Uint8Array(128)};let now=0;
  const d=new TemporalDecoder({maxBytes:1024,maxAgeMs:10,clock:()=>now});
  let g=d.accumulate(fs[0],entry,9);assert.equal(g.target,2);
  d.accumulate(fs[1],entry,9);now=11;g=d.accumulate(fs[0],entry,9);assert.deepEqual(g.phases.map(p=>p.length),[1,0]);
  for(let i=0;i<9;i++)d.accumulate({...fs[0],index:i},entry,9);
  assert.ok(d.bytes<=1024);assert.ok(d.evicted>0);
  assert.throws(()=>new TemporalDecoder({maxBytes:1}).accumulate(fs[0],entry,3),/budget/);
});
test('different sequences require explicit reset and no old evidence survives',()=>{
  const f=make(),other={...f,id:'different'},d=new TemporalDecoder(),v=values(f);d.process(f,v,{filter:'mean',window:3});
  assert.throws(()=>d.process(other,v,{filter:'mean',window:3}),/Different sequence/);d.reset();
  assert.deepEqual(d.process(other,v,{filter:'mean',window:3}).stats.temporal.counts,[1,0]);
});
test('bad headers never contribute; phase changes respect the transition guard',()=>{
  let now=0;const fs=phaseFrames([make()]),d=new TemporalDecoder({clock:()=>now}),image=renderFrame(fs[0],512);
  for(let y=16;y<112;y++)for(let x=96;x<416;x++)image.data.set([248,248,248,255],4*(y*512+x));
  assert.equal(d.decode(image,{settleMs:300,preview:false}).info,undefined);assert.equal(d.bytes,0);
  const options={settleMs:300,filter:'none',preview:false};assert.equal(d.decode(renderFrame(fs[0],512),options).stage,'settle');
  now=301;assert.deepEqual(d.decode(renderFrame(fs[0],512),options).stats.temporal.counts,[1,0]);
  assert.equal(d.decode(renderFrame(fs[1],512),options).stage,'settle');now=602;
  assert.equal(d.decode(renderFrame(fs[1],512),options).ok,true);
});
test('one photograph can decode a self-contained phase without pretending to fuse it',()=>{
  const fs=phaseFrames([make()]),r=new TemporalDecoder().decode(renderFrame(fs[1]),{singlePhase:true,preview:false});
  assert.equal(r.ok,true);assert.deepEqual(r.chunk,fs[0].chunk);assert.equal(r.colorComparison.fusedLocal,undefined);
});
test('drifting curved moire and blur are reproducible and independent of geometry',()=>{
  const source=renderFrame(make(),512),o={width:320,height:253,noise:12,moire:44,moirePeriod:16,curvature:1,colorDependence:1,seed:1234,phase:1.1,blur:.6};
  const a=simulateCapture(source,o);assert.deepEqual(a,simulateCapture(source,o));
  assert.notDeepEqual(a.data,simulateCapture(source,{...o,phase:2.2}).data);assert.notDeepEqual(a.data,simulateCapture(source,{...o,blur:0}).data);
});
test('worker reset drops phase evidence; separate workers retain independent state',async()=>{
  const worker=new Worker(new URL('./worker-harness.mjs',import.meta.url)),fs=phaseFrames([make()]);let id=0;
  const run=f=>new Promise((resolve,reject)=>{worker.once('message',data=>resolve(data.result));worker.once('error',reject);worker.postMessage({id:++id,epoch:1,image:renderFrame(f,512),options:{filter:'none',preview:false}});});
  try{assert.equal((await run(fs[0])).ok,false);worker.postMessage({type:'reset'});let r=await run(fs[1]);assert.equal(r.ok,false);assert.deepEqual(r.stats.temporal.counts,[0,1]);r=await run(fs[0]);assert.equal(r.ok,true,r.error);}finally{await worker.terminate();}
});
test('512/16 four-phase retained-evidence budget reduces a nine-sample request predictably',()=>{
  const f=make({size:512,phases:4}),n=206116*16,d=new TemporalDecoder(),entry={global:new Uint8Array(n),local:new Uint8Array(n)};
  for(let phase=0;phase<4;phase++)for(let i=0;i<3;i++){const group=d.accumulate({...f,phase},entry,9);assert.equal(group.target,2);}
  assert.ok(d.bytes<=64*1024*1024);assert.deepEqual([...d.groups.values()][0].phases.map(p=>p.length),[2,2,2,2]);
});
test('a valid border candidate delivers content when noisy internal references mislead the local model',()=>{
  const f=make({phases:1,permute:false,parity:0}),v=values(f),d=new TemporalDecoder();
  for(let y=0;y<f.size;y++)for(let x=0;x<f.size;x++)if(x%9===0||y%9===0||x===f.size-1||y===f.size-1){
    const cell=y*f.size+x;v[cell*3]=255-v[cell*3];v[cell*3+1]=255-v[cell*3+1];v[cell*3+2]=255-v[cell*3+2];
  }
  const r=d.process(f,v,{filter:'none'});assert.equal(r.colorComparison.fusedLocal.ok,false);assert.equal(r.colorComparison.fusedGlobal.ok,true);
  assert.equal(r.ok,true);assert.equal(r.stats.acceptedDecoder,'global');assert.deepEqual(r.chunk,f.chunk);
});

test('16-phase progressive fusion starts after four distinct ready phases and keeps refining',()=>{
  const fs=phaseFrames([make({size:16,phases:16})]),d=new TemporalDecoder(),options={filter:'robust',window:3,overhead:2,minPhases:4};let r;
  for(let phase=0;phase<5;phase++){
    const v=values(fs[phase]);
    for(let i=0;i<3;i++)r=d.process(fs[phase],v,options);
    assert.equal(r.ok,phase>=3,r.error);
    assert.equal(r.stats.temporal.eligiblePhases.length,phase+1);
  }
  assert.deepEqual(r.chunk,fs[0].chunk);assert.equal(r.stats.temporal.poolCapacity,5);
  assert.equal(r.stats.temporal.robust.local.length,5);assert.equal(r.colorComparison.robustLocal.ok,true);
  assert.equal(r.colorComparison.fusedLocal.ok,true);assert.equal(r.symbols,undefined);assert.equal(r.scores,undefined);
});
test('a full robust pool continues replacing a wrong consensus until exact content recovers',()=>{
  const f=make({size:16,phases:1,permute:false,parity:0}),good=values(f),bad=good.slice(),d=new TemporalDecoder();
  // Damage data only; references are still independently measured. No truth is
  // provided to the decoder; the expected bytes only assert the final result.
  for(let y=0;y<f.size;y++)for(let x=0;x<f.size;x++)if(x%9&&y%9&&x!==f.size-1&&y!==f.size-1){
    const j=(y*f.size+x)*3;bad[j]=255-good[j];bad[j+1]=255-good[j+1];bad[j+2]=255-good[j+2];
  }
  const options={filter:'robust',window:5,overhead:2};let r;
  for(let i=0;i<7;i++)r=d.process(f,bad,options);
  assert.equal(r.ok,false);assert.equal(r.stats.temporal.ready,true);
  for(let i=0;i<7;i++)r=d.process(f,good,options);
  assert.equal(r.ok,true,r.error);assert.deepEqual(r.chunk,f.chunk);
  assert.deepEqual(r.stats.temporal.counts,[7]);assert.equal(r.stats.temporal.target,5);
  assert.equal(r.stats.temporal.robust.local[0].kept,5);
});
test('header cadence retains early phases through long cycles, then expires individual samples',()=>{
  const f={...make(),phases:16,holdMs:4000},entry={global:new Uint8Array(32),local:new Uint8Array(32)};let now=0;
  const d=new TemporalDecoder({clock:()=>now});d.accumulate({...f,phase:0},entry,5);
  now=65000;let g=d.accumulate({...f,phase:15},entry,5);
  assert.equal(g.ageLimitMs,128000);assert.equal(g.phases[0].length,1);
  now=128001;g=d.accumulate({...f,phase:1},entry,5);
  assert.equal(g.phases[0].length,0);assert.equal(g.phases[15].length,1);
});
test('large 16-phase profiles retain a bounded subset and report the reduced pool explicitly',()=>{
  const f={...make({size:512,phases:4}),phases:16},n=206116*16,d=new TemporalDecoder(),entry={global:new Uint8Array(n),local:new Uint8Array(n)};let g;
  for(let phase=0;phase<16;phase++)for(let i=0;i<3;i++){
    g=d.accumulate({...f,phase},entry,5,{overhead:2,minPhases:4});assert.ok(d.bytes<=64*1024*1024);
  }
  assert.equal(g.requestedPool,7);assert.equal(g.poolCapacity,2);assert.equal(g.target,2);assert.equal(g.retainedPhaseLimit,5);
  assert.equal(g.phases.filter(p=>p.length).length,5);assert.ok(d.evicted>0);
  assert.throws(()=>new TemporalDecoder({maxBytes:n*2*3}).accumulate(f,entry,5,{minPhases:4}),/budget/);
});
test('receiver rejects invalid refinement settings before accepting evidence',()=>{
  const f=make(),d=new TemporalDecoder(),v=values(f);
  for(const options of [{overhead:3},{window:10},{minPhases:3},{recovery:'guess'},{filter:'legacy'}])assert.throws(()=>d.process(f,v,options),/settings/);
  assert.equal(d.bytes,0);
});

test('different capture counts cannot give one phase more fusion weight',()=>{
  const fs=phaseFrames([make({size:16})]),vs=fs.map(values),settings={filter:'robust',window:3,overhead:2},a=new TemporalDecoder(),b=new TemporalDecoder();let ra,rb;
  for(let phase=0;phase<2;phase++)for(let i=0;i<3;i++)ra=a.process(fs[phase],vs[phase],settings);
  for(let phase=0;phase<2;phase++)for(let i=0;i<(phase===0?5:3);i++)rb=b.process(fs[phase],vs[phase],settings);
  assert.deepEqual(ra.stats.temporal.counts,[3,3]);assert.deepEqual(rb.stats.temporal.counts,[5,3]);
  for(const name of ['fusedGlobal','fusedLocal','robustGlobal','robustLocal']){
    assert.ok(Math.abs(ra.colorComparison[name].stats.confidence-rb.colorComparison[name].stats.confidence)<1e-6);
    assert.equal(ra.colorComparison[name].ok,rb.colorComparison[name].ok);
  }
  assert.deepEqual(ra.chunk,rb.chunk);
});
