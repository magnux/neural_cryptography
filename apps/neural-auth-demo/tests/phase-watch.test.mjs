import test from 'node:test';
import assert from 'node:assert/strict';
import {PhaseWatch,createPhaseSampler} from '../dist/phase-watch.js';

const W=640,H=480;
const locked={stage:'collect',info:{size:32,phase:0},corners:[[50,50],[300,70],[290,290],[35,260]]};
const stable=pattern(0),different=pattern(1);
function pattern(offset,drift=0){
  const v=new Uint8ClampedArray(12*12*4);
  for(let k=0;k<144;k++)v.set(((k+offset)%2?[160,100,40,255]:[40,100,160,255]).map((x,i)=>i===3?x:x+drift),4*k);
  return v;
}
function before(){const watch=new PhaseWatch({captureMs:300,statistics:true});assert.equal(watch.decide(0,1,()=>assert.fail('no blind ROI'),()=>true),'search');return watch;}
function arm(){const watch=before();watch.observe(locked,W,H,()=>stable);assert.equal(watch.locked,true);return watch;}
function tick(w,t,n,patch){return w.decide(t,n,()=>patch,()=>assert.fail('pace applies only before a lock'));}

test('latches a persistent changed exposure; never captures on the edge itself, an identical callback, or before settling',()=>{
  const w=arm();
  assert.equal(tick(w,40,2,different),null);
  assert.equal(tick(w,65,2,different),null,'same camera exposure');
  assert.equal(tick(w,47,3,different),null);
  assert.equal(tick(w,57,4,different),'change');
  assert.equal(tick(w,100,5,different),null,'not a second full read of the held change');
  const m=w.snapshot();assert.equal(m.settleToCaptureMs.avgMs,17);assert.equal(m.probes,4);assert.equal(m.transitions,1);
});

test('ignores minor exposure drift, isolated outliers and transient spikes; periodic watchdog still reads',()=>{
  const w=arm(),hot=stable.slice();hot.fill(255,0,24);
  for(const [i,p] of [pattern(0,12),hot,different,stable].entries())assert.equal(tick(w,20*(i+1),2+i,p),null);
  assert.equal(tick(w,605,7,stable),'watchdog');assert.equal(w.snapshot().changeReads,0);
  assert.equal(w.snapshot().watchdogReads,1);
  w.observe(locked,W,H,()=>stable);
  assert.equal(tick(w,1605,8,different),null,'even an overdue watchdog gives an observed edge a chance to settle');
  assert.equal(tick(w,1622,9,different),'change');
});

test('validates lock before sampling and falls back to periodic reads when lock or sample is lost',()=>{
  const w=before(),never=()=>assert.fail('discard tentative ROI');
  w.observe({...locked,stage:'profile'},W,H,never);assert.equal(w.locked,false);
  w.observe({...locked,corners:undefined,tentativeCorners:locked.corners},W,H,never);assert.equal(w.locked,false);
  w.observe(locked,W,H,()=>stable);
  const miss={stage:'phase',tentativeCorners:locked.corners,stats:{geometrySource:'previous'}};
  w.observe(miss,W,H,never);assert.equal(w.locked,true);
  w.observe(miss,W,H,never);assert.equal(w.locked,false);
  w.observe(locked,W,H,()=>stable);
  assert.equal(w.decide(100,2,()=>{throw Error('video inaccessible');},()=>true),'search');
  assert.equal(w.locked,false);
  assert.equal(w.snapshot().lostLocks,2);
  assert.equal(w.snapshot().probeFailures,1);
  w.reset();assert.equal(w.snapshot().searchReads,2,'new turn preserves only cumulative stats');
  const out=new PhaseWatch({captureMs:50});assert.equal(out.snapshot(),null);
});

test('reads only the tiny projected cell cluster in unmirrored source coordinates even on a downscaled portrait canvas',()=>{
  const draws=[],gets=[];
  const ctx={drawImage(...args){draws.push(args);},getImageData(...args){gets.push(args);return {data:stable};}};
  const canvas={getContext:()=>ctx};let created=0;
  const sampler=createPhaseSampler(()=>{created++;return canvas;});
  assert.equal(created,0);
  const w=before(),W=400,H=800,video={width:100,height:200,videoWidth:1000,videoHeight:2000};
  // Perspective center = intersection of diagonals, not center of the camera.
  const quad=[[50,200],[300,200],[250,400],[100,400]];
  const result={...locked,corners:quad};
  const geometry=(size)=>{
    let roi=null;w.observe({...result,info:{size,phase:1}},W,H,r=>{roi=r;return sampler.read({width:W,height:H},r);});return roi;
  };
  const g32=geometry(32),g128=geometry(128);
  assert.ok(Math.abs((g32.x+g32.w/2)*400-175)<1e-6);
  assert.ok(Math.abs((g32.y+g32.h/2)*800-325)<1e-6);
  assert.ok(Math.abs(g32.w/g128.w-4)<1e-6,'crop spans a similar number of cells regardless of grid size');
  assert.ok(w.locked);assert.equal(created,1);
  sampler.read(video,g128);
  assert.equal(draws[2][0],video);assert.equal(draws[2][1],g128.x*1000);assert.equal(draws[2][2],g128.y*2000);
  assert.deepEqual(draws[2].slice(5),[0,0,12,12]);assert.deepEqual(gets[2],[0,0,12,12]);
  assert.equal(sampler.read({videoWidth:2000,videoHeight:1000},g128),null,'rotating the camera invalidates the crop');
});
