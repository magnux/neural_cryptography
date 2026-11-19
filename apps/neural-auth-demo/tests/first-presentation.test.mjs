import test from 'node:test';
import assert from 'node:assert/strict';
import {FirstPresentation} from '../dist/first-presentation.js';

function clock(){
  let t=0,id=0;const frames=new Map(),timers=new Map();
  const scheduler=new FirstPresentation({now:()=>t,raf:fn=>{frames.set(++id,fn);return id;},
    cancelRAF:key=>frames.delete(key),defer:(fn,delay)=>{timers.set(++id,{fn,due:t+delay});return id;},
    cancelDefer:key=>timers.delete(key)});
  const refresh=time=>{
    t=time;
    for(const [key,entry] of [...timers])if(entry.due<=t){timers.delete(key);entry.fn();}
    // Browser rAF requests made during a refresh run on a later refresh.
    for(const [key,fn] of [...frames]){frames.delete(key);fn(t);}
    assert.ok(frames.size+timers.size<=1,'at most one outstanding display callback');
  };
  return {scheduler,refresh,frames,timers};
}

test('slow paints cannot skip a phase, backfill a missed deadline or release the camera in the phase-3 draw',()=>{
  const {scheduler:q,refresh}=clock(),seen=[],releases=[];
  q.start({phaseMs:100,paint:(i,first)=>seen.push([i,first]),onReady:()=>releases.push(q.holding)});
  assert.equal(q.holding,true);refresh(10);refresh(96);refresh(110);refresh(510);refresh(526);
  assert.deepEqual(seen,[[0,true],[1,false],[2,false]]);
  refresh(610);assert.equal(q.holding,true);assert.deepEqual(releases,[]);
  refresh(627);assert.equal(q.holding,false);assert.deepEqual(releases,[false]);
  refresh(710);refresh(810);refresh(910);refresh(1010);refresh(1027);
  assert.deepEqual(seen.map(([i])=>i),[0,1,2,3,0,1,2,3]);
  assert.deepEqual(releases,[false],'repetitions of the same packet never re-arm the gate');
  q.clear();
});

test('replacement or blank invalidates queued paints and credit from the old packet',()=>{
  const {scheduler:q,refresh,frames,timers}=clock(),seen=[];
  q.start({phaseMs:50,paint:()=>{},onReady:()=>assert.fail('stale release')});
  refresh(1);refresh(51);refresh(101);refresh(151);
  const oldRelease=[...frames.values()][0];assert.equal(typeof oldRelease,'function');
  q.start({phaseMs:50,paint:(i,first)=>seen.push([i,first])});oldRelease();
  assert.equal(q.holding,true);refresh(168);assert.deepEqual(seen,[[0,true]]);
  const oldTimer=[...timers.values()][0].fn;q.clear();oldTimer();
  assert.equal(frames.size+timers.size,0);assert.equal(q.holding,false);
});

test('progressive dwell increases per four paints, caps, and starts over on a new packet',()=>{
  const {scheduler:q,refresh}=clock(),seen=[];
  const start=()=>q.start({phaseMs:100,incrementMs:25,maxPhaseMs:140,
    paint:(i,first,dwell,round)=>seen.push([i,first,dwell,round])});
  start();[0,100,200,300,320,399].forEach(refresh);
  assert.deepEqual(seen,[[0,true,100,1],[1,false,100,1],[2,false,100,1],[3,false,100,1]]);
  assert.equal(q.holding,false);
  [400,500,525,650,775,899].forEach(refresh);
  assert.deepEqual(seen.slice(4),[[0,false,125,2],[1,false,125,2],[2,false,125,2],[3,false,125,2]]);
  [900,1040,1180,1320,1460].forEach(refresh);
  assert.deepEqual(seen.slice(8),[[0,false,140,3],[1,false,140,3],[2,false,140,3],[3,false,140,3],[0,false,140,4]]);
  start();refresh(1477);assert.equal(q.holding,true);assert.deepEqual(seen.at(-1),[0,true,100,1]);q.clear();
});

test('stopping during drawing or release leaves no live paint or timer; paint failures do not claim a presentation',()=>{
  const {scheduler:q,refresh,frames,timers}=clock(),seen=[];
  q.start({phaseMs:50,paint:()=>q.clear()});refresh(1);
  assert.equal(frames.size+timers.size,0);
  q.start({phaseMs:50,paint:i=>seen.push(i),onReady:()=>q.clear()});
  [10,60,110,160,177].forEach(refresh);
  assert.deepEqual(seen,[0,1,2,3]);assert.equal(frames.size+timers.size,0);
  let cause;q.start({phaseMs:50,paint:()=>{throw Error('lost canvas');},
    onReady:()=>assert.fail('a failed draw did not complete a pass'),onError:e=>{cause=e;}});
  refresh(200);assert.match(cause.message,/lost canvas/);assert.equal(q.holding,false);assert.equal(frames.size+timers.size,0);
});
